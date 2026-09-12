"""Export DAF-admitted NOAA water-level observations to data/daf/.

This repository consumes DAF evidence; it never re-derives it. The tool runs DAF's
OWN acquisition path, unmodified -- the per-measurement NOAA binding
(`daf.orchestration.bindings.noaa_water_level_measurement_binding`: DAF's
`NoaaWaterLevelSourceAdapter` paired with `NoaaWaterLevelMeasurementExtractor`),
`daf.scheduling.runner.execute_plan`, SCOUT admission into a `DurablePool` -- exactly
as DAF's own tests/test_live_scientific_observation.py does, replaying DAF's committed
fixture bytes through the adapter's own `fetch_bytes` injection point, so no request
leaves the machine. Every admitted Observation is serialised with DAF's own
`daf.storage.serialization.observation_to_dict` and re-read with DAF's
`strict_json_loads` + `observation_from_dict`, which recomputes its content-addressed
id before anything is written.

    DAF_ROOT=/path/to/daf uv run --python 3.13 python tools/export_daf_fixtures.py [--out DIR]

DAF_ROOT must be a Data Acquisition Fabric checkout with its vendored substrate
initialised (git submodule update --init). What DAF's conftest.py sets up -- `import
daf`, which puts vendor/scout-retrieval-agent on sys.path (daf/_vendor.py) -- is
replicated here; nothing in DAF_ROOT is modified: bytecode writing is switched off
before DAF is imported, the fixture bytes are read from DAF's git object store (the
committed blob, independent of the checkout's line-ending conversion), and the
evidence store and checkpoints live in a temporary directory removed afterwards. The
tool refuses a DAF checkout whose tracked files differ from its HEAD, or whose
vendored substrate is not at the commit DAF pins.

Outputs (default --out data/daf): one `<name>.observations.json` per fixture -- a JSON
array of DAF observation dicts, one per line, sorted by (measurement_time, id) -- plus
manifest.json and PROVENANCE.md. The manifest also counts, per fixture, NOAA's revision
flag `q` and QC flag vector `f` as the raw bytes carry them: DAF's extractor keeps both out
of Observation.content by design (revision and acquisition metadata), so they are recorded
as provenance of the raw artifact only, and nothing on the bridging path reads them.
Files prefixed SYNTHETIC_ come from DAF's synthetic test fixtures (station 9999999, "NOT A
REAL NOAA STATION") and exist only to exercise revision conflicts. Re-running against the same DAF commit reproduces every output
byte for byte (nothing wall-clock is written).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "daf"
DAF_URL = "https://github.com/atomtrapping/Data-Acquisition-Channel"
VENDOR_REL = "vendor/scout-retrieval-agent"
SOURCE_ID = "noaa-water-level-measurements"   # the operator's DAF source id; not part of any evidence id
PRODUCT = "water_level"


@dataclass(frozen=True)
class Job:
    fixture: str        # path inside DAF_ROOT
    out: str            # output file name
    station: str
    start_date: str     # the DAF plan's parameters (YYYYMMDD); the adapter derives the window
    end_date: str
    datum: str
    units: str
    requested_at: str   # DAF's caller-supplied retrieval stamp; run_scout copies it to extracted_at
    synthetic: bool
    note: str


JOBS = (
    Job("tests/fixtures/noaa_live_8454000_20240115_mllw.json", "noaa_live_8454000_20240115_mllw.observations.json",
        "8454000", "20240115", "20240115", "MLLW", "metric", "2026-08-25T00:00:00Z", False,
        "Real NOAA CO-OPS response, station 8454000 (Providence, RI), 2024-01-15, verified (q=v), recorded "
        "verbatim from the live API by DAF (docs/PHASE_17_LIVE_SCIENTIFIC_OBSERVATION.md, section 7: "
        "datum=MLLW units=metric time_zone=gmt)."),
    Job("tests/fixtures/noaa_live_8454000_20240115_stnd.json", "noaa_live_8454000_20240115_stnd.observations.json",
        "8454000", "20240115", "20240115", "STND", "metric", "2026-08-25T00:00:00Z", False,
        "Real NOAA CO-OPS response, same station and day on the STND datum (DAF Phase R's datum case, "
        "docs/PHASE_18_NOAA_ARTIFACT_IDENTITY.md): a different physical quantity, never pooled with MLLW."),
    Job("tests/fixtures/noaa_live_8454000_preliminary.json", "noaa_live_8454000_preliminary.observations.json",
        "8454000", "20260823", "20260823", "MLLW", "metric", "2026-08-25T00:00:00Z", False,
        "Real NOAA CO-OPS response, station 8454000, 2026-08-23, preliminary (q=p). DAF does not record this "
        "recording's request; the readings' own timestamps fix the day, and MLLW/metric is the binding's "
        "default and the datum DAF's own test extracts this fixture with (tests/test_live_scientific_"
        "observation.py, _extract)."),
    Job("tests/fixtures/noaa_window_synthetic_20260101_20260103.json",
        "SYNTHETIC_noaa_window_20260101_20260103.observations.json",
        "9999999", "20260101", "20260103", "MLLW", "metric", "2026-08-25T00:00:00Z", True,
        "SYNTHETIC. DAF's hand-written test window (station 9999999, 'SYNTHETIC TEST STATION -- NOT A REAL "
        "NOAA STATION'), four readings; requested_at as in DAF's tests/test_noaa_water_level_integration.py."),
    Job("tests/fixtures/noaa_window_synthetic_20260101_20260103_revised.json",
        "SYNTHETIC_noaa_window_20260101_20260103_revised.observations.json",
        "9999999", "20260101", "20260103", "MLLW", "metric", "2026-08-26T00:00:00Z", True,
        "SYNTHETIC. The same window re-fetched after a (synthetic) QC revision: the 2026-01-03 00:00 reading "
        "moves from 1.200 (s 0.011, q=p) to 1.207 (s 0.006, q=v); the other three are unchanged. "
        "requested_at a day later, as in DAF's integration test."),
)


def _git(root: Path, *args: str, binary: bool = False):
    out = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True, timeout=60).stdout
    return out if binary else out.decode("utf-8").strip()


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _daf_root() -> Path:
    raw = os.environ.get("DAF_ROOT")
    if not raw:
        raise SystemExit("DAF_ROOT is not set: point it at a Data Acquisition Fabric checkout "
                         f"({DAF_URL}) with its submodule initialised")
    root = Path(raw).resolve()
    for need in ("daf/__init__.py", "daf/extractors/noaa_water_level_measurements.py",
                 f"{VENDOR_REL}/evidence/types.py"):
        if not (root / need).is_file():
            raise SystemExit(f"DAF_ROOT={root} is not a DAF checkout with its vendored substrate: missing {need}")
    return root


def _published_base(root: Path, head: str) -> dict:
    """Which part of the exporting checkout's history is PUBLISHED, and which is not.

    DAF's default branch is the published history. A checkout carrying local commits -- the
    USGS daily-values adapter lives on one, and by the user's standing instruction is never
    pushed to DAF -- would otherwise stamp every output with a commit id no one else can
    resolve. So the manifest records the published ancestor as well, and names the local
    commits and the patch series that carries them into THIS repository.
    """
    refs = [r for r in _git(root, "for-each-ref", "--format=%(refname)", "refs/remotes/origin").splitlines()
            if r and not r.endswith("/HEAD")]
    published = None
    for ref in refs:
        try:
            _git(root, "merge-base", "--is-ancestor", head, ref)
            return {"published": True, "published_base": head, "published_ref": ref.split("/", 3)[-1],
                    "local_commits": []}
        except subprocess.CalledProcessError:
            continue
    for ref in refs:
        try:
            base = _git(root, "merge-base", head, ref)
        except subprocess.CalledProcessError:
            continue
        local = _git(root, "log", "--format=%H %s", f"{base}..{head}").splitlines()
        if published is None or len(local) < len(published["local_commits"]):
            published = {"published": False, "published_base": base, "published_ref": ref.split("/", 3)[-1],
                         "local_commits": [{"commit": ln.split(" ", 1)[0], "subject": ln.split(" ", 1)[1]}
                                           for ln in local if ln]}
    return published or {"published": False, "published_base": None, "published_ref": None,
                         "local_commits": []}


def _pins(root: Path) -> dict:
    """The commits that produced the outputs, refusing a checkout that would make that a lie."""
    daf_commit = _git(root, "rev-parse", "HEAD")
    dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise SystemExit(f"DAF checkout has tracked changes against {daf_commit}; refusing to export:\n{dirty}")
    vendor = root / VENDOR_REL
    vendor_commit = _git(vendor, "rev-parse", "HEAD")
    pinned = _git(root, "ls-tree", "HEAD", VENDOR_REL).split()[2]
    if vendor_commit != pinned:
        raise SystemExit(f"vendored substrate is at {vendor_commit}; DAF {daf_commit} pins {pinned}")
    if _git(vendor, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("vendored substrate has tracked changes; refusing to export")
    return {"daf_commit": daf_commit, "vendor_commit": vendor_commit,
            **_published_base(root, daf_commit)}


def _import_daf(root: Path):
    """What DAF's conftest.py does (`import daf`, whose _vendor module puts the vendored
    substrate on sys.path), with bytecode writing off so DAF_ROOT gains no files."""
    sys.dont_write_bytecode = True
    for p in (str(root / VENDOR_REL), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import daf  # noqa: F401  (side effect: vendored substrate importable)
    from daf.adapters.noaa_water_level import DATAGETTER_BASE
    from daf.catalog.checkpoint import CheckpointStore
    from daf.catalog.plan import AcquisitionPlan
    from daf.extractors.noaa_water_level_measurements import NoaaWaterLevelMeasurementExtractor
    from daf.orchestration.adapter_registry import AdapterRegistry
    from daf.orchestration.bindings import noaa_water_level_measurement_binding
    from daf.orchestration.result import AcquisitionOutcome
    from daf.orchestration.source_registry import SourceDefinition, SourceRegistry
    from daf.orchestration.bindings import usgs_daily_values_binding
    from daf.scheduling.runner import execute_plan
    from daf.storage.durable_pool import DurablePool
    from daf.storage.filesystem_store import FilesystemEvidenceStore
    from daf.storage.serialization import observation_from_dict, observation_to_dict, strict_json_loads
    return {
        "DATAGETTER_BASE": DATAGETTER_BASE, "CheckpointStore": CheckpointStore, "AcquisitionPlan": AcquisitionPlan,
        "NoaaWaterLevelMeasurementExtractor": NoaaWaterLevelMeasurementExtractor,
        "AdapterRegistry": AdapterRegistry,
        "noaa_water_level_measurement_binding": noaa_water_level_measurement_binding,
        "usgs_daily_values_binding": usgs_daily_values_binding,
        "AcquisitionOutcome": AcquisitionOutcome, "SourceDefinition": SourceDefinition,
        "SourceRegistry": SourceRegistry, "execute_plan": execute_plan, "DurablePool": DurablePool,
        "FilesystemEvidenceStore": FilesystemEvidenceStore, "observation_from_dict": observation_from_dict,
        "observation_to_dict": observation_to_dict, "strict_json_loads": strict_json_loads,
    }


def _export_one(job: Job, root: Path, d: dict) -> tuple[list[dict], dict]:
    blob = _git(root, "rev-parse", f"HEAD:{job.fixture}")
    raw = _git(root, "cat-file", "blob", blob, binary=True)   # the committed bytes, not the working tree's
    readings = json.loads(raw)["data"]
    # NOAA's per-reading revision flag q (p / v) and QC flag vector f, which DAF's extractor
    # deliberately leaves out of Observation.content: counted from the raw bytes, provenance only.
    raw_flags = {k: dict(sorted(Counter(str(r.get(k)) for r in readings).items())) for k in ("q", "f")}
    seen: list[str] = []

    def replay(url: str) -> bytes:
        q = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
        want = {"product": PRODUCT, "station": job.station, "datum": job.datum, "units": job.units,
                "time_zone": "gmt", "format": "json"}
        bad = {k: (q.get(k), v) for k, v in want.items() if q.get(k) != v}
        if bad or not url.startswith(d["DATAGETTER_BASE"] + "?"):
            raise RuntimeError(f"{job.fixture}: the adapter requested {url!r}; mismatched {bad}")
        seen.append(url)
        return raw

    binding = d["noaa_water_level_measurement_binding"](datum=job.datum, units=job.units, fetch_bytes=replay)
    sources = d["SourceRegistry"]()
    sources.register(d["SourceDefinition"](
        source_id=SOURCE_ID, name="NOAA CO-OPS Tides & Currents", domain="environmental-observations",
        adapter_id=binding.adapter_id, required_parameters=("station", "product", "start_date", "end_date"),
        capabilities=("incremental",)))
    adapters = d["AdapterRegistry"]()
    adapters.register(binding)
    params = {"station": job.station, "product": PRODUCT, "start_date": job.start_date, "end_date": job.end_date}
    # DAF's metadata index opens a short-lived sqlite connection per call and leaves closing it
    # to the garbage collector; on Windows an unclosed one blocks the temporary directory's
    # removal, hence the collect before cleanup (and ignore_cleanup_errors as the last resort).
    with tempfile.TemporaryDirectory(prefix="daf-export-", ignore_cleanup_errors=True) as tmp:
        pool = d["DurablePool"](d["FilesystemEvidenceStore"](Path(tmp) / "evidence"))
        plan = d["AcquisitionPlan"](plan_id="set-lcm-export", source_id=SOURCE_ID, parameters=params)
        result = d["execute_plan"](plan, sources, adapters, pool, d["CheckpointStore"](Path(tmp) / "checkpoints"),
                                   requested_at=job.requested_at)
        if result.outcome is not d["AcquisitionOutcome"].ACQUIRED:
            raise RuntimeError(f"{job.fixture}: DAF outcome {result.outcome} ({result.error})")
        observations = pool.all_observations()
        del pool
        gc.collect()
    # DAF reports one AcquiredArtifact per admitted finding; for one window they all name the
    # same artifact version (one Document, one Record), which is what is checked here.
    versions = {(a.locator, a.version_id) for a in result.artifacts}
    if len(seen) != 1 or len(versions) != 1:
        raise RuntimeError(f"{job.fixture}: expected one request and one artifact version, "
                           f"got {len(seen)} request(s) and versions {sorted(versions)}")
    if len(observations) != len(readings):
        raise RuntimeError(f"{job.fixture}: {len(readings)} readings but {len(observations)} observations admitted")

    dicts = []
    for o in observations:
        dd = d["observation_to_dict"](o)
        text = json.dumps(dd, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        back = d["observation_from_dict"](d["strict_json_loads"](text))   # DAF recomputes the id, or raises
        assert back.id == o.id == dd["id"]
        dicts.append(dd)
    dicts.sort(key=lambda r: (r["content"]["measurement_time"], r["id"]))
    art = result.artifacts[0]
    meta = {
        "output": job.out,
        "synthetic": job.synthetic,
        "note": job.note,
        "source_fixture": job.fixture,
        "source_fixture_sha256": _sha256(raw),
        "source_fixture_git_blob": blob,
        "daf_commit": _git(root, "rev-parse", "HEAD"),
        "n_readings": len(readings),
        "n_observations": len(dicts),
        "raw_flag_counts": raw_flags,
        "request_url": seen[0],
        "binding": {"factory": "daf.orchestration.bindings.noaa_water_level_measurement_binding",
                    "adapter_id": binding.adapter_id, "version": binding.version,
                    "station": job.station, "product": PRODUCT, "start_date": job.start_date,
                    "end_date": job.end_date, "datum": job.datum, "units": job.units,
                    "time_zone": {k: v[0] for k, v in parse_qs(urlsplit(seen[0]).query).items()}["time_zone"]},
        "extractor": f"{d['NoaaWaterLevelMeasurementExtractor'].__module__}."
                     f"{d['NoaaWaterLevelMeasurementExtractor'].__qualname__}"
                     f"(datum={job.datum!r}, units={job.units!r})",
        "extraction_methods": sorted({r["extraction_method"] for r in dicts}),
        "requested_at": job.requested_at,
        "daf_locator": art.locator,
        "daf_version_id": art.version_id,
    }
    return dicts, meta


# ---------------------------------------------------------------------------
# recorded live sessions (data/daf/raw/<session>), replayed offline
# ---------------------------------------------------------------------------

BINDING_FACTORIES = {
    "daf.orchestration.bindings.noaa_water_level_measurement_binding": "noaa_water_level_measurement_binding",
    "daf.orchestration.bindings.usgs_daily_values_binding": "usgs_daily_values_binding",
}


def _session_replay(index: dict, session_dir: Path):
    """Serve each recorded response for the URL it was recorded under, and refuse any other.

    The sha256 in index.json is checked against the bytes on disk before they are served, so
    a session whose recordings have been edited cannot be replayed into observations that
    claim to come from what was fetched."""
    by_url: dict[str, bytes] = {}
    for entry in index["responses"]:
        body = (session_dir / entry["file"]).read_bytes()
        digest = _sha256(body)
        if digest != entry["sha256"]:
            raise SystemExit(f"{session_dir / entry['file']} hashes to {digest}, index.json says "
                             f"{entry['sha256']}: refusing to replay an edited recording")
        if len(body) != entry["n_bytes"]:
            raise SystemExit(f"{session_dir / entry['file']} is {len(body)} bytes, index.json says "
                             f"{entry['n_bytes']}")
        by_url[entry["url"]] = body
    served: list[str] = []

    def replay(url: str) -> bytes:
        if url not in by_url:
            raise SystemExit(f"the adapter requested {url!r}, which this session never recorded; "
                             "refusing to reach the network from the exporter")
        served.append(url)
        return by_url[url]

    return replay, by_url, served


def _export_session(session_dir: Path, root: Path, d: dict) -> tuple[list[dict], dict]:
    """One recorded session -> the observations DAF admits from it, replayed offline.

    The same binding, the same plan parameters and the same `requested_at` the live fetch
    used, driven by the same `execute_plan` loop, against a temporary pool. Nothing is
    fetched: every response comes from the recording, matched by URL.
    """
    index = json.loads((session_dir / "index.json").read_text(encoding="utf-8"))
    # `index.json` names the binding factory at the top level; the NOAA month session was
    # recorded by an earlier revision of the writer that nested it under "binding", and its
    # per-response item count under "n_readings". Both spellings are read rather than the
    # recording being edited: the responses are the evidence, and a provenance file is not
    # something to rewrite by hand.
    factory_path = index.get("binding_factory") or index["binding"]["factory"]
    for entry in index["responses"]:
        if "n_items" not in entry and "n_readings" in entry:
            entry["n_items"] = entry["n_readings"]
    try:
        factory = d[BINDING_FACTORIES[factory_path]]
    except KeyError:
        raise SystemExit(f"{session_dir.name} names binding factory {factory_path!r}, which this exporter "
                         "does not know how to rebuild") from None
    replay, by_url, served = _session_replay(index, session_dir)

    if factory_path.endswith("noaa_water_level_measurement_binding"):
        binding = factory(datum=index["datum"], units=index["units"], fetch_bytes=replay)
        source_name = "NOAA CO-OPS Tides & Currents"
        required = ("station", "product", "start_date", "end_date")
        plans = [("fsre-noaa-month", index["plan_parameters"])]
    else:
        binding = factory(fetch_bytes=replay)
        source_name = "USGS Water Data OGC API -- daily values"
        required = ("monitoring_location_id", "parameter_code", "statistic_id", "start_date", "end_date")
        plans = [(f"fsre-usgs-{s['monitoring_location_id']}-{s['parameter_code']}-{s['statistic_id']}",
                  s["plan_parameters"]) for s in index["series"]]
    # The DAF commit a session was recorded at and the one it is exported at need not be the
    # same -- a commit that adds an unrelated adapter changes neither. What must be the same is
    # the BINDING VERSION, which is a hash of the adapter's and extractor's own source: that is
    # what decides the bytes requested and the content extracted, hence every evidence id here.
    if binding.version != index["binding"]["version"]:
        raise SystemExit(f"{session_dir.name} was recorded with binding version "
                         f"{index['binding']['version'][:16]} and this checkout builds "
                         f"{binding.version[:16]}: the adapter or extractor code has changed, so the "
                         "recording can no longer be replayed into the observations it produced")

    source_id = index.get("source_id") or ("noaa-water-level-measurements"
                                           if "noaa" in factory_path else "usgs-daily-values")
    sources = d["SourceRegistry"]()
    sources.register(d["SourceDefinition"](
        source_id=source_id, name=source_name, domain="environmental-observations",
        adapter_id=binding.adapter_id, required_parameters=required, capabilities=("incremental",)))
    adapters = d["AdapterRegistry"]()
    adapters.register(binding)

    outcomes: list[str] = []
    with tempfile.TemporaryDirectory(prefix="daf-export-", ignore_cleanup_errors=True) as tmp:
        pool = d["DurablePool"](d["FilesystemEvidenceStore"](Path(tmp) / "evidence"))
        checkpoints = d["CheckpointStore"](Path(tmp) / "checkpoints")
        for plan_id, params in plans:
            plan = d["AcquisitionPlan"](plan_id=plan_id, source_id=source_id, parameters=dict(params),
                                        mode="incremental")
            for _ in range(len(by_url) + 4):
                result = d["execute_plan"](plan, sources, adapters, pool, checkpoints,
                                           requested_at=index["requested_at"])
                outcomes.append(result.outcome.name)
                if result.outcome is not d["AcquisitionOutcome"].ACQUIRED:
                    break
            else:
                raise SystemExit(f"{session_dir.name}: plan {plan_id} did not settle")
        observations = pool.all_observations()
        del pool
        gc.collect()

    unserved = sorted(set(by_url) - set(served))
    if unserved:
        raise SystemExit(f"{session_dir.name}: {len(unserved)} recorded response(s) were never requested by "
                         f"the replay, so the export is not the session that was fetched: {unserved[:3]}")
    dicts = []
    for o in observations:
        dd = d["observation_to_dict"](o)
        text = json.dumps(dd, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        back = d["observation_from_dict"](d["strict_json_loads"](text))   # DAF recomputes the id, or raises
        assert back.id == o.id == dd["id"]
        dicts.append(dd)
    dicts.sort(key=lambda r: (r["content"]["measurement_time"], r["id"]))

    n_recorded_items = sum(e.get("n_items", 0) for e in index["responses"])
    meta = {
        "output": f"{session_dir.name}.observations.json",
        "synthetic": False,
        "note": index.get("note", ""),
        "source_session": f"data/daf/raw/{session_dir.name}",
        "source_session_index_sha256": _sha256((session_dir / "index.json").read_bytes()),
        "recorded_at_daf_commit": index["daf_commit"],
        "exported_at_daf_commit": _git(root, "rev-parse", "HEAD"),
        "fetched_live": True,
        "n_recorded_responses": len(by_url),
        "n_recorded_items": n_recorded_items,
        "n_observations": len(dicts),
        "n_distinct_measurement_times": len({r["content"]["measurement_time"] for r in dicts}),
        "outcomes": outcomes,
        "binding": {"factory": factory_path, "adapter_id": binding.adapter_id, "version": binding.version,
                    **{k: v for k, v in index.items()
                       if k in ("datum", "units", "time_zone", "window_days", "revision_lookback_days",
                                "page_limit", "scope")}},
        "plans": [{"plan_id": pid, "parameters": dict(par)} for pid, par in plans],
        "extraction_methods": sorted({r["extraction_method"] for r in dicts}),
        "requested_at": index["requested_at"],
        "responses": [{k: e[k] for k in ("url", "file", "sha256", "n_bytes", "n_items") if k in e}
                      for e in index["responses"]],
    }
    return dicts, meta


def _write(path: Path, text: str) -> str:
    data = text.encode("utf-8")
    path.write_bytes(data)          # LF only, exactly these bytes
    return _sha256(data)


def _provenance_md(man: dict) -> str:
    L = [
        "# DAF evidence exported for the bridge",
        "",
        "Generated by `tools/export_daf_fixtures.py`; do not edit by hand (re-run the tool). "
        "`manifest.json` carries the same facts in machine-readable form.",
        "",
        "## Pins",
        "",
        f"- DAF (Data Acquisition Fabric): {man['daf_repository']} at commit `{man['daf_commit']}`"
        + ("" if man["daf_commit_published"] else
           f" -- **not published**. Its published ancestor is `{man['daf_published_base']}` on "
           f"`{man['daf_published_ref']}`; the "
           + ", ".join(f"commit `{c['commit'][:12]}` ({c['subject']})" for c in man["daf_local_commits"])
           + " above it is local only and travels as the patch series in `patches/daf/`, which is how "
             "this repository carries it without pushing anything to DAF."),
        f"- Vendored substrate `{man['vendored_substrate']['path']}` ({man['vendored_substrate']['url']}) at commit "
        f"`{man['vendored_substrate']['commit']}` (`git -C {man['vendored_substrate']['path']} rev-parse HEAD`; "
        "equal to the commit DAF pins)",
        f"- Exported with Python {man['python']}",
        "",
        "## Command",
        "",
        "```bash",
        man["command"],
        "```",
        "",
        "Nothing is fetched: DAF's own per-measurement NOAA binding "
        "(`daf.orchestration.bindings.noaa_water_level_measurement_binding`, i.e. the unmodified "
        "`NoaaWaterLevelSourceAdapter` paired with `NoaaWaterLevelMeasurementExtractor`) runs through "
        "`daf.scheduling.runner.execute_plan` into a temporary `DurablePool`, and the adapter's own "
        "`fetch_bytes` injection point replays the committed fixture bytes (read from DAF's git object store, "
        "so the checkout's line-ending conversion cannot change them). Each admitted Observation is written "
        "with DAF's `observation_to_dict` and re-read with DAF's `strict_json_loads` + `observation_from_dict`, "
        "which recomputes its content-addressed id. `extracted_at` is DAF's caller-supplied retrieval stamp "
        "(`run_scout` copies `retrieved_at` into it): the value DAF's own tests use when replaying these "
        "fixtures, NOT the time NOAA served the bytes (DAF does not record that), and excluded from every id.",
        "",
        "## Files",
        "",
        "| output | source | observations | acquired as | extracted_at |",
        "|---|---|---|---|---|",
    ]
    for f in man["files"]:
        b = f["binding"]
        if "source_fixture" in f:
            source = f"`{f['source_fixture']}` (a DAF fixture, replayed)"
            how = f"{b['station']} / {b['datum']} / {b['units']} / {b['time_zone']}"
        else:
            source = f"`{f['source_session']}` ({f['n_recorded_responses']} recorded live response(s))"
            how = f"{b['adapter_id']}, {f['n_distinct_measurement_times']} distinct measurement times"
        L.append(f"| `{f['output']}`{' (SYNTHETIC)' if f['synthetic'] else ''} | {source} | "
                 f"{f['n_observations']} | {how} | {f['requested_at']} |")
    L += ["", "Per file:", ""]
    for f in man["files"]:
        b = f["binding"]
        if "source_session" in f:
            L += [
                f"### `{f['output']}` -- replayed from a recorded live session",
                "",
                f"- {f['note']}",
                f"- Recorded by `tools/fetch_noaa_month.py` / `tools/fetch_usgs_reservoir.py` into "
                f"`{f['source_session']}`: {f['n_recorded_responses']} HTTPS response(s) kept byte for byte "
                f"under the sha256 of their own bytes, carrying {f['n_recorded_items']} item(s) in total. "
                f"`index.json` sha256 `{f['source_session_index_sha256']}`; each response's own sha256 is "
                "checked against it before the replay serves it, and a URL the session never recorded is "
                "refused rather than fetched.",
                f"- Replayed through `{b['factory']}` (adapter id `{b['adapter_id']}`, DAF code version "
                f"`{b['version']}`) and `execute_plan`, plan(s) "
                + ", ".join(f"`{p['plan_id']}`" for p in f["plans"]) + f", outcomes {f['outcomes']}.",
                f"- Recorded against DAF `{f['recorded_at_daf_commit']}`, exported at "
                f"`{f['exported_at_daf_commit']}`. These need not be equal: what must match, and is checked, "
                "is the binding version -- a hash of the adapter's and extractor's own source, which is what "
                "decides the bytes requested and the content extracted.",
                f"- {f['n_observations']} observation(s) over {f['n_distinct_measurement_times']} distinct "
                "measurement times. The two differ because DAF re-requests a trailing safety window, so a "
                "reading acquired through two overlapping windows is two observations under two records; "
                "collapsing them on content is the consumer's job (`set_lcm.bridge.daf`).",
                f"- Extraction method(s) {', '.join(f'`{m}`' for m in f['extraction_methods'])}.",
                f"- Output sha256 `{f['output_sha256']}`.",
                "",
            ]
            continue
        L += [
            f"### `{f['output']}`{' -- SYNTHETIC' if f['synthetic'] else ''}",
            "",
            f"- {f['note']}",
            f"- Source fixture `{f['source_fixture']}`: sha256 `{f['source_fixture_sha256']}` of the committed "
            f"bytes (git blob `{f['source_fixture_git_blob']}`; a checkout that converts line endings holds "
            f"different bytes on disk), {f['n_readings']} readings.",
            f"- Extractor `{f['extractor']}`, extraction method {', '.join(f'`{m}`' for m in f['extraction_methods'])}.",
            f"- Binding `{b['factory']}` (adapter id `{b['adapter_id']}`, DAF code version `{b['version']}`), "
            f"plan parameters station={b['station']} product={b['product']} start_date={b['start_date']} "
            f"end_date={b['end_date']}, datum={b['datum']} units={b['units']}.",
            f"- Adapter request URL (replayed, never sent): `{f['request_url']}` -- the `time_zone={b['time_zone']}` "
            "the measurement times are expressed in comes from this URL; DAF's content carries no zone.",
            f"- DAF locator `{f['daf_locator']}`, version id (Document.id) `{f['daf_version_id']}`.",
            "- NOAA flags in the raw fixture, per reading (DAF's extractor keeps `q`, revision metadata, and "
            "`f`, the QC flag vector, out of Observation.content, so no exported observation carries them; "
            "counted here as provenance of the raw artifact, read by nothing on the bridging path): q "
            + ", ".join(f"`{k}` {v}" for k, v in f["raw_flag_counts"]["q"].items()) + "; f "
            + ", ".join(f"`{k}` {v}" for k, v in f["raw_flag_counts"]["f"].items()) + ".",
            f"- Output sha256 `{f['output_sha256']}`.",
            "",
        ]
    L += [
        "## Licence",
        "",
        "NOAA CO-OPS water-level data are a work of the U.S. Government (NOAA National Ocean Service, Center for "
        "Operational Oceanographic Products and Services) and are in the public domain in the United States "
        "(17 U.S.C. 105). The three `noaa_live_*` files are NOAA data. The two `SYNTHETIC_*` files are not NOAA "
        "data: they are extracted from DAF's hand-written test fixtures for a non-existent station 9999999 and "
        "exist only to exercise revision conflicts.",
        "",
    ]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--raw", type=Path, default=DEFAULT_OUT / "raw",
                    help="directory of recorded live sessions to replay (data/daf/raw)")
    ap.add_argument("--only", choices=("fixtures", "sessions"), default=None,
                    help="export only DAF's committed fixtures, or only the recorded sessions")
    args = ap.parse_args(argv)
    root = _daf_root()
    pins = _pins(root)
    d = _import_daf(root)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    gitmodules = (root / ".gitmodules").read_text(encoding="utf-8")
    vendor_url = next((ln.split("=", 1)[1].strip() for ln in gitmodules.splitlines() if ln.strip().startswith("url")),
                      None)
    files = []
    jobs = () if args.only == "sessions" else JOBS
    for job in jobs:
        dicts, meta = _export_one(job, root, d)
        body = "[\n" + ",\n".join(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                                             allow_nan=False) for r in dicts) + "\n]\n"
        meta["output_sha256"] = _write(out / job.out, body)
        files.append(meta)
        print(f"{job.out}: {len(dicts)} observations{' (SYNTHETIC)' if job.synthetic else ''}")
    sessions = []
    if args.only != "fixtures" and args.raw.is_dir():
        sessions = sorted(p for p in args.raw.iterdir() if (p / "index.json").is_file())
    for session_dir in sessions:
        dicts, meta = _export_session(session_dir, root, d)
        body = "[\n" + ",\n".join(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                                             allow_nan=False) for r in dicts) + "\n]\n"
        meta["output_sha256"] = _write(out / meta["output"], body)
        files.append(meta)
        print(f"{meta['output']}: {len(dicts)} observations from {meta['n_recorded_responses']} recorded "
              f"response(s), {meta['n_distinct_measurement_times']} distinct measurement times")
    man = {
        "daf_repository": DAF_URL,
        "daf_commit": pins["daf_commit"],
        "daf_commit_published": pins["published"],
        "daf_published_base": pins["published_base"],
        "daf_published_ref": pins["published_ref"],
        "daf_local_commits": pins["local_commits"],
        "daf_local_commits_note":
            "Commits in the exporting checkout that are NOT in DAF's published history. Nothing is ever "
            "pushed to the DAF repository, so these travel as the git format-patch series under "
            "patches/daf/, applied to daf_published_base. A file whose bytes are determined only by "
            "DAF's own committed fixtures and an unchanged binding is unaffected by them -- the fixture "
            "blob sha and the binding version recorded per file are what fix those bytes."
            if pins["local_commits"] else
            "The exporting checkout is entirely within DAF's published history.",
        "vendored_substrate": {"path": VENDOR_REL, "url": vendor_url, "commit": pins["vendor_commit"]},
        "exporter": "tools/export_daf_fixtures.py",
        "command": "DAF_ROOT=/path/to/daf-checkout uv run --python 3.13 python tools/export_daf_fixtures.py",
        "python": platform.python_version(),
        "files": files,
    }
    _write(out / "manifest.json", json.dumps(man, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
    _write(out / "PROVENANCE.md", _provenance_md(man))
    print(f"wrote {out} (DAF {pins['daf_commit'][:10]}, vendored substrate {pins['vendor_commit'][:10]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
