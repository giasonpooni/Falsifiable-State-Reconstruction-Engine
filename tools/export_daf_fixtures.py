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
manifest.json and PROVENANCE.md. Files prefixed SYNTHETIC_ come from DAF's synthetic
test fixtures (station 9999999, "NOT A REAL NOAA STATION") and exist only to exercise
revision conflicts. Re-running against the same DAF commit reproduces every output
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
    return {"daf_commit": daf_commit, "vendor_commit": vendor_commit}


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
    from daf.scheduling.runner import execute_plan
    from daf.storage.durable_pool import DurablePool
    from daf.storage.filesystem_store import FilesystemEvidenceStore
    from daf.storage.serialization import observation_from_dict, observation_to_dict, strict_json_loads
    return {
        "DATAGETTER_BASE": DATAGETTER_BASE, "CheckpointStore": CheckpointStore, "AcquisitionPlan": AcquisitionPlan,
        "NoaaWaterLevelMeasurementExtractor": NoaaWaterLevelMeasurementExtractor,
        "AdapterRegistry": AdapterRegistry,
        "noaa_water_level_measurement_binding": noaa_water_level_measurement_binding,
        "AcquisitionOutcome": AcquisitionOutcome, "SourceDefinition": SourceDefinition,
        "SourceRegistry": SourceRegistry, "execute_plan": execute_plan, "DurablePool": DurablePool,
        "FilesystemEvidenceStore": FilesystemEvidenceStore, "observation_from_dict": observation_from_dict,
        "observation_to_dict": observation_to_dict, "strict_json_loads": strict_json_loads,
    }


def _export_one(job: Job, root: Path, d: dict) -> tuple[list[dict], dict]:
    blob = _git(root, "rev-parse", f"HEAD:{job.fixture}")
    raw = _git(root, "cat-file", "blob", blob, binary=True)   # the committed bytes, not the working tree's
    readings = json.loads(raw)["data"]
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
        "n_readings": len(readings),
        "n_observations": len(dicts),
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
        f"- DAF (Data Acquisition Fabric): {man['daf_repository']} at commit `{man['daf_commit']}`",
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
        "| output | source fixture (in DAF) | fixture sha256 (committed bytes) | observations | binding: station / datum / units / time_zone | extracted_at |",
        "|---|---|---|---|---|---|",
    ]
    for f in man["files"]:
        b = f["binding"]
        L.append(f"| `{f['output']}`{' (SYNTHETIC)' if f['synthetic'] else ''} | `{f['source_fixture']}` | "
                 f"`{f['source_fixture_sha256']}` | {f['n_observations']} | {b['station']} / {b['datum']} / "
                 f"{b['units']} / {b['time_zone']} | {f['requested_at']} |")
    L += ["", "Per file:", ""]
    for f in man["files"]:
        b = f["binding"]
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
    for job in JOBS:
        dicts, meta = _export_one(job, root, d)
        body = "[\n" + ",\n".join(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                                             allow_nan=False) for r in dicts) + "\n]\n"
        meta["output_sha256"] = _write(out / job.out, body)
        files.append(meta)
        print(f"{job.out}: {len(dicts)} observations{' (SYNTHETIC)' if job.synthetic else ''}")
    man = {
        "daf_repository": DAF_URL,
        "daf_commit": pins["daf_commit"],
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
