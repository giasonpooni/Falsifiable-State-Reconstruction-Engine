"""Shared machinery for the two live-fetch tools: run DAF's own acquisition, record it.

`tools/fetch_noaa_month.py` and `tools/fetch_usgs_reservoir.py` differ only in which DAF
binding they drive and which plan parameters they pass. Everything else -- refusing a DAF
checkout that is modified or unpinned, importing DAF and its vendored substrate without
writing bytecode into it, teeing every response to disk under the sha256 of its own bytes,
driving `execute_plan` until the plan stops reporting new data, and writing an `index.json`
that a later offline replay can be checked against -- is here.

Nothing in this module is imported by `set_lcm`, by any test, or by the export path. It
exists only so that the two tools that DO reach the network share one implementation of
what "record faithfully" means.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = REPO_ROOT / "data" / "daf" / "raw"
VENDOR_REL = "vendor/scout-retrieval-agent"
DAF_URL = "https://github.com/atomtrapping/Data-Acquisition-Channel"
MAX_CALLS = 400          # a hard stop: the plan loop must terminate on its own long before this
SLEEP_S = 0.5            # between live requests, so a long scope is not a burst


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args],
                          capture_output=True, check=True, timeout=60).stdout.decode("utf-8").strip()


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def check_requested_at(value: str) -> str:
    """DAF's caller-supplied retrieval stamp. Never wall-clock: `run_scout` copies it into
    `extracted_at`, so a fixed value is what makes a re-run reproduce the same evidence ids."""
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        raise SystemExit(f"--requested-at must be 'YYYY-MM-DDTHH:MM:SSZ'; got {value!r}") from None
    return value


def daf_root(*needs: str) -> Path:
    """The DAF checkout named by DAF_ROOT, refused unless it is clean and its vendored
    substrate is at the commit DAF pins -- the same guard tools/export_daf_fixtures.py uses,
    for the same reason: an output stamped with a commit must actually come from it."""
    raw = os.environ.get("DAF_ROOT")
    if not raw:
        raise SystemExit(f"DAF_ROOT is not set: point it at a Data Acquisition Fabric checkout "
                         f"({DAF_URL}) with its submodule initialised")
    root = Path(raw).resolve()
    for need in ("daf/__init__.py", f"{VENDOR_REL}/evidence/types.py", *needs):
        if not (root / need).is_file():
            raise SystemExit(f"DAF_ROOT={root} is not a DAF checkout with its vendored substrate: missing {need}")
    dirty = git(root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise SystemExit(f"DAF checkout has tracked changes; refusing to fetch against a modified DAF:\n{dirty}")
    vendor_commit = git(root / VENDOR_REL, "rev-parse", "HEAD")
    pinned = git(root, "ls-tree", "HEAD", VENDOR_REL).split()[2]
    if vendor_commit != pinned:
        raise SystemExit(f"vendored substrate is at {vendor_commit}; DAF pins {pinned}")
    return root


def import_daf(root: Path) -> dict:
    """What DAF's conftest.py does (`import daf`, whose _vendor module puts the vendored
    substrate on sys.path), with bytecode writing off so DAF_ROOT gains no files."""
    sys.dont_write_bytecode = True
    for p in (str(root / VENDOR_REL), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import daf  # noqa: F401  (side effect: vendored substrate importable)
    from daf.catalog.checkpoint import CheckpointStore
    from daf.catalog.plan import AcquisitionPlan
    from daf.orchestration.adapter_registry import AdapterRegistry
    from daf.orchestration.result import AcquisitionOutcome
    from daf.orchestration.source_registry import SourceDefinition, SourceRegistry
    from daf.scheduling.runner import execute_plan
    from daf.storage.durable_pool import DurablePool
    from daf.storage.filesystem_store import FilesystemEvidenceStore
    return {
        "CheckpointStore": CheckpointStore, "AcquisitionPlan": AcquisitionPlan,
        "AdapterRegistry": AdapterRegistry, "AcquisitionOutcome": AcquisitionOutcome,
        "SourceDefinition": SourceDefinition, "SourceRegistry": SourceRegistry,
        "execute_plan": execute_plan, "DurablePool": DurablePool,
        "FilesystemEvidenceStore": FilesystemEvidenceStore,
    }


def live_fetch(url: str, user_agent: str, timeout_s: int = 120) -> bytes:
    """One HTTPS GET, returning the body bytes verbatim. Deliberately plain: DAF's adapter
    owns retry and error-envelope handling, and this tool must not reinterpret either."""
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:      # noqa: S310 -- fixed https endpoints
        if resp.status != 200:
            raise RuntimeError(f"{url!r} returned HTTP {resp.status}")
        return resp.read()


@dataclass
class Recorder:
    """Tees every response the adapter asks for to `out_dir`, under the sha256 of its own
    bytes. `check` is given the URL and must raise if the adapter asked for something the
    tool did not intend -- a recorded session is only evidence of what was intended if the
    requests were checked as they were made."""
    out_dir: Path
    user_agent: str
    check: Callable[[str], Mapping[str, Any]]
    describe: Callable[[bytes], Mapping[str, Any]] = lambda b: {}
    quiet: bool = False
    responses: list[dict] = field(default_factory=list)
    _seen: dict[str, str] = field(default_factory=dict)

    def __call__(self, url: str) -> bytes:
        fields = dict(self.check(url))
        if url in self._seen:                       # DAF re-requesting one window: recorded once
            return (self.out_dir / self._seen[url]).read_bytes()
        time.sleep(SLEEP_S)
        body = live_fetch(url, self.user_agent)
        digest = sha256(body)
        name = f"{digest}.json"
        (self.out_dir / name).write_bytes(body)
        self._seen[url] = name
        entry = {"url": url, "file": name, "sha256": digest, "n_bytes": len(body),
                 **fields, **dict(self.describe(body))}
        self.responses.append(entry)
        if not self.quiet:
            extra = " ".join(f"{k}={v}" for k, v in fields.items())
            print(f"  {extra}  {len(body):>8} bytes  {entry.get('n_items', '?')} items")
        return body


def run_plan_to_exhaustion(d: dict, *, binding, source_id: str, source_name: str,
                           required_parameters: Iterable[str], plan_id: str,
                           parameters: Mapping[str, Any], requested_at: str) -> tuple[list[str], int]:
    """Drive DAF's own incremental acquisition until the plan stops reporting new data.

    `mode="incremental"` is what makes `execute_plan` hand the checkpoint position back to
    the adapter as `since`; without it every call re-requests the first window. The loop
    ends when an outcome is not ACQUIRED -- for these adapters that is DUPLICATE, reported
    once the cursor has settled and the same window is re-fetched to the same bytes.

    The evidence store is temporary: this tool records RESPONSES, and the observations are
    rebuilt offline by tools/export_daf_fixtures.py. Returns (outcomes, n_observations).
    """
    sources = d["SourceRegistry"]()
    sources.register(d["SourceDefinition"](
        source_id=source_id, name=source_name, domain="environmental-observations",
        adapter_id=binding.adapter_id, required_parameters=tuple(required_parameters),
        capabilities=("incremental",)))
    adapters = d["AdapterRegistry"]()
    adapters.register(binding)

    outcomes: list[str] = []
    # DAF's metadata index opens a short-lived sqlite connection per call and leaves closing
    # it to the garbage collector; hence the collect before cleanup, as the exporter does.
    with tempfile.TemporaryDirectory(prefix="daf-fetch-", ignore_cleanup_errors=True) as tmp:
        pool = d["DurablePool"](d["FilesystemEvidenceStore"](Path(tmp) / "evidence"))
        checkpoints = d["CheckpointStore"](Path(tmp) / "checkpoints")
        plan = d["AcquisitionPlan"](plan_id=plan_id, source_id=source_id, parameters=dict(parameters),
                                    mode="incremental")
        for _ in range(MAX_CALLS):
            result = d["execute_plan"](plan, sources, adapters, pool, checkpoints, requested_at=requested_at)
            outcomes.append(result.outcome.name)
            if result.outcome is not d["AcquisitionOutcome"].ACQUIRED:
                break
        else:
            raise SystemExit(f"the plan still reported new data after {MAX_CALLS} calls; refusing to keep fetching")
        n_obs = len(pool.all_observations())
        del pool
        gc.collect()
    return outcomes, n_obs


def prepare_session(raw_dir: Path, session: str) -> Path:
    out_dir = raw_dir / session
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} already exists and is not empty; refusing to overwrite a recorded session")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_index(out_dir: Path, index: Mapping[str, Any]) -> None:
    body = json.dumps(dict(index), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    (out_dir / "index.json").write_bytes(body.encode("utf-8"))


def base_index(root: Path, *, session: str, tool: str, binding, requested_at: str,
               parameters: Mapping[str, Any]) -> dict:
    return {
        "session": session,
        "tool": tool,
        "fetched_live": True,
        "daf_repository": DAF_URL,
        "daf_commit": git(root, "rev-parse", "HEAD"),
        "vendored_substrate_commit": git(root / VENDOR_REL, "rev-parse", "HEAD"),
        "binding": {"adapter_id": binding.adapter_id, "version": binding.version},
        "plan_parameters": dict(parameters),
        "requested_at": requested_at,
        "python": platform.python_version(),
    }
