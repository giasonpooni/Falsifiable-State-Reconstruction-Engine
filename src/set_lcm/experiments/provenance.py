"""Provenance stamp for results files: which code, which Python and numpy, which
machine. Stamped into every results JSON and printed at the top of every
results markdown so a table can be traced to the source that produced it.

`git_head` is HEAD at generation time, i.e. the parent of the commit that
contains the regenerated results. The source hash is over the normalised
(LF) bytes of src/**/*.py and run_experiments.py, so it does not depend on
the checkout's line endings.
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]   # src/set_lcm/experiments/ -> repo root
SOURCE_GLOBS = ("src/**/*.py", "run_experiments.py")
SOURCE_STATUS_PATHS = ("src", "run_experiments.py", "pyproject.toml", "tests")


def source_sha256(root: Path = REPO_ROOT) -> str:
    h = hashlib.sha256()
    files = sorted({p for g in SOURCE_GLOBS for p in root.glob(g) if p.is_file()})
    for p in files:
        h.update(p.relative_to(root).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
        h.update(b"\0")
    return h.hexdigest()


def _git(*args: str, root: Path = REPO_ROOT) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()
    except Exception:   # git absent, not a repo, or any other failure: provenance is best-effort
        return None


def provenance(root: Path = REPO_ROOT) -> dict:
    status = _git("status", "--porcelain", "--", *SOURCE_STATUS_PATHS, root=root)
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "source_sha256": source_sha256(root),
        "git_head": _git("rev-parse", "HEAD", root=root),
        "source_dirty": bool(status) if status is not None else None,
        "note": "git_head is HEAD at generation time (the parent of the commit containing these results); "
                "source_dirty covers src/, run_experiments.py, pyproject.toml and tests/ only",
    }


def header_line(p: dict) -> str:
    head = str(p["git_head"])[:10] if p["git_head"] else "unknown"
    dirty = " (source dirty)" if p["source_dirty"] else ""
    return (f"Generated with Python {p['python']}, numpy {p['numpy']} on {p['platform']}; "
            f"source sha256 {p['source_sha256'][:12]}, git {head}{dirty}. "
            "Latency columns are wall-clock on this machine and are not a claim.")
