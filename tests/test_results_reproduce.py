"""Regenerate the Phase 1 grid at full size and check it against the committed
results/summary.json -- every aggregate and every per-seed metric, with only the latency
columns and the provenance stamp excluded.

What "check" means depends on the build, and set_lcm.experiments.compare says why: on the
build that generated the file, bitwise equality; on any other, the declared per-file
tolerance measured there (summary.json: 1e-12, against 5.878e-15 observed between Windows
and Linux). Either way a deviation past the declared tolerance fails.

Slow (about ten minutes):  uv run --python 3.13 --dev pytest -q -m slow
"""
import json

import pytest

from set_lcm.experiments.compare import reproduction_failure
from set_lcm.experiments.phase1 import main
from set_lcm.experiments.provenance import REPO_ROOT


@pytest.mark.slow
def test_phase1_results_reproduce(tmp_path):
    main(tmp_path, quiet=True)
    committed = json.loads((REPO_ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
    fresh = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    failure = reproduction_failure("summary.json", fresh, committed)
    assert failure is None, failure
