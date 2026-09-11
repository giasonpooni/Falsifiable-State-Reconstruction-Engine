"""Regenerate the Phase 1 grid at full size and assert it equals the committed
results/summary.json value for value -- every aggregate and every per-seed
metric -- with only the latency columns and the provenance stamp excluded.

Slow (about a minute):  uv run --python 3.13 --dev pytest -m slow
"""
import json

import pytest

from set_lcm.experiments.phase1 import main
from set_lcm.experiments.provenance import REPO_ROOT

EXCLUDED_KEYS = ("provenance",)
EXCLUDED_PREFIXES = ("latency_us",)


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items()
                if k not in EXCLUDED_KEYS and not k.startswith(EXCLUDED_PREFIXES)}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


@pytest.mark.slow
def test_phase1_results_reproduce_exactly(tmp_path):
    main(tmp_path, quiet=True)
    committed = json.loads((REPO_ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
    fresh = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert _strip(fresh) == _strip(committed), "results/summary.json does not match a fresh run of the source tree"
