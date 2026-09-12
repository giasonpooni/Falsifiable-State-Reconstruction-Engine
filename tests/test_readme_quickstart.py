"""Execute the documented entry point and verify its two decision paths."""
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np

from set_lcm.schema import Status


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "quickstart.py"


def test_documented_example_runs_from_the_checkout():
    result = subprocess.run([sys.executable, str(EXAMPLE)], cwd=ROOT, check=True,
                            capture_output=True, text=True)
    assert "Action: ok" in result.stdout
    assert "Action: model_inconsistent" in result.stdout
    assert "Correction held: investigate" in result.stdout


def test_example_keeps_originals_and_holds_a_rejected_correction():
    check = runpy.run_path(str(EXAMPLE))["check_estimate"]
    accepted = check([52.0, 46.0])
    held = check([60.0, 50.0])
    assert accepted.status is Status.OK
    np.testing.assert_array_equal(accepted.x_unprojected, [52.0, 46.0])
    assert 0 < abs(accepted.residual_post[0]) < abs(accepted.residual_pre[0])
    assert held.status is Status.MODEL_INCONSISTENT
    assert held.consistency_stat > held.consistency_threshold
    np.testing.assert_array_equal(held.x, held.x_unprojected)
    assert held.correction is None
