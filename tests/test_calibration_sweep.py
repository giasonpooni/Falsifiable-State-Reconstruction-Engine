"""Smoke tests for the calibration and sweep modules on 2 seeds (seconds), plus the
full-size runs marked slow (minutes; `uv run --python 3.13 --dev pytest -m slow`)."""
import pytest

from set_lcm.experiments import calibration, sweep


def test_null_stats_small():
    r = calibration.null_stats(n_seeds=2)
    assert set(r) == set(calibration.NULL_WINDOWS)
    for v in r.values():
        assert v["n_samples"] > 0
        assert 0.0 < v["mean"] < 5.0
        assert set(v["exceedance"]) == set(calibration.THRESHOLDS)
        assert -1.0 <= v["lag1_autocorr"] <= 1.0


def test_threshold_sweep_small():
    rows = calibration.threshold_sweep(n_seeds=2, qs=(0.999,), debounces=(3,))
    assert len(rows) == 1
    r = rows[0]
    assert r["detected_within"] == r["n"] == 2
    assert r["held"] > 100


def test_sweep_declared_total_error_small():
    r = sweep.run_axis("declared_total_error", n_seeds=2, points=(0.0, 4.0))
    assert set(r) == {0.0, 4.0}
    assert r[0.0]["kf+hard+guard"]["held_steps"]["mean"] == 0          # nothing to hold when the total is right
    assert r[4.0]["kf+hard+guard"]["held_steps"]["mean"] > 100         # a 4 kg wrong total is caught and held
    assert r[4.0]["kf+hard"]["rmse_by_window"]["steady"]["mean"] > r[4.0]["kf"]["rmse_by_window"]["steady"]["mean"]


def test_sweep_uncertain_total_adds_soft_with_declared_variance():
    r = sweep.run_axis("uncertain_total", n_seeds=2, points=(1.0,))
    a = r[1.0]
    assert sweep.SOFT_NAME in a
    assert a[sweep.SOFT_NAME]["mean_abs_res_post"]["mean"] > 0.0       # soft leaves a residual by design


@pytest.mark.slow
def test_full_calibration_and_sweep_write_files(tmp_path):
    calibration.main(tmp_path, quiet=True)
    sweep.main(tmp_path, quiet=True)
    for name in ("calibration.md", "calibration.json", "sweep.md", "sweep.json"):
        assert (tmp_path / name).exists()
