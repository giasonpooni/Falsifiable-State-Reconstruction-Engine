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


def test_cusum_null_small():
    r = calibration.cusum_null(n_seeds=2, hs=(8.0,))
    assert set(r["windows"]) == set(calibration.NULL_WINDOWS)
    assert r["shipped_h"] == 8.0 and r["hs"] == [8.0]
    for w in r["windows"].values():
        assert set(w["by_h"]) == {"8", "inf"}
        assert w["by_h"]["inf"]["alarms"] == [0, 0]                # no reset, no alarm
        assert all(m >= 0.0 for m in w["by_h"]["inf"]["max_stat"])
        assert w["by_h"]["8"]["max_stat"][0] <= w["by_h"]["inf"]["max_stat"][0] + 1e-12
    assert r["smallest_h_zero_alarms"] in (8.0, None)
    assert r["total_alarms_by_h"]["8"] == r["shipped_h_alarms"]


def _steady(a: dict, spec: str, key: str = "rmse_by_window") -> float:
    return a[spec][key]["steady"]["mean"]


def test_sweep_declared_total_error_small():
    r = sweep.run_axis("declared_total_error", n_seeds=2, points=(0.0, 4.0))
    assert set(r) == {0.0, 4.0}
    assert r[0.0]["kf+hard+guard"]["held_steps"]["mean"] == 0          # nothing to hold when the total is right
    assert r[4.0]["kf+hard+guard"]["held_steps"]["mean"] > 100         # a 4 kg wrong total is caught and held
    assert r[4.0]["kf+hard"]["rmse_by_window"]["steady"]["mean"] > r[4.0]["kf"]["rmse_by_window"]["steady"]["mean"]
    # the same wrong total declared with sigma_b = 0.5 kg: a partial correction, still rejected at 4 kg
    hb, gb = sweep.HARD_BV_DTE.name, sweep.GUARD_BV_DTE.name
    assert {hb, gb} <= set(r[4.0])
    assert r[0.0][hb]["mean_abs_res_post"]["mean"] > 0.0              # b_var > 0: the residual is left in place
    assert r[0.0][gb]["held_steps"]["mean"] == 0
    assert r[4.0][gb]["held_steps"]["mean"] > 100
    assert _steady(r[4.0], "kf") < _steady(r[4.0], hb) < _steady(r[4.0], "kf+hard")


def test_sweep_uncertain_total_adds_soft_with_declared_variance():
    r = sweep.run_axis("uncertain_total", n_seeds=2, points=(1.0,))
    a = r[1.0]
    assert sweep.SOFT_NAME in a
    assert a[sweep.SOFT_NAME]["mean_abs_res_post"]["mean"] > 0.0       # soft leaves a residual by design


def test_sweep_uncertain_total_declares_b_var_on_the_constraint_set():
    """Declaring b_var = sigma_b^2 on the set gives the estimate the soft variant gets from
    lam = 1/sigma_b^2 on an exact set (the same pseudo-measurement, two algebraic forms),
    and a guard whose hypothesis holds: its flags are scored as false alarms."""
    r = sweep.run_axis("uncertain_total", n_seeds=2, points=(2.0,))
    a = r[2.0]
    hb, gb = sweep.HARD_BV.name, sweep.GUARD_BV.name
    assert {hb, gb, sweep.SOFT_NAME, "kf+hard", "kf+hard+guard"} <= set(a)
    for key in ("rmse_by_window", "coverage95_by_window", "nz_rms_by_window"):
        assert _steady(a, hb, key) == pytest.approx(_steady(a, sweep.SOFT_NAME, key), abs=1e-9)
    assert not a[gb]["detection"]["applicable"] and a["kf+hard+guard"]["detection"]["applicable"]
    assert a[gb]["false_alarms"]["max"] == 0 and a[gb]["held_steps"]["mean"] == 0
    assert _steady(a, hb, "coverage95_by_window") > 0.9 > _steady(a, "kf+hard", "coverage95_by_window")


@pytest.mark.slow
def test_full_calibration_and_sweep_reproduce_the_committed_files(tmp_path):
    """Full-size calibration and sweep, compared with the committed results/calibration.json
    and results/sweep.json value for value, with only the provenance stamp and the latency
    columns excluded (as tests/test_results_reproduce.py does for the grid)."""
    import json

    from set_lcm.experiments.provenance import REPO_ROOT

    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k != "provenance" and not k.startswith("latency_us")}
        if isinstance(o, list):
            return [strip(v) for v in o]
        return o

    calibration.main(tmp_path, quiet=True)
    sweep.main(tmp_path, quiet=True)
    for name in ("calibration.md", "calibration.json", "sweep.md", "sweep.json"):
        assert (tmp_path / name).exists()
    for name in ("calibration.json", "sweep.json"):
        fresh = json.loads((tmp_path / name).read_text(encoding="utf-8"))
        committed = json.loads((REPO_ROOT / "results" / name).read_text(encoding="utf-8"))
        assert strip(fresh) == strip(committed), f"results/{name} does not match a fresh run of the source tree"
