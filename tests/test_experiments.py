"""End-to-end tests over the experiment grid. These encode the claims the slice
is supposed to be able to make -- and the negative control it must not fail.

Claims are asserted on means over several seeds; a single seed is a realization.
"""
from functools import lru_cache

import numpy as np
import pytest

from set_lcm.experiments.phase1 import SCENARIOS, SPECS, constraint_for, run_scenario
from set_lcm.testbed.degrade import observe
from set_lcm.testbed.runner import run
from set_lcm.testbed.simulator import simulate

TEST_SEEDS = 8


@lru_cache(maxsize=None)
def agg(name: str) -> dict:
    return run_scenario(name, n_seeds=TEST_SEEDS)[0]


def win(a: dict, est: str, w: str) -> float:
    return a[est]["rmse_by_window"][w]["mean"]


def _single(name, spec_name):
    sc = SCENARIOS[name]
    truth = simulate(sc.sim)
    obs = observe(truth, sc.deg)
    spec = next(s for s in SPECS if s.name == spec_name)
    return run(truth, obs, constraint_for(truth), spec, sc.sim.m0, sc.deg.delay_steps)


def test_determinism_same_seed_identical_output():
    a = _single("leak_stale_constraint", "kf+hard+guard")
    b = _single("leak_stale_constraint", "kf+hard+guard")
    assert np.array_equal(a.x, b.x)
    assert np.array_equal(a.P, b.P)
    assert np.array_equal(a.stat, b.stat)
    assert a.status == b.status


def test_unconstrained_kf_tracks_leak_with_bounded_lag():
    """The estimator only learns about the leak through observations. The KF lags an
    unmodeled 0.05 kg/s drift (~1 kg) but stays far below the 10 kg leak and below the
    hold-last baseline."""
    a = agg("leak_stale_constraint")
    assert win(a, "kf", "leak_and_after") < 1.5
    assert win(a, "kf", "leak_and_after") < win(a, "hold_last", "leak_and_after")


def test_true_constraint_helps_stale_constraint_hurts():
    a = agg("leak_stale_constraint")
    assert win(a, "kf+hard", "pre_leak") < win(a, "kf", "pre_leak")
    assert win(a, "kf+hard", "leak_and_after") > 3.0 * win(a, "kf", "leak_and_after")
    assert a["kf+hard"]["mean_abs_res_post"]["mean"] < 1e-9     # numerically perfect, physically wrong


def test_guard_detects_leak_and_preserves_evidence():
    a = agg("leak_stale_constraint")
    g = a["kf+hard+guard"]
    # KF errors decorrelate over ~40 steps, so a 3-step debounce does not suppress a rare
    # 3.5-sigma excursion; the honest metric is the per-step rate, not "never".
    assert g["false_alarms"]["rate"] < 1e-3
    assert g["detection_delay_steps"]["missed"] == 0
    assert g["detection_delay_steps"]["mean"] < 100
    assert g["held_steps"]["mean"] > 200
    # once held, the reported estimate is the unprojected one and tracks the leak
    assert win(a, "kf+hard+guard", "leak_and_after") < 1.5 * win(a, "kf", "leak_and_after")


def test_constraint_recovers_observability_during_blackout():
    """Pure observability loss: sensor 2 dark, an unmodeled valve moves mass at random.
    The constraint carries sensor 1's information over to reservoir 2."""
    a = agg("closed_blackout_noisy_valve")
    assert win(a, "kf+hard", "blackout") < 0.7 * win(a, "kf", "blackout")


def test_constraint_cannot_fix_a_wrong_parameter():
    """Damaged assumption, not damaged measurement: the pump delivers 20% more than
    commanded. The measured reservoir lags, and the constraint mirrors that lag into the
    dark reservoir. It helps a little, and both estimators are over-confident."""
    a = agg("closed_blackout_pumpbias")
    assert win(a, "kf+hard", "blackout") < win(a, "kf", "blackout")
    assert win(a, "kf+hard", "blackout") > 0.7 * win(a, "kf", "blackout")
    assert a["kf"]["coverage95"]["mean"] < 0.85 and a["kf+hard"]["coverage95"]["mean"] < 0.85


def test_soft_leaves_residual_hard_does_not():
    a = agg("closed_noise")
    assert a["kf+soft(1/lam=4)"]["mean_abs_res_post"]["mean"] > 1e-3
    assert a["kf+hard"]["mean_abs_res_post"]["mean"] < 1e-9


def test_undeclared_sensor_bias_is_flagged_as_inconsistent():
    """The guard cannot tell a leak from a biased sensor; it can only say the
    evidence and the constraint disagree. That is the honest output."""
    g = agg("bias_quant_delay")["kf+hard+guard"]
    assert g["false_alarms"]["rate"] < 1e-3
    assert g["detection_delay_steps"]["missed"] == 0


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_no_solver_failures(name):
    for est, a in agg(name).items():
        assert a["solver_failures"] == 0, est
