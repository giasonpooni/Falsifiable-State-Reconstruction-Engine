"""End-to-end tests over the experiment grid. These encode the claims the slice
is supposed to be able to make -- and the negative control it must not fail.

Claims are asserted on means over several seeds; a single seed is a realization.
Detection claims use censored summaries (k/n detected within N steps, censored
median), never a mean over the seeds that happened to detect.
"""
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from set_lcm.experiments.phase1 import (
    SCENARIOS, SEED_STRIDE, SPECS, constraint_for, declared_prior, run_scenario, scenario_constraint,
)
from set_lcm.schema import ConstraintSet, Observation
from set_lcm.testbed.degrade import DegradeConfig, observe
from set_lcm.testbed.inputs import PublicInputs
from set_lcm.testbed.runner import run
from set_lcm.testbed.simulator import SimConfig, simulate

TEST_SEEDS = 8


@lru_cache(maxsize=None)
def agg(name: str) -> dict:
    return run_scenario(name, n_seeds=TEST_SEEDS)[0]


def win(a: dict, est: str, w: str) -> float:
    return a[est]["rmse_by_window"][w]["mean"]


def cov(a: dict, est: str, w: str) -> float:
    return a[est]["coverage95_by_window"][w]["mean"]


def direction(a: dict, est: str, w: str, which: str, unproj: bool = False) -> float:
    key = "rms_err_unproj_by_direction" if unproj else "rms_err_by_direction"
    return a[est][key][w][which]["mean"]


def _single(name, spec_name):
    sc = SCENARIOS[name]
    truth = simulate(sc.sim)
    obs = observe(truth, sc.deg)
    spec = next(s for s in SPECS if s.name == spec_name)
    m0, m0_std = declared_prior(sc, sc.sim)
    return run(PublicInputs.from_truth(truth), obs, constraint_for(truth), spec, m0, m0_std)


@pytest.mark.parametrize("spec_name", ["kf+hard+guard", "kf_aug"])
def test_determinism_same_seed_identical_output(spec_name):
    a = _single("leak_stale_constraint", spec_name)
    b = _single("leak_stale_constraint", spec_name)
    assert np.array_equal(a.x, b.x)
    assert np.array_equal(a.P, b.P)
    assert np.array_equal(a.stat, b.stat)
    assert a.status == b.status
    assert (a.extra is None) == (b.extra is None)
    if a.extra is not None:
        assert set(a.extra) == set(b.extra)
        for k in a.extra:
            assert np.array_equal(a.extra[k], b.extra[k]), k


def test_observations_are_used_only_after_they_arrive():
    """With a 5-step arrival delay, nothing sampled after step k-5 may influence the
    report at step k. Corrupt every observation sampled after step K-5: reports up to
    K must be bit-identical; the report at K+1 must differ."""
    d, K = 5, 200
    truth = simulate(SimConfig(seed=1))
    obs = observe(truth, DegradeConfig(seed=2, delay_steps=d))
    corrupted = [
        o if j <= K - d else Observation(o.t, o.arrival_t, o.y + 1000.0, o.R, o.mask, o.source_ids)
        for j, o in enumerate(obs)
    ]
    spec = next(s for s in SPECS if s.name == "kf")
    inputs = PublicInputs.from_truth(truth)
    a = run(inputs, obs, constraint_for(truth), spec, (70.0, 30.0), 5.0)
    b = run(inputs, corrupted, constraint_for(truth), spec, (70.0, 30.0), 5.0)
    assert np.array_equal(a.x_unproj[: K + 1], b.x_unproj[: K + 1])
    assert not np.array_equal(a.x_unproj[K + 1], b.x_unproj[K + 1])
    # before anything has arrived the report is the declared prior, predicted forward
    assert np.array_equal(a.x_unproj[0], np.array([70.0, 30.0]))


def test_estimator_forgets_a_wrong_declared_prior():
    """closed_wrong_prior starts the estimator at 74/26 (truth 70/30). The KF must end
    up where closed_noise's KF ends up; hold-last must be worse than the KF while settling."""
    a = agg("closed_wrong_prior")
    ref = agg("closed_noise")
    assert win(a, "kf", "steady") == pytest.approx(win(ref, "kf", "steady"), rel=0.2)
    assert win(a, "kf", "settle") < 2.0
    assert win(a, "kf", "settle") < win(a, "hold_last", "settle")


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
    # ... and confidently so: in-window coverage collapses while the whole-run number hides it
    assert cov(a, "kf+hard", "leak_and_after") < 0.15
    assert a["kf+hard"]["coverage95"]["mean"] > 0.4


def test_guard_detects_leak_and_preserves_evidence():
    a = agg("leak_stale_constraint")
    g = a["kf+hard+guard"]
    # KF errors decorrelate over ~40 steps, so a 3-step debounce does not suppress a rare
    # 3.5-sigma excursion; the honest metric is the per-step rate, not "never".
    assert g["false_alarms"]["rate"] < 1e-3
    det = g["detection"]
    assert det["detected_any"] == det["n"]
    assert det["detected_within"] == det["n"]
    assert not det["median_censored"] and det["median_delay"] < 100
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
    assert cov(a, "kf+hard", "blackout") < 0.1        # the whole-run 0.67 hides an in-window 0.02


def test_error_decomposes_along_the_constraint():
    """Hard projection removes error along row(A) exactly. It is an OBLIQUE projection:
    the P^-1-weighted correction moves along P A^T, which has a null(A) component
    whenever P is anisotropic. So null(A) error is unchanged where P is near-isotropic
    (closed_noise steady state) and changes where it is not (pump-bias blackout, where
    the dark reservoir carries most of the uncertainty and most of the error)."""
    a = agg("closed_blackout_pumpbias")
    assert direction(a, "kf", "blackout", "null", unproj=True) > direction(a, "kf", "blackout", "row", unproj=True)
    assert direction(a, "kf+hard", "blackout", "row") < 1e-9
    assert direction(a, "kf+hard", "blackout", "null") != pytest.approx(
        direction(a, "kf+hard", "blackout", "null", unproj=True), rel=0.05)
    b = agg("closed_noise")
    assert direction(b, "kf+hard", "steady", "row") < 1e-9
    assert direction(b, "kf+hard", "steady", "null") == pytest.approx(
        direction(b, "kf+hard", "steady", "null", unproj=True), rel=0.05)


def test_sum_constraint_is_blind_to_difference_direction_faults():
    """d(f) is exactly zero for the pump-bias and valve faults and positive for leak and bias.
    The blindness shows up as a calibration failure with no flag."""
    for name in ("closed_blackout_pumpbias", "closed_blackout_noisy_valve"):
        a = agg(name)
        for est in ("kf", "kf+hard", "kf+hard+guard"):
            assert a[est]["detectability"]["mean"] == 0.0
        assert a["kf+hard+guard"]["detection"]["detected_within"] <= 1
    for name in ("leak_stale_constraint", "bias_quant_delay"):
        assert agg(name)["kf"]["detectability"]["mean"] > 0.0


def test_soft_leaves_residual_hard_does_not():
    a = agg("closed_noise")
    assert a["kf+soft(1/lam=4)"]["mean_abs_res_post"]["mean"] > 1e-3
    assert a["kf+hard"]["mean_abs_res_post"]["mean"] < 1e-9


def test_undeclared_sensor_bias_is_flagged_as_inconsistent():
    """The guard cannot tell a leak from a biased sensor; it can only say the
    evidence and the constraint disagree. That is the honest output."""
    g = agg("bias_quant_delay")["kf+hard+guard"]
    assert g["false_alarms"]["rate"] < 1e-3
    assert g["detection"]["detected_any"] == g["detection"]["n"]


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_no_solver_failures(name):
    for est, a in agg(name).items():
        assert a["solver_failures"] == 0, est


def test_every_scenario_but_one_declares_the_exact_total():
    """The per-scenario builder hands the six pre-b_var scenarios exactly constraint_for(truth)
    (so their results are unchanged), and closed_uncertain_total an offset total that carries
    its declared variance."""
    for name, sc in SCENARIOS.items():
        truth = simulate(sc.sim)
        cs = scenario_constraint(sc, truth, 3)
        if name == "closed_uncertain_total":
            continue
        ref = constraint_for(truth)
        assert cs.b_var is None and cs.version == ref.version
        np.testing.assert_array_equal(cs.A, ref.A)
        np.testing.assert_array_equal(cs.b, ref.b)
    sc = SCENARIOS["closed_uncertain_total"]
    truth = simulate(sc.sim)
    bs = [float(scenario_constraint(sc, truth, i).b[0]) for i in range(8)]
    cs = scenario_constraint(sc, truth, 0)
    np.testing.assert_array_equal(cs.b_var, [1.0])
    assert all(b != truth.total0 for b in bs) and len(set(bs)) == 8
    assert bs == [float(scenario_constraint(sc, truth, i).b[0]) for i in range(8)]   # seeded, reproducible
    assert sc.fault_onset is None


def test_declared_constraint_uncertainty_keeps_hard_projection_calibrated():
    """closed_uncertain_total: b is off by N(0, 1 kg^2) per seed and says so (b_var = 1).
    Hard projection honours the declaration -- a partial correction, residual left in
    place, calibrated -- and the guard, testing r^T (A P A^T + 1)^-1 r, does not fire.
    The same seed with the same wrong b declared EXACT is over-confident. Feeding the
    declared projection back is not rescued by b_var: the filter is told the same
    uncertain b every step as if it were fresh evidence and becomes over-confident."""
    a = agg("closed_uncertain_total")
    h = a["kf+hard"]
    assert cov(a, "kf+hard", "steady") > 0.9
    assert h["nz_rms_by_window"]["steady"]["mean"] < 1.0
    assert h["mean_abs_res_post"]["mean"] > 0.1                   # not an exact projection
    g = a["kf+hard+guard"]
    assert g["false_alarms"]["rate"] < 1e-3 and g["held_steps"]["mean"] < 1.0
    assert not g["detection"]["applicable"]                         # the joint hypothesis holds
    assert cov(a, "kf+hard+fb", "steady") < 0.8
    assert a["kf+hard+fb"]["nz_rms_by_window"]["steady"]["mean"] > 1.3

    sc = SCENARIOS["closed_uncertain_total"]
    i = 3                                                           # seed index 3: b off by +1.42 kg
    sim = replace(sc.sim, seed=sc.sim.seed + SEED_STRIDE * i)
    truth = simulate(sim)
    obs = observe(truth, replace(sc.deg, seed=sc.deg.seed + SEED_STRIDE * i))
    m0, m0_std = declared_prior(sc, sim)
    declared = scenario_constraint(sc, truth, i)
    assert float(declared.b[0]) - truth.total0 > 1.0
    exact = ConstraintSet(declared.version, declared.A, declared.b, declared.description)
    spec = next(s for s in SPECS if s.name == "kf+hard")
    lo, hi = sc.windows["steady"]

    def steady_cov(rr):
        sd = np.sqrt(np.stack([np.diag(p) for p in rr.P[lo:hi]]))
        return float((np.abs(rr.x[lo:hi] - truth.m[lo:hi]) <= 1.96 * sd).mean())

    inputs = PublicInputs.from_truth(truth)
    with_var = run(inputs, obs, declared, spec, m0, m0_std)
    without = run(inputs, obs, exact, spec, m0, m0_std)
    assert steady_cov(with_var) > 0.9 > 0.5 > steady_cov(without)
    # the unprojected filter is the same either way; only the reconciliation differs
    np.testing.assert_array_equal(with_var.x_unproj, without.x_unproj)
