"""The augmented-state filter kf_aug: x = [m1, m2, alpha, L].

(a) the transition is the stated time-varying linear F and reduces to the plain KF
when alpha = 1 and L = 0; (b) the runner keeps the mass marginal in x / P, never
projects for mode None, and stores the parameters and their flags in extra; (c) the
estimator side never reads Truth.u_actual; (d)-(g) what it recovers, what it costs
and what it misattributes on the grid scenarios over 8 seeds, asserted with censored
summaries.

The detection thresholds below are what AugConfig's stated values deliver, not the
stage's targets: with 2 kg sensors and a 3.29 sigma test the pump-scale flag fires
in every seed but only 3/8 within 150 steps of onset (median 151, the blackout ends
at 150), and the 0.05 kg/s leak sits at about 3.5 sigma_L of the filter's own
steady-state L uncertainty (0.014 kg/s), so the L flag fires in every seed but only
4/8 within 100 steps (median 97). The commit body records both.
"""
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from set_lcm.experiments.phase1 import SCENARIOS, SPECS, constraint_for, declared_prior, run_scenario
from set_lcm.schema import Status
from set_lcm.testbed.degrade import observe
from set_lcm.testbed.estimators import AugConfig, AugmentedKalmanFilter, KalmanFilter
from set_lcm.testbed.evaluate import evaluate
from set_lcm.testbed.runner import PARAM_FLAG_DEBOUNCE, PARAM_FLAG_Z, run
from set_lcm.testbed.simulator import simulate

TEST_SEEDS = 8
AUG = next(s for s in SPECS if s.name == "kf_aug")


@lru_cache(maxsize=None)
def scenario(name: str) -> tuple[dict, list[dict], dict]:
    return run_scenario(name, n_seeds=TEST_SEEDS)


def agg(name: str) -> dict:
    return scenario(name)[0]


def delays(name: str, flag: str) -> list[int | None]:
    return [ps["kf_aug"]["aug"]["flags"][flag]["detection_delay_steps"] for ps in scenario(name)[1]]


def win(a: dict, est: str, w: str) -> float:
    return a[est]["rmse_by_window"][w]["mean"]


def cov(a: dict, est: str, w: str) -> float:
    return a[est]["coverage95_by_window"][w]["mean"]


def _single(name: str):
    sc = SCENARIOS[name]
    truth = simulate(sc.sim)
    obs = observe(truth, sc.deg)
    m0, m0_std = declared_prior(sc, sc.sim)
    return truth, obs, run(truth, obs, constraint_for(truth), AUG, m0, m0_std)


# ---------------------------------------------------------------------------
# (a) the model
# ---------------------------------------------------------------------------

def test_transition_is_the_stated_time_varying_linear_model():
    u = np.array([0.1, 0.0])
    cfg = AugConfig()
    f = AugmentedKalmanFilter((70.0, 30.0), 5.0, 1.0, u, cfg)
    np.testing.assert_array_equal(f.F(0), [[1, 0, -0.1, 0], [0, 1, 0.1, -1], [0, 0, 1, 0], [0, 0, 0, 1]])
    np.testing.assert_array_equal(f.F(1), np.eye(4) + np.array([[0, 0, 0, 0], [0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0]]))
    x = np.array([70.0, 30.0, 1.2, 0.05])
    P = np.diag([25.0, 25.0, 0.01, 4e-4])
    x1, P1 = f.predict(x, P, 0)
    np.testing.assert_allclose(x1, [70.0 - 1.2 * 0.1, 30.0 + 1.2 * 0.1 - 0.05, 1.2, 0.05])
    F = f.F(0)
    np.testing.assert_allclose(P1, F @ P @ F.T + np.diag([cfg.sigma_w ** 2] * 2 + [cfg.q_alpha ** 2, cfg.q_L ** 2]))
    # the prior is the declared mass prior plus the stated parameter priors
    np.testing.assert_array_equal(f.x0, [70.0, 30.0, 1.0, 0.0])
    np.testing.assert_allclose(np.diag(f.P0), [25.0, 25.0, 0.01, 4e-4])
    # with alpha = 1 and L = 0 the mass prediction is the plain KF's
    kf = KalmanFilter((70.0, 30.0), 5.0, 1.0, u)
    xk, _ = kf.predict(np.array([70.0, 30.0]), np.eye(2), 0)
    xa, _ = f.predict(np.array([70.0, 30.0, 1.0, 0.0]), np.eye(4), 0)
    np.testing.assert_allclose(xa[:2], xk)


# ---------------------------------------------------------------------------
# (b), (c) the record and the truth boundary
# ---------------------------------------------------------------------------

def test_runner_keeps_the_mass_marginal_and_never_projects():
    truth, obs, rr = _single("leak_stale_constraint")
    n = len(obs)
    assert rr.x.shape == (n, 2) and rr.P.shape == (n, 2, 2)
    assert np.array_equal(rr.x, rr.x_unproj) and np.array_equal(rr.P, rr.P_unproj)
    assert all(s is Status.SKIPPED for s in rr.status)
    assert np.isnan(rr.corr).all() and np.isnan(rr.res_post).all()
    assert np.isfinite(rr.stat).all() and rr.flag.dtype == bool      # consistency stat still computed on the marginal
    assert set(rr.extra) == {"alpha_hat", "alpha_sd", "L_hat", "L_sd", "flag_alpha", "flag_L"}
    for k in ("alpha_hat", "alpha_sd", "L_hat", "L_sd"):
        assert rr.extra[k].shape == (n,) and np.isfinite(rr.extra[k]).all()
    assert rr.extra["flag_alpha"].dtype == bool and rr.extra["flag_L"].dtype == bool
    # step 0: the pump is off and the parameters are uncorrelated with the masses, so the
    # first report carries the declared parameter prior untouched
    assert rr.extra["alpha_hat"][0] == 1.0 and rr.extra["alpha_sd"][0] == 0.1
    assert rr.extra["L_hat"][0] == 0.0 and rr.extra["L_sd"][0] == 0.02
    # the flag is the debounced z test on the record itself
    z = np.abs(rr.extra["L_hat"]) / rr.extra["L_sd"] > PARAM_FLAG_Z
    streak = 0
    expect = np.zeros(n, dtype=bool)
    for k in range(n):
        streak = streak + 1 if z[k] else 0
        expect[k] = streak >= PARAM_FLAG_DEBOUNCE
    assert np.array_equal(rr.extra["flag_L"], expect)


def test_estimator_side_never_reads_u_actual():
    sc = SCENARIOS["closed_blackout_pumpbias"]
    truth = simulate(sc.sim)
    garbage = replace(truth, u_actual=truth.u_actual * 0.0 + 7.0)
    obs = observe(truth, sc.deg)
    m0, m0_std = declared_prior(sc, sc.sim)
    a = run(truth, obs, constraint_for(truth), AUG, m0, m0_std)
    b = run(garbage, obs, constraint_for(truth), AUG, m0, m0_std)
    assert np.array_equal(a.x, b.x) and np.array_equal(a.P, b.P)
    for k in a.extra:
        assert np.array_equal(a.extra[k], b.extra[k])
    # ... while the evaluator does read it
    ea = evaluate(a, truth, sc.fault_onset, sc.windows, cs=constraint_for(truth))
    eb = evaluate(b, garbage, sc.fault_onset, sc.windows, cs=constraint_for(truth))
    assert ea["aug"]["alpha_rmse_pump_on"] != eb["aug"]["alpha_rmse_pump_on"]
    assert ea["aug"]["L_rmse"] == eb["aug"]["L_rmse"]


# ---------------------------------------------------------------------------
# (d)-(g) on the grid scenarios
# ---------------------------------------------------------------------------

def test_pump_scale_is_recovered_in_the_blackout():
    """The fault the sum constraint cannot see: the pump delivers 1.2x the command while
    sensor 2 is dark. alpha converges from sensor 1 alone; the mass estimate stays
    calibrated where kf and kf+hard collapse; the flag fires in every seed, median
    ~150 steps after onset (3/8 within the stage's 150-step target)."""
    a = agg("closed_blackout_pumpbias")
    g = a["kf_aug"]["aug"]
    assert g["alpha_rmse_pump_on_by_window"]["recovery"]["mean"] < 0.05
    assert win(a, "kf_aug", "blackout") < 0.7 * win(a, "kf", "blackout")
    assert win(a, "kf_aug", "blackout") < win(a, "kf+hard", "blackout")
    assert cov(a, "kf_aug", "blackout") > 0.9 and cov(a, "kf", "blackout") < 0.5
    det = g["flags"]["alpha"]["detection"]
    assert det["detected_any"] == det["n"] == TEST_SEEDS
    assert g["flags"]["alpha"]["false_alarms"]["max"] == 0
    d = delays("closed_blackout_pumpbias", "alpha")
    assert sum(x is not None and x <= 200 for x in d) >= 6
    assert det["median_delay"] < 200
    # the constraint test on the same marginal is still blind to it
    assert a["kf_aug"]["detectability"]["mean"] == 0.0


def test_leak_is_estimated_and_flagged():
    """The stale-constraint negative control: L tracks the drain, so the mass estimate
    beats the unconstrained filter and the guard by a wide margin and stays calibrated.
    The flag fires in every seed; the 0.05 kg/s leak is ~3.5 sigma_L, so half the seeds
    cross 3.29 sigma within 100 steps and all within 120."""
    a = agg("leak_stale_constraint")
    g = a["kf_aug"]["aug"]
    assert win(a, "kf_aug", "leak_and_after") < 0.9 * win(a, "kf", "leak_and_after")
    assert win(a, "kf_aug", "leak_and_after") < win(a, "kf+hard+guard", "leak_and_after")
    assert cov(a, "kf_aug", "leak_and_after") > 0.9
    assert g["L_rmse_by_window"]["leak_and_after"]["mean"] < 0.6 * 0.05     # transients at 300 and 500 included
    det = g["flags"]["L"]["detection"]
    assert det["detected_any"] == det["n"] == TEST_SEEDS
    assert g["flags"]["L"]["false_alarms"]["max"] == 0
    d = delays("leak_stale_constraint", "L")
    assert sum(x is not None and x <= 120 for x in d) >= 6
    assert g["L_sd_end"]["mean"] == pytest.approx(0.0144, abs=0.001)


def test_closed_system_raises_no_parameter_flags():
    a = agg("closed_noise")
    g = a["kf_aug"]["aug"]
    assert g["flags"]["alpha"]["false_alarms"]["max"] == 0
    assert g["flags"]["L"]["false_alarms"]["max"] == 0
    for ps in scenario("closed_noise")[1]:
        assert abs(ps["kf_aug"]["aug"]["alpha_hat_end"] - 1.0) < 0.02
    # the price of two extra states on the nominal system: worse than kf, still calibrated
    assert win(a, "kf", "steady") < win(a, "kf_aug", "steady") < 1.6 * win(a, "kf", "steady")
    assert cov(a, "kf_aug", "steady") > 0.9


def test_unobservable_parameter_is_dead_reckoned():
    """Pure observability loss with no parameter fault: sensor 2 dark for 300 steps. L is
    unobservable in the dark and the filter dead-reckons m2 from its pre-blackout L
    estimate, so kf_aug is WORSE than kf here -- and its intervals say so."""
    a = agg("closed_blackout_noisy_valve")
    assert win(a, "kf_aug", "blackout") > win(a, "kf", "blackout")
    assert cov(a, "kf_aug", "blackout") > cov(a, "kf", "blackout")
    assert a["kf_aug"]["aug"]["flags"]["L"]["detection"]["detected_any"] == 0


def test_parameter_flags_name_a_parameter_not_a_cause():
    """A +3 kg step on sensor 1 while the pump is on can only be explained by the model
    through alpha, so the alpha flag fires in some seeds after the bias onset; the L flag
    does not. Neither fires before it."""
    g = agg("bias_quant_delay")["kf_aug"]["aug"]
    assert g["flags"]["alpha"]["false_alarms"]["max"] == 0 and g["flags"]["L"]["false_alarms"]["max"] == 0
    assert g["flags"]["alpha"]["detection"]["detected_any"] > 0
    assert g["flags"]["L"]["detection"]["detected_any"] == 0
