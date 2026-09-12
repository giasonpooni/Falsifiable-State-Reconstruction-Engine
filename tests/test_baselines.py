"""The baselines that could remove the case for a constraint row.

kf_closedq   closure written into Q instead of into a constraint row
oracle       a KF given the hidden actual pump rate and leak: a BOUND, not a candidate
kf+hard+fb   the projected (x*, P*) fed back into the filter instead of kept as a
             post-stage over a retained unprojected state

Thresholds below are what the stated (untuned) configs deliver over 8 seeds,
not targets. Two of the stage's expectations did not survive the numbers and
the tests say so where it matters:

- kf_closedq does not land "within 15 %" of kf+hard on the nominal system; it
  is about a third BELOW it (0.104 vs 0.159 kg over 8 seeds). Its declared
  sigma_q = 0.01 kg/s also makes its null-direction process noise 0.014 kg per
  step against KFConfig's untuned 0.05, so it averages longer. The test asserts
  the one-sided claim (the 1/sqrt2 gain is recovered without a row) and that
  the row on top of it still buys the exact total.
- the oracle is below every other estimator in the leak window but not on the
  nominal system, where kf_closedq+hard (0.075) beats it (0.097): the constraint
  row carries the exact total b, which is knowledge of the STATE, and a perfect
  model of the INPUTS cannot supply it. The test names that exception.
"""
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from set_lcm.experiments import sweep
from set_lcm.experiments.phase1 import (
    SCENARIOS, SPECS, constraint_for, declared_prior, oracle_inputs_for, run_scenario, run_spec,
)
from set_lcm.lcm import check_spd, project_hard
from set_lcm.schema import Status
from set_lcm.testbed.degrade import observe
from set_lcm.testbed.estimators import (
    B, ClosedQConfig, ClosedQKalmanFilter, KalmanFilter, OracleConfig, OracleKalmanFilter,
)
from set_lcm.testbed.inputs import PublicInputs
from set_lcm.testbed.runner import EstimatorSpec, run
from set_lcm.testbed.simulator import simulate

TEST_SEEDS = 8
A = np.array([[1.0, 1.0]])


def spec(name: str) -> EstimatorSpec:
    return next(s for s in SPECS if s.name == name)


@lru_cache(maxsize=None)
def scenario(name: str):
    return run_scenario(name, n_seeds=TEST_SEEDS)


def agg(name: str) -> dict:
    return scenario(name)[0]


def win(a: dict, est: str, w: str) -> float:
    return a[est]["rmse_by_window"][w]["mean"]


def cov(a: dict, est: str, w: str) -> float:
    return a[est]["coverage95_by_window"][w]["mean"]


def row(a: dict, est: str, w: str) -> float:
    return a[est]["rms_err_by_direction"][w]["row"]["mean"]


def _single(name: str, sp: EstimatorSpec, truth=None):
    sc = SCENARIOS[name]
    truth = simulate(sc.sim) if truth is None else truth
    obs = observe(simulate(sc.sim), sc.deg)
    m0, m0_std = declared_prior(sc, sc.sim)
    return truth, obs, run_spec(truth, obs, constraint_for(truth), sp, m0, m0_std)


# ---------------------------------------------------------------------------
# kf_closedq: closure in the noise model
# ---------------------------------------------------------------------------

def test_closedq_process_noise_is_the_stated_structure():
    cfg = ClosedQConfig()
    assert cfg.sigma_q == 0.01 and cfg.eps == 1e-8
    f = ClosedQKalmanFilter((70.0, 30.0), 5.0, 1.0, np.zeros(3), cfg)
    np.testing.assert_allclose(f.Q, 1e-4 * np.outer(B, B) + 1e-8 * np.eye(2))
    assert (A @ f.Q @ A.T)[0, 0] == pytest.approx(2e-8)            # the sum direction gets eps only
    # same declared prior and H as kf; only Q differs
    kf = KalmanFilter((70.0, 30.0), 5.0, 1.0, np.zeros(3))
    np.testing.assert_array_equal(f.x0, kf.x0)
    np.testing.assert_array_equal(f.P0, kf.P0)
    assert not np.allclose(f.Q, kf.Q)


def test_closedq_recovers_the_constraint_gain_without_a_row():
    """closed_noise, steady: kf+hard's gain over kf is the 1/sqrt2 of averaging two
    sensors. A filter that asserts closure through Q gets it with no constraint row --
    and more, because its declared sigma_q is smaller than kf's untuned sigma_w in the
    difference direction (8 seeds: kf_closedq 0.104, kf+hard 0.159, kf 0.230). It does
    not know b, so its sum-direction error is not zero; adding the row zeroes it."""
    a = agg("closed_noise")
    assert win(a, "kf_closedq", "steady") <= 1.15 * win(a, "kf+hard", "steady")
    assert win(a, "kf_closedq", "steady") < 0.8 * win(a, "kf", "steady")
    assert cov(a, "kf_closedq", "steady") >= 0.9
    assert row(a, "kf_closedq", "steady") > 0.05                      # no row: b is not known
    assert row(a, "kf_closedq+hard", "steady") < 1e-9
    assert win(a, "kf_closedq+hard", "steady") < win(a, "kf_closedq", "steady")
    # it never projects; its consistency stat is computed on its own marginal
    assert a["kf_closedq"]["mean_correction_norm"] is None
    assert a["kf_closedq"]["false_alarms"]["max"] == 0


def test_closedq_is_confidently_wrong_under_a_stale_constraint():
    """The negative control: closure in Q has nothing to stop enforcing. In the leak
    window it is as over-confident as hard projection (nz far above 1, cov95 near 0),
    and its own consistency statistic still rejects the stale constraint in every seed."""
    a = agg("leak_stale_constraint")
    for est in ("kf_closedq", "kf_closedq+hard"):
        assert win(a, est, "leak_and_after") > 3.0 * win(a, "kf", "leak_and_after")
        assert cov(a, est, "leak_and_after") < 0.2
        assert a[est]["nz_rms_by_window"]["leak_and_after"]["mean"] > 10.0
        det = a[est]["detection"]
        assert det["detected_any"] == det["n"] == TEST_SEEDS


# ---------------------------------------------------------------------------
# oracle (bound)
# ---------------------------------------------------------------------------

def test_hidden_inputs_reach_the_oracle_and_nothing_else():
    """Corrupt Truth.u_actual and Truth.leak and route both truths through the experiment
    code's own path (run_spec): every non-oracle estimator's record is bit-identical; the
    oracle's is not. The runner is handed the same public inputs either way; the hidden
    arrays are routed to the oracle and to nothing else. The oracle cannot be built
    without them, and no other estimator accepts them."""
    sc = SCENARIOS["leak_stale_constraint"]
    truth = simulate(sc.sim)
    garbage = replace(truth, u_actual=truth.u_actual + 0.5, leak=truth.leak + 0.1)
    obs = observe(truth, sc.deg)
    cs = constraint_for(truth)
    m0, m0_std = declared_prior(sc, sc.sim)
    pa, pb = PublicInputs.from_truth(truth), PublicInputs.from_truth(garbage)
    assert np.array_equal(pa.t, pb.t) and np.array_equal(pa.u_commanded, pb.u_commanded)
    for sp in SPECS:
        assert (oracle_inputs_for(sp, garbage) is None) == (sp.kind != "oracle"), sp.name
        a = run_spec(truth, obs, cs, sp, m0, m0_std)
        b = run_spec(garbage, obs, cs, sp, m0, m0_std)
        same = np.array_equal(a.x, b.x) and np.array_equal(a.P, b.P) and np.array_equal(a.stat, b.stat)
        assert same == (sp.kind != "oracle"), sp.name
    with pytest.raises(ValueError):
        OracleKalmanFilter((70.0, 30.0), 5.0, 1.0, truth.u_commanded, OracleConfig())
    with pytest.raises(TypeError):
        KalmanFilter((70.0, 30.0), 5.0, 1.0, truth.u_commanded, oracle_inputs=(truth.u_actual, truth.leak))
    with pytest.raises(TypeError):
        ClosedQKalmanFilter((70.0, 30.0), 5.0, 1.0, truth.u_commanded, oracle_inputs=(truth.u_actual, truth.leak))


def test_oracle_drift_uses_the_hidden_inputs_and_eps_only():
    u_cmd = np.array([0.1, 0.1, 0.0])
    f = OracleKalmanFilter((70.0, 30.0), 5.0, 1.0, u_cmd, OracleConfig(),
                           oracle_inputs=(np.array([0.12, 0.12, 0.0]), np.array([0.0, 0.05, 0.05])))
    np.testing.assert_allclose(f.drift(0), [-0.12, 0.12])
    np.testing.assert_allclose(f.drift(1), [-0.12, 0.12 - 0.05])
    np.testing.assert_allclose(f.drift(2), [0.0, -0.05])
    np.testing.assert_allclose(f.Q, 1e-8 * np.eye(2))


def test_oracle_is_the_bound_where_it_bounds():
    """leak_stale_constraint, leak window: the oracle knows the leak and is below every
    other estimator (8 seeds: 0.128 against kf_aug's 0.394). closed_noise, steady: it is
    below everything except kf_closedq+hard (0.097 vs 0.075), whose constraint row carries
    the exact total b -- knowledge of the state that a perfect model of the inputs does
    not have. Not a candidate: the estimator side never sees what it sees."""
    a = agg("leak_stale_constraint")
    others = [e for e in a if e != "oracle (bound)"]
    assert win(a, "oracle (bound)", "leak_and_after") < min(win(a, e, "leak_and_after") for e in others)
    b = agg("closed_noise")
    others = [e for e in b if e not in ("oracle (bound)", "kf_closedq+hard")]
    assert win(b, "oracle (bound)", "steady") < min(win(b, e, "steady") for e in others)
    assert win(b, "kf_closedq+hard", "steady") < win(b, "oracle (bound)", "steady")


# ---------------------------------------------------------------------------
# feedback
# ---------------------------------------------------------------------------

def test_feedback_requires_a_projection_mode():
    with pytest.raises(ValueError):
        EstimatorSpec("kf+fb", "kf", None, feedback=True)


@pytest.mark.parametrize("cls,cfg", [(KalmanFilter, None), (ClosedQKalmanFilter, ClosedQConfig())])
def test_set_state_is_rank_deficient_until_the_next_predict(cls, cfg):
    """P* from a hard projection has A P* A^T = 0 and the kernel refuses it; one predict
    adds Q and the reported covariance is SPD again with A P A^T = A Q A^T."""
    sc = SCENARIOS["closed_noise"]
    truth = simulate(sc.sim)
    obs = observe(truth, sc.deg)
    cs = constraint_for(truth)
    f = cls((70.0, 30.0), 5.0, 1.0, truth.u_commanded) if cfg is None else cls((70.0, 30.0), 5.0, 1.0, truth.u_commanded, cfg)
    with pytest.raises(ValueError):
        f.set_state(np.array([70.0, 30.0]), np.eye(2))              # never before the first ingest
    f.ingest(obs[0], 0)
    x, P = f.report(0)
    check_spd(P)
    xs, Ps = project_hard(x, P, cs)
    f.set_state(xs, Ps)
    x0, P0 = f.report(0)
    np.testing.assert_array_equal(x0, xs)
    np.testing.assert_array_equal(P0, Ps)
    with pytest.raises(ValueError):
        check_spd(P0)
    x1, P1 = f.report(1)
    check_spd(P1)
    assert (A @ P1 @ A.T)[0, 0] == pytest.approx((A @ f.Q @ A.T)[0, 0])
    assert (A @ x1 - cs.b)[0] == pytest.approx(0.0, abs=1e-9)        # the drift is in null(A)


@pytest.mark.parametrize("name", ["leak_stale_constraint", "bias_quant_delay"])
def test_feedback_reported_P_is_spd_at_every_step(name):
    """kf+hard+fb with and without arrival delay (bias_quant_delay has 5 steps): the run
    completes (the kernel would raise on a non-SPD P), every reported unprojected P is SPD,
    and every applied projection is exact. After the first feedback the filter's OWN
    state satisfies the constraint to within its tiny sum-direction gain, where the
    non-fed-back filter's residual is what the evidence says."""
    truth, obs, rr = _single(name, spec("kf+hard+fb"))
    cs = constraint_for(truth)
    for k in range(len(obs)):
        check_spd(rr.P_unproj[k])
    ok = np.array([s is Status.OK for s in rr.status])
    assert ok.all()                                              # no guard: projected at every step
    assert np.abs(rr.x @ A.T - cs.b).max() < 1e-9
    assert np.abs(np.einsum("ij,kjl,ml->kim", A, rr.P, A)).max() < 1e-9
    _, _, plain = _single(name, spec("kf+hard"))
    r_fb = np.abs(rr.x_unproj[10:] @ A.T - cs.b)
    r_plain = np.abs(plain.x_unproj[10:] @ A.T - cs.b)
    assert r_fb.max() < 0.05 and r_plain.mean() > 0.1
    assert not np.array_equal(rr.x_unproj, plain.x_unproj)


def test_feedback_is_not_better_and_kills_the_consistency_test():
    """leak_stale_constraint: feeding the projection back cannot beat projecting a filter
    that already tracks -- the leak-window RMSE is the same to within 0.5 % (8 seeds:
    3.791 vs 3.790). And the guard on the fed-back filter never fires: a filter that has
    been told the constraint every step no longer disagrees with it, so kf+hard+fb+guard
    stays confidently wrong where kf+hard+guard detects in every seed and lets go."""
    a = agg("leak_stale_constraint")
    fb, hard = win(a, "kf+hard+fb", "leak_and_after"), win(a, "kf+hard", "leak_and_after")
    assert fb >= 0.995 * hard
    assert fb <= 1.005 * hard
    g = a["kf+hard+fb+guard"]
    assert g["detection"]["detected_any"] == 0 and g["held_steps"]["mean"] == 0
    assert win(a, "kf+hard+fb+guard", "leak_and_after") == pytest.approx(fb, rel=1e-3)
    assert cov(a, "kf+hard+fb+guard", "leak_and_after") < 0.15
    ref = a["kf+hard+guard"]["detection"]
    assert ref["detected_any"] == ref["n"] == TEST_SEEDS
    # the evidence-side channel, which reads no constraint, still sees the leak
    assert g["cusum"]["s2"]["detection"]["detected_any"] == TEST_SEEDS
    # same story for the undeclared sensor bias
    b = agg("bias_quant_delay")
    assert b["kf+hard+fb+guard"]["detection"]["detected_any"] == 0
    assert b["kf+hard+guard"]["detection"]["detected_any"] == TEST_SEEDS


def test_feedback_survives_a_stalled_ingest_clock():
    """One observation arrives late and, because ingestion is in sampling order, blocks
    the ones behind it: the filter's last ingested step j stays put for several report
    steps. The state at j has already received its projection and is rank-deficient, so
    it must not be projected again (the kernel's SPD check would refuse it); the reports
    in the stall are predictions from that fed-back state and still get projected."""
    sc = SCENARIOS["closed_noise"]
    truth = simulate(sc.sim)
    obs = observe(truth, sc.deg)
    obs[100] = replace(obs[100], arrival_t=float(truth.t[104]))   # blocks obs 100..103 until step 104
    m0, m0_std = declared_prior(sc, sc.sim)
    cs = constraint_for(truth)
    inputs = PublicInputs.from_truth(truth)
    rr = run(inputs, obs, cs, spec("kf+hard+fb"), m0, m0_std)
    assert all(s is Status.OK for s in rr.status)
    for k in range(len(obs)):
        check_spd(rr.P_unproj[k])
    assert np.abs(rr.x @ A.T - cs.b).max() < 1e-9
    # the stalled reports 100..103 are predictions from the fed-back state at 99: sum-direction
    # uncertainty is A Q A^T per predicted step, growing with the stall length
    aqa = float((A @ KalmanFilter((0.0, 0.0), 1.0, 1.0, truth.u_commanded).Q @ A.T)[0, 0])
    for k in range(100, 104):
        assert float((A @ rr.P_unproj[k] @ A.T)[0, 0]) == pytest.approx((k - 99) * aqa, rel=1e-6)
    # on the shipped scenarios the clock never stalls, so the record is unchanged
    plain = run(inputs, observe(truth, sc.deg), cs, spec("kf+hard+fb"), m0, m0_std)
    assert np.array_equal(rr.x[:100], plain.x[:100])


def test_feedback_into_kf_aug_and_hold_last_raises():
    for kind in ("kf_aug", "hold_last"):
        with pytest.raises(NotImplementedError):
            _single("closed_noise", EstimatorSpec(f"{kind}+hard+fb", kind, "hard", feedback=True))


def test_fed_back_guard_is_dead_on_a_wrong_declared_total():
    """Sweep, declared_total_error at 4 kg, 2 seeds: kf+hard+guard holds for most of the
    run; kf+hard+fb+guard, pinned to the wrong total from step 0, never holds."""
    r = sweep.run_axis("declared_total_error", n_seeds=2, points=(4.0,))[4.0]
    assert {"kf_closedq", "kf+hard+fb+guard"} <= set(r)
    assert r["kf+hard+guard"]["held_steps"]["mean"] > 100
    assert r["kf+hard+fb+guard"]["held_steps"]["mean"] == 0
    assert r["kf+hard+fb+guard"]["rmse_by_window"]["steady"]["mean"] > 5.0 * r["kf"]["rmse_by_window"]["steady"]["mean"]


@pytest.mark.parametrize("name", ["kf+hard+fb+guard", "kf_closedq+hard", "oracle (bound)"])
def test_determinism_of_the_new_specs(name):
    _, _, a = _single("leak_stale_constraint", spec(name))
    _, _, b = _single("leak_stale_constraint", spec(name))
    assert np.array_equal(a.x, b.x) and np.array_equal(a.P, b.P) and np.array_equal(a.stat, b.stat)
    assert a.status == b.status
