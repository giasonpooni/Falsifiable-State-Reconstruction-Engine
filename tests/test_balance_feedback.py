"""Feedback must preserve the Gaussian conditional distribution of unreported rates."""
import numpy as np
import pytest

from set_lcm.lcm import project_hard
from set_lcm.schema import ConstraintSet, Observation
from set_lcm.testbed.estimators_balance import (
    BalanceConfig, WaterBalanceAugmented, WaterBalanceClosed, WaterBalanceOpen,
)


def _filter(cls=WaterBalanceAugmented, *, missing=True):
    clock = np.arange(3, dtype=float) * 86400.0
    cfg = BalanceConfig(q_storage=1.0, q_flow=1.0, q_ungauged=1.0,
                        cumulative0_std=1.0, flow0_std=1.0)
    est = cls((100.0,), 1.0, 86400.0, np.zeros(3), cfg, clock=clock)
    observations = []
    for j in range(2):
        obs = Observation(
            t=clock[j], arrival_t=clock[j],
            y=np.full(4, np.nan) if missing else np.array([110.0 + j, 2.0, 7.0, 3.0]),
            R=np.diag([4.0, 1.0, 1.0, 1.0]), mask=np.full(4, not missing),
            source_ids=("storage", "outflow", "inflow1", "inflow2"),
        )
        observations.append(obs)
        est.ingest(obs, j)
    return est, observations


def _constraint(est, variance):
    row = np.array([[1.0] + [-1.0] * (est.n_report - 1)])
    # A nonzero correction checks that the rate means, not only their covariance, move.
    b = row @ est._x[:est.n_report] - 7.0
    return ConstraintSet(version="feedback-test", A=row, b=b, description="test balance",
                         b_var=None if variance is None else np.array([variance]))


def _full_joseph(x, P, cs):
    """Independent reference: condition the entire state on the noisy balance reading."""
    H = np.zeros((1, x.size))
    H[:, :cs.A.shape[1]] = cs.A
    noise = np.zeros((1, 1)) if cs.b_var is None else np.diag(cs.b_var)
    innovation_var = H @ P @ H.T + noise
    gain = np.linalg.solve(innovation_var, H @ P).T
    mean = x + gain @ (cs.b - H @ x)
    residual_map = np.eye(x.size) - gain @ H
    covariance = residual_map @ P @ residual_map.T + gain @ noise @ gain.T
    return mean, covariance


@pytest.mark.parametrize("cls", [WaterBalanceOpen, WaterBalanceAugmented, WaterBalanceClosed])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("variance", [None, 1.0])
def test_feedback_matches_full_state_joseph_update(cls, missing, variance):
    est, observations = _filter(cls, missing=missing)
    n = est.n_report
    prior_x, prior_P = est.report(1)
    prior_x, prior_P = prior_x.copy(), prior_P.copy()
    initial_x, initial_P = est.x0.copy(), est.P0.copy()
    history_x, history_P = est.xf[0].copy(), est.Pf[0].copy()
    observation_copies = [(o.y.copy(), o.R.copy(), o.mask.copy()) for o in observations]
    cs = _constraint(est, variance)
    expected_x, expected_P = _full_joseph(prior_x, prior_P, cs)
    marginal_x, marginal_P = project_hard(prior_x[:n], prior_P[:n, :n], cs)
    passed_x, passed_P = marginal_x.copy(), marginal_P.copy()
    marginal_x.setflags(write=False)
    marginal_P.setflags(write=False)

    est.set_state(marginal_x, marginal_P)

    np.testing.assert_allclose(est._x, expected_x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(est._P, expected_P, rtol=1e-12, atol=1e-12)
    assert np.linalg.norm(est._x[n:] - prior_x[n:]) > 1e-3
    assert np.linalg.eigvalsh(est._P)[0] >= -1e-12
    # report() must use the corrected full state, including across a later prediction.
    np.testing.assert_array_equal(est.xf[-1], est._x)
    np.testing.assert_array_equal(est.Pf[-1], est._P)
    next_x, next_P = est.report(2)
    np.testing.assert_allclose(next_x, est.F @ expected_x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(next_P, est.F @ expected_P @ est.F.T + est.Q,
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(marginal_x, passed_x)
    np.testing.assert_array_equal(marginal_P, passed_P)
    np.testing.assert_array_equal(est.x0, initial_x)
    np.testing.assert_array_equal(est.P0, initial_P)
    np.testing.assert_array_equal(est.xf[0], history_x)
    np.testing.assert_array_equal(est.Pf[0], history_P)
    for obs, (y, R, mask) in zip(observations, observation_copies):
        np.testing.assert_array_equal(obs.y, y)
        np.testing.assert_array_equal(obs.R, R)
        np.testing.assert_array_equal(obs.mask, mask)


def test_missing_observations_no_longer_produce_indefinite_feedback_covariance():
    est, _ = _filter()
    n = est.n_report
    prior_x, prior_P = est.report(1)
    cs = _constraint(est, 1.0)
    marginal_x, marginal_P = project_hard(prior_x[:n], prior_P[:n, :n], cs)
    invalid_block_replacement = prior_P.copy()
    invalid_block_replacement[:n, :n] = marginal_P
    assert np.linalg.eigvalsh(prior_P)[0] > 0.0
    assert np.linalg.eigvalsh(marginal_P)[0] > 0.0
    assert np.linalg.eigvalsh(invalid_block_replacement)[0] < -1.0

    est.set_state(marginal_x, marginal_P)

    assert np.linalg.eigvalsh(est._P)[0] > 0.0


@pytest.mark.parametrize("bad", ["state_nan", "covariance_nan", "asymmetric", "indefinite"])
def test_invalid_feedback_is_refused_without_changing_filter(bad):
    est, _ = _filter()
    x, P = est._x[:3].copy(), est._P[:3, :3].copy()
    if bad == "state_nan":
        x[0] = np.nan
    elif bad == "covariance_nan":
        P[0, 0] = np.nan
    elif bad == "asymmetric":
        P[0, 1] += 1.0
    else:
        P[0, 0] = -1.0
    before_x, before_P = est._x.copy(), est._P.copy()
    with pytest.raises(ValueError):
        est.set_state(x, P)
    np.testing.assert_array_equal(est._x, before_x)
    np.testing.assert_array_equal(est._P, before_P)
    np.testing.assert_array_equal(est.xf[-1], before_x)
    np.testing.assert_array_equal(est.Pf[-1], before_P)


@pytest.mark.parametrize("cls", [WaterBalanceOpen, WaterBalanceAugmented, WaterBalanceClosed])
def test_feedback_before_first_ingest_preserves_the_declared_prior(cls):
    clock = np.arange(3, dtype=float) * 86400.0
    cfg = BalanceConfig(q_storage=1.0, q_flow=1.0, q_ungauged=1.0,
                        cumulative0_std=1.0, flow0_std=1.0)
    est = cls((100.0,), 1.0, 86400.0, np.zeros(3), cfg, clock=clock)
    initial_x, initial_P = est.x0.copy(), est.P0.copy()
    before_x, before_P = est.report(0)

    with pytest.raises(ValueError, match="before the first ingest"):
        est.set_state(initial_x[:est.n_report] + 7.0, np.eye(est.n_report) * 0.5)

    after_x, after_P = est.report(0)
    np.testing.assert_array_equal(after_x, before_x)
    np.testing.assert_array_equal(after_P, before_P)
    np.testing.assert_array_equal(est._x, initial_x)
    np.testing.assert_array_equal(est._P, initial_P)
    np.testing.assert_array_equal(est.x0, initial_x)
    np.testing.assert_array_equal(est.P0, initial_P)
    assert est.xf == [] and est.Pf == []
