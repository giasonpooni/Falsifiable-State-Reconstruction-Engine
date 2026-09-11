"""Kernel-level tests for the LCM projection and consistency check.

The review's numerical sanity check (100k draws, sigma = 2 kg, seed 20260910)
is reproduced here as a regression, with the analytic expectations:
    correct constraint: RMSE 2 -> 2/sqrt(2) = 1.414
    stale constraint  : RMSE 2 -> sqrt(25 + 2) = 5.196
"""
import numpy as np
import pytest

from set_lcm.lcm import (
    chi2_quantile, consistency_stat, is_feasible, project_hard, project_soft, reconcile, reduced,
)
from set_lcm.schema import ConstraintSet, Status

SUM100 = ConstraintSet("closed-v1", np.array([[1.0, 1.0]]), np.array([100.0]), "m1+m2=100")


def test_hard_projection_satisfies_constraint_exactly():
    x = np.array([62.0, 41.0]); P = np.diag([4.0, 4.0])
    xs, Ps = project_hard(x, P, SUM100)
    assert abs(SUM100.A @ xs - SUM100.b)[0] < 1e-12
    np.testing.assert_allclose(xs, [60.5, 39.5])          # equal uncertainty -> equal split
    assert abs(SUM100.A @ Ps @ SUM100.A.T)[0, 0] < 1e-12  # constrained direction has zero variance


def test_hard_projection_weights_by_uncertainty():
    x = np.array([62.0, 41.0]); P = np.diag([1.0, 100.0])
    xs, _ = project_hard(x, P, SUM100)
    corr = xs - x
    assert abs(corr[1] / corr[0] - 100.0) < 1e-9          # correction lands on the uncertain component


def test_soft_projection_leaves_residual_1d():
    cs = ConstraintSet("v", np.array([[1.0]]), np.array([1.0]), "x=1")
    for lam in (0.5, 1.0, 10.0, 1000.0):
        xs, _ = project_soft(np.array([2.0]), np.eye(1), cs, lam)
        assert abs(xs[0] - (2 + lam) / (1 + lam)) < 1e-12
        assert xs[0] != pytest.approx(1.0, abs=1e-6) or lam > 1e5


def test_soft_tends_to_hard_as_lam_grows():
    """Soft -> hard as lam -> inf, but the solve is ill-conditioned (cond ~ lam * A P A^T),
    so lam = 1e9 already loses ~1e-5 kg. Another reason hard is its own formulation."""
    x = np.array([62.0, 41.0]); P = np.diag([1.0, 9.0])
    xh, Ph = project_hard(x, P, SUM100)
    xs, Ps = project_soft(x, P, SUM100, lam=1e6)
    np.testing.assert_allclose(xs, xh, atol=1e-5)
    np.testing.assert_allclose(Ps, Ph, atol=1e-5)


def test_infeasible_constraint_set_is_reported_not_solved():
    cs = ConstraintSet("bad", np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([100.0, 90.0]), "contradiction")
    assert not is_feasible(cs)
    se = reconcile(np.array([60.0, 40.0]), np.diag([4.0, 4.0]), cs, mode="hard")
    assert se.status is Status.INFEASIBLE
    assert se.consistency_stat is None                     # cannot be computed for a contradiction
    np.testing.assert_array_equal(se.x, se.x_unprojected)


def test_reconcile_preserves_unprojected_state_and_reports_both_residuals():
    x = np.array([62.0, 41.0]); P = np.diag([4.0, 4.0])
    se = reconcile(x, P, SUM100, mode="hard", t=1.0, model_version="m")
    assert se.status is Status.OK
    np.testing.assert_array_equal(se.x_unprojected, x)
    assert se.residual_pre[0] == pytest.approx(3.0)
    assert abs(se.residual_post[0]) < 1e-12
    np.testing.assert_allclose(se.correction, [-1.5, -1.5])
    assert se.constraint_set_version == "closed-v1"


def test_hold_returns_model_inconsistent_without_projecting():
    se = reconcile(np.array([55.0, 35.0]), np.diag([0.1, 0.1]), SUM100, mode="hard", hold=True, threshold=10.828)
    assert se.status is Status.MODEL_INCONSISTENT
    np.testing.assert_array_equal(se.x, se.x_unprojected)
    assert se.consistency_stat > se.consistency_threshold


def test_consistency_stat_is_chi2_under_null():
    rng = np.random.default_rng(1)
    P = np.diag([4.0, 4.0]); truth = np.array([60.0, 40.0])
    draws = truth + rng.normal(0, 2.0, (100_000, 2))
    stats = np.array([consistency_stat(d, P, SUM100) for d in draws[:20_000]])
    assert stats.mean() == pytest.approx(1.0, abs=0.05)                       # E[chi2(1)] = 1
    assert (stats > chi2_quantile(1, 0.999)).mean() == pytest.approx(0.001, abs=0.001)


def test_review_sanity_check_reproduced():
    """Vectorised replica of the equal-split projection from the design review."""
    rng = np.random.default_rng(20260910)
    e = rng.normal(0, 2.0, (100_000, 2))

    def rmse_after(truth):
        y = truth + e
        corr = (100.0 - y.sum(axis=1)) / 2.0
        m_hat = y + corr[:, None]
        assert np.abs(m_hat.sum(axis=1) - 100.0).max() < 1e-10
        return np.sqrt(np.mean((y - truth) ** 2)), np.sqrt(np.mean((m_hat - truth) ** 2))

    before, after_ok = rmse_after(np.array([60.0, 40.0]))
    _, after_stale = rmse_after(np.array([55.0, 35.0]))
    assert before == pytest.approx(2.0, abs=0.03)
    assert after_ok == pytest.approx(np.sqrt(2.0), abs=0.03)
    assert after_stale == pytest.approx(np.sqrt(27.0), abs=0.05)

    # and the kernel agrees with the hand formula on a few draws
    for i in range(5):
        xs, _ = project_hard(np.array([60.0, 40.0]) + e[i], np.diag([4.0, 4.0]), SUM100)
        y = np.array([60.0, 40.0]) + e[i]
        np.testing.assert_allclose(xs, y + (100.0 - y.sum()) / 2.0)


# ---------------------------------------------------------------------------
# Input guards: the kernel must refuse what it cannot interpret, not return 0.0
# ---------------------------------------------------------------------------

def test_consistency_stat_refuses_rank_deficient_P():
    """A hard-projected covariance has A P* A^T = 0. Feeding it back used to give
    stat = 0.0 with Status.OK -- a silent lie. Now it raises."""
    x = np.array([62.0, 41.0]); P = np.diag([4.0, 4.0])
    xs, Ps = project_hard(x, P, SUM100)
    with pytest.raises(ValueError):
        consistency_stat(xs, Ps, SUM100)
    with pytest.raises(ValueError):
        reconcile(xs, Ps, SUM100, mode="hard")


@pytest.mark.parametrize("bad", [np.zeros((2, 2)), np.diag([4.0, -1.0]), np.array([[4.0, 1.0], [0.0, 4.0]])])
def test_kernel_refuses_non_spd_P(bad):
    x = np.array([62.0, 41.0])
    with pytest.raises(ValueError):
        consistency_stat(x, bad, SUM100)
    with pytest.raises(ValueError):
        project_hard(x, bad, SUM100)
    with pytest.raises(ValueError):
        project_soft(x, bad, SUM100, 1.0)


def test_duplicated_constraint_row_uses_rank_and_projects_identically():
    """Rows(A) = 2 but rank(A) = 1: the reduced system is m1 + m2 = 100 and the chi-square
    dof is 1. Projection and statistic must match the single-row constraint."""
    cs = ConstraintSet("dup", np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([100.0, 100.0]), "dup")
    assert cs.dof == 2 and cs.rank == 1
    A_r, b_r = reduced(cs)
    assert A_r.shape == (1, 2)
    x = np.array([62.0, 41.0]); P = np.diag([4.0, 4.0])
    xs, Ps = project_hard(x, P, cs)
    np.testing.assert_allclose(xs, [60.5, 39.5])
    assert consistency_stat(x, P, cs) == pytest.approx(consistency_stat(x, P, SUM100))
    se = reconcile(x, P, cs, mode="hard")
    assert se.status is Status.OK and se.residual_pre.shape == (2,)
