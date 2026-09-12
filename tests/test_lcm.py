"""Kernel-level tests for the LCM projection and consistency check.

The review's numerical sanity check (100k draws, sigma = 2 kg, seed 20260910)
is reproduced here as a regression, with the analytic expectations:
    correct constraint: RMSE 2 -> 2/sqrt(2) = 1.414
    stale constraint  : RMSE 2 -> sqrt(25 + 2) = 5.196
"""
import numpy as np
import pytest

from set_lcm.lcm import (
    check_spd, chi2_quantile, consistency_stat, constraint_bases, detectability, is_feasible,
    project_hard, project_soft, reconcile, reduced, residual,
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


# ---------------------------------------------------------------------------
# Detectability: what a sum constraint can and cannot see
# ---------------------------------------------------------------------------

def test_constraint_bases_for_sum_constraint():
    v_row, v_null = constraint_bases(SUM100)
    assert v_row.shape == (1, 2) and v_null.shape == (1, 2)
    np.testing.assert_allclose(np.abs(v_row[0]), [1, 1] / np.sqrt(2))
    np.testing.assert_allclose(np.abs(v_null[0]), [1, 1] / np.sqrt(2))
    assert abs(v_row[0] @ v_null[0]) < 1e-12
    assert abs(SUM100.A @ v_null[0]) < 1e-12          # null vector is annihilated by A


@pytest.mark.parametrize("P", [np.diag([4.0, 4.0]), np.diag([0.1, 3.0]), np.array([[2.0, 0.5], [0.5, 1.0]])])
def test_difference_direction_is_exactly_undetectable_by_a_sum_constraint(P):
    """f = (-1, 1) lies in null(A) for A = [1, 1]: d(f) is identically zero, whatever P."""
    assert detectability((-1.0, 1.0), P, SUM100) == 0.0
    assert detectability((0.0, -1.0), P, SUM100) > 0.0
    assert detectability((1.0, 0.0), P, SUM100) > 0.0


def test_detectability_scales_the_statistic():
    """An error c f shifts the consistency statistic by exactly c^2 d(f) from a consistent state."""
    P = np.diag([0.1, 0.3]); x_true = np.array([60.0, 40.0])
    for f, c in (((0.0, -1.0), 2.0), ((1.0, 0.0), -1.5), ((0.3, 0.9), 0.7)):
        d = detectability(f, P, SUM100)
        shifted = consistency_stat(x_true + c * np.asarray(f), P, SUM100)
        assert shifted == pytest.approx(c * c * d)


# ---------------------------------------------------------------------------
# Declared constraint uncertainty (ConstraintSet.b_var)
# ---------------------------------------------------------------------------

def _declared(cs: ConstraintSet, b_var, b=None) -> ConstraintSet:
    return ConstraintSet(cs.version, cs.A, cs.b if b is None else b, cs.description, b_var=b_var)


PS = [np.diag([4.0, 4.0]), np.diag([0.1, 3.0]), np.array([[2.0, 0.5], [0.5, 1.0]]),
      np.array([[0.2, -0.05], [-0.05, 0.3]])]


@pytest.mark.parametrize("P", PS)
def test_soft_equals_hard_with_declared_variance_one_over_lam(P):
    """soft(lam) on an exact set IS the pseudo-measurement with variance 1/lam: the
    penalised (information) form and the Kalman (Joseph) form agree to 1e-12, with
    b_var as a vector or as a matrix. (At lam = 1e4 the penalised form itself loses
    ~1e-10 to its conditioning -- see test_soft_tends_to_hard_as_lam_grows.)"""
    x = np.array([62.0, 41.0])
    for lam in (0.01, 0.25, 1.0, 4.0, 100.0):
        xs, Ps = project_soft(x, P, SUM100, lam)
        for b_var in (np.array([1.0 / lam]), np.array([[1.0 / lam]])):
            xh, Ph = project_hard(x, P, _declared(SUM100, b_var))
            np.testing.assert_allclose(xh, xs, rtol=0, atol=1e-12)
            np.testing.assert_allclose(Ph, Ps, rtol=0, atol=1e-12)


def test_soft_equals_hard_with_declared_variance_two_independent_rows():
    A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]); b = np.array([100.0, 50.0])
    cs = ConstraintSet("two", A, b, "two rows")
    x = np.array([60.0, 41.0, 11.0]); P = np.array([[2.0, 0.3, 0.1], [0.3, 1.5, -0.2], [0.1, -0.2, 0.7]])
    for lam in (0.1, 1.0, 10.0):
        xs, Ps = project_soft(x, P, cs, lam)
        xh, Ph = project_hard(x, P, ConstraintSet("two", A, b, "two rows", b_var=np.eye(2) / lam))
        np.testing.assert_allclose(xh, xs, rtol=0, atol=1e-12)
        np.testing.assert_allclose(Ph, Ps, rtol=0, atol=1e-12)


@pytest.mark.parametrize("P", PS)
def test_hard_with_zero_declared_variance_is_the_exact_projection(P):
    x = np.array([62.0, 41.0])
    xe, Pe = project_hard(x, P, SUM100)
    for b_var in (np.zeros(1), np.zeros((1, 1))):
        xz, Pz = project_hard(x, P, _declared(SUM100, b_var))
        np.testing.assert_allclose(xz, xe, rtol=0, atol=1e-12)
        np.testing.assert_allclose(Pz, Pe, rtol=0, atol=1e-12)
    assert consistency_stat(x, P, _declared(SUM100, np.zeros(1))) == pytest.approx(
        consistency_stat(x, P, SUM100), rel=1e-12)


def test_hard_with_declared_variance_is_a_calibrated_partial_correction():
    """b_var > 0: the residual after projection is not zero, the correction shrinks by
    S_P / (S_P + Sigma_b) along the constraint, and P* is SPD (Joseph form)."""
    x = np.array([62.0, 41.0]); P = np.diag([4.0, 4.0])
    se = reconcile(x, P, _declared(SUM100, np.array([8.0])), mode="hard")
    assert se.status is Status.OK
    assert se.residual_pre[0] == pytest.approx(3.0)
    assert se.residual_post[0] == pytest.approx(3.0 * 8.0 / (8.0 + 8.0))     # A P A^T = 8
    np.testing.assert_allclose(se.correction, [-0.75, -0.75])
    check_spd(se.P)
    assert (SUM100.A @ se.P @ SUM100.A.T)[0, 0] == pytest.approx(8.0 * 8.0 / 16.0)
    np.testing.assert_array_equal(se.x_unprojected, x)
    # soft on a declared set adds an undeclared 1/lam on top of b_var
    xs, Ps = project_soft(x, P, _declared(SUM100, np.array([8.0])), lam=0.25)
    xh, Ph = project_hard(x, P, _declared(SUM100, np.array([12.0])))
    np.testing.assert_allclose(xs, xh, rtol=0, atol=1e-12)
    np.testing.assert_allclose(Ps, Ph, rtol=0, atol=1e-12)
    with pytest.raises(ValueError):
        project_soft(x, P, _declared(SUM100, np.array([8.0])), lam=0.0)


def test_consistency_stat_with_declared_variance_is_chi2_under_null():
    """i.i.d. null: x ~ N(truth, P) and b ~ N(A truth, Sigma_b). The statistic with
    S = A P A^T + Sigma_b is chi2(1); the exact-set statistic on the same draws is not."""
    rng = np.random.default_rng(2)
    P = np.diag([0.1, 0.3]); truth = np.array([60.0, 40.0]); sb2 = 0.5
    xs = truth + rng.multivariate_normal(np.zeros(2), P, 20_000)
    bs = 100.0 + rng.normal(0.0, np.sqrt(sb2), 20_000)
    stats = np.array([consistency_stat(x, P, _declared(SUM100, np.array([sb2]), b=np.array([b])))
                      for x, b in zip(xs, bs)])
    assert stats.mean() == pytest.approx(1.0, abs=0.05)
    assert (stats > chi2_quantile(1, 0.999)).mean() == pytest.approx(0.001, abs=0.001)
    exact = np.array([consistency_stat(x, P, ConstraintSet("e", SUM100.A, np.array([b]), "e"))
                      for x, b in zip(xs[:2000], bs[:2000])])
    assert exact.mean() > 2.0                     # (0.4 + 0.5) / 0.4: b's spread read as model failure


def test_detectability_shrinks_as_declared_variance_grows():
    P = np.diag([0.1, 0.3]); f = (0.0, -1.0)
    d = [detectability(f, P, _declared(SUM100, np.array([v]))) for v in (0.0, 0.1, 1.0, 10.0)]
    assert d[0] == pytest.approx(detectability(f, P, SUM100), rel=1e-12)
    assert all(a > b for a, b in zip(d, d[1:]))
    assert d[2] == pytest.approx(1.0 / (0.4 + 1.0))
    assert detectability((-1.0, 1.0), P, _declared(SUM100, np.array([1.0]))) == 0.0   # null(A) stays blind


def test_declared_variance_makes_S_nonsingular_but_not_P():
    """Two nearly parallel rows: A P A^T is numerically singular and the exact kernel
    refuses; with Sigma_b = I, S is well-conditioned and the statistic is computed. A
    rank-deficient P is still refused whatever b_var says."""
    A = np.array([[1.0, 0.0], [1.0, 1e-7]]); b = np.array([60.0, 60.0])
    x = np.array([60.5, 1.0]); P = np.eye(2)
    with pytest.raises(ValueError):
        consistency_stat(x, P, ConstraintSet("near", A, b, "near"))
    cs = ConstraintSet("near", A, b, "near", b_var=np.ones(2))
    assert np.isfinite(consistency_stat(x, P, cs))
    xs, Ps = project_hard(np.array([62.0, 41.0]), np.diag([4.0, 4.0]), SUM100)
    for c in (_declared(SUM100, np.array([1.0])), _declared(SUM100, np.array([[1.0]]))):
        with pytest.raises(ValueError):
            consistency_stat(xs, Ps, c)
        with pytest.raises(ValueError):
            project_hard(xs, Ps, c)


def test_feasibility_only_exact_rows_can_contradict():
    """Identical rows with b = (100, 90): infeasible when both are exact (zero variance,
    a zero matrix, or perfectly correlated errors, which make b1 - b2 exact); feasible
    as soon as one of them is a noisy statement. The kernel then refuses the dependent
    rows rather than reduce them (see the lcm docstring); reconcile reports INFEASIBLE
    for the contradictory sets without computing anything."""
    A = np.array([[1.0, 1.0], [1.0, 1.0]]); b = np.array([100.0, 90.0])
    x = np.array([62.0, 41.0]); P = np.diag([4.0, 4.0])
    for bv in (np.zeros(2), np.zeros((2, 2)), np.array([[1.0, 1.0], [1.0, 1.0]])):
        cs = ConstraintSet("c", A, b, "c", b_var=bv)
        assert not is_feasible(cs)
        se = reconcile(x, P, cs, mode="hard")
        assert se.status is Status.INFEASIBLE and se.consistency_stat is None
    for bv in (np.array([0.0, 1.0]), np.eye(2), np.array([[1.0, 0.5], [0.5, 1.0]])):
        cs = ConstraintSet("c", A, b, "c", b_var=bv)
        assert is_feasible(cs)
        for fn in (lambda: consistency_stat(x, P, cs), lambda: project_hard(x, P, cs),
                   lambda: project_soft(x, P, cs, 1.0), lambda: reconcile(x, P, cs, mode="hard")):
            with pytest.raises(ValueError, match="dependent rows"):
                fn()
    # independent rows are always feasible, exact or not
    assert is_feasible(ConstraintSet("i", np.eye(2), b, "i", b_var=np.zeros(2)))


@pytest.mark.parametrize("bad", [
    np.array([1.0, 1.0]),                    # wrong length for one row
    np.ones((1, 2)),                         # wrong matrix shape
    np.array(1.0),                           # scalar: must be (rows,) or (rows, rows)
    np.ones((1, 1, 1)),
    np.array([-0.1]),                        # negative variance
    np.array([np.nan]),
    np.array([np.inf]),
    np.array([[-1.0]]),                      # negative 1x1 covariance
])
def test_invalid_b_var_raises_on_one_row(bad):
    with pytest.raises(ValueError):
        ConstraintSet("v", SUM100.A, SUM100.b, "s", b_var=bad)


@pytest.mark.parametrize("bad", [
    np.array([[1.0, 0.5], [0.0, 1.0]]),      # asymmetric
    np.array([[1.0, 2.0], [2.0, 1.0]]),      # symmetric but indefinite
    np.array([1.0, -1.0]),                   # one negative variance
    np.ones(3),                              # wrong length
])
def test_invalid_b_var_raises_on_two_rows(bad):
    with pytest.raises(ValueError):
        ConstraintSet("v", np.eye(2), np.array([60.0, 40.0]), "two", b_var=bad)


def test_b_var_is_validated_copied_and_read_only():
    src = np.array([1.0])
    cs = ConstraintSet("v", SUM100.A, SUM100.b, "s", b_var=src)
    src[0] = -5.0                                    # mutating the caller's array does not reach the set
    assert cs.b_var[0] == 1.0 and not cs.exact and SUM100.exact and SUM100.b_cov is None
    with pytest.raises(ValueError):
        cs.b_var[0] = -5.0                           # the stored copy is read-only
    np.testing.assert_array_equal(cs.b_cov, [[1.0]])
    two = ConstraintSet("v", np.eye(2), np.zeros(2), "two", b_var=np.array([[2.0, 0.5], [0.5, 1.0]]))
    np.testing.assert_array_equal(two.b_cov, [[2.0, 0.5], [0.5, 1.0]])
    assert ConstraintSet("v", np.eye(2), np.zeros(2), "two", b_var=np.zeros((2, 2))).b_cov.shape == (2, 2)


# Audit regressions: malformed inputs must never become successful estimates.

@pytest.mark.parametrize("lam", [-2.0, -0.5, 0.0, -np.inf, np.nan, True, [1.0]])
@pytest.mark.parametrize("b_var", [None, np.array([1.0])])
def test_soft_rejects_invalid_weight_for_exact_and_uncertain_constraints(lam, b_var):
    cs = ConstraintSet("one", np.array([[1.0]]), np.array([1.0]), "x=1", b_var=b_var)
    # lam=-2 formerly returned x=0, P=-1 and Status.OK; NaN returned all NaNs.
    with pytest.raises(ValueError, match="lam"):
        project_soft(np.array([2.0]), np.eye(1), cs, lam)
    with pytest.raises(ValueError, match="lam"):
        reconcile(np.array([2.0]), np.eye(1), cs, mode="soft", lam=lam, stat=0.0)


@pytest.mark.parametrize("b_var", [None, np.array([0.0]), np.array([1.0])])
def test_positive_infinite_soft_weight_is_exactly_hard(b_var):
    cs = ConstraintSet("one", np.array([[1.0]]), np.array([1.0]), "x=1", b_var=b_var)
    hard = project_hard(np.array([2.0]), np.eye(1), cs)
    soft = project_soft(np.array([2.0]), np.eye(1), cs, np.inf)
    for a, b in zip(hard, soft):
        np.testing.assert_array_equal(a, b)
    out = reconcile(np.array([2.0]), np.eye(1), cs, mode="soft", lam=np.inf)
    assert out.status is Status.OK
    np.testing.assert_array_equal(out.x, hard[0])


@pytest.mark.parametrize("bad", [
    np.array([np.nan, 40.0]), np.array([np.inf, 40.0]),
    np.array([[60.0], [40.0]]), np.array([60.0]), np.empty(0),
])
def test_state_and_direction_must_be_finite_matching_vectors(bad):
    calls = [
        lambda: consistency_stat(bad, np.eye(2), SUM100),
        lambda: detectability(bad, np.eye(2), SUM100),
        lambda: project_hard(bad, np.eye(2), SUM100),
        lambda: project_soft(bad, np.eye(2), SUM100, 1.0),
        lambda: residual(bad, SUM100),
        lambda: reconcile(bad, np.eye(2), SUM100, mode="hard", stat=0.0),
        lambda: reconcile(bad, np.eye(2), None, mode=None),
    ]
    for call in calls:
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("bad", [
    np.diag([np.inf, 1.0]), np.diag([np.nan, 1.0]),
    np.array([[1.0, 2.0], [2.0, 1.0]]), np.array([[1.0, 1.0], [0.0, 1.0]]),
    np.eye(3), np.empty((0, 0)), np.ones(2),
])
def test_covariance_validation_cannot_be_bypassed_by_supplied_stat_or_skipping(bad):
    for cs, mode, hold in [(SUM100, "hard", False), (SUM100, None, False),
                           (SUM100, "hard", True), (None, None, False)]:
        with pytest.raises(ValueError):
            reconcile(np.array([60.0, 40.0]), bad, cs, mode=mode, hold=hold, stat=0.0)
    if bad.shape != (3, 3):  # identity(3) is SPD, but mismatches the two-component state
        with pytest.raises(ValueError):
            check_spd(bad)


def test_no_constraint_passthrough_accepts_psd_and_preserves_copies():
    x, P = project_hard(np.array([62.0, 41.0]), np.eye(2), SUM100)
    out = reconcile(x, P, None, mode=None)
    assert out.status is Status.SKIPPED
    np.testing.assert_array_equal(out.x, x)
    np.testing.assert_array_equal(out.P, P)
    assert not np.shares_memory(out.x, x) and not np.shares_memory(out.P, P)
    with pytest.raises(ValueError):
        reconcile(x, P, SUM100, mode="hard", stat=0.0)
    zero = reconcile(np.zeros(2), np.zeros((2, 2)), None, mode=None)
    assert zero.status is Status.SKIPPED


@pytest.mark.parametrize("name", ["stat", "threshold"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -1.0, [1.0]])
def test_supplied_diagnostics_are_finite_nonnegative_scalars(name, bad):
    for cs in (None, SUM100):
        with pytest.raises(ValueError, match=name):
            reconcile(np.array([60.0, 40.0]), np.eye(2), cs, mode=None, **{name: bad})


@pytest.mark.parametrize("A,b", [
    (np.array([[np.nan, 1.0]]), np.array([100.0])),
    (np.array([[1.0, 1.0]]), np.array([np.inf])),
    (np.array([[1.0, 1.0]]), np.array([100.0, 100.0])),
    (np.eye(3), np.zeros(3)),
    (np.empty((0, 2)), np.empty(0)),
])
def test_constraint_validation_prevents_broadcasting_and_nonfinite_estimates(A, b):
    cs = ConstraintSet("invalid", A, b, "invalid")
    with pytest.raises(ValueError):
        reconcile(np.array([60.0, 40.0]), np.eye(2), cs, mode="hard", stat=0.0)


@pytest.mark.parametrize("kwargs", [{"mode": "typo"}, {"mode": "soft", "lam": -1.0},
                                    {"mode": None, "t": np.nan}])
def test_configuration_is_checked_even_without_a_constraint(kwargs):
    with pytest.raises(ValueError):
        reconcile(np.zeros(2), np.eye(2), None, **kwargs)


def test_overflow_does_not_produce_a_successful_estimate():
    cs = ConstraintSet("huge", np.array([[2.0]]), np.array([0.0]), "2x=0")
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="finite"):
            reconcile(np.array([1e308]), np.eye(1), cs, mode="hard", stat=0.0)


def test_full_rank_exact_projection_accepts_covariance_roundoff():
    P = np.array([[2.0, 0.7], [0.7, 1.0]])
    cs = ConstraintSet("two", np.array([[1.0, 1.0], [1.0, -1.0]]), np.array([3.0, 1.0]), "two")
    xs, Ps = project_hard(np.array([8.0, 4.0]), P, cs)
    np.testing.assert_allclose(xs, [2.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(Ps, np.zeros((2, 2)), atol=1e-12)
    out = reconcile(xs, Ps, None, mode=None)
    assert out.status is Status.SKIPPED
    np.testing.assert_array_equal(out.P, Ps)
