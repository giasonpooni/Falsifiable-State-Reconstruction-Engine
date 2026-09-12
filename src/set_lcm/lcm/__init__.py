"""Constraint-reconciliation kernel, Phase 1: linear equality constraints only.

Two formulations, kept explicitly separate because they make different promises.

hard:   x* = argmin (x - x~)^T P^-1 (x - x~)   s.t.  A x = b
        Closed-form KKT solution. Residual after projection is zero to floating
        point. Equivalent to a Kalman update with a noise-free pseudo-measurement.

soft:   x* = argmin (x - x~)^T P^-1 (x - x~) + lam * ||A x - b||^2
        Closed-form linear solve. Residual after projection is NOT zero for any
        finite lam. 1-D check: x~ = 2, constraint x = 1  =>  x* = (2 + lam)/(1 + lam).
        Equivalent to a pseudo-measurement with variance 1/lam.

Weighting by P^-1 is what makes the correction land on the *uncertain*
component instead of being split by fiat, and what stops a 1 kg correction from
being treated like a 1 degree correction.

Consistency check:  r = A x~ - b,  S = A P A^T,  stat = r^T S^-1 r.
Under the hypothesis "the constraint set and the estimate describe the same
system", stat ~ chi^2(rank A). A large stat means the estimate and the declared
constraint disagree by more than the estimate's own uncertainty explains. That
is evidence of model failure -- stale constraint, undeclared sensor bias,
unmodeled leak -- and it is surfaced as MODEL_INCONSISTENT. It does not say which.

Input guards. Calculations require finite, shape-compatible vectors and a finite,
symmetric positive-definite P; residual solves also require a numerically
non-singular S. Invalid inputs raise ValueError. No-constraint reconciliation
permits a positive-semidefinite P because it only copies the state. Soft projection
requires lam > 0; positive infinity explicitly selects hard projection. A covariance
that has already been hard-projected is rank-deficient
by construction (A P* A^T = 0); feeding it back in used to yield stat = 0.0
with Status.OK, which is a silent lie. Duplicated or dependent constraint rows
are reduced to an independent set first, so the dof is rank(A), not rows(A).

Declared constraint uncertainty (ConstraintSet.b_var). When b_var is None the
set is exact and every function below runs exactly the arithmetic described
above -- the b_var branches are separate code, so results for exact sets are
bit-identical to what they were before b_var existed. When b_var is declared,
with Sigma_b its (rows, rows) covariance and S = A P A^T + Sigma_b:

    consistency   stat = r^T S^-1 r ~ chi^2(rank A) under the joint hypothesis,
                  which now includes "b is within its declared uncertainty";
    detectability d(f) = f^T A^T S^-1 A f, which shrinks as Sigma_b grows;
    hard          the Kalman update with pseudo-measurement noise Sigma_b:
                  K = P A^T S^-1, x* = x - K r,
                  P* = (I - K A) P (I - K A)^T + K Sigma_b K^T   (Joseph form).
                  Sigma_b = 0 reduces to the exact projection (tested to 1e-12);
                  for Sigma_b > 0 the residual after projection is NOT zero and
                  P* stays positive definite;
    soft          hard with Sigma_b + (1/lam) I: an extra, undeclared slack on top
                  of the declared uncertainty. (With b_var None, soft is the
                  penalised form above, which equals hard with Sigma_b = (1/lam) I.)

Dependent rows with b_var declared are REFUSED (ValueError), not reduced. The
SVD reduction used for exact sets keeps U_r^T b, which is unbiased with covariance
U_r^T Sigma_b U_r, but it discards U_perp^T b, and that part carries information
about the errors whenever Sigma_b is not isotropic across the dependent rows: an
exact row duplicated by an uncertain one would come out with a positive variance
instead of zero, and two unequal variances would be averaged with equal weights.
The right reduction conditions on U_perp^T b (a Schur complement of U^T Sigma_b U);
it is not built, so merge dependent rows before declaring their uncertainty.
Feasibility is still reported for such sets: a row (or combination of rows) with
nonzero declared variance is a noisy statement and cannot contradict anything;
only the exact part -- the combinations in null(Sigma_b), i.e. the zero-variance
rows of a vector b_var -- can be infeasible.

The guards are unchanged: check_spd(P) still refuses a rank-deficient P whatever
b_var says, and S is still checked for singularity; with Sigma_b > 0, S is
non-singular even where A P A^T is not.

Nothing in this module overwrites the incoming estimate. `reconcile` returns a
StateEstimate that carries the unprojected state, the correction, and both
residuals.
"""
from __future__ import annotations

import numpy as np

from ..schema import ConstraintSet, StateEstimate, Status

__all__ = [
    "chi2_quantile", "is_feasible", "residual", "reduced", "consistency_stat",
    "detectability", "constraint_bases", "project_hard", "project_soft", "reconcile", "check_spd",
]

# Upper quantiles of chi^2(dof). Enough for Phase 1; scipy would replace this table.
_CHI2 = {
    1: {0.95: 3.841, 0.99: 6.635, 0.999: 10.828},
    2: {0.95: 5.991, 0.99: 9.210, 0.999: 13.816},
    3: {0.95: 7.815, 0.99: 11.345, 0.999: 16.266},
}

_SPD_REL_TOL = 1e-12   # smallest eigenvalue must exceed this fraction of the largest
_COND_MAX = 1e12       # S = A P A^T beyond this condition number is treated as singular


def chi2_quantile(dof: int, q: float) -> float:
    try:
        return _CHI2[dof][q]
    except KeyError as e:
        raise ValueError(f"no tabulated chi^2 quantile for dof={dof}, q={q}") from e


def check_spd(P: np.ndarray, name: str = "P") -> None:
    """Raise ValueError unless P is a symmetric positive-definite matrix.

    Validates only; never returns a modified copy, so callers' arithmetic is
    bit-identical to what it was without the check.
    """
    _, w = _covariance_eigenvalues(P, name)
    if w[-1] <= 0.0 or w[0] <= _SPD_REL_TOL * w[-1]:
        raise ValueError(
            f"{name} is not positive definite (eigenvalues {w}); a covariance that has "
            f"already been hard-projected is rank-deficient by construction"
        )


def _finite(value, name: str) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")


def _covariance_eigenvalues(P, name: str) -> tuple[np.ndarray, np.ndarray]:
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1] or P.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty square matrix, got shape {P.shape}")
    _finite(P, name)
    if not np.allclose(P, P.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} is not symmetric")
    # Halve first so even a finite covariance near float's upper limit is validated
    # without overflowing merely while checking symmetry/eigenvalues.
    w = np.linalg.eigvalsh(0.5 * P + 0.5 * P.T)
    _finite(w, f"{name} eigenvalues")
    return P, w


def _check_psd(P, name: str = "P", *, reference_scale: float | None = None) -> None:
    _, w = _covariance_eigenvalues(P, name)
    # Passthrough may receive an all-constrained covariance consisting only of
    # subtraction roundoff. Use the same 1e-12 absolute floor as symmetry checking.
    scale = max(1.0, float(np.abs(w).max())) if reference_scale is None else reference_scale
    if w[0] < -_SPD_REL_TOL * scale:
        raise ValueError(f"{name} is not positive semidefinite (eigenvalues {w})")


def _vector(value, name: str, size: int | None = None) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.ndim != 1 or value.size == 0 or (size is not None and value.size != size):
        expected = "a nonempty 1-D vector" if size is None else f"a 1-D vector of length {size}"
        raise ValueError(f"{name} must be {expected}, got shape {value.shape}")
    _finite(value, name)
    return value


def _state_arrays(x, P, *, psd: bool = False) -> tuple[np.ndarray, np.ndarray]:
    x = _vector(x, "x")
    P = np.asarray(P, dtype=float)
    if P.shape != (x.size, x.size):
        raise ValueError(f"P must have shape {(x.size, x.size)} for x, got {P.shape}")
    (_check_psd if psd else check_spd)(P)
    return x, P


def _constraint_arrays(cs: ConstraintSet) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(cs.A, dtype=float)
    if A.ndim not in (1, 2) or A.size == 0:
        raise ValueError(f"A must be a nonempty row or matrix, got shape {A.shape}")
    A = np.atleast_2d(A)
    _finite(A, "A")
    b = _vector(cs.b, "b", A.shape[0])
    return A, b


def _check_columns(A: np.ndarray, size: int) -> None:
    if A.shape[1] != size:
        raise ValueError(f"A has {A.shape[1]} columns but the state has {size} components")


def _soft_weight(lam) -> float:
    if np.ndim(lam) != 0 or isinstance(lam, (bool, np.bool_)):
        raise ValueError("soft projection needs scalar lam > 0 (positive infinity selects hard)")
    lam = float(lam)
    if not lam > 0.0:
        raise ValueError(f"soft projection needs lam > 0, got {lam!r}")
    return lam


def _nonnegative_stat(value, name: str) -> float:
    if np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative scalar, got {value!r}")
    return value


def _projection_result(x: np.ndarray, P: np.ndarray, prior_P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _finite(x, "projected state")
    # Exact projection can leave only roundoff when every direction is constrained.
    # Judge that roundoff against the incoming covariance, not the near-zero result.
    _check_psd(P, "projected covariance", reference_scale=float(np.abs(prior_P).max()))
    return x, P


def is_feasible(cs: ConstraintSet) -> bool:
    """A x = b has at least one solution  <=>  rank([A | b]) == rank(A).

    With b_var declared only the exact part of the set can contradict itself: for a
    basis N of null(Sigma_b) (the zero-variance rows of a vector b_var), the set is
    feasible iff rank([N^T A | N^T b]) == rank(N^T A). Rows with nonzero declared
    variance never make a set infeasible.
    """
    A, b = _constraint_arrays(cs)
    if cs.b_var is not None:
        N = _exact_directions(cs)
        if N.shape[1] == 0:
            return True
        Ae = N.T @ A
        be = N.T @ b
        return np.linalg.matrix_rank(np.hstack([Ae, be.reshape(-1, 1)])) == np.linalg.matrix_rank(Ae)
    Ab = np.hstack([A, b.reshape(-1, 1)])
    return np.linalg.matrix_rank(Ab) == np.linalg.matrix_rank(A)


def _exact_directions(cs: ConstraintSet) -> np.ndarray:
    """Columns spanning null(Sigma_b): the row combinations b_var declares exact."""
    v = cs.b_var
    if v.ndim == 1:
        return np.eye(v.shape[0])[:, v == 0.0]
    w, V = np.linalg.eigh(v)
    tol = max(float(w[-1]), 0.0) * v.shape[0] * np.finfo(float).eps
    return V[:, w <= tol]


def residual(x: np.ndarray, cs: ConstraintSet) -> np.ndarray:
    A, b = _constraint_arrays(cs)
    x = _vector(x, "x", A.shape[1])
    r = A @ x - b
    _finite(r, "constraint residual")
    return r


def reduced(cs: ConstraintSet) -> tuple[np.ndarray, np.ndarray]:
    """An independent row set (A_r, b_r) equivalent to A x = b.

    Full-row-rank A is returned unchanged (bit-identical arithmetic downstream).
    Dependent rows are reduced through the SVD: A = U S V^T  =>  S_r V_r^T x = U_r^T b,
    which is valid only when the system is consistent; an infeasible set raises.
    A set with dependent rows AND declared b_var raises: U_r^T b alone is not the right
    reduction when b is uncertain (see the module docstring); merge the rows first.
    """
    A, b = _constraint_arrays(cs)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    tol = s.max() * max(A.shape) * np.finfo(float).eps if s.size else 0.0
    r = int((s > tol).sum())
    if r == A.shape[0]:
        return A, b
    if cs.b_var is not None:
        raise ValueError(
            f"constraint set {cs.version!r} has dependent rows (rank {r} < {A.shape[0]} rows) and a "
            "declared b_var; the SVD reduction would discard information about b's errors, so the "
            "kernel refuses it -- merge dependent rows before declaring their uncertainty")
    if not is_feasible(cs):
        raise ValueError("constraint set is infeasible; dependent rows disagree")
    return s[:r, None] * Vt[:r], U[:, :r].T @ b


def _S(A: np.ndarray, P: np.ndarray) -> np.ndarray:
    S = A @ P @ A.T
    _finite(S, "A P A^T")
    if S.size == 0:
        raise ValueError("constraint set has no independent constraint rows")
    if np.linalg.cond(S) > _COND_MAX:
        raise ValueError("A P A^T is numerically singular: P carries no uncertainty along a constraint")
    return S


def _S_declared(A: np.ndarray, P: np.ndarray, Sigma_b: np.ndarray) -> np.ndarray:
    """S = A P A^T + Sigma_b for a set with declared b_var, with the same singularity guard."""
    S = A @ P @ A.T + Sigma_b
    _finite(S, "A P A^T + Sigma_b")
    if np.linalg.cond(S) > _COND_MAX:
        raise ValueError("A P A^T + Sigma_b is numerically singular: neither P nor the declared b_var "
                         "carries uncertainty along a constraint")
    return S


def _pseudo_measurement_update(x: np.ndarray, P: np.ndarray, A: np.ndarray, b: np.ndarray,
                               Sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kalman update with the pseudo-measurement b = A x + e, e ~ N(0, Sigma):
    K = P A^T S^-1 with S = A P A^T + Sigma, x* = x - K r, and P* in Joseph form
    (I - K A) P (I - K A)^T + K Sigma K^T, which stays symmetric PSD by construction."""
    r = A @ x - b
    S = _S_declared(A, P, Sigma)
    K = P @ A.T @ np.linalg.inv(S)
    x_star = x - K @ r
    I_KA = np.eye(P.shape[0]) - K @ A
    P_star = I_KA @ P @ I_KA.T + K @ Sigma @ K.T
    P_star = 0.5 * (P_star + P_star.T)
    return _projection_result(x_star, P_star, P)


def consistency_stat(x: np.ndarray, P: np.ndarray, cs: ConstraintSet) -> float:
    """r^T S^-1 r with S = A P A^T (+ Sigma_b when the set declares b_var); chi^2(rank A)
    under the joint hypothesis."""
    x, P = _state_arrays(x, P)
    A, b = reduced(cs)
    _check_columns(A, x.size)
    r = A @ x - b
    _finite(r, "constraint residual")
    if cs.b_var is not None:
        S = _S_declared(A, P, cs.b_cov)
        return _nonnegative_stat(r @ np.linalg.solve(S, r), "consistency statistic")
    S = _S(A, P)
    return _nonnegative_stat(r @ np.linalg.pinv(S) @ r, "consistency statistic")


def detectability(f, P: np.ndarray, cs: ConstraintSet) -> float:
    """How visible a unit error along direction f is to the consistency statistic:

        d(f) = f^T A^T (A P A^T + Sigma_b)^-1 A f        (Sigma_b = 0 when b is exact)

    An error e = c f shifts the statistic by c^2 d(f). d(f) = 0 exactly when f lies in
    null(A): the constraint test is structurally blind to it, whatever P is. For a
    single sum constraint A = [1, 1] that is every fault that moves mass between the
    reservoirs (pump-rate error, valve transfer). Declared uncertainty on b makes every
    other direction less visible too: d(f) shrinks as Sigma_b grows.
    """
    f = _vector(f, "f")
    _, P = _state_arrays(f, P)
    A, _ = reduced(cs)
    _check_columns(A, f.size)
    Af = A @ f
    _finite(Af, "fault residual")
    if cs.b_var is not None:
        S = _S_declared(A, P, cs.b_cov)
        return _nonnegative_stat(Af @ np.linalg.solve(S, Af), "detectability")
    S = _S(A, P)
    return _nonnegative_stat(Af @ np.linalg.pinv(S) @ Af, "detectability")


def constraint_bases(cs: ConstraintSet) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal bases of row(A) and null(A) from the SVD of A: (V_row, V_null),
    each with basis vectors as rows. For A = [1, 1]: [1, 1]/sqrt2 and [1, -1]/sqrt2."""
    A, _ = _constraint_arrays(cs)
    _, s, Vt = np.linalg.svd(A, full_matrices=True)
    tol = s.max() * max(A.shape) * np.finfo(float).eps if s.size else 0.0
    r = int((s > tol).sum())
    return Vt[:r], Vt[r:]


def project_hard(x: np.ndarray, P: np.ndarray, cs: ConstraintSet) -> tuple[np.ndarray, np.ndarray]:
    """Minimum P^-1-weighted correction that satisfies A x = b exactly -- or, when the
    set declares b_var, the Kalman update with pseudo-measurement noise Sigma_b (Joseph
    form); for Sigma_b > 0 that leaves a nonzero residual and a positive-definite P*."""
    x, P = _state_arrays(x, P)
    A, b = reduced(cs)
    _check_columns(A, x.size)
    if cs.b_var is not None:
        return _pseudo_measurement_update(x, P, A, b, cs.b_cov)
    r = A @ x - b
    _finite(r, "constraint residual")
    S = _S(A, P)
    K = P @ A.T @ np.linalg.pinv(S)
    x_star = x - K @ r
    P_star = P - K @ A @ P
    P_star = 0.5 * (P_star + P_star.T)
    return _projection_result(x_star, P_star, P)


def project_soft(x: np.ndarray, P: np.ndarray, cs: ConstraintSet, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Penalised correction; leaves a nonzero residual for every finite lam.

    With b_var declared: hard with Sigma_b + (1/lam) I, i.e. an extra, undeclared slack of
    variance 1/lam per row on top of the declared uncertainty. For every constraint set,
    lam must be positive; positive infinity explicitly dispatches to project_hard."""
    lam = _soft_weight(lam)
    if np.isposinf(lam):
        return project_hard(x, P, cs)
    x, P = _state_arrays(x, P)
    if cs.b_var is not None:
        A, b = reduced(cs)
        _check_columns(A, x.size)
        Sigma = cs.b_cov + np.eye(A.shape[0]) / lam
        return _pseudo_measurement_update(x, P, A, b, Sigma)
    A, b = _constraint_arrays(cs)
    _check_columns(A, x.size)
    Pinv = np.linalg.inv(P)
    M = Pinv + lam * A.T @ A
    rhs = Pinv @ x + lam * A.T @ b
    _finite(M, "soft precision")
    _finite(rhs, "soft right-hand side")
    x_star = np.linalg.solve(M, rhs)
    P_star = np.linalg.inv(M)
    P_star = 0.5 * (P_star + P_star.T)
    return _projection_result(x_star, P_star, P)


def reconcile(
    x: np.ndarray,
    P: np.ndarray,
    cs: ConstraintSet | None,
    *,
    mode: str | None,
    lam: float = 1.0,
    hold: bool = False,
    threshold: float | None = None,
    stat: float | None = None,
    t: float = 0.0,
    model_version: str = "",
) -> StateEstimate:
    """Produce a derived StateEstimate. Never mutates x or P.

    mode:  None  -> no projection (status SKIPPED; residual still reported if cs given)
           "hard" / "soft"
    hold:  caller has decided (e.g. after debouncing the consistency stat) that
           the constraint should not be enforced this step -> MODEL_INCONSISTENT

    Input validation also applies when skipping/holding and when the caller supplies
    stat or threshold. Without a constraint P may be PSD (this is a passthrough);
    with a constraint it must be SPD. Supplied diagnostics must be finite/nonnegative.
    """
    x = np.array(x, dtype=float, copy=True)
    P = np.array(P, dtype=float, copy=True)
    x, P = _state_arrays(x, P, psd=cs is None)
    if mode not in (None, "hard", "soft"):
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "soft":
        lam = _soft_weight(lam)
    if stat is not None:
        stat = _nonnegative_stat(stat, "stat")
    if threshold is not None:
        threshold = _nonnegative_stat(threshold, "threshold")
    if np.ndim(t) != 0 or not np.isfinite(t):
        raise ValueError("t must be a finite scalar")

    def envelope(status, xs, Ps, r_pre, r_post, corr):
        return StateEstimate(
            t=t, x=xs, P=Ps, model_version=model_version,
            constraint_set_version=cs.version if cs is not None else None,
            status=status, x_unprojected=x, P_unprojected=P,
            residual_pre=r_pre, residual_post=r_post, correction=corr,
            consistency_stat=stat, consistency_threshold=threshold,
        )

    if cs is None:
        return envelope(Status.SKIPPED, x.copy(), P.copy(), None, None, None)

    r_pre = residual(x, cs)
    feasible = is_feasible(cs)
    if stat is None and feasible:
        stat = consistency_stat(x, P, cs)

    if mode is None:
        return envelope(Status.SKIPPED, x.copy(), P.copy(), r_pre, None, None)
    if not feasible:
        return envelope(Status.INFEASIBLE, x.copy(), P.copy(), r_pre, None, None)
    if hold:
        return envelope(Status.MODEL_INCONSISTENT, x.copy(), P.copy(), r_pre, None, None)

    if mode == "hard":
        xs, Ps = project_hard(x, P, cs)
    elif mode == "soft":
        xs, Ps = project_soft(x, P, cs, lam)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    corr = xs - x
    _finite(corr, "correction")
    return envelope(Status.OK, xs, Ps, r_pre, residual(xs, cs), corr)
