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

Input guards. Every entry point checks that P is symmetric positive definite
and that S = A P A^T is numerically non-singular, and raises ValueError
otherwise. A covariance that has already been hard-projected is rank-deficient
by construction (A P* A^T = 0); feeding it back in used to yield stat = 0.0
with Status.OK, which is a silent lie. Duplicated or dependent constraint rows
are reduced to an independent set first, so the dof is rank(A), not rows(A).

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
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {P.shape}")
    if not np.allclose(P, P.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} is not symmetric")
    w = np.linalg.eigvalsh(0.5 * (P + P.T))
    if w[-1] <= 0.0 or w[0] <= _SPD_REL_TOL * w[-1]:
        raise ValueError(
            f"{name} is not positive definite (eigenvalues {w}); a covariance that has "
            f"already been hard-projected is rank-deficient by construction"
        )


def is_feasible(cs: ConstraintSet) -> bool:
    """A x = b has at least one solution  <=>  rank([A | b]) == rank(A)."""
    A = np.atleast_2d(np.asarray(cs.A, dtype=float))
    Ab = np.hstack([A, np.asarray(cs.b, dtype=float).reshape(-1, 1)])
    return np.linalg.matrix_rank(Ab) == np.linalg.matrix_rank(A)


def residual(x: np.ndarray, cs: ConstraintSet) -> np.ndarray:
    return np.atleast_2d(cs.A) @ x - np.asarray(cs.b, dtype=float).reshape(-1)


def reduced(cs: ConstraintSet) -> tuple[np.ndarray, np.ndarray]:
    """An independent row set (A_r, b_r) equivalent to A x = b.

    Full-row-rank A is returned unchanged (bit-identical arithmetic downstream).
    Dependent rows are reduced through the SVD: A = U S V^T  =>  S_r V_r^T x = U_r^T b,
    which is valid only when the system is consistent; an infeasible set raises.
    """
    A = np.atleast_2d(np.asarray(cs.A, dtype=float))
    b = np.asarray(cs.b, dtype=float).reshape(-1)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    tol = s.max() * max(A.shape) * np.finfo(float).eps if s.size else 0.0
    r = int((s > tol).sum())
    if r == A.shape[0]:
        return A, b
    if not is_feasible(cs):
        raise ValueError("constraint set is infeasible; dependent rows disagree")
    return s[:r, None] * Vt[:r], U[:, :r].T @ b


def _S(A: np.ndarray, P: np.ndarray) -> np.ndarray:
    S = A @ P @ A.T
    if np.linalg.cond(S) > _COND_MAX:
        raise ValueError("A P A^T is numerically singular: P carries no uncertainty along a constraint")
    return S


def consistency_stat(x: np.ndarray, P: np.ndarray, cs: ConstraintSet) -> float:
    check_spd(P)
    A, b = reduced(cs)
    r = A @ x - b
    S = _S(A, P)
    return float(r @ np.linalg.pinv(S) @ r)


def detectability(f, P: np.ndarray, cs: ConstraintSet) -> float:
    """How visible a unit error along direction f is to the consistency statistic:

        d(f) = f^T A^T (A P A^T)^-1 A f

    An error e = c f shifts the statistic by c^2 d(f). d(f) = 0 exactly when f lies in
    null(A): the constraint test is structurally blind to it, whatever P is. For a
    single sum constraint A = [1, 1] that is every fault that moves mass between the
    reservoirs (pump-rate error, valve transfer).
    """
    check_spd(P)
    A, _ = reduced(cs)
    Af = A @ np.asarray(f, dtype=float).reshape(-1)
    S = _S(A, P)
    return float(Af @ np.linalg.pinv(S) @ Af)


def constraint_bases(cs: ConstraintSet) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal bases of row(A) and null(A) from the SVD of A: (V_row, V_null),
    each with basis vectors as rows. For A = [1, 1]: [1, 1]/sqrt2 and [1, -1]/sqrt2."""
    A = np.atleast_2d(np.asarray(cs.A, dtype=float))
    _, s, Vt = np.linalg.svd(A, full_matrices=True)
    tol = s.max() * max(A.shape) * np.finfo(float).eps if s.size else 0.0
    r = int((s > tol).sum())
    return Vt[:r], Vt[r:]


def project_hard(x: np.ndarray, P: np.ndarray, cs: ConstraintSet) -> tuple[np.ndarray, np.ndarray]:
    """Minimum P^-1-weighted correction that satisfies A x = b exactly."""
    check_spd(P)
    A, b = reduced(cs)
    r = A @ x - b
    S = _S(A, P)
    K = P @ A.T @ np.linalg.pinv(S)
    x_star = x - K @ r
    P_star = P - K @ A @ P
    P_star = 0.5 * (P_star + P_star.T)
    return x_star, P_star


def project_soft(x: np.ndarray, P: np.ndarray, cs: ConstraintSet, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Penalised correction; leaves a nonzero residual for every finite lam."""
    check_spd(P)
    A = np.atleast_2d(np.asarray(cs.A, dtype=float))
    b = np.asarray(cs.b, dtype=float).reshape(-1)
    Pinv = np.linalg.inv(P)
    M = Pinv + lam * A.T @ A
    rhs = Pinv @ x + lam * A.T @ b
    x_star = np.linalg.solve(M, rhs)
    P_star = np.linalg.inv(M)
    P_star = 0.5 * (P_star + P_star.T)
    return x_star, P_star


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
    """
    x = np.array(x, dtype=float, copy=True)
    P = np.array(P, dtype=float, copy=True)

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

    return envelope(Status.OK, xs, Ps, r_pre, residual(xs, cs), xs - x)
