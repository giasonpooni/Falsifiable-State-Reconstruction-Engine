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
system", stat ~ chi^2(rows(A)). A large stat means the estimate and the declared
constraint disagree by more than the estimate's own uncertainty explains. That
is evidence of model failure -- stale constraint, undeclared sensor bias,
unmodeled leak -- and it is surfaced as MODEL_INCONSISTENT. It does not say which.

Nothing in this module overwrites the incoming estimate. `reconcile` returns a
StateEstimate that carries the unprojected state, the correction, and both
residuals.
"""
from __future__ import annotations

import numpy as np

from ..schema import ConstraintSet, StateEstimate, Status

__all__ = [
    "chi2_quantile", "is_feasible", "residual", "consistency_stat",
    "project_hard", "project_soft", "reconcile",
]

# Upper quantiles of chi^2(dof). Enough for Phase 1; scipy would replace this table.
_CHI2 = {
    1: {0.95: 3.841, 0.99: 6.635, 0.999: 10.828},
    2: {0.95: 5.991, 0.99: 9.210, 0.999: 13.816},
    3: {0.95: 7.815, 0.99: 11.345, 0.999: 16.266},
}


def chi2_quantile(dof: int, q: float) -> float:
    try:
        return _CHI2[dof][q]
    except KeyError as e:
        raise ValueError(f"no tabulated chi^2 quantile for dof={dof}, q={q}") from e


def is_feasible(cs: ConstraintSet) -> bool:
    """A x = b has at least one solution  <=>  rank([A | b]) == rank(A)."""
    A = np.atleast_2d(cs.A)
    Ab = np.hstack([A, np.atleast_2d(cs.b).reshape(-1, 1)])
    return np.linalg.matrix_rank(Ab) == np.linalg.matrix_rank(A)


def residual(x: np.ndarray, cs: ConstraintSet) -> np.ndarray:
    return cs.A @ x - cs.b


def consistency_stat(x: np.ndarray, P: np.ndarray, cs: ConstraintSet) -> float:
    r = residual(x, cs)
    S = cs.A @ P @ cs.A.T
    return float(r @ np.linalg.pinv(S) @ r)


def project_hard(x: np.ndarray, P: np.ndarray, cs: ConstraintSet) -> tuple[np.ndarray, np.ndarray]:
    """Minimum P^-1-weighted correction that satisfies A x = b exactly."""
    r = residual(x, cs)
    S = cs.A @ P @ cs.A.T
    K = P @ cs.A.T @ np.linalg.pinv(S)
    x_star = x - K @ r
    P_star = P - K @ cs.A @ P
    P_star = 0.5 * (P_star + P_star.T)
    return x_star, P_star


def project_soft(x: np.ndarray, P: np.ndarray, cs: ConstraintSet, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Penalised correction; leaves a nonzero residual for every finite lam."""
    Pinv = np.linalg.inv(P)
    M = Pinv + lam * cs.A.T @ cs.A
    rhs = Pinv @ x + lam * cs.A.T @ cs.b
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

    if stat is None:
        stat = consistency_stat(x, P, cs)
    r_pre = residual(x, cs)

    if mode is None:
        return envelope(Status.SKIPPED, x.copy(), P.copy(), r_pre, None, None)
    if not is_feasible(cs):
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
