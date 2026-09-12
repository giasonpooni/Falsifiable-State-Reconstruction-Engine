"""Typed envelopes for the Phase 1 slice.

The separation these types encode is the one both design reviews insisted on:

    source observation != estimated state != reconciled state != verification result

Nothing here overwrites an observation. LCM emits a *derived* StateEstimate
that carries the unprojected state, the correction it applied, and the
constraint residuals before and after, so a disagreement between estimate and
declared model is preserved rather than erased.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

__all__ = ["Status", "Observation", "ConstraintSet", "StateEstimate"]


class Status(str, Enum):
    OK = "ok"                        # projection applied
    SKIPPED = "skipped"              # no constraint set, or estimator has no projection stage
    INFEASIBLE = "infeasible"        # constraint set is self-contradictory
    NOT_CONVERGED = "not_converged"  # reserved for iterative solvers (unused by closed-form kernels)
    MODEL_INCONSISTENT = "model_inconsistent"  # estimate and constraint disagree beyond tolerance; held


@dataclass(frozen=True)
class Observation:
    """One sampling step of evidence: one value per sensor (NaN where missing).

    `t` is the physical sampling time -- for real evidence the source's own event
    time, read out of the evidence content, never a retrieval or extraction stamp.
    `evidence_ids` names the admitted evidence this observation was built from (for
    example the content-addressed ids of the evidence records it came from). It is
    provenance only: no estimator reads it, it is not part of any state's identity,
    and the runner carries it through to RunResult.ingested_evidence at the report
    step the observation was ingested. Simulated observations carry ()."""
    t: float                     # physical sampling time
    arrival_t: float             # earliest time the estimator may use it (>= t)
    y: np.ndarray                # measurement vector; NaN where missing
    R: np.ndarray                # declared measurement covariance (sensor spec, incl. quantization)
    mask: np.ndarray             # bool; True where a value is present
    source_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...] = ()   # ids of the admitted evidence behind this observation

    def __post_init__(self):
        ids = self.evidence_ids
        if isinstance(ids, str):
            raise TypeError("evidence_ids must be a tuple of str, not a single str")
        ids = tuple(ids)
        if not all(isinstance(e, str) for e in ids):
            raise TypeError(f"evidence_ids must be str; got {ids!r}")
        object.__setattr__(self, "evidence_ids", ids)


_PSD_REL_TOL = 1e-12   # b_var's smallest eigenvalue may undershoot 0 by this fraction of its largest


@dataclass(frozen=True)
class ConstraintSet:
    """Linear equality constraints A x = b, optionally with declared uncertainty on b.

    b_var None (the default) declares the set EXACT: A x = b by declaration, which is
    what every constraint in the tree declared before b_var existed. Otherwise b_var is
    the declared uncertainty of b -- b = A x_true + e with e ~ N(0, Sigma_b) -- given as
    a (rows,) vector of variances (Sigma_b = diag(b_var)) or as a (rows, rows) symmetric
    positive-semidefinite covariance. A zero variance (a null direction of Sigma_b) is an
    exact row (combination of rows). The shape, finiteness, symmetry and PSD-ness are
    validated here and a violation raises ValueError; the validated value is stored as a
    read-only float copy so it cannot drift after validation. The kernel (set_lcm.lcm)
    decides what it can do with a declared set; the schema only says what was declared.
    """
    version: str
    A: np.ndarray                # linear equality constraints  A x = b
    b: np.ndarray
    description: str
    b_var: np.ndarray | None = None   # declared variance of b: (rows,) or (rows, rows); None = exact

    def __post_init__(self):
        if self.b_var is None:
            return
        rows = self.dof
        v = np.array(self.b_var, dtype=float)          # a copy, never a view of the caller's array
        if v.ndim not in (1, 2):
            raise ValueError(f"b_var must be a (rows,) vector of variances or a (rows, rows) covariance; "
                             f"got shape {v.shape} for {rows} row(s)")
        if v.shape != (rows,) * v.ndim:
            raise ValueError(f"b_var has shape {v.shape}; the set has {rows} row(s), so it must be "
                             f"({rows},) or ({rows}, {rows})")
        if not np.all(np.isfinite(v)):
            raise ValueError("b_var must be finite")
        if v.ndim == 1:
            if np.any(v < 0.0):
                raise ValueError(f"b_var has a negative variance: {v}")
        else:
            if not np.allclose(v, v.T, rtol=1e-10, atol=1e-12):
                raise ValueError("b_var is not symmetric")
            v = 0.5 * (v + v.T)                        # exact for an exactly symmetric input
            w = np.linalg.eigvalsh(v)
            if w[0] < -_PSD_REL_TOL * float(np.abs(w).max()):
                raise ValueError(f"b_var is not positive semidefinite (eigenvalues {w})")
        v.setflags(write=False)
        object.__setattr__(self, "b_var", v)

    @property
    def dof(self) -> int:
        """Number of declared rows. Use `rank` for chi-square degrees of freedom."""
        return int(np.atleast_2d(self.A).shape[0])

    @property
    def rank(self) -> int:
        """Number of independent constraints; the chi-square dof of the consistency statistic."""
        return int(np.linalg.matrix_rank(np.atleast_2d(np.asarray(self.A, dtype=float))))

    @property
    def exact(self) -> bool:
        """True when b carries no declared uncertainty (b_var is None)."""
        return self.b_var is None

    @property
    def b_cov(self) -> np.ndarray | None:
        """Sigma_b as a (rows, rows) matrix -- diag(b_var) for a vector -- or None when exact."""
        if self.b_var is None:
            return None
        return np.diag(self.b_var) if self.b_var.ndim == 1 else np.array(self.b_var)


@dataclass
class StateEstimate:
    t: float
    x: np.ndarray                    # reported state (post-projection when one was applied)
    P: np.ndarray                    # reported covariance
    model_version: str
    constraint_set_version: str | None
    status: Status
    x_unprojected: np.ndarray        # always retained
    P_unprojected: np.ndarray
    residual_pre: np.ndarray | None  # A x_unprojected - b
    residual_post: np.ndarray | None # A x - b
    correction: np.ndarray | None    # x - x_unprojected
    consistency_stat: float | None   # r^T (A P A^T + Sigma_b)^-1 r (Sigma_b = 0 when b is exact),
                                     # chi-square(rank A) under "same system, b within its declared uncertainty"
    consistency_threshold: float | None
