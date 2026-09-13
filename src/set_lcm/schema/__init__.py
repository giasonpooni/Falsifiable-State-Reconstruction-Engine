"""Typed envelopes for the Phase 1 slice.

The separation these types encode is the one both design reviews insisted on:

    source observation != estimated state != reconciled state != verification result

Nothing here overwrites an observation. LCM emits a *derived* StateEstimate
that carries the unprojected state, the correction it applied, and the
constraint residuals before and after, so a disagreement between estimate and
declared model is preserved rather than erased.
"""
from __future__ import annotations

from collections.abc import Sequence
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

    A_var None (the default) declares the MATRIX exact: the relation itself is known, and only
    b carries error. That is true of a conservation law written on declared states, and false
    of any constitutive relation whose coefficients are measured or fitted -- a Muskingum
    routing row carrying K and x, a heat-exchanger duty row carrying an effectiveness, a
    stage-capacity slope. Otherwise A_var is the declared uncertainty of the DECLARED rows of
    A, as either a (rows, n, n) stack giving row i's covariance over the states (declaring the
    rows independent of each other) or a full (rows*n, rows*n) covariance of vec(A) in
    row-major order, which is the form that can carry dependence between rows. Shape,
    finiteness, symmetry and PSD-ness are validated here and a violation raises ValueError.

    What it does NOT declare, and the kernel does not infer: any dependence between A's error
    and b's, or between A's error and the state estimate. Shared evidence produces exactly
    those dependencies -- a flow reading that enters both a coefficient and a measurement --
    and an independently declared A_var does not represent them.

    row_units optionally declares the unit of each row of A x and b, in declared row
    order (including dependent rows). None leaves units undeclared for compatibility.
    Declarations are stored as an immutable tuple of non-empty strings. They are metadata:
    no unit conversion, dimensional validation or numerical rescaling is performed. The
    caller remains responsible for consistent units in A, b and b_var; covariance entry
    (i, j) has the product of the declared units of rows i and j.
    """
    version: str
    A: np.ndarray                # linear equality constraints  A x = b
    b: np.ndarray
    description: str
    b_var: np.ndarray | None = None   # declared variance of b: (rows,) or (rows, rows); None = exact
    A_var: np.ndarray | None = None   # declared covariance of A: (rows, n, n) or (rows*n, rows*n)
    row_units: tuple[str, ...] | None = None  # one unit per declared row; metadata only

    def __post_init__(self):
        if self.A_var is not None:
            rows, n = self.dof, int(np.atleast_2d(self.A).shape[1])
            v = np.array(self.A_var, dtype=float)
            if v.shape == (rows, n, n):
                full = np.zeros((rows * n, rows * n))
                for i in range(rows):                       # declared independent across rows
                    full[i * n:(i + 1) * n, i * n:(i + 1) * n] = v[i]
                v = full
            elif v.shape != (rows * n, rows * n):
                raise ValueError(
                    f"A_var must be a ({rows}, {n}, {n}) stack of per-row covariances or a "
                    f"({rows * n}, {rows * n}) covariance of vec(A) in row-major order; "
                    f"got shape {v.shape}")
            if not np.all(np.isfinite(v)):
                raise ValueError("A_var must be finite")
            if not np.allclose(v, v.T, rtol=1e-10, atol=1e-12):
                raise ValueError("A_var is not symmetric")
            v = 0.5 * (v + v.T)
            w = np.linalg.eigvalsh(v)
            if w[0] < -_PSD_REL_TOL * float(np.abs(w).max() or 1.0):
                raise ValueError(f"A_var is not positive semidefinite (eigenvalues {w})")
            v.setflags(write=False)
            object.__setattr__(self, "A_var", v)
        if self.row_units is not None:
            units = self.row_units
            if isinstance(units, (str, bytes)) or not isinstance(units, Sequence):
                raise ValueError("row_units must be an ordered sequence of non-empty strings")
            units = tuple(units)
            if len(units) != self.dof:
                raise ValueError(f"row_units has {len(units)} declarations; A has {self.dof} row(s)")
            if not all(isinstance(unit, str) and unit.strip() for unit in units):
                raise ValueError("row_units entries must be non-empty strings")
            object.__setattr__(self, "row_units", units)
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
    def matrix_exact(self) -> bool:
        """True when A carries no declared uncertainty (A_var is None)."""
        return self.A_var is None

    def A_block(self, i: int, j: int) -> np.ndarray:
        """Cov(row i of A, row j of A) as an (n, n) matrix; raises when A is declared exact."""
        if self.A_var is None:
            raise ValueError("this constraint set declares A exact (A_var is None)")
        n = int(np.atleast_2d(self.A).shape[1])
        return np.array(self.A_var[i * n:(i + 1) * n, j * n:(j + 1) * n])

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
