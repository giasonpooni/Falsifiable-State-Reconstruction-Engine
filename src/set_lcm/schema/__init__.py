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
    t: float                     # physical sampling time
    arrival_t: float             # earliest time the estimator may use it (>= t)
    y: np.ndarray                # measurement vector; NaN where missing
    R: np.ndarray                # declared measurement covariance (sensor spec, incl. quantization)
    mask: np.ndarray             # bool; True where a value is present
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class ConstraintSet:
    version: str
    A: np.ndarray                # linear equality constraints  A x = b
    b: np.ndarray
    description: str

    @property
    def dof(self) -> int:
        """Number of declared rows. Use `rank` for chi-square degrees of freedom."""
        return int(np.atleast_2d(self.A).shape[0])

    @property
    def rank(self) -> int:
        """Number of independent constraints; the chi-square dof of the consistency statistic."""
        return int(np.linalg.matrix_rank(np.atleast_2d(np.asarray(self.A, dtype=float))))


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
    consistency_stat: float | None   # r^T (A P A^T)^-1 r, chi-square(rank A) under "same system"
    consistency_threshold: float | None
