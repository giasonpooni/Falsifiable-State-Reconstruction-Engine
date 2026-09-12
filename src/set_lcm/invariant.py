"""Invariant error-state filtering on the additive Lie group (R^n, +).

The group error is state - reference and its retraction is reference + error.
For the affine model x_next = F x + drift + w and z = H x + offset + v,
prediction propagates the invariant error by F. Measurement conditioning estimates
that error before retracting its correction onto the prior mean. Left and right
errors coincide on this commutative group. These are the ordinary exact affine
Gaussian filtering equations; this module claims no nonlinear IEKF advantage.

P, Q and R are declared covariances in their respective coordinate units. Process
noise is independent of the prior state error; measurement noise is independent
of the predicted state error. Cross-correlation within each covariance is retained.
No uncertainty is inferred, no conservation projection or fault classification is
performed, and the pre-update innovation is preserved for separate diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

__all__ = ["GaussianState", "UpdateResult", "invariant_error", "retract", "predict", "update"]

_PSD_REL_TOL = 1e-12


def _array(value, name: str, ndim: int, shape=None) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite numbers") from exc
    if result.ndim != ndim or (shape is not None and result.shape != shape):
        expected = f"shape {shape}" if shape is not None else f"{ndim} dimensions"
        raise ValueError(f"{name} must have {expected}, got shape {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _immutable(value: np.ndarray) -> np.ndarray:
    """Own an immutable buffer, including protection against setflags(write=True)."""
    return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)


def _covariance(value, name: str, size: int) -> np.ndarray:
    """Validate PSD in correlation coordinates, retaining declared exact zeros.

    A global raw-unit tolerance can hide negative variances in smaller-unit rows.
    No eigenvalue clipping, jitter, or nearest-PSD repair is performed. Only
    symmetry roundoff is averaged; PSD uses a relative 1e-12 roundoff tolerance
    on the positive-variance correlation block.
    """
    covariance = _array(value, name, 2, (size, size))
    variances = np.diag(covariance)
    if np.any(variances < 0.0):
        raise ValueError(f"{name} variances must be nonnegative")
    zero = variances == 0.0
    if np.any(covariance[zero, :] != 0.0) or np.any(covariance[:, zero] != 0.0):
        raise ValueError(f"{name} zero-variance rows must have zero cross-covariance")
    positive = ~zero
    if np.any(positive):
        scales = np.sqrt(variances[positive])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            correlation = covariance[np.ix_(positive, positive)] / scales[:, None] / scales[None, :]
        if not np.all(np.isfinite(correlation)):
            raise ValueError(f"{name} has invalid normalized correlations")
        if not np.allclose(correlation, correlation.T, rtol=1e-10, atol=1e-12):
            raise ValueError(f"{name} must be symmetric in correlation coordinates")
        correlation = 0.5 * correlation + 0.5 * correlation.T
        if np.any(np.abs(correlation) > 1.0 + _PSD_REL_TOL):
            raise ValueError(f"{name} must be positive semidefinite")
        try:
            eigenvalues = np.linalg.eigvalsh(correlation)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"{name} positive-semidefinite validation did not converge") from exc
        if eigenvalues[0] < -_PSD_REL_TOL * float(np.max(np.abs(eigenvalues))):
            raise ValueError(f"{name} must be positive semidefinite")
    return 0.5 * covariance + 0.5 * covariance.T


def _state(value, name="state") -> GaussianState:
    if not isinstance(value, GaussianState):
        raise TypeError(f"{name} must be a GaussianState")
    return value


@dataclass(frozen=True)
class GaussianState:
    """Mean and full PSD covariance of an additive state, with immutable copies.

    The covariance may be singular, including exact known components. No unit or
    physical constraint is inferred from the vector: callers supply consistent
    coordinates and all corresponding model transformations.
    """
    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self):
        mean = _array(self.mean, "mean", 1)
        if mean.size == 0:
            raise ValueError("mean must contain at least one state component")
        covariance = _covariance(self.covariance, "state covariance", mean.size)
        object.__setattr__(self, "mean", _immutable(mean))
        object.__setattr__(self, "covariance", _immutable(covariance))


def invariant_error(reference, state) -> np.ndarray:
    """Return state - reference, the left/right invariant error on (R^n, +).

    Arguments are real finite 1-D vectors. Adding the same translation to both
    arguments leaves this error unchanged. The returned vector is immutable.
    """
    reference = _array(reference, "reference", 1)
    state = _array(state, "state vector", 1, reference.shape)
    with np.errstate(over="ignore", invalid="ignore"):
        error = state - reference
    return _immutable(_array(error, "invariant error", 1))


def retract(reference, error) -> np.ndarray:
    """Map an additive error to the state as reference + error, without projection."""
    reference = _array(reference, "reference", 1)
    error = _array(error, "error", 1, reference.shape)
    with np.errstate(over="ignore", invalid="ignore"):
        state = reference + error
    return _immutable(_array(state, "retracted state", 1))


def predict(state: GaussianState, F, drift, Q) -> GaussianState:
    """Predict an affine state and its invariant error covariance.

    x_next = F x + drift + w, with independent zero-mean noise Cov(w)=Q.
    Consequently e_next = F e + w, P_next = F P F.T + Q. F is (n,n), drift
    (n,), and Q a full PSD (n,n) covariance; none is inferred or defaulted.
    """
    state = _state(state)
    n = state.mean.size
    F = _array(F, "F", 2, (n, n))
    drift = _array(drift, "drift", 1, (n,))
    Q = _covariance(Q, "Q", n)
    with np.errstate(over="ignore", invalid="ignore"):
        mean = F @ state.mean + drift
        covariance = F @ state.covariance @ F.T + Q
        covariance = 0.5 * covariance + 0.5 * covariance.T
    return GaussianState(mean, covariance)


@dataclass(frozen=True)
class UpdateResult:
    """A conditioned state and its preserved pre-update evidence.

    status is 'updated' or 'no_observations'; neither is a health/fault verdict.
    statistic is the squared whitened pre-update innovation, with dof equal to
    the observed dimension. Its chi-square interpretation additionally requires
    a correctly specified Gaussian prior and independent Gaussian measurement
    noise with the declared covariance. No threshold or alarm is supplied here.
    """
    prior: GaussianState
    posterior: GaussianState
    innovation: np.ndarray
    innovation_covariance: np.ndarray
    gain: np.ndarray
    correction: np.ndarray
    statistic: float | None
    dof: int
    status: str
    observed_indices: tuple[int, ...]

    def __post_init__(self):
        prior, posterior = _state(self.prior, "prior"), _state(self.posterior, "posterior")
        n = prior.mean.size
        if posterior.mean.shape != (n,):
            raise ValueError("prior and posterior must have the same state dimension")
        innovation = _array(self.innovation, "innovation", 1)
        m = innovation.size
        covariance = _array(self.innovation_covariance, "innovation covariance", 2, (m, m))
        gain = _array(self.gain, "gain", 2, (n, m))
        correction = _array(self.correction, "correction", 1, (n,))
        if isinstance(self.dof, (bool, np.bool_)) or not isinstance(self.dof, (int, np.integer)) or self.dof != m:
            raise ValueError("dof must equal the observed measurement dimension")
        if self.status != ("updated" if m else "no_observations"):
            raise ValueError("status must agree with the observed measurement dimension")
        try:
            indices = tuple(self.observed_indices)
        except TypeError as exc:
            raise ValueError("observed_indices must contain the selected measurement indices") from exc
        if (len(indices) != m or any(isinstance(i, (bool, np.bool_)) or
                                    not isinstance(i, (int, np.integer)) or i < 0 for i in indices)
                or any(a >= b for a, b in zip(indices, indices[1:]))):
            raise ValueError("observed_indices must be distinct nonnegative indices in increasing order")
        object.__setattr__(self, "observed_indices", tuple(int(i) for i in indices))
        if m:
            if self.statistic is None or not np.isscalar(self.statistic) or np.iscomplexobj(self.statistic):
                raise ValueError("statistic must be finite and nonnegative")
            statistic = float(self.statistic)
            if not math.isfinite(statistic) or statistic < 0.0:
                raise ValueError("statistic must be finite and nonnegative")
            object.__setattr__(self, "statistic", statistic)
        elif self.statistic is not None:
            raise ValueError("statistic must be None when there are no observations")
        for name, array in (("innovation", innovation), ("innovation_covariance", covariance),
                            ("gain", gain), ("correction", correction)):
            object.__setattr__(self, name, _immutable(array))
        object.__setattr__(self, "dof", int(self.dof))


def update(state: GaussianState, measurement, H, R, offset=None, mask=None) -> UpdateResult:
    """Condition the additive invariant error and retract it onto the prior mean.

    z = H x + offset + v. Thus innovation = z - (H mean + offset) = H e + v,
    estimated error = K innovation, and posterior mean = retract(mean, error).
    Covariance uses the Joseph expression with Cholesky solves, never an inverse.

    measurement is (m,), H (m,n), R a full PSD (m,m) covariance, and optional
    offset (m,) defaults to zero. mask is a strict boolean (m,) array. Missing
    measurements may be NaN only in masked-out rows; complex values and infinities
    are always refused. All other inputs must be finite and valid even in masked
    rows. The observed subvector and full principal R submatrix are selected
    together; observed_indices preserves their original order. Innovation covariance
    must be SPD with numerically resolved eigenvalues in correlation coordinates:
    dependent exact or numerically unresolved observations are refused, not
    pseudoinverted or assigned artificial residual degrees of freedom.
    Empty/all-masked measurements retain the prior and return 'no_observations'
    with empty innovation/covariance, (n,0) gain, zero correction, dof=0 and no
    statistic. There is no hidden state, gating, adaptation or fault decision.
    """
    state = _state(state)
    n = state.mean.size
    if np.iscomplexobj(measurement):
        raise ValueError("measurement must be real-valued")
    try:
        measurement = np.asarray(measurement, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("measurement must be a numeric vector") from exc
    if measurement.ndim != 1 or np.any(np.isinf(measurement)):
        raise ValueError("measurement must be a 1-D vector without infinities")
    m = measurement.size
    H = _array(H, "H", 2, (m, n))
    R = _covariance(R, "R", m)
    offset = np.zeros(m) if offset is None else _array(offset, "offset", 1, (m,))
    if mask is None:
        observed = np.ones(m, dtype=bool)
    else:
        observed = np.asarray(mask)
        if observed.shape != (m,) or observed.dtype != np.dtype(bool):
            raise ValueError(f"mask must be a boolean array with shape ({m},)")
    if not np.all(np.isfinite(measurement[observed])):
        raise ValueError("observed measurements must be finite; NaN requires a masked-out row")
    observed_indices = tuple(int(i) for i in np.flatnonzero(observed))
    if not np.any(observed):
        return UpdateResult(state, state, np.empty(0), np.empty((0, 0)), np.empty((n, 0)),
                            np.zeros(n), None, 0, "no_observations", observed_indices)
    measurement, H, offset = measurement[observed], H[observed], offset[observed]
    R = R[np.ix_(observed, observed)]
    with np.errstate(over="ignore", invalid="ignore"):
        expected = H @ state.mean + offset
        PHt = state.covariance @ H.T
        S = H @ PHt + R
        S = 0.5 * S + 0.5 * S.T
    innovation = invariant_error(expected, measurement)
    S = _array(S, "innovation covariance", 2, (measurement.size, measurement.size))
    if np.any(np.diag(S) <= 0.0):
        raise ValueError("innovation covariance must be positive definite")
    scales = np.sqrt(np.diag(S))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        correlation = S / scales[:, None] / scales[None, :]
    if not np.all(np.isfinite(correlation)):
        raise ValueError("innovation covariance has invalid normalized correlations")
    correlation = 0.5 * correlation + 0.5 * correlation.T
    try:
        # A mathematically singular H P H.T can round to a matrix whose Cholesky
        # has a tiny positive pivot. Such a pivot does not establish another
        # independent Gaussian observation or justify incrementing its dof.
        eigenvalues = np.linalg.eigvalsh(correlation)
        resolution = measurement.size * np.finfo(float).eps * float(np.max(np.abs(eigenvalues)))
        if eigenvalues[0] <= resolution:
            raise ValueError("innovation covariance must be positive definite with numerically resolved eigenvalues")
        chol = np.linalg.cholesky(correlation)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            gain = (np.linalg.solve(chol.T, np.linalg.solve(chol, PHt.T / scales[:, None]))
                    / scales[:, None]).T
            whitened = np.linalg.solve(chol, innovation / scales)
    except np.linalg.LinAlgError as exc:
        raise ValueError("innovation covariance must be positive definite") from exc
    gain = _array(gain, "gain", 2, (n, measurement.size))
    whitened = _array(whitened, "whitened innovation", 1)
    with np.errstate(over="ignore", invalid="ignore"):
        correction = gain @ innovation
        residual_map = np.eye(n) - gain @ H
        covariance = (residual_map @ state.covariance @ residual_map.T + gain @ R @ gain.T)
        covariance = 0.5 * covariance + 0.5 * covariance.T
        statistic = float(whitened @ whitened)
    if not math.isfinite(statistic):
        raise ValueError("innovation statistic exceeds the supported finite numeric range")
    posterior = GaussianState(retract(state.mean, correction), covariance)
    return UpdateResult(state, posterior, innovation, S, gain, correction, statistic,
                        measurement.size, "updated", observed_indices)
