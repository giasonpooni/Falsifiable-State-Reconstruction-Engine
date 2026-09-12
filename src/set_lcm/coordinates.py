"""Fixed affine coordinate charts for the additive invariant filter.

Changing coordinates changes the state, model and covariance together. A chart
x_chart = matrix @ x + offset is deterministic, fixed over a record, and carries
no calibration uncertainty. The offset changes the coordinate origin; it is not
an automorphism of the zero-identity additive group. Additive errors transform by
matrix alone. Units are supplied by the caller; this module does not infer them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .invariant import GaussianState, _array, _covariance, _immutable

__all__ = ["AffineCoordinates"]

# A declared numeric support limit, not a physical identifiability criterion.
# It prevents treating a nearly singular chart as an equivalent representation.
MAX_CONDITION_NUMBER = 1e12
ROUND_TRIP_TOLERANCE = 1e-8


def _check_mean_fidelity(reference, recovered, covariance, name: str) -> None:
    """Bound coordinate error in declared mean/standard-deviation scales."""
    recovered = _array(recovered, f"round-trip {name}", 1, reference.shape)
    scales = np.maximum(np.abs(reference), np.sqrt(np.diag(covariance)))
    positive = scales > 0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        error = np.abs((recovered[positive] - reference[positive]) / scales[positive])
    if (np.any(recovered[~positive] != reference[~positive]) or
            not np.all(np.isfinite(error)) or np.any(error > ROUND_TRIP_TOLERANCE)):
        raise ValueError(f"{name} loses precision in the coordinate round trip (tolerance 1e-8)")


def _check_covariance_fidelity(reference, recovered, name: str) -> None:
    """Compare entries in the reference covariance's standard-deviation products."""
    recovered = _array(recovered, f"round-trip {name}", 2, reference.shape)
    positive = np.diag(reference) > 0
    # An exact deterministic component supplies no nonzero unit scale to borrow.
    if np.any(recovered[~positive, :] != 0) or np.any(recovered[:, ~positive] != 0):
        raise ValueError(f"{name} creates uncertainty in an exact zero direction during the coordinate round trip")
    scales = np.sqrt(np.diag(reference)[positive])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        error = ((recovered[np.ix_(positive, positive)] - reference[np.ix_(positive, positive)])
                 / scales[:, None] / scales[None, :])
    if not np.all(np.isfinite(error)) or np.any(np.abs(error) > ROUND_TRIP_TOLERANCE):
        raise ValueError(f"{name} loses precision in the coordinate round trip (tolerance 1e-8)")


@dataclass(frozen=True)
class AffineCoordinates:
    """An immutable invertible chart x_chart = matrix @ x + offset.

    Singular matrices and 2-norm condition numbers above 1e12 are refused. This
    is a numerical support limit: algebraic invariance does not guarantee good
    floating-point conditioning. State and process-covariance conversions also
    require a round-trip error <= 1e-8 in the input covariance's standard-deviation
    product units. Means and drift use max(abs(input), input standard deviation)
    per component. An exactly zero scale requires exactly zero introduced error.
    Computed covariances are symmetrized before checking. These conservative
    checks refuse underflow and destructive origin cancellation; they do not
    promise uniform precision for every later model calculation. Time-varying
    charts are not supported here. A dense chart can round an exactly singular
    covariance to a zero variance with nonzero cross-covariance; such an invalid
    result is refused, without jitter or projection, even for a well-conditioned
    chart. A particular singular covariance may therefore have narrower numeric
    support than the chart matrix itself.
    """
    matrix: np.ndarray
    offset: np.ndarray

    def __post_init__(self):
        offset = _array(self.offset, "chart offset", 1)
        n = offset.size
        if n == 0:
            raise ValueError("chart must contain at least one state component")
        matrix = _array(self.matrix, "chart matrix", 2, (n, n))
        try:
            condition = np.linalg.cond(matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError("chart matrix condition could not be resolved") from exc
        if not np.isfinite(condition) or condition > MAX_CONDITION_NUMBER:
            raise ValueError("chart matrix must be invertible with condition number <= 1e12")
        object.__setattr__(self, "matrix", _immutable(matrix))
        object.__setattr__(self, "offset", _immutable(offset))

    def _check_state(self, state: GaussianState) -> None:
        if not isinstance(state, GaussianState):
            raise TypeError("state must be a GaussianState")
        if state.mean.shape != self.offset.shape:
            raise ValueError("state dimension must match the coordinate chart")

    def _map_covariance(self, covariance: np.ndarray, inverse: bool) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            if inverse:
                left = np.linalg.solve(self.matrix, covariance)
                mapped = np.linalg.solve(self.matrix, left.T).T
            else:
                mapped = self.matrix @ covariance @ self.matrix.T
            mapped = 0.5 * mapped + 0.5 * mapped.T
        return _array(mapped, "computed covariance", 2, covariance.shape)

    def _checked_covariance(self, covariance: np.ndarray, inverse: bool, name: str) -> np.ndarray:
        mapped = self._map_covariance(covariance, inverse)
        recovered = self._map_covariance(mapped, not inverse)
        _check_covariance_fidelity(covariance, recovered, name)
        return _covariance(mapped, name, self.offset.size)

    def transform_state(self, state: GaussianState) -> GaussianState:
        """Push mean and full covariance into this chart: T x+c, T P T.T."""
        self._check_state(state)
        with np.errstate(over="ignore", invalid="ignore"):
            mean = self.matrix @ state.mean + self.offset
            recovered = np.linalg.solve(self.matrix, mean - self.offset)
        _check_mean_fidelity(state.mean, recovered, state.covariance, "state mean")
        covariance = self._checked_covariance(state.covariance, False, "state covariance")
        return GaussianState(mean, covariance)

    def restore_state(self, state: GaussianState) -> GaussianState:
        """Pull a chart state back to original coordinates, using linear solves."""
        self._check_state(state)
        with np.errstate(over="ignore", invalid="ignore"):
            mean = np.linalg.solve(self.matrix, state.mean - self.offset)
            recovered = self.matrix @ mean + self.offset
        _check_mean_fidelity(state.mean, recovered, state.covariance, "state mean")
        covariance = self._checked_covariance(state.covariance, True, "state covariance")
        return GaussianState(mean, covariance)

    def transform_dynamics(self, F, drift, Q) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return F'=T F T^-1, d'=T d+c-F'c, Q'=T Q T.T.

        F and Q are (n,n); drift is (n,). Q must be a declared PSD covariance.
        Both sides of the transition must use this same fixed chart.
        """
        n = self.offset.size
        F = _array(F, "F", 2, (n, n))
        drift = _array(drift, "drift", 1, (n,))
        Q = _covariance(Q, "Q", n)
        with np.errstate(over="ignore", invalid="ignore"):
            transformed_F = np.linalg.solve(self.matrix.T, (self.matrix @ F).T).T
            transformed_drift = self.matrix @ drift + self.offset - transformed_F @ self.offset
            recovered_drift = np.linalg.solve(
                self.matrix, transformed_drift - self.offset + transformed_F @ self.offset)
        _check_mean_fidelity(drift, recovered_drift, Q, "dynamics drift")
        transformed_Q = self._checked_covariance(Q, False, "transformed Q")
        return (_immutable(_array(transformed_F, "transformed F", 2, (n, n))),
                _immutable(_array(transformed_drift, "transformed drift", 1, (n,))),
                _immutable(_covariance(transformed_Q, "transformed Q", n)))

    def transform_observation(self, H, offset) -> tuple[np.ndarray, np.ndarray]:
        """Return H'=H T^-1 and a'=a-H'c for z = H x+a+v.

        Measurement values, R and mask remain in their original coordinates.
        H is (m,n), offset is (m,), including the empty-measurement case. For a
        separate change of measurement units/order, transform z, H, a and R
        together. General mixing of missing channels cannot reuse a boolean mask.
        """
        offset = _array(offset, "observation offset", 1)
        H = _array(H, "H", 2, (offset.size, self.offset.size))
        with np.errstate(over="ignore", invalid="ignore"):
            transformed_H = np.linalg.solve(self.matrix.T, H.T).T
            transformed_offset = offset - transformed_H @ self.offset
        return (_immutable(_array(transformed_H, "transformed H", 2, H.shape)),
                _immutable(_array(transformed_offset, "transformed observation offset", 1, offset.shape)))
