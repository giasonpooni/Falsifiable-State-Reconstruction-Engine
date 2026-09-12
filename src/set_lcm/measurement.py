"""Offline fluid-balance residuals from measurements with explicit interval support.

This baseline applies one linear operator H to the SAME raw measurement vector for
both residuals and uncertainty: r = H y and Cov(r) = H Cov(y) H.T. Raw measurements
are ordered as storage, then flows.ravel(order="C") (interval first, channel second).
Covariance is required; no unknown uncertainty is silently replaced with zero.

Storage at interval boundaries supports exact volume balances against interval mean
flow rates or interval total volumes. Mean storage requires the explicitly declared
constant_net_flow model within each interval. Without that assumption, interval
means alone do not determine the averaging operator used here.

All times are seconds and all edges describe contiguous intervals. Supported units
are explicitly supplied m3 for storage and m3/s for mean flow, or m3 for total flow.
Convert other units in a reviewed adapter before constructing a record.

This is an offline batch calculation, not an arrival-time or real-time estimator.
Output times label interval ends (instant storage) or the midpoint of the second
compared interval (mean storage). A midpoint label does NOT mean the interval's
mean was available at that time. The propagated covariance is conditional on the
declared measurement-error covariance and within-interval model; this module makes
no claim that either is calibrated and emits no inferential alarm or diagnosis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = ["BalanceRecord", "ResidualSeries", "balance_residuals"]

_PSD_REL_TOL = 1e-12


def _finite_array(value, name: str, ndim: int) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain real numbers")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite numbers") from exc
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite numbers")
    return array


def _immutable(array: np.ndarray) -> np.ndarray:
    """Copy into an immutable buffer, so even setflags(write=True) cannot mutate it."""
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _covariance(value, size: int) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError("covariance must contain real numbers")
    try:
        covariance = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("covariance must be declared as variances or a joint covariance") from exc
    if covariance.ndim not in (1, 2) or covariance.shape != (size,) * covariance.ndim:
        raise ValueError(f"covariance must have shape ({size},) or ({size}, {size}), "
                         f"got {covariance.shape}; raw order is storage then time-major flows")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must be finite; unknown uncertainty must be declared explicitly")
    if covariance.ndim == 1:
        if np.any(covariance < 0.0):
            raise ValueError("covariance variances must be nonnegative")
        return covariance
    variances = np.diag(covariance)
    if np.any(variances < 0.0):
        raise ValueError("joint covariance variances must be nonnegative")
    exact = variances == 0.0
    if np.any(covariance[exact, :] != 0.0) or np.any(covariance[:, exact] != 0.0):
        raise ValueError("zero-variance measurements must have zero cross-covariance")
    positive = ~exact
    if np.any(positive):
        # Raw components mix volume and rate units. A global covariance scale can hide
        # a materially indefinite small-variance block. Validate the positive block in
        # dimensionless correlation coordinates instead, without modifying its values.
        sd = np.sqrt(variances[positive])
        with np.errstate(over="ignore", invalid="ignore"):
            correlation = covariance[np.ix_(positive, positive)] / sd[:, None] / sd[None, :]
        if not np.all(np.isfinite(correlation)):
            raise ValueError("joint covariance has invalid normalized correlations")
        if not np.allclose(correlation, correlation.T, rtol=1e-10, atol=1e-12):
            raise ValueError("joint covariance must be symmetric in correlation coordinates")
        correlation = 0.5 * correlation + 0.5 * correlation.T
        if np.any(np.abs(correlation) > 1.0 + _PSD_REL_TOL):
            raise ValueError("joint covariance must be positive semidefinite")
        eigenvalues = np.linalg.eigvalsh(correlation)
        if eigenvalues[0] < -_PSD_REL_TOL * float(np.max(np.abs(eigenvalues))):
            raise ValueError("joint covariance must be positive semidefinite")
    return 0.5 * covariance + 0.5 * covariance.T


def _evidence_groups(value, size: int) -> tuple[tuple[str, ...], ...]:
    if isinstance(value, str):
        raise ValueError("evidence_ids must contain one ID group per raw measurement")
    try:
        groups = tuple(value)
    except TypeError as exc:
        raise ValueError("evidence_ids must contain one ID group per raw measurement") from exc
    if not groups:
        return ((),) * size
    if len(groups) != size:
        raise ValueError(f"evidence_ids must contain {size} raw-measurement groups, got {len(groups)}")
    normalized = []
    for group in groups:
        try:
            ids = (group,) if isinstance(group, str) else tuple(group)
        except TypeError as exc:
            raise ValueError("each evidence ID group must be a string or an iterable of strings") from exc
        if not all(isinstance(item, str) and item for item in ids):
            raise ValueError("evidence IDs must be nonempty strings")
        normalized.append(tuple(dict.fromkeys(ids)))
    return tuple(normalized)


@dataclass(frozen=True, kw_only=True)
class BalanceRecord:
    """Measurements over contiguous intervals, with immutable numerical copies.

    edges has length n+1; flows has shape (n,m); flow_signs has m entries (+1 in,
    -1 out). Storage has n+1 entries for 'instant', or n for 'mean'. Full covariance
    entries have the products of their raw component units; a variance vector
    declares independent component errors. Explicit zero variance is allowed.

    For mean storage, within_interval_model='constant_net_flow' is required.
    For instant storage no within-interval model is needed; that same model may
    optionally be declared, but no other model is implemented.

    evidence_ids optionally supplies a string or ID tuple for each raw component.
    Omission normalizes to empty groups. These IDs are provenance, not additional
    independent observations or information about covariance.
    """
    edges: np.ndarray
    storage: np.ndarray
    flows: np.ndarray
    flow_signs: np.ndarray
    covariance: np.ndarray
    storage_support: Literal["instant", "mean"]
    volume_unit: str
    flow_unit: str
    flow_support: Literal["mean", "total"] = "mean"
    within_interval_model: str | None = None
    evidence_ids: tuple[str | tuple[str, ...], ...] = ()

    def __post_init__(self):
        edges = _finite_array(self.edges, "edges", 1)
        storage = _finite_array(self.storage, "storage", 1)
        flows = _finite_array(self.flows, "flows", 2)
        signs = _finite_array(self.flow_signs, "flow_signs", 1)
        if edges.size < 2:
            raise ValueError("edges must contain at least two interval boundaries")
        with np.errstate(over="ignore", invalid="ignore"):
            durations = np.diff(edges)
        if not np.all(np.isfinite(durations) & (durations > 0.0)):
            raise ValueError("edges must be strictly increasing with finite positive durations")
        n = edges.size - 1
        if flows.shape[0] != n or flows.shape[1] == 0:
            raise ValueError(f"flows must have shape ({n}, m) for at least one flow channel")
        if signs.shape != (flows.shape[1],) or not np.all(np.isin(signs, [-1.0, 1.0])):
            raise ValueError("flow_signs must contain one +1 or -1 per flow channel")
        if self.storage_support not in ("instant", "mean"):
            raise ValueError("storage_support must be 'instant' or 'mean'")
        if self.flow_support not in ("mean", "total"):
            raise ValueError("flow_support must be 'mean' or 'total'")
        expected_storage = n + 1 if self.storage_support == "instant" else n
        if storage.size != expected_storage:
            raise ValueError(f"{self.storage_support} storage needs {expected_storage} values, "
                             f"got {storage.size}")
        if self.storage_support == "mean" and n < 2:
            raise ValueError("mean storage needs at least two intervals to form a balance residual")
        if self.within_interval_model not in (None, "constant_net_flow"):
            raise ValueError("only within_interval_model='constant_net_flow' is implemented")
        if self.storage_support == "mean" and self.within_interval_model != "constant_net_flow":
            raise ValueError("mean storage requires explicit within_interval_model='constant_net_flow'")
        expected_flow_unit = "m3/s" if self.flow_support == "mean" else "m3"
        if self.volume_unit != "m3" or self.flow_unit != expected_flow_unit:
            raise ValueError(f"supported units are volume_unit='m3' and flow_unit='{expected_flow_unit}' "
                             f"for {self.flow_support} flow; convert other units explicitly")
        size = storage.size + flows.size
        covariance = _covariance(self.covariance, size)
        evidence = _evidence_groups(self.evidence_ids, size)
        for name, array in (("edges", edges), ("storage", storage), ("flows", flows),
                            ("flow_signs", signs), ("covariance", covariance)):
            object.__setattr__(self, name, _immutable(array))
        object.__setattr__(self, "evidence_ids", evidence)


@dataclass(frozen=True, kw_only=True)
class ResidualSeries:
    """Offline balance residuals and their joint declared covariance, in m3 and m3^2.

    operator acts on storage followed by time-major flows. times are support labels,
    not data availability times. evidence_ids is a per-row union of nonzero operator
    contributors; canceled measurements are absent. Numerical arrays are immutable.
    """
    residual: np.ndarray
    covariance: np.ndarray
    operator: np.ndarray
    times: np.ndarray
    evidence_ids: tuple[tuple[str, ...], ...]

    def __post_init__(self):
        residual = _finite_array(self.residual, "residual", 1)
        covariance = _finite_array(self.covariance, "residual covariance", 2)
        operator = _finite_array(self.operator, "operator", 2)
        times = _finite_array(self.times, "times", 1)
        n = residual.size
        if covariance.shape != (n, n) or operator.shape[0] != n or times.shape != (n,):
            raise ValueError("residual, covariance, operator and times must have matching rows")
        evidence = _evidence_groups(self.evidence_ids, n)
        for name, array in (("residual", residual), ("covariance", covariance),
                            ("operator", operator), ("times", times)):
            object.__setattr__(self, name, _immutable(array))
        object.__setattr__(self, "evidence_ids", evidence)


def balance_residuals(record: BalanceRecord, *, cumulative: bool = False) -> ResidualSeries:
    """Apply the support-aware balance operator to measurements and joint uncertainty.

    Instant storage: S[k+1]-S[k] minus signed interval volumes. Mean storage:
    Sbar[k+1]-Sbar[k] minus half the signed volumes of EACH adjacent interval,
    conditional on constant net flow within each interval. Unequal interval durations
    are retained. Mean flow rates become volumes through their own interval duration;
    total flow values are already volumes.

    cumulative=True prefixes the ROWS OF H before applying either calculation. This
    shares the first storage reference and all reused measurements correctly, rather
    than pretending cumulative residuals have independent reference errors. Full raw
    covariance is supported; diagonal variances use column weighting without expanding
    a dense raw covariance matrix. Neither path estimates covariance from residuals.
    """
    if not isinstance(record, BalanceRecord):
        raise TypeError("balance_residuals requires a BalanceRecord")
    if not isinstance(cumulative, (bool, np.bool_)):
        raise ValueError("cumulative must be a boolean")
    durations = np.diff(record.edges)
    n, m = record.flows.shape
    count = n if record.storage_support == "instant" else n - 1
    storage_count = record.storage.size
    H = np.zeros((count, storage_count + record.flows.size))
    rows = np.arange(count)
    H[rows, rows] = -1.0
    H[rows, rows + 1] = 1.0
    columns = storage_count + np.arange(record.flows.size).reshape(n, m)
    factors = np.broadcast_to(record.flow_signs, (n, m))
    if record.flow_support == "mean":
        factors = factors * durations[:, None]
    if record.storage_support == "instant":
        H[rows[:, None], columns] = -factors
        times = record.edges[1:]
    else:
        H[rows[:, None], columns[:-1]] = -0.5 * factors[:-1]
        H[rows[:, None], columns[1:]] = -0.5 * factors[1:]
        times = record.edges[1:-1] + 0.5 * durations[1:]
    if cumulative:
        H = np.cumsum(H, axis=0)
    measurements = np.concatenate((record.storage, record.flows.ravel(order="C")))
    with np.errstate(over="ignore", invalid="ignore"):
        residual = H @ measurements
        weighted = H * record.covariance if record.covariance.ndim == 1 else H @ record.covariance
        covariance = weighted @ H.T
        covariance = 0.5 * covariance + 0.5 * covariance.T
    provenance = tuple(tuple(dict.fromkeys(
        evidence_id
        for column in np.flatnonzero(row)
        for evidence_id in record.evidence_ids[column]
    )) for row in H)
    return ResidualSeries(residual=residual, covariance=covariance, operator=H,
                          times=times, evidence_ids=provenance)
