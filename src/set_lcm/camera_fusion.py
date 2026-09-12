"""Offline comparison of synchronized gauge and camera level measurements.

At each time the combined estimate is the two-source best linear unbiased
estimate under the declared covariance. The complete record covariance is
propagated through those weights, retaining shared calibration across times.
There is no interpolation, temporal smoothing, automatic fault exclusion or
reference-instrument input. Raw gauge-camera disagreement remains available.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .camera import LevelSeries
from .invariant import _array, _covariance, _immutable

__all__ = ["LevelEstimate", "LevelComparison", "compare_level_sources"]


@dataclass(frozen=True)
class LevelEstimate:
    times: np.ndarray
    heights: np.ndarray
    covariance: np.ndarray
    observed_indices: tuple[int, ...]
    operator: np.ndarray

    def __post_init__(self):
        times = _array(self.times, "times", 1)
        if np.any(np.diff(times) <= 0):
            raise ValueError("estimate times must be strictly increasing")
        if np.iscomplexobj(self.heights):
            raise ValueError("heights must be real")
        heights = np.asarray(self.heights, dtype=float)
        if heights.shape != times.shape or np.any(np.isinf(heights)):
            raise ValueError("heights must match times and contain finite values or NaN")
        indices = tuple(int(i) for i in np.flatnonzero(np.isfinite(heights)))
        if tuple(self.observed_indices) != indices:
            raise ValueError("observed_indices must identify finite heights in order")
        covariance = _covariance(self.covariance, "level covariance", len(indices))
        operator = _array(self.operator, "level operator", 2)
        if operator.shape[0] != len(indices):
            raise ValueError("operator rows must match observed heights")
        for name, value in (("times", times), ("heights", heights),
                            ("covariance", covariance), ("operator", operator)):
            object.__setattr__(self, name, _immutable(value))
        object.__setattr__(self, "observed_indices", indices)


@dataclass(frozen=True)
class LevelComparison:
    series: Mapping[str, LevelEstimate]
    difference: LevelEstimate
    camera_weight: np.ndarray
    raw_covariance: np.ndarray
    raw_order: tuple[tuple[str, int], ...]
    assumptions: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "series", MappingProxyType(dict(self.series)))
        object.__setattr__(self, "camera_weight", _immutable(np.asarray(self.camera_weight, dtype=float)))
        object.__setattr__(self, "raw_covariance", _immutable(np.asarray(self.raw_covariance, dtype=float)))


def _declared_covariance(value, name: str, size: int) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        array = np.diag(_array(array, name, 1, (size,)))
    return _covariance(array, name, size)


def compare_level_sources(gauge_times, gauge_values, gauge_covariance,
                          camera_times, camera: LevelSeries, *, cross_covariance) -> LevelComparison:
    """Compare and fuse level readings in metres on an explicit common time axis.

    Camera heights and gauge values use the same N positions. Valid camera
    timestamps must exactly match the corresponding gauge timestamps; excluded
    frames may retain their original bad timestamps or NaN when absent. No nearest-time matching is
    done. Gauge times are finite, strictly increasing seconds on the same origin.
    NaN gauge values are missing, not interpolated.

    Gauge covariance is a variance vector or full N-by-N matrix; camera covariance
    follows LevelSeries's valid-subset ordering. cross_covariance is required:
    explicitly pass None to declare independent gauge/camera errors, otherwise
    supply N-by-K entries Cov(gauge[i], camera_valid[j]) in m^2. The entire joint
    covariance is checked, not only its diagonal blocks.

    The combined estimator uses per-time BLUE weights, which can be negative for
    correlated sources. Cross-time covariance is propagated, not used to smooth
    the estimates. Shared-reference errors must be supplied, never inferred.
    """
    if not isinstance(camera, LevelSeries):
        raise TypeError("camera must be a LevelSeries")
    times = _array(gauge_times, "gauge times", 1)
    n = times.size
    if np.any(np.diff(times) <= 0):
        raise ValueError("gauge times must be strictly increasing")
    if np.iscomplexobj(camera_times):
        raise ValueError("camera times must be real")
    camera_times = np.asarray(camera_times, dtype=float)
    if camera_times.shape != (n,) or np.any(np.isinf(camera_times)):
        raise ValueError("camera times must match the axis and be finite or NaN for excluded frames")
    if camera.heights.shape != (n,):
        raise ValueError("camera heights must match the common time axis")
    if np.iscomplexobj(gauge_values):
        raise ValueError("gauge values must be real")
    gauge_values = np.asarray(gauge_values, dtype=float)
    if gauge_values.shape != (n,) or np.any(np.isinf(gauge_values)):
        raise ValueError("gauge values must match times and be finite or NaN")
    gauge_indices = tuple(int(i) for i in np.flatnonzero(np.isfinite(gauge_values)))
    camera_indices = camera.observed_indices
    if any(camera_times[i] != times[i] for i in camera_indices):
        raise ValueError("valid camera and gauge capture times must match exactly; align or exclude explicitly")
    ng, nc = len(gauge_indices), len(camera_indices)
    gauge_C = _declared_covariance(gauge_covariance, "gauge covariance", n)
    if cross_covariance is None:
        cross = np.zeros((n, nc))
        independence = "Gauge/camera errors declared independent, including calibration references."
    else:
        cross = _array(cross_covariance, "gauge-camera cross covariance", 2, (n, nc))
        independence = "Gauge/camera dependence declared through the supplied full cross covariance."
    # Validate unused gauge rows too: missing values do not license invalid uncertainty.
    full_C = np.block([[gauge_C, cross], [cross.T, camera.covariance]])
    full_C = _covariance(full_C, "joint gauge-camera covariance", n + nc)
    select = list(gauge_indices) + list(range(n, n + nc))
    C = full_C[np.ix_(select, select)]
    y = np.concatenate((gauge_values[list(gauge_indices)], camera.heights[list(camera_indices)]))
    g_lookup = {index: j for j, index in enumerate(gauge_indices)}
    c_lookup = {index: ng + j for j, index in enumerate(camera_indices)}

    def estimate(indices, rows):
        operator = np.asarray(rows, dtype=float).reshape(len(indices), ng + nc)
        heights = np.full(n, np.nan)
        heights[list(indices)] = operator @ y
        with np.errstate(over="ignore", invalid="ignore"):
            covariance = operator @ C @ operator.T
            covariance = .5 * covariance + .5 * covariance.T
        return LevelEstimate(times, heights, covariance, tuple(indices), operator)

    basis = np.eye(ng + nc)
    gauge = estimate(gauge_indices, [basis[j] for j in range(ng)])
    camera_estimate = estimate(camera_indices, [basis[ng + j] for j in range(nc)])
    combined_indices, combined_rows, difference_indices, difference_rows = [], [], [], []
    camera_weight = np.full(n, np.nan)
    for i in range(n):
        g, c = g_lookup.get(i), c_lookup.get(i)
        if g is None and c is None:
            continue
        if g is None:
            row, weight = basis[c], 1.0
        elif c is None:
            row, weight = basis[g], 0.0
        else:
            scale = max(C[g, g], C[c, c], abs(C[g, c]))
            vg, vc, cross = ((C[g, g] / scale, C[c, c] / scale, C[g, c] / scale)
                             if scale else (0.0, 0.0, 0.0))
            variance_difference = vg + vc - 2 * cross
            if variance_difference < 0:
                raise ValueError("gauge-camera difference variance is numerically negative")
            if variance_difference == 0:
                if y[g] != y[c]:
                    raise ValueError("sources disagree despite declared exact agreement; do not fuse them")
                weight = .5
            else:
                weight = (vg - cross) / variance_difference
            row = (1 - weight) * basis[g] + weight * basis[c]
            difference_indices.append(i)
            difference_rows.append(basis[g] - basis[c])
        combined_indices.append(i)
        combined_rows.append(row)
        camera_weight[i] = weight
    return LevelComparison(
        {"gauge_only": gauge, "camera_only": camera_estimate,
         "combined": estimate(combined_indices, combined_rows)},
        estimate(difference_indices, difference_rows), camera_weight, C,
        tuple(("gauge", i) for i in gauge_indices) + tuple(("camera", i) for i in camera_indices),
        (independence, "All admitted readings describe instantaneous level in metres at the same declared capture time.",
         "Covariance and weights are conditional on calibration, frame admission and measurement-error assumptions.",
         "No reference observations or fault labels enter estimation; disagreement is preserved before fusion."),
    )
