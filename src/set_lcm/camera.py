"""Declared-ROI camera level measurements and conditional linear calibration.

This NumPy-only baseline detects a horizontal, antialiased intensity step in a
fixed grayscale image. It is not a general segmentation model. Pixel row i is
the center of the band [i-.5, i+.5]; an edge at p has below-edge coverage
clip(i+.5-p, 0, 1). Contrast/profile/column checks are quality gates, not calibrated
probabilities or uncertainty estimates. Failed detections are missing measurements.

A separately declared marker edge can register vertical image translation. Its
displacement is retained separately and subtracted from the measured water row.
This supports one translation coordinate, not rotation, perspective, or 3-D odometry.

Linear calibration treats anchor pixel coordinates as fixed and exact, with known
physical-reference covariance. Mapping uncertain rows uses first-order covariance
propagation with shared calibration parameters. It does not provide exact Gaussian
coverage after products, quality gating, or calibration selection. All uncertainty
is supplied by the caller; this module does not estimate it from image contrast.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .invariant import _array, _covariance, _immutable

__all__ = ["PixelDetection", "MarkerRegistration", "LevelSeries", "LinearLevelCalibration",
           "detect_level", "register_vertical_marker", "compensate_level", "fit_linear_calibration"]


def _number(value, name: str, *, positive=False, nonnegative=False) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result) or (positive and result <= 0) or (nonnegative and result < 0):
        raise ValueError(f"{name} must be finite with a valid sign")
    return result


def _bounds(value, name: str, limit: int) -> tuple[int, int]:
    try:
        pair = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be two integer bounds") from exc
    if len(pair) != 2 or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in pair):
        raise ValueError(f"{name} must be two integer bounds")
    left, right = (int(i) for i in pair)
    if not 0 <= left < right <= limit:
        raise ValueError(f"{name} must be a nonempty half-open range inside the frame")
    return left, right


def _ids(value) -> tuple[str, ...]:
    if isinstance(value, str):
        raise ValueError("calibration_ids must be a sequence of nonempty strings")
    try:
        ids = tuple(value)
    except TypeError as exc:
        raise ValueError("calibration_ids must be a sequence of nonempty strings") from exc
    if not all(isinstance(item, str) and item.strip() for item in ids):
        raise ValueError("calibration_ids must be a sequence of nonempty strings")
    return tuple(dict.fromkeys(ids))


def _possibly_missing(value, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric vector") from exc
    if result.ndim != 1 or np.any(np.isinf(result)):
        raise ValueError(f"{name} must be a 1-D vector; only NaN denotes missing data")
    return result


def _declared_covariance(value, name: str, size: int) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be declared as variances or full covariance") from exc
    if array.ndim == 1:
        array = _array(array, name, 1, (size,))
        if np.any(array < 0):
            raise ValueError(f"{name} variances must be nonnegative")
        return np.diag(array)
    return _covariance(array, name, size)


@dataclass(frozen=True)
class PixelDetection:
    row: float | None
    status: str
    contrast: float | None
    column_spread: float | None
    explanation: str

    def __post_init__(self):
        if self.status not in {"ok", "low_contrast", "inconsistent_columns", "invalid_edge", "missing_marker"}:
            raise ValueError("unknown pixel-detection status")
        if (self.row is not None) != (self.status == "ok"):
            raise ValueError("only a successful detection may contain a row")
        if self.row is not None:
            object.__setattr__(self, "row", _number(self.row, "row"))
        for name in ("contrast", "column_spread"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _number(value, name, nonnegative=name == "column_spread"))


def detect_level(frame, *, water_columns, expected_polarity, min_contrast,
                 max_column_spread, row_range=None, plateau_rows=3,
                 max_profile_deviation=0.2) -> PixelDetection:
    """Measure a declared horizontal step; return missing on failed quality checks.

    Column and optional row bounds are half-open integer ranges. Polarity is
    'dark_below' or 'bright_below'. min_contrast is in frame intensity units;
    max_column_spread is the allowed max-minus-min row spread in pixels.
    plateau_rows supplies the upper/lower intensity plateaus. The strongest
    three-gradient neighborhood estimates each antialiased edge by its weighted
    position; the full column must agree with that step to within
    max_profile_deviation times its contrast. All columns must pass. The reported
    row is their arithmetic mean. These thresholds do not supply a pixel variance.
    """
    frame = _array(frame, "frame", 2)
    height, width = frame.shape
    left, right = _bounds(water_columns, "water_columns", width)
    top, bottom = (0, height) if row_range is None else _bounds(row_range, "row_range", height)
    if expected_polarity not in ("dark_below", "bright_below"):
        raise ValueError("expected_polarity must be 'dark_below' or 'bright_below'")
    min_contrast = _number(min_contrast, "min_contrast", positive=True)
    max_column_spread = _number(max_column_spread, "max_column_spread", nonnegative=True)
    max_profile_deviation = _number(max_profile_deviation, "max_profile_deviation", positive=True)
    if max_profile_deviation >= 0.5:
        raise ValueError("max_profile_deviation must be less than 0.5")
    if isinstance(plateau_rows, (bool, np.bool_)) or not isinstance(plateau_rows, (int, np.integer)) or plateau_rows < 1:
        raise ValueError("plateau_rows must be a positive integer")
    if bottom - top < 2 * plateau_rows + 3:
        raise ValueError("row range is too short for plateaus and an interior edge")
    roi = frame[top:bottom, left:right]
    above = np.median(roi[:plateau_rows], axis=0)
    below = np.median(roi[-plateau_rows:], axis=0)
    sign = 1.0 if expected_polarity == "bright_below" else -1.0
    with np.errstate(over="ignore", invalid="ignore"):
        contrasts = sign * (below - above)
    if not np.all(np.isfinite(contrasts)):
        raise ValueError("frame contrast exceeds the supported numeric range")
    contrast = float(np.min(contrasts))
    if contrast < min_contrast:
        return PixelDetection(None, "low_contrast", contrast, None,
                              "A declared column lacks the required contrast or polarity.")
    rows = []
    positions = np.arange(top, bottom, dtype=float)
    for j in range(roi.shape[1]):
        gradient = sign * np.diff(roi[:, j])
        peak = int(np.argmax(gradient))
        start, end = max(0, peak - 1), min(gradient.size, peak + 2)
        weights = np.maximum(gradient[start:end], 0.0)
        mass = float(np.sum(weights))
        if not math.isfinite(mass) or mass <= 0:
            return PixelDetection(None, "invalid_edge", contrast, None, "No resolved interior edge.")
        row = float(np.dot(weights / mass, top + np.arange(start, end) + 0.5))
        if not top + plateau_rows - 0.5 <= row <= bottom - plateau_rows - 0.5:
            return PixelDetection(None, "invalid_edge", contrast, None,
                                  "The edge overlaps the required intensity plateaus.")
        expected = above[j] + sign * contrasts[j] * np.clip(positions + 0.5 - row, 0.0, 1.0)
        deviation = float(np.max(np.abs(roi[:, j] - expected))) / contrasts[j]
        if not math.isfinite(deviation) or deviation > max_profile_deviation:
            return PixelDetection(None, "invalid_edge", contrast, None,
                                  "A column is inconsistent with the declared antialiased single-step model.")
        rows.append(row)
    spread = float(np.max(rows) - np.min(rows))
    if spread > max_column_spread:
        return PixelDetection(None, "inconsistent_columns", contrast, spread,
                              "Column edge positions disagree beyond the declared tolerance.")
    return PixelDetection(float(np.mean(rows)), "ok", contrast, spread,
                          "Declared horizontal edge passed deterministic quality gates; uncertainty is caller-supplied.")


@dataclass(frozen=True)
class MarkerRegistration:
    row: float | None
    displacement: float | None
    reference_row: float
    status: str
    explanation: str

    def __post_init__(self):
        object.__setattr__(self, "reference_row", _number(self.reference_row, "reference_row"))
        if self.status not in {"ok", "low_contrast", "inconsistent_columns", "invalid_edge"}:
            raise ValueError("unknown marker-registration status")
        if (self.row is not None) != (self.status == "ok") or (self.displacement is not None) != (self.status == "ok"):
            raise ValueError("failed marker registration cannot contain a row or displacement")
        if self.status == "ok":
            object.__setattr__(self, "row", _number(self.row, "marker row"))
            object.__setattr__(self, "displacement", _number(self.displacement, "marker displacement"))
            if self.displacement != self.row - self.reference_row:
                raise ValueError("marker displacement must equal row minus reference_row")


def register_vertical_marker(frame, *, marker_columns, reference_row, **detector_options) -> MarkerRegistration:
    """Register a separate fixed horizontal marker edge under vertical translation only.

    detector_options are detect_level's required polarity/quality thresholds and
    optional row bounds/profile settings. The marker reference row is declared;
    any uncertainty in it must be included in the caller's marker-displacement
    covariance, including its shared cross-time contribution.
    """
    reference_row = _number(reference_row, "reference_row")
    detection = detect_level(frame, water_columns=marker_columns, **detector_options)
    displacement = None if detection.row is None else _number(detection.row - reference_row, "marker displacement")
    return MarkerRegistration(detection.row, displacement, reference_row, detection.status, detection.explanation)


def compensate_level(level: PixelDetection, registration: MarkerRegistration) -> PixelDetection:
    """Subtract declared marker translation; failed registration never falls back to raw pixels."""
    if not isinstance(level, PixelDetection) or not isinstance(registration, MarkerRegistration):
        raise TypeError("compensate_level requires PixelDetection and MarkerRegistration")
    if level.status != "ok":
        return level
    if registration.status != "ok":
        return PixelDetection(None, "missing_marker", level.contrast, level.column_spread,
                              "Marker registration failed; uncompensated level is not substituted.")
    row = _number(level.row - registration.displacement, "corrected row")
    return PixelDetection(row, "ok", level.contrast, level.column_spread,
                          "Marker vertical displacement was subtracted; both errors require declared covariance.")


@dataclass(frozen=True)
class LevelSeries:
    """Full-length values/statuses; joint covariance covers observed_indices only."""
    heights: np.ndarray
    covariance: np.ndarray
    corrected_rows: np.ndarray
    statuses: tuple[str, ...]
    observed_indices: tuple[int, ...]
    calibration_ids: tuple[str, ...]
    assumptions: tuple[str, ...]

    def __post_init__(self):
        heights = _possibly_missing(self.heights, "heights")
        rows = _possibly_missing(self.corrected_rows, "corrected_rows")
        if rows.shape != heights.shape:
            raise ValueError("heights and corrected_rows must have matching shapes")
        statuses = tuple(self.statuses)
        if len(statuses) != heights.size or any(s not in {"ok", "missing_level", "missing_marker", "outside_calibration_range"} for s in statuses):
            raise ValueError("statuses must label every level sample")
        valid = np.array([s == "ok" for s in statuses])
        indices = tuple(self.observed_indices)
        if (any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in indices)
                or indices != tuple(int(i) for i in np.flatnonzero(valid))):
            raise ValueError("observed_indices must identify the valid samples in order")
        indices = tuple(int(i) for i in indices)
        if not np.array_equal(np.isfinite(heights), valid) or not np.array_equal(np.isfinite(rows), valid):
            raise ValueError("invalid samples must have NaN heights and corrected rows")
        covariance = _covariance(self.covariance, "level covariance", len(indices))
        for name, array in (("heights", heights), ("corrected_rows", rows), ("covariance", covariance)):
            object.__setattr__(self, name, _immutable(array))
        object.__setattr__(self, "statuses", statuses)
        object.__setattr__(self, "observed_indices", indices)
        object.__setattr__(self, "calibration_ids", _ids(self.calibration_ids))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))


@dataclass(frozen=True)
class LinearLevelCalibration:
    """h = slope_m_per_pixel * p + intercept_m over a declared pixel range.

    parameter_covariance is ordered [slope, intercept] and shared across all
    mapped observations. Exact zero covariance is permitted only as an explicit
    declaration. Lens/perspective effects and errors-in-variables are not modeled.
    """
    slope_m_per_pixel: float
    intercept_m: float
    parameter_covariance: np.ndarray
    valid_pixel_range: tuple[float, float]
    calibration_ids: tuple[str, ...] = ()

    def __post_init__(self):
        slope = _number(self.slope_m_per_pixel, "slope_m_per_pixel")
        if slope == 0:
            raise ValueError("a level calibration must have a nonzero slope")
        intercept = _number(self.intercept_m, "intercept_m")
        covariance = _covariance(self.parameter_covariance, "parameter_covariance", 2)
        bounds = _array(self.valid_pixel_range, "valid_pixel_range", 1, (2,))
        if bounds[0] >= bounds[1]:
            raise ValueError("valid_pixel_range must be increasing")
        object.__setattr__(self, "slope_m_per_pixel", slope)
        object.__setattr__(self, "intercept_m", intercept)
        object.__setattr__(self, "parameter_covariance", _immutable(covariance))
        object.__setattr__(self, "valid_pixel_range", tuple(float(x) for x in bounds))
        object.__setattr__(self, "calibration_ids", _ids(self.calibration_ids))

    def map_rows(self, pixel_rows, *, localization_covariance, marker_displacement=None,
                 marker_covariance=None, localization_marker_covariance=None) -> LevelSeries:
        """Map rows, preserving missing samples and full shared calibration covariance.

        Supplied localization/marker covariances are variance vectors or full
        matrices in pixels squared, even for missing samples. Marker reference
        uncertainty belongs in marker_covariance. The cross covariance is
        Cov(localization_error, marker_displacement_error); omission explicitly
        assumes their independence. Corrected rows have covariance
        Cp = Clocal + Cmarker - Ccross - Ccross.T. The entire joint declaration
        must be PSD. Without a marker, marker covariance arguments are refused.

        Calibration and pixel errors are assumed independent. First-order mapped
        covariance is J Cparameter J.T + slope**2 Cp, with J rows [p,1]. The
        second-order slope/pixel product is omitted; quality-gated errors are not
        asserted Gaussian. Rows outside calibration bounds are missing, never
        extrapolated. An all-missing input returns an empty covariance.
        """
        rows = _possibly_missing(pixel_rows, "pixel_rows")
        n = rows.size
        local_cov = _declared_covariance(localization_covariance, "localization_covariance", n)
        valid = np.isfinite(rows)
        statuses = np.where(valid, "ok", "missing_level").astype(object)
        corrected = rows.copy()
        if marker_displacement is None:
            if marker_covariance is not None or localization_marker_covariance is not None:
                raise ValueError("marker covariance requires marker_displacement")
            pixel_covariance = local_cov
        else:
            marker = _possibly_missing(marker_displacement, "marker_displacement")
            if marker.shape != rows.shape:
                raise ValueError("marker_displacement must match pixel_rows")
            if marker_covariance is None:
                raise ValueError("marker_covariance must be explicitly declared")
            marker_cov = _declared_covariance(marker_covariance, "marker_covariance", n)
            cross = (np.zeros((n, n)) if localization_marker_covariance is None else
                     _array(localization_marker_covariance, "localization_marker_covariance", 2, (n, n)))
            _covariance(np.block([[local_cov, cross], [cross.T, marker_cov]]), "joint localization/marker covariance", 2*n)
            pixel_covariance = local_cov + marker_cov - cross - cross.T
            statuses[valid & ~np.isfinite(marker)] = "missing_marker"
            valid &= np.isfinite(marker)
            with np.errstate(over="ignore", invalid="ignore"):
                corrected = rows - marker
        if np.any(valid & ~np.isfinite(corrected)):
            raise ValueError("corrected pixels exceed the supported finite numeric range")
        low, high = self.valid_pixel_range
        outside = valid & ((corrected < low) | (corrected > high))
        statuses[outside] = "outside_calibration_range"
        valid &= ~outside
        corrected[~valid] = np.nan
        observed = tuple(int(i) for i in np.flatnonzero(valid))
        J = np.column_stack((corrected[valid], np.ones(len(observed))))
        heights = np.full(n, np.nan)
        with np.errstate(over="ignore", invalid="ignore"):
            heights[valid] = self.slope_m_per_pixel * corrected[valid] + self.intercept_m
            # Scale in two steps: a finite a*a may overflow even when a*Cp*a is
            # representable for small pixel variances in a different unit system.
            localization_term = (self.slope_m_per_pixel * pixel_covariance[np.ix_(valid, valid)]
                                 * self.slope_m_per_pixel)
            covariance = J @ self.parameter_covariance @ J.T + localization_term
            covariance = 0.5 * covariance + 0.5 * covariance.T
        if not np.all(np.isfinite(heights[valid])):
            raise ValueError("mapped heights exceed the supported finite numeric range")
        assumptions = (
            "Calibration is linear within its declared pixel range; no extrapolation is performed.",
            "Calibration anchor pixel coordinates are fixed and exact; errors-in-variables are not implemented.",
            "Calibration parameter errors are independent of localization and marker errors.",
            "Covariance propagation is first order; slope/pixel product terms are omitted.",
            "Localization/marker cross covariance is declared; omission assumes independence.",
            "Quality gates and products do not establish exact Gaussian coverage or calibrated image confidence.",
            "Marker correction, when supplied, models vertical image translation only.",
        )
        return LevelSeries(heights, covariance, corrected, tuple(statuses), observed,
                           self.calibration_ids, assumptions)


def fit_linear_calibration(pixel_rows, reference_heights, reference_covariance, *, calibration_ids=()) -> LinearLevelCalibration:
    """GLS calibration from fixed exact pixel anchors and noisy physical references.

    reference_covariance may be a variance vector or full known SPD covariance
    in meters squared. It includes shared physical-reference uncertainty. The
    fit does not estimate a noise scale from residuals. A centered/scaled design
    and whitened least squares avoid raw normal-equation conditioning. Returned
    validity bounds are the extreme supplied anchors. No automatic anchor
    extraction or selection occurs here: noisy pixel anchors require a separate
    errors-in-variables method, which is deliberately not implemented.
    """
    pixels = _array(pixel_rows, "pixel_rows", 1)
    n = pixels.size
    heights = _array(reference_heights, "reference_heights", 1, (n,))
    if n < 2:
        raise ValueError("calibration requires at least two distinct pixel anchors")
    covariance = _declared_covariance(reference_covariance, "reference_covariance", n)
    low, high = float(np.min(pixels)), float(np.max(pixels))
    if low == high:
        raise ValueError("calibration requires at least two distinct pixel anchors")
    with np.errstate(over="ignore", invalid="ignore"):
        center = 0.5 * low + 0.5 * high
        span = high - low
    if not math.isfinite(span):
        raise ValueError("calibration pixel span exceeds the supported numeric range")
    design = np.column_stack(((pixels - center) / span, np.ones(n)))
    if np.any(np.diag(covariance) <= 0):
        raise ValueError("reference_covariance must be positive definite")
    scales = np.sqrt(np.diag(covariance))
    correlation = covariance / scales[:, None] / scales[None, :]
    try:
        eigenvalues = np.linalg.eigvalsh(correlation)
        if eigenvalues[0] <= n * np.finfo(float).eps * float(np.max(np.abs(eigenvalues))):
            raise ValueError("reference covariance must have numerically resolved positive eigenvalues")
        chol = np.linalg.cholesky(correlation)
        W = np.linalg.solve(chol, design / scales[:, None])
        y = np.linalg.solve(chol, heights / scales)
        U, singular, Vt = np.linalg.svd(W, full_matrices=False)
        if singular[-1] <= max(W.shape) * np.finfo(float).eps * singular[0]:
            raise ValueError("calibration design is numerically unresolved")
        beta = Vt.T @ ((U.T @ y) / singular)
        beta_cov = (Vt.T / singular) @ (Vt.T / singular).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("calibration whitening or least-squares solution failed") from exc
    transform = np.array([[1.0 / span, 0.0], [-center / span, 1.0]])
    with np.errstate(over="ignore", invalid="ignore"):
        parameters = transform @ beta
        parameter_covariance = transform @ beta_cov @ transform.T
    return LinearLevelCalibration(parameters[0], parameters[1], parameter_covariance,
                                  (low, high), _ids(calibration_ids))
