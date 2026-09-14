"""Synthetic calibrated-camera/gauge comparison with an independent scoring reference.

The camera observations are rendered grayscale frames, not ground-truth level values
passed to inference. Fixed image-quality and clock policies precede an offline BLUE
comparison. This experiment tests a small declared measurement model; it is neither
field validation nor general visual odometry or automatic sensor-fault isolation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

from set_lcm.camera import (LevelSeries, LinearLevelCalibration, detect_level,
                            fit_linear_calibration, register_vertical_marker)
from set_lcm.camera_fusion import LevelComparison, compare_level_sources
from set_lcm.diagnostics import DiagnosticResult, diagnose
from set_lcm.frame_quality import FrameAudit, audit_frames, frame_content_hash, frame_spectrum

from .provenance import header_line, provenance

N_FRAMES = 32
ONSET_SECONDS = 12.0
EXPECTED_PERIOD = 1.0
MAX_JITTER_SECONDS = 0.15
SPECTRAL_WINDOW = 8
CALIBRATION_SEEDS = tuple(range(8))
EVALUATION_SEEDS = tuple(range(10000, 10016))
SCENARIOS = ("healthy", "gauge_drift", "camera_obstruction", "camera_vertical_drift",
             "duplicate_frame", "dropped_frame", "time_jitter", "shared_calibration_offset",
             "physical_level_change")
METHODS = ("gauge_only", "camera_only", "combined")
STATUSES = ("consistent", "identified", "ambiguous", "unexplained", "insufficient_evidence")
FRAME_SHAPE = (56, 88)
INTENSITY_GRID = 1e-9
WATER_COLUMNS = (8, 40)
MARKER_COLUMNS = (56, 80)
MARKER_REFERENCE_ROW = 8.0
TRUE_SLOPE_M_PER_PIXEL = -0.02
TRUE_INTERCEPT_M = 0.95
PIXEL_SD = 0.03
GAUGE_SD_M = 0.006
REFERENCE_SD_M = 0.002
ANCHOR_REFERENCE_SD_M = 0.002
SHARED_REFERENCE_SD_M = 0.004
GAUGE_DRIFT_M_PER_SECOND = 0.0015
CAMERA_TRANSLATION_PIXELS = 4.0
SHARED_OFFSET_M = 0.03
ALPHA = 0.01
INTERVAL_LEVEL = 0.95
DETECTOR_OPTIONS = {"expected_polarity": "dark_below", "min_contrast": 0.3,
                    "max_column_spread": 0.15, "plateau_rows": 3,
                    "max_profile_deviation": 0.2}


@dataclass(frozen=True)
class CalibrationEvidence:
    frames: tuple[np.ndarray, ...]
    reference_heights: np.ndarray
    reference_covariance: np.ndarray
    calibration_ids: tuple[str, ...]


@dataclass(frozen=True)
class SyntheticTrial:
    seed: int
    scenario: str
    frames: tuple[np.ndarray | None, ...]
    times: np.ndarray
    frame_ids: tuple[int, ...]
    gauge_times: np.ndarray
    gauge_values: np.ndarray
    gauge_covariance: np.ndarray
    calibration_evidence: CalibrationEvidence
    # These two independent-reference fields and truth belong to scoring only.
    reference_values: np.ndarray
    reference_covariance: np.ndarray
    truth: np.ndarray


@dataclass(frozen=True)
class CameraAnalysis:
    calibration: LinearLevelCalibration
    audit: FrameAudit
    water_rows: np.ndarray
    marker_displacements: np.ndarray
    water_statuses: tuple[str, ...]
    marker_statuses: tuple[str, ...]
    camera: LevelSeries
    unregistered_camera: LevelSeries
    comparison: LevelComparison
    diagnostic: dict
    spectral_windows: tuple[dict, ...]


def render_frame(water_row: float, marker_row: float = MARKER_REFERENCE_ROW) -> np.ndarray:
    """Simulator: area-antialiased edges on a fixed 1e-9 intensity grid.

    Quantization makes payload hashes insensitive to tiny platform math-library
    differences. Its <=5e-10 intensity error is negligible against the declared
    0.03-pixel localization scale; exact integer-row calibration anchors survive.
    """
    positions = np.arange(FRAME_SHAPE[0], dtype=float)
    frame = np.full(FRAME_SHAPE, 0.5)
    for row, columns, above, below in ((water_row, WATER_COLUMNS, .8, .2),
                                       (marker_row, MARKER_COLUMNS, .9, .1)):
        fraction = np.clip(positions + .5 - row, 0, 1)
        frame[:, columns[0]:columns[1]] = (above + (below - above) * fraction)[:, None]
    return np.rint(frame / INTENSITY_GRID) * INTENSITY_GRID


def _shared_reference_error() -> float:
    return float(SHARED_REFERENCE_SD_M * np.random.default_rng(CALIBRATION_SEEDS[0]).standard_normal())


def calibration_evidence(*, added_offset_m: float = 0.0) -> CalibrationEvidence:
    """Eight separate exact-design anchor images; only reference heights are noisy.

    The shared reference realization is reused across all evaluation records.
    Each anchor's independent error uses a separate development seed; its second
    draw avoids reusing the shared-error draw from the first seed.
    """
    rows = np.linspace(12.0, 40.0, len(CALIBRATION_SEEDS))
    independent_errors = np.array([np.random.default_rng(seed).standard_normal(2)[1]
                                   for seed in CALIBRATION_SEEDS]) * ANCHOR_REFERENCE_SD_M
    references = (TRUE_SLOPE_M_PER_PIXEL * rows + TRUE_INTERCEPT_M
                  + independent_errors + _shared_reference_error() + added_offset_m)
    covariance = (ANCHOR_REFERENCE_SD_M**2 * np.eye(len(rows))
                  + SHARED_REFERENCE_SD_M**2 * np.ones((len(rows), len(rows))))
    return CalibrationEvidence(tuple(render_frame(row) for row in rows), references, covariance,
                               tuple(f"synthetic-anchor-seed-{seed}" for seed in CALIBRATION_SEEDS))


def build_case(seed: int, scenario: str) -> SyntheticTrial:
    """Simulator/scorer bundle, also exposed for saving a small inspectable fixture.

    Inference must receive only the explicit measurement/calibration arguments of
    infer_measurements, never this bundle's truth or independent reference.
    """
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if scenario not in SCENARIOS:
        raise ValueError("unknown camera-baseline scenario")
    rng = np.random.default_rng(seed)
    times = np.arange(N_FRAMES, dtype=float) * EXPECTED_PERIOD
    if scenario == "time_jitter":
        times += .05 * np.sin(.7 * times)
    ramp = np.maximum(times - ONSET_SECONDS, 0.0)
    truth = .40 + .03 * np.sin(2 * np.pi * times / 16) + .001 * times
    if scenario == "physical_level_change":
        truth += .04 * (1 - np.exp(-ramp / 3))
    shared_offset = SHARED_OFFSET_M if scenario == "shared_calibration_offset" else 0.0
    water_error, marker_error, gauge_error, reference_error = rng.standard_normal((4, N_FRAMES))
    translation = (CAMERA_TRANSLATION_PIXELS * ramp / (times[-1] - ONSET_SECONDS)
                   if scenario == "camera_vertical_drift" else np.zeros(N_FRAMES))
    rows = (truth - TRUE_INTERCEPT_M) / TRUE_SLOPE_M_PER_PIXEL
    frames = [render_frame(row + move + PIXEL_SD * water,
                           MARKER_REFERENCE_ROW + move + PIXEL_SD * marker)
              for row, move, water, marker in zip(rows, translation, water_error, marker_error, strict=True)]
    if scenario == "camera_obstruction":
        for index in range(12, 18):
            frames[index][:, WATER_COLUMNS[0]:WATER_COLUMNS[1]] = .5
    if scenario == "duplicate_frame":
        frames[16] = frames[15].copy()  # A repeated payload at a new declared capture time.
    if scenario == "dropped_frame":
        frames[10] = frames[20] = None  # Timeline retained; the payload is missing.
    gauge = truth + GAUGE_SD_M * gauge_error + _shared_reference_error() + shared_offset
    if scenario == "gauge_drift":
        gauge += GAUGE_DRIFT_M_PER_SECOND * ramp
    gauge_covariance = (GAUGE_SD_M**2 * np.eye(N_FRAMES)
                        + SHARED_REFERENCE_SD_M**2 * np.ones((N_FRAMES, N_FRAMES)))
    return SyntheticTrial(int(seed), scenario, tuple(frames), times, tuple(range(N_FRAMES)), times.copy(),
                          gauge, gauge_covariance, calibration_evidence(added_offset_m=shared_offset),
                          truth + REFERENCE_SD_M * reference_error,
                          REFERENCE_SD_M**2 * np.eye(N_FRAMES), truth)


def fit_camera(evidence: CalibrationEvidence) -> LinearLevelCalibration:
    """Extract separate anchor images, then fit the declared exact-pixel design."""
    detections = [detect_level(frame, water_columns=WATER_COLUMNS, **DETECTOR_OPTIONS)
                  for frame in evidence.frames]
    if any(detection.status != "ok" for detection in detections):
        raise ValueError("a calibration anchor failed the fixed image-quality policy")
    return fit_linear_calibration([detection.row for detection in detections],
                                  evidence.reference_heights, evidence.reference_covariance,
                                  calibration_ids=evidence.calibration_ids)


def _spectra(values: np.ndarray, audit: FrameAudit) -> tuple[dict, ...]:
    windows = []
    for start in range(0, len(values), SPECTRAL_WINDOW):
        stop = min(start + SPECTRAL_WINDOW, len(values))
        outcome = {"source_indices": list(range(start, stop)), "status": "refused"}
        # Retain a refusal inherited from the complete audit, including a repeated
        # payload whose first occurrence lies in a preceding window.
        if not all(audit.admitted[start:stop]) or not np.all(np.isfinite(values[start:stop])):
            outcome["reason"] = "A frame or extracted feature failed admission in the full record."
        else:
            window_audit = audit_frames(audit.frame_ids[start:stop], audit.capture_times[start:stop],
                                        audit.content_hashes[start:stop], time_origin=audit.time_origin,
                                        source_kind=audit.source_kinds[start:stop],
                                        expected_period=audit.expected_period, max_jitter=audit.max_jitter,
                                        source_indices=audit.source_indices[start:stop],
                                        missing=audit.missing[start:stop],
                                        reject_repeated_content=audit.reject_repeated_content)
            try:
                spectrum = frame_spectrum(values[start:stop], window_audit, feature_unit="m",
                                          detrend="linear", window="hann")
            except ValueError as error:
                outcome["reason"] = str(error)
            else:
                outcome.update(status="computed", spectrum=spectrum.as_dict())
        windows.append(outcome)
    return tuple(windows)


def infer_measurements(frames, times, frame_ids, gauge_times, gauge_values, gauge_covariance,
                       evidence: CalibrationEvidence) -> CameraAnalysis:
    """Inference uses images, declared times/covariances and separate calibration only.

    Constants specify this synthetic protocol, including shared reference covariance.
    There is no truth, scoring-reference, intervention-label or scenario argument.
    """
    n = len(frames)
    if np.iscomplexobj(times) or np.iscomplexobj(gauge_times):
        raise ValueError("camera and gauge capture times must be real-valued")
    times = np.asarray(times, dtype=float)
    gauge_times = np.asarray(gauge_times, dtype=float)
    if gauge_times.shape != (n,) or not np.all(np.isfinite(gauge_times)):
        raise ValueError("gauge_times must explicitly declare one finite time per frame position")
    calibration = fit_camera(evidence)
    hashes = [None if frame is None else frame_content_hash(frame) for frame in frames]
    audit = audit_frames(frame_ids, times, hashes, time_origin="synthetic trial start, seconds",
                         source_kind="synthetic_test", expected_period=EXPECTED_PERIOD,
                         max_jitter=MAX_JITTER_SECONDS, missing=[frame is None for frame in frames],
                         reject_repeated_content=True)
    water_rows, displacements = np.full(n, np.nan), np.full(n, np.nan)
    water_statuses, marker_statuses = [], []
    for index, (frame, admitted) in enumerate(zip(frames, audit.admitted, strict=True)):
        if not admitted:
            water_statuses.append("frame_not_admitted")
            marker_statuses.append("frame_not_admitted")
            continue
        if times[index] != gauge_times[index]:
            water_statuses.append("time_not_aligned")
            marker_statuses.append("time_not_aligned")
            continue
        water = detect_level(frame, water_columns=WATER_COLUMNS, **DETECTOR_OPTIONS)
        marker = register_vertical_marker(frame, marker_columns=MARKER_COLUMNS,
                                           reference_row=MARKER_REFERENCE_ROW, **DETECTOR_OPTIONS)
        water_statuses.append(water.status)
        marker_statuses.append(marker.status)
        if water.row is not None:
            water_rows[index] = water.row
        if marker.displacement is not None:
            displacements[index] = marker.displacement
    camera = calibration.map_rows(water_rows, localization_covariance=np.full(n, PIXEL_SD**2),
                                   marker_displacement=displacements,
                                   marker_covariance=np.full(n, PIXEL_SD**2),
                                   localization_marker_covariance=np.zeros((n, n)))
    unregistered = calibration.map_rows(water_rows, localization_covariance=np.full(n, PIXEL_SD**2))
    # The physical reference offset affects every gauge and the camera intercept.
    cross_covariance = np.full((n, len(camera.observed_indices)), SHARED_REFERENCE_SD_M**2)
    comparison = compare_level_sources(gauge_times, gauge_values, gauge_covariance, times, camera,
                                        cross_covariance=cross_covariance)
    indices = list(comparison.difference.observed_indices)
    ramp = np.maximum(gauge_times[indices] - ONSET_SECONDS, 0.0)
    if indices:
        diagnostic = diagnose(comparison.difference.heights[indices], comparison.difference.covariance,
                              {"gauge_drift": ramp, "camera_drift": -ramp},
                              alpha=ALPHA, interval_level=INTERVAL_LEVEL).as_dict()
    else:
        diagnostic = DiagnosticResult(
            "insufficient_evidence", (), None, 0, None, (), 0, ALPHA, INTERVAL_LEVEL,
            "No common admitted gauge-camera observations; no consistency statistic or candidate fit was computed.",
            ("Missing camera evidence is not evidence of instrument health.",)).as_dict()
    return CameraAnalysis(calibration, audit, water_rows, displacements,
                          tuple(water_statuses), tuple(marker_statuses), camera, unregistered,
                          comparison, diagnostic, _spectra(camera.heights, audit))


def analyze(trial: SyntheticTrial) -> CameraAnalysis:
    """Select measurement fields only; do not pass the simulation/scoring bundle onward."""
    return infer_measurements(trial.frames, trial.times, trial.frame_ids, trial.gauge_times,
                              trial.gauge_values, trial.gauge_covariance, trial.calibration_evidence)


def _nullable(values) -> list:
    return [None if not np.isfinite(value) else float(value) for value in values]


def _score(heights, covariance, observed_indices, common, truth, reference, reference_covariance) -> dict:
    """Scorer only: independent reference uncertainty adds to estimate covariance."""
    lookup = {index: j for j, index in enumerate(observed_indices)}
    positions = [lookup[index] for index in common]
    C = covariance[np.ix_(positions, positions)]
    error = heights[common] - truth[common]
    difference = heights[common] - reference[common]
    reference_difference_covariance = C + reference_covariance[np.ix_(common, common)]
    z = NormalDist().inv_cdf((1 + INTERVAL_LEVEL) / 2)
    n = len(common)
    return {"common_samples": n, "squared_error_sum_m2": float(error @ error),
            "truth_rmse_m": float(np.sqrt(np.mean(error**2))) if n else None,
            "truth_interval_covered": int(np.count_nonzero(np.abs(error) <= z * np.sqrt(np.diag(C)))),
            "reference_discrepancy_squared_sum_m2": float(difference @ difference),
            "reference_discrepancy_rmse_m": float(np.sqrt(np.mean(difference**2))) if n else None,
            "reference_interval_covered": int(np.count_nonzero(np.abs(difference) <= z *
                                                               np.sqrt(np.diag(reference_difference_covariance)))),
            "reference_joint_squared_discrepancy": (float(difference @ np.linalg.solve(reference_difference_covariance,
                                                                                       difference)) if n else None),
            "reference_joint_dimension": n,
            "mean_reference_difference_variance_m2": (float(np.mean(np.diag(reference_difference_covariance))) if n else None),
            "mean_estimate_variance_m2": float(np.mean(np.diag(C))) if n else None}


def score_trial(trial: SyntheticTrial, analysis: CameraAnalysis) -> dict:
    """All truth/reference access and intervention-based scoring is confined here."""
    comparison = analysis.comparison
    common = sorted(set.intersection(*(set(comparison.series[name].observed_indices) for name in METHODS)))
    methods = {}
    for name in METHODS:
        estimate = comparison.series[name]
        methods[name] = {"available_samples": len(estimate.observed_indices),
                         **_score(estimate.heights, estimate.covariance, estimate.observed_indices, common,
                                  trial.truth, trial.reference_values, trial.reference_covariance)}
    motion_common = sorted(set(common) & set(analysis.unregistered_camera.observed_indices))
    motion_scores = {}
    for name, camera in (("registered", analysis.camera), ("unregistered", analysis.unregistered_camera)):
        motion_scores[name] = _score(camera.heights, camera.covariance, camera.observed_indices, motion_common,
                                    trial.truth, trial.reference_values, trial.reference_covariance)
    raw_difference = comparison.difference
    intervals = analysis.diagnostic["fits"].get("gauge_drift", {}).get("interval")
    drift_coverage = None
    if trial.scenario == "gauge_drift" and intervals is not None:
        drift_coverage = bool(intervals[0] <= GAUGE_DRIFT_M_PER_SECOND <= intervals[1])
    return {"seed": trial.seed, "scenario": trial.scenario, "common_indices": common,
            "methods": methods, "registration_comparison": motion_scores,
            "gauge_drift_interval_covers_injected_rate": drift_coverage,
            "diagnostic": analysis.diagnostic,
            "frames": analysis.audit.as_dict(), "water_statuses": list(analysis.water_statuses),
            "marker_statuses": list(analysis.marker_statuses), "camera_statuses": list(analysis.camera.statuses),
            "spectral_windows": list(analysis.spectral_windows),
            "trace": {"times_seconds": trial.times.tolist(), "gauge_times_seconds": trial.gauge_times.tolist(),
                      "scoring_truth_m": trial.truth.tolist(),
                      "scoring_reference_m": trial.reference_values.tolist(),
                      "water_rows_pixels": _nullable(analysis.water_rows),
                      "marker_displacements_pixels": _nullable(analysis.marker_displacements),
                      "corrected_rows_pixels": _nullable(analysis.camera.corrected_rows),
                      "camera_weight": _nullable(comparison.camera_weight),
                      "estimates_m": {name: _nullable(comparison.series[name].heights) for name in METHODS},
                      "raw_gauge_minus_camera_m": _nullable(raw_difference.heights),
                      "difference_observed_indices": list(raw_difference.observed_indices),
                      "difference_variances_m2": np.diag(raw_difference.covariance).tolist()}}


def _fraction(count: int, denominator: int) -> dict:
    return {"count": count, "denominator": denominator,
            "fraction": count / denominator if denominator else None}


def _aggregate(records: list[dict]) -> dict:
    n_frames = sum(len(record["frames"]["frame_ids"]) for record in records)
    counts = Counter(record["diagnostic"]["status"] for record in records)
    methods = {}
    for name in METHODS:
        scores = [record["methods"][name] for record in records]
        n = sum(score["common_samples"] for score in scores)
        methods[name] = {
            "availability": _fraction(sum(score["available_samples"] for score in scores), n_frames),
            "common_samples": n,
            "truth_rmse_m": float(np.sqrt(sum(score["squared_error_sum_m2"] for score in scores) / n)) if n else None,
            "reference_discrepancy_rmse_m": float(np.sqrt(sum(score["reference_discrepancy_squared_sum_m2"] for score in scores) / n)) if n else None,
            "truth_interval_coverage": _fraction(sum(score["truth_interval_covered"] for score in scores), n),
            "reference_interval_coverage": _fraction(sum(score["reference_interval_covered"] for score in scores), n)}
    motion = {}
    for name in ("registered", "unregistered"):
        scores = [record["registration_comparison"][name] for record in records]
        n = sum(score["common_samples"] for score in scores)
        motion[name] = {"common_samples": n,
                        "truth_rmse_m": float(np.sqrt(sum(score["squared_error_sum_m2"] for score in scores) / n)) if n else None}
    drift_intervals = [record["gauge_drift_interval_covers_injected_rate"] for record in records
                       if record["gauge_drift_interval_covers_injected_rate"] is not None]
    selected_drift_intervals = [record["gauge_drift_interval_covers_injected_rate"] for record in records
                                if record["gauge_drift_interval_covers_injected_rate"] is not None
                                and "gauge_drift" in record["diagnostic"]["candidates"]
                                and record["diagnostic"]["status"] != "consistent"]
    null_tests = [record["diagnostic"] for record in records if record["diagnostic"]["null_threshold"] is not None]
    return {"records": len(records), "frame_opportunities": n_frames,
            "diagnostic_status_counts": {status: counts[status] for status in STATUSES},
            "record_null_rejections": _fraction(sum(test["null_statistic"] > test["null_threshold"]
                                                    for test in null_tests), len(null_tests)),
            "methods": methods, "registration_comparison": motion,
            "gauge_drift_interval_coverage_all_fitted": _fraction(sum(drift_intervals), len(drift_intervals)),
            "gauge_drift_interval_coverage_given_adequate_rejected_null": _fraction(sum(selected_drift_intervals), len(selected_drift_intervals)),
            "frame_flag_counts": dict(sorted(Counter(flag for record in records for flags in record["frames"]["flags"] for flag in flags).items())),
            "water_status_counts": dict(sorted(Counter(status for record in records for status in record["water_statuses"]).items())),
            "spectral_window_counts": dict(sorted(Counter(window["status"] for record in records for window in record["spectral_windows"]).items()))}


def run_experiment() -> dict:
    cases = {}
    calibrations = {}
    for scenario in SCENARIOS:
        records = []
        for seed in EVALUATION_SEEDS:
            trial = build_case(seed, scenario)
            analysis = analyze(trial)
            records.append(score_trial(trial, analysis))
            if scenario not in calibrations:
                calibrations[scenario] = {
                    "slope_m_per_pixel": analysis.calibration.slope_m_per_pixel,
                    "intercept_m": analysis.calibration.intercept_m,
                    "parameter_covariance": analysis.calibration.parameter_covariance.tolist(),
                    "valid_pixel_range": list(analysis.calibration.valid_pixel_range),
                    "calibration_ids": list(analysis.calibration.calibration_ids),
                    "anchor_reference_heights_m": trial.calibration_evidence.reference_heights.tolist(),
                    "anchor_reference_covariance_m2": trial.calibration_evidence.reference_covariance.tolist()}
        cases[scenario] = {"aggregate": _aggregate(records), "per_seed": records}
    return {"schema_version": "fsre-camera-baseline-v1", "provenance": provenance(),
            "design": {
                "scope": "Rendered synthetic tank-level observations only; no field validation, general image segmentation, 3-D odometry or operational fault-diagnosis claim.",
                "n_frames": N_FRAMES, "calibration_development_seeds": list(CALIBRATION_SEEDS),
                "evaluation_seeds": list(EVALUATION_SEEDS), "scenarios": list(SCENARIOS),
                "calibration": "Eight separate noiseless anchor images at exact fixed pixel designs; noisy reference heights fit one shared linear calibration. Thresholds and covariance are declared before evaluation, not tuned on evaluation outcomes.",
                "pairing": "Evaluation seeds supply independent per-frame errors, paired across scenarios. All records reuse one calibration realization from disjoint development seeds. Pooled coverage is empirical conditional on that realization, not independent calibration trials or a certified nominal coverage rate.",
                "physics": "Prescribed bounded level trajectory .40+.03*sin(2*pi*t/16)+.001*t metres; a unit-area tank can realize it with interval net volumes equal to successive level differences. No flow sensor or dynamics prior is used in this observation comparison.",
                "image": {"shape": list(FRAME_SHAPE), "water_columns": list(WATER_COLUMNS),
                          "marker_columns": list(MARKER_COLUMNS), "marker_reference_row": MARKER_REFERENCE_ROW,
                          "intensity_grid": INTENSITY_GRID,
                          "model": "Horizontal area-antialiased dark-below edges; water intensities .8/.2 and marker .9/.1. Per-frame Gaussian geometric edge jitter is rendered into pixels; no extra intensity noise. Final intensities are rounded to a fixed 1e-9 grid for reproducible payload hashes; <=5e-10 quantization error is negligible at the declared 0.03-pixel noise scale. Exact integer-row anchor designs remain exact.",
                          "true_slope_m_per_pixel": TRUE_SLOPE_M_PER_PIXEL, "true_intercept_m": TRUE_INTERCEPT_M,
                          "detector_options": DETECTOR_OPTIONS},
                "noise_sd": {"water_localization_pixels": PIXEL_SD, "marker_localization_pixels": PIXEL_SD,
                             "gauge_m": GAUGE_SD_M, "scoring_reference_m": REFERENCE_SD_M,
                             "anchor_reference_m": ANCHOR_REFERENCE_SD_M, "shared_calibration_reference_m": SHARED_REFERENCE_SD_M},
                "covariance": "Independent time noise plus one shared reference offset affecting both the gauge and camera calibration. Cg=sigma_g^2*I+sigma_b^2*11'; Cov(g_i,c_j)=sigma_b^2. Camera C=J*Ctheta*J'+a_hat^2*(Cwater+Cmarker), J_i=[corrected_pixel_i,1]. Full cross-time matrices are used throughout; no covariance is estimated from held-out errors.",
                "covariance_limits": "Camera propagation is first-order: the slope-error/pixel-error product term is omitted. Intervals use the declared marginal shared-calibration covariance; conditional coverage across this fixed calibration, image gates and fault cases is empirical, not exact Gaussian calibration.",
                "motion": "One-dimensional vertical translation from a fixed image fiducial is subtracted. Marker localization uncertainty is propagated; its reference row is exact in this simulation. This is not rotation, perspective correction or 3-D visual odometry.",
                "timing": {"expected_period_seconds": EXPECTED_PERIOD, "max_jitter_seconds": MAX_JITTER_SECONDS,
                           "spectral_window_frames": SPECTRAL_WINDOW,
                           "policy": "Original timestamps and missing placeholders are retained without interpolation. Exact repeated payloads are refused under an explicit conservative policy; repeated content alone does not prove a fault. Jittered frames remain synchronized with gauge/reference but their nonuniform windows have no spectrum."},
                "interventions": {"onset_seconds": ONSET_SECONDS, "gauge_drift_m_per_second": GAUGE_DRIFT_M_PER_SECOND,
                                  "camera_translation_final_pixels": CAMERA_TRANSLATION_PIXELS,
                                  "shared_calibration_offset_m": SHARED_OFFSET_M,
                                  "obstructed_indices": list(range(12, 18)), "duplicate_payload_index": 16,
                                  "dropped_indices": [10, 20], "jitter_seconds": ".05*sin(.7*t)",
                                  "physical_level_change_m": ".04*(1-exp(-max(t-12,0)/3))",
                                  "interpretation": "Arbitrary synthetic magnitudes, not industry detection targets. Shared offset changes both calibration references and gauge while the held-out reference remains independent."},
                "estimation": "Camera-only and gauge-only readings and per-time correlated BLUE combined estimates; full covariance propagated, no temporal smoothing, no automatic fault exclusion. All accuracy comparisons use the same physical times and common valid support; availability also counts source fallbacks.",
                "diagnosis": {"alpha": ALPHA, "interval_level": INTERVAL_LEVEL,
                              "residual": "Raw gauge minus registered camera, with full covariance, before combination.",
                              "hypotheses": "Known onset/profile: gauge drift +max(t-12,0) and camera-level drift -max(t-12,0), each with unrestricted signed amplitude in m/s. They are collinear: evidence can reject consistency without choosing the faulty source. Physical camera translation is separately corrected by the marker.",
                              "limits": "Record-level approximate Gaussian lack of fit; candidates are adequate explanations, not source probabilities. No online delay, searched-onset, multiple-test or causal isolation guarantee. A shared level offset cancels in the difference."},
                "scoring": "Truth and independent reference are read only by simulator/scorer. RMSE versus truth differs from reference discrepancy RMSE. Reference discrepancy covariance is Cestimate+Creference under the declared independence. Pointwise coverage counts use common support, with all denominators retained; the joint reference squared discrepancy uses its full covariance and is reported without a calibrated decision threshold.",
                "spectra": "Advisory linear-detrended Hann periodograms of registered camera level in fixed disjoint eight-frame windows. Only admitted, contiguous, uniform windows are computed; no spectral threshold or clock recovery is claimed.",
            }, "calibrations": calibrations, "cases": cases}


def render(report: dict) -> str:
    def number(value):
        return "—" if value is None else f"{value:.6f}"

    lines = ["# Synthetic camera/gauge level baseline", "", header_line(report["provenance"]), "",
             "**Rendered synthetic frames; no field validation or automatic source isolation.**", "",
             "A separately calibrated camera detects a water edge and fixed fiducial. The marker registers vertical translation. "
             "Gauge-only, camera-only and correlated BLUE estimates are scored on identical valid times. "
             "An independent reference enters scoring only.", "",
             "| scenario | method | available / possible | common samples | truth RMSE [m] | reference discrepancy RMSE [m] | truth interval coverage |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for name, case in report["cases"].items():
        for method, result in case["aggregate"]["methods"].items():
            availability, coverage = result["availability"], result["truth_interval_coverage"]
            lines.append(f"| {name} | {method} | {availability['count']}/{availability['denominator']} | {result['common_samples']} | "
                         f"{number(result['truth_rmse_m'])} | {number(result['reference_discrepancy_rmse_m'])} | "
                         f"{coverage['count']}/{coverage['denominator']} |")
    lines += ["", "Coverage uses pointwise 95% intervals under the declared first-order covariance and is empirical "
              "conditional on one reused calibration. Time samples and paired scenarios are not independent calibration trials.", "",
              "| scenario | consistent | ambiguous | identified | unexplained | insufficient | null rejections / records | computed / attempted spectral windows |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, case in report["cases"].items():
        result = case["aggregate"]
        counts, rejection, windows = result["diagnostic_status_counts"], result["record_null_rejections"], result["spectral_window_counts"]
        lines.append(f"| {name} | {counts['consistent']} | {counts['ambiguous']} | {counts['identified']} | {counts['unexplained']} | "
                     f"{counts['insufficient_evidence']} | {rejection['count']}/{rejection['denominator']} | "
                     f"{windows.get('computed', 0)}/{sum(windows.values())} |")
    motion = report["cases"].get("camera_vertical_drift")
    if motion is not None:
        scores = motion["aggregate"]["registration_comparison"]
        lines += ["", f"In the vertical-motion case, camera RMSE on the same {scores['registered']['common_samples']} samples is "
                  f"{number(scores['unregistered']['truth_rmse_m'])} m without registration and {number(scores['registered']['truth_rmse_m'])} m with it."]
    lines += ["", "## What these results establish", "",
              "Gauge and camera drift profiles have opposite signs but the same span with unknown signed amplitude. "
              "They remain ambiguous when supported. A shared calibration offset can leave their difference consistent "
              "while both disagree with the independent reference; combination cannot remove that shared bias.", "",
              "The JSON retains every seed, frame hashes/times/admission flags, extracted rows, estimates, raw differences, "
              "calibration covariance, quality and diagnostic outcomes, spectral windows and scoring denominators. "
              "Quality refusal is not a fault label; no faulty-source truth is supplied to inference.", ""]
    for key in ("calibration", "pairing", "covariance", "covariance_limits", "motion", "estimation", "scoring", "spectra"):
        lines.append(f"- **{key.replace('_', ' ').capitalize()}:** {report['design'][key]}")
    lines.append("")
    return "\n".join(lines)


def main(out_dir: Path, *, quiet: bool = False) -> int:
    report = run_experiment()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "camera_baseline.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    text = render(report)
    (out_dir / "camera_baseline.md").write_text(text, encoding="utf-8")
    if not quiet:
        print(text)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path.cwd() / "results")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(args.out_dir, quiet=args.quiet))
