"""Declared frame admission and advisory spectra, without transport or clock repair.

Capture times are seconds relative to an explicitly named origin. These checks
cannot recognize a consistently wrong time unit or frame rate without an
independent reference. Identical bytes can describe a static scene; repeated
content is a possible freeze, never proof of a camera fault.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Integral, Real

import numpy as np

__all__ = ["FrameAudit", "Spectrum", "audit_frames", "frame_content_hash", "frame_spectrum"]

_SOURCES = {"measured", "synthetic_test", "generated"}
_UNIFORM_TOLERANCE = 1e-9


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _number(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or (value <= 0 if positive else value < 0)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def _vector(value, name, n=None):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real vector") from exc
    if result.ndim != 1 or (n is not None and result.size != n) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 1-D vector of the declared length")
    return result


def _integers(value, name, n=None):
    try:
        result = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain nonnegative integer indices") from exc
    if ((n is not None and len(result) != n) or
            any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) or i < 0 for i in result)):
        raise ValueError(f"{name} must contain nonnegative integer indices of the declared length")
    return tuple(int(i) for i in result)


def _roundoff(first, second, period):
    return 8 * np.finfo(float).eps * max(abs(first), abs(second), period)


def _uniform(times, period, tolerance):
    return all(abs((second - first) - period) <= tolerance + _roundoff(first, second, period)
               for first, second in zip(times, times[1:]))


def frame_content_hash(frame: np.ndarray) -> str:
    """SHA-256 over exact logical C-order pixels, dtype/byte order, and shape.

    No image compression, conversion, quantization or perceptual similarity is
    applied. This records content identity, not acquisition authenticity. Array
    memory layout is immaterial; dtype and shape are part of the hash contract.
    """
    if (not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3)
            or frame.size == 0 or frame.dtype.kind not in "buif"):
        raise ValueError("frame must be a nonempty 2-D or 3-D real numeric array")
    metadata = json.dumps({"dtype": frame.dtype.str, "shape": list(frame.shape)},
                          sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b"fsre-frame-v1\0" + metadata + b"\0" + frame.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class FrameAudit:
    frame_ids: tuple[int, ...]
    capture_times: tuple[float, ...]
    content_hashes: tuple[str | None, ...]
    source_kinds: tuple[str, ...]
    source_indices: tuple[int, ...]
    missing: tuple[bool, ...]
    admitted: tuple[bool, ...]
    flags: tuple[tuple[str, ...], ...]
    time_origin: str
    expected_period: float
    max_jitter: float
    reject_repeated_content: bool
    spectral_eligible: bool
    assumptions: tuple[str, ...]

    def as_dict(self) -> dict:
        """JSON-safe snapshot; original row order and refusal flags are retained."""
        return json.loads(json.dumps(asdict(self), allow_nan=False))


def audit_frames(frame_ids, capture_times, content_hashes, *, time_origin: str,
                 source_kind, expected_period: float, max_jitter: float = 0.0,
                 missing=None, source_indices=None, reject_repeated_content: bool = False) -> FrameAudit:
    """Audit supplied metadata in original order, without sorting or inserting rows.

    IDs are nonnegative integer capture counters; source_indices identify the
    original rows and must increase. Times are finite seconds, including any
    caller-declared missing placeholders; no missing time is inferred. Hashes
    are nonempty strings, or None only for externally marked missing rows.
    source_kind is one allowed kind for all rows or a same-length sequence.

    Missing/generated rows, duplicate/backward IDs or times, and intervals outside
    max_jitter are refused. Expected elapsed time is the positive capture-counter
    difference times expected_period. Gaps in IDs/source indices are flagged;
    a following observation at the corresponding actual time is still admitted
    individually. Source indices never alter expected physical time. Long/short
    intervals flag a clock violation, not its physical cause. Repeated hashes
    are advisory unless reject_repeated_content is explicitly true. Floating
    clock comparisons include eight machine epsilons at the timestamp scale.

    spectral_eligible uses the default 1e-9-second numerical uniformity tolerance
    and requires >=4 admitted, contiguous rows. frame_spectrum rechecks these
    requirements using its own explicitly supplied numerical tolerance.
    """
    ids = _integers(frame_ids, "frame_ids")
    n = len(ids)
    times = tuple(float(t) for t in _vector(capture_times, "capture_times", n))
    origin = _text(time_origin, "time_origin")
    period = _number(expected_period, "expected_period", positive=True)
    jitter = _number(max_jitter, "max_jitter")
    if jitter >= period / 2:
        raise ValueError("max_jitter must be less than half the expected period")
    if not isinstance(reject_repeated_content, (bool, np.bool_)):
        raise ValueError("reject_repeated_content must be boolean")
    indices = tuple(range(n)) if source_indices is None else _integers(source_indices, "source_indices", n)
    if any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError("source_indices must increase, preserving original row order")
    if missing is None:
        absent = (False,) * n
    else:
        array = np.asarray(missing)
        if array.shape != (n,) or array.dtype != np.dtype(bool):
            raise ValueError("missing must be a boolean vector of the declared length")
        absent = tuple(bool(v) for v in array)
    try:
        hashes = tuple(content_hashes)
        kinds = (source_kind,) * n if isinstance(source_kind, str) else tuple(source_kind)
    except TypeError as exc:
        raise ValueError("content_hashes and source_kind must match the frame rows") from exc
    if (isinstance(content_hashes, (str, bytes)) or len(hashes) != n or
            any(h is None and not absent[i] or h is not None and
                (not isinstance(h, str) or not h.strip()) for i, h in enumerate(hashes))):
        raise ValueError("content_hashes must contain nonempty strings, or None for declared missing rows")
    if ((isinstance(source_kind, str) and source_kind not in _SOURCES) or len(kinds) != n
            or any(not isinstance(k, str) or k not in _SOURCES for k in kinds)):
        raise ValueError("source_kind must be measured, synthetic_test, or generated for every row")
    flags, admitted = [], []
    seen_ids, seen_times, seen_hashes = set(), set(), set()
    highest_id, highest_time = -1, -np.inf
    contiguous = True
    for i in range(n):
        row, accept = [], True
        if absent[i]:
            row.append("missing")
            accept = False
        if kinds[i] == "generated":
            row.append("generated_source")
            accept = False
        if ids[i] in seen_ids:
            row.append("duplicate_frame_id")
            accept = False
        if times[i] in seen_times:
            row.append("duplicate_timestamp")
            accept = False
        if i:
            if ids[i] < highest_id:
                row.append("backward_frame_id")
                accept = False
            if times[i] < highest_time:
                row.append("backward_timestamp")
                accept = False
            if ids[i] > ids[i-1] + 1:
                row.append("frame_id_gap")
            if indices[i] > indices[i-1] + 1:
                row.append("source_index_gap")
            contiguous &= ids[i] == ids[i-1] + 1 and indices[i] == indices[i-1] + 1
            difference = times[i] - times[i-1]
            try:
                expected_elapsed = max(1, ids[i] - ids[i-1]) * period
            except OverflowError:
                expected_elapsed = np.inf
            allowance = jitter + _roundoff(times[i-1], times[i], expected_elapsed)
            if not np.isfinite(expected_elapsed) or abs(difference - expected_elapsed) > allowance:
                row.append("clock_outside_tolerance")
                if difference > expected_elapsed:
                    row.append("long_interval")
                elif difference > 0:
                    row.append("short_interval")
                accept = False
        if not absent[i] and hashes[i] in seen_hashes:
            row.append("repeated_content")
            if reject_repeated_content:
                accept = False
        seen_ids.add(ids[i])
        seen_times.add(times[i])
        highest_id = max(highest_id, ids[i])
        highest_time = max(highest_time, times[i])
        if not absent[i]:
            seen_hashes.add(hashes[i])
        flags.append(tuple(row))
        admitted.append(accept)
    eligible = n >= 4 and all(admitted) and contiguous and _uniform(times, period, _UNIFORM_TOLERANCE)
    assumptions = (
        "Capture times are caller-declared seconds relative to time_origin; no clock units or frame rate are inferred.",
        "Capture-counter gaps multiply the expected elapsed interval; no rows are sorted, interpolated, or silently repaired.",
        "Repeated exact content can be a static scene; it is only evidence of possible freezing.",
        "Admission is a metadata policy, not proof of sensor health; synthetic_test remains labeled synthetic.",
    )
    return FrameAudit(ids, times, hashes, kinds, indices, absent, tuple(admitted), tuple(flags),
                      origin, period, jitter, bool(reject_repeated_content), bool(eligible), assumptions)


@dataclass(frozen=True)
class Spectrum:
    frequencies_hz: tuple[float, ...]
    power_density: tuple[float, ...]
    values: tuple[float, ...]
    source_indices: tuple[int, ...]
    frame_ids: tuple[int, ...]
    capture_times: tuple[float, ...]
    content_hashes: tuple[str | None, ...]
    source_kinds: tuple[str, ...]
    time_origin: str
    feature_unit: str
    sample_period: float
    detrend: str
    window: str
    uniform_tolerance: float
    peak_frequency_hz: float | None
    detrended_rms: float
    normalization: str
    assumptions: tuple[str, ...]

    def as_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self), allow_nan=False))


def frame_spectrum(values, audit: FrameAudit, *, feature_unit: str, detrend: str = "linear",
                   window: str = "hann", uniform_tolerance: float = _UNIFORM_TOLERANCE) -> Spectrum:
    """One-sided periodogram of a complete contiguous scalar-feature window.

    All rows must pass the declared admission policy, frame/source indices must
    be contiguous, and actual intervals must match expected_period within the
    numerical uniform_tolerance (seconds) plus timestamp roundoff. No missing
    row, arbitrary reindexing, or generated frame is accepted. Audit jitter
    tolerance does not relax this spectral requirement. There is no resampling.

    Detrending precedes windowing. The symmetric Hann window is np.hanning(n);
    boxcar is all ones. PSD = |rfft(window*detrended)|^2 / (fs*sum(window^2)),
    doubling only non-DC/non-Nyquist bins. Units are feature_unit^2/Hz. Integrated
    PSD equals window-weighted mean square, not generally unwindowed variance.
    peak_frequency_hz is the largest non-DC bin, or None when all such power is
    zero. It is not an anomaly score, significance test, or estimated fault rate.
    """
    if not isinstance(audit, FrameAudit):
        raise TypeError("audit must be a FrameAudit")
    # Rebuild from retained declarations, so a manually constructed dataclass
    # cannot bypass admission by supplying invented admitted/eligible fields.
    checked = audit_frames(audit.frame_ids, audit.capture_times, audit.content_hashes,
                           time_origin=audit.time_origin, source_kind=audit.source_kinds,
                           expected_period=audit.expected_period, max_jitter=audit.max_jitter,
                           missing=audit.missing, source_indices=audit.source_indices,
                           reject_repeated_content=audit.reject_repeated_content)
    unit = _text(feature_unit, "feature_unit")
    signal = _vector(values, "values", len(checked.frame_ids))
    n = signal.size
    tolerance = _number(uniform_tolerance, "uniform_tolerance")
    if tolerance >= checked.expected_period / 2:
        raise ValueError("uniform_tolerance must be less than half the expected period")
    if n < 4 or not all(checked.admitted):
        raise ValueError("spectrum requires at least four admitted frames with no missing or generated rows")
    if (any(b != a + 1 for a, b in zip(checked.frame_ids, checked.frame_ids[1:])) or
            any(b != a + 1 for a, b in zip(checked.source_indices, checked.source_indices[1:]))):
        raise ValueError("spectrum requires contiguous frame IDs and original source indices")
    if not _uniform(checked.capture_times, checked.expected_period, tolerance):
        raise ValueError("spectrum requires a uniform capture clock; irregular timestamps cannot be repaired here")
    if detrend not in {"none", "constant", "linear"} or window not in {"hann", "boxcar"}:
        raise ValueError("unsupported detrend or window declaration")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        centered = signal.copy() if detrend == "none" else signal - np.mean(signal)
        if detrend == "linear":
            axis = np.arange(n, dtype=float) - (n - 1) / 2
            centered = centered - axis * (axis @ centered) / (axis @ axis)
        weights = np.hanning(n) if window == "hann" else np.ones(n)
        transformed = np.fft.rfft(centered * weights)
        frequency = np.fft.rfftfreq(n, d=checked.expected_period)
        power = np.abs(transformed)**2 * checked.expected_period / (weights @ weights)
        power[1:-1 if n % 2 == 0 else None] *= 2
        rms = float(np.sqrt(np.mean(centered**2)))
    if not np.all(np.isfinite(power)) or not np.all(np.isfinite(frequency)) or not np.isfinite(rms):
        raise ValueError("spectrum exceeds the supported finite numeric range")
    peak = float(frequency[1 + np.argmax(power[1:])]) if np.any(power[1:] > 0) else None
    assumptions = (
        "Feature values and their units are supplied by the caller; no physical image quantity is inferred.",
        "Frequency uses the declared expected period after checking actual capture intervals.",
        "Clock/feature aliasing and a consistently wrong time scale require independent reference evidence.",
        "Spectrum and peaks are advisory features, not calibrated anomaly probabilities or sensor-fault verdicts.",
    )
    return Spectrum(tuple(float(f) for f in frequency), tuple(float(p) for p in power),
                    tuple(float(v) for v in signal), checked.source_indices, checked.frame_ids,
                    checked.capture_times, checked.content_hashes, checked.source_kinds, checked.time_origin,
                    unit, checked.expected_period, detrend, window, tolerance, peak, rms,
                    "One-sided PSD: |FFT(w*x_detrended)|^2 / (fs*sum(w^2)); double interior bins only.",
                    assumptions)
