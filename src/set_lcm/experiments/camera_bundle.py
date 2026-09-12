"""Replayable decoded-frame fixture for the explicitly synthetic camera protocol.

Observation loading never opens evaluation.npz. Checksums detect changed bytes;
they do not authenticate a sensor, establish a clock, or make this a real-data adapter.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import struct
import zlib

import numpy as np

from set_lcm.frame_quality import frame_content_hash
from .camera_baseline import CalibrationEvidence, FRAME_SHAPE, build_case
from .provenance import provenance

PROTOCOL = "synthetic-camera-baseline-v1"
UNITS = {"times": "s", "gauge_times": "s", "gauge_values": "m",
         "reference_values": "m", "truth": "m", "covariances": "m^2",
         "pixel_rows": "pixel", "frames": "dimensionless grayscale intensity"}
OBSERVATION_KEYS = frozenset(("frames", "frame_present", "times", "frame_ids", "gauge_times",
    "gauge_values", "gauge_covariance", "calibration_frames", "calibration_reference_heights",
    "calibration_reference_covariance"))
EVALUATION_KEYS = frozenset(("reference_values", "reference_covariance", "truth"))


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _preview_png(frame: np.ndarray) -> bytes:
    """Display-only 8-bit preview; original float pixels are retained in the archive."""
    pixels = np.rint(np.clip(frame, 0, 1) * 255).astype(np.uint8)
    def chunk(kind, payload):
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff))
    h, w = pixels.shape
    scanlines = b"".join(b"\x00" + row.tobytes() for row in pixels)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b""))


def write_synthetic_bundle(out_dir: Path, *, seed: int = 10000,
                           scenario: str = "camera_vertical_drift") -> Path:
    """Save original generated frames, clocks and calibration separately from scoring."""
    trial = build_case(seed, scenario)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence = trial.calibration_evidence
    present = np.array([frame is not None for frame in trial.frames], dtype=bool)
    packed = np.zeros((len(present), *FRAME_SHAPE), dtype=float)
    packed[present] = np.stack([frame for frame in trial.frames if frame is not None])
    np.savez_compressed(out_dir / "observations.npz", frames=packed, frame_present=present,
        times=trial.times, frame_ids=trial.frame_ids, gauge_times=trial.gauge_times,
        gauge_values=trial.gauge_values, gauge_covariance=trial.gauge_covariance,
        calibration_frames=np.stack(evidence.frames),
        calibration_reference_heights=evidence.reference_heights,
        calibration_reference_covariance=evidence.reference_covariance)
    np.savez_compressed(out_dir / "evaluation.npz", reference_values=trial.reference_values,
                        reference_covariance=trial.reference_covariance, truth=trial.truth)
    first = next(frame for frame in trial.frames if frame is not None)
    (out_dir / "frame-preview.png").write_bytes(_preview_png(first))
    manifest = {
        "schema_version": 1, "protocol": PROTOCOL, "source_kind": "synthetic_test",
        "seed": trial.seed, "scenario": scenario, "provenance": provenance(),
        "time_origin": "synthetic trial start, seconds",
        "units": UNITS,
        "calibration_ids": list(evidence.calibration_ids),
        "frame_hashes": [None if frame is None else frame_content_hash(frame) for frame in trial.frames],
        "calibration_frame_hashes": [frame_content_hash(frame) for frame in evidence.frames],
        "sha256": {name: _sha((out_dir / name).read_bytes()) for name in
                   ("observations.npz", "evaluation.npz", "frame-preview.png")},
        "notes": ["All records and images are labeled synthetic; no captured tank video is supplied.",
                  "Missing payloads have zero storage placeholders plus frame_present=false; loaders return None.",
                  "This fixed protocol uses the declared constants in camera_baseline.py, not a general acquisition adapter.",
                  "Calibration reference heights are training evidence; evaluation reference and truth are scoring only.",
                  "The PNG is a display-only intensity quantization; inference uses the original NPZ float arrays.",
                  "Checksums establish byte consistency, not origin authenticity or measurement validity."]}
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _manifest(out_dir: Path) -> dict:
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("protocol") != PROTOCOL
            or manifest.get("source_kind") != "synthetic_test"
            or manifest.get("time_origin") != "synthetic trial start, seconds"
            or manifest.get("units") != UNITS):
        raise ValueError("bundle must declare the supported synthetic protocol, clock and units")
    return manifest


def _archive(out_dir: Path, manifest: dict, filename: str, keys: frozenset) -> dict:
    payload = (out_dir / filename).read_bytes()
    if _sha(payload) != manifest.get("sha256", {}).get(filename):
        raise ValueError(f"checksum mismatch: {filename}")
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        if len(archive.files) != len(keys) or set(archive.files) != keys:
            raise ValueError(f"unexpected arrays in {filename}")
        arrays = {key: archive[key] for key in keys}
    if any(value.dtype.kind not in "biuf" for value in arrays.values()):
        raise ValueError("bundle arrays must be real numeric values")
    return arrays


def load_observations(out_dir: Path) -> dict:
    """Load only manifest and measurements; return infer_measurements keyword arguments.

    Protocol covariance, geometry, detector and clock policies are fixed in the experiment.
    This loader deliberately refuses other source kinds rather than relabeling them.
    """
    out_dir = Path(out_dir)
    manifest = _manifest(out_dir)
    data = _archive(out_dir, manifest, "observations.npz", OBSERVATION_KEYS)
    frames, present, times, ids = (data[key] for key in ("frames", "frame_present", "times", "frame_ids"))
    n = len(times) if times.ndim == 1 else 0
    if (n == 0 or frames.shape != (n, *FRAME_SHAPE) or frames.dtype != np.dtype(float)
            or present.shape != (n,) or present.dtype.kind != "b"
            or ids.shape != (n,) or ids.dtype.kind not in "iu" or np.any(ids < 0)
            or not np.all(np.isfinite(frames)) or np.any(frames[~present] != 0)):
        raise ValueError("invalid frame payload, presence mask or clock shape")
    for key in ("times", "gauge_times", "gauge_values"):
        value = data[key]
        if value.shape != (n,) or np.any(np.isinf(value)) or (key != "gauge_values" and np.any(np.isnan(value))):
            raise ValueError(f"invalid {key}")
    if data["gauge_covariance"].shape != (n, n):
        raise ValueError("invalid gauge covariance shape")
    unpacked = tuple(frame if exists else None for frame, exists in zip(frames, present, strict=True))
    hashes = [None if frame is None else frame_content_hash(frame) for frame in unpacked]
    if hashes != manifest.get("frame_hashes"):
        raise ValueError("frame content hashes disagree with the manifest")
    anchors = data["calibration_frames"]
    calibration_ids = manifest.get("calibration_ids")
    if (anchors.ndim != 3 or anchors.shape[1:] != FRAME_SHAPE or len(anchors) < 2
            or not np.all(np.isfinite(anchors)) or not isinstance(calibration_ids, list)
            or len(calibration_ids) != len(anchors)
            or any(not isinstance(identifier, str) or not identifier.strip() for identifier in calibration_ids)
            or len(set(calibration_ids)) != len(calibration_ids)
            or data["calibration_reference_heights"].shape != (len(anchors),)
            or data["calibration_reference_covariance"].shape != (len(anchors), len(anchors))):
        raise ValueError("invalid calibration evidence")
    if [frame_content_hash(frame) for frame in anchors] != manifest.get("calibration_frame_hashes"):
        raise ValueError("calibration frame hashes disagree with the manifest")
    evidence = CalibrationEvidence(tuple(anchors), data["calibration_reference_heights"],
                                   data["calibration_reference_covariance"], tuple(calibration_ids))
    return dict(frames=unpacked, times=times, frame_ids=tuple(ids), gauge_times=data["gauge_times"],
                gauge_values=data["gauge_values"], gauge_covariance=data["gauge_covariance"], evidence=evidence)


def load_evaluation(out_dir: Path) -> dict:
    """Explicit scoring-only read. Nothing in observation loading calls this function."""
    out_dir = Path(out_dir)
    data = _archive(out_dir, _manifest(out_dir), "evaluation.npz", EVALUATION_KEYS)
    truth = data["truth"]
    if (truth.ndim != 1 or len(truth) == 0 or data["reference_values"].shape != truth.shape
            or data["reference_covariance"].shape != (len(truth), len(truth))
            or any(not np.all(np.isfinite(value)) for value in data.values())):
        raise ValueError("invalid evaluation arrays")
    return data
