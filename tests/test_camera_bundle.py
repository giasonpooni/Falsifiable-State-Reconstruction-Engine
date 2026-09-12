"""Recording roundtrip, evidence separation and payload-integrity regressions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from set_lcm.experiments.camera_baseline import analyze, build_case, infer_measurements
from set_lcm.experiments.camera_bundle import load_evaluation, load_observations, write_synthetic_bundle


def rewrite_manifest(folder, change):
    path = folder / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize("scenario", ["healthy", "camera_vertical_drift", "dropped_frame", "time_jitter"])
def test_replaying_original_arrays_reproduces_inference(tmp_path, scenario):
    write_synthetic_bundle(tmp_path, scenario=scenario)
    original = build_case(10000, scenario)
    expected = analyze(original)
    observations = load_observations(tmp_path)
    assert set(observations) == {"frames", "times", "frame_ids", "gauge_times", "gauge_values", "gauge_covariance", "evidence"}
    actual = infer_measurements(**observations)
    assert actual.audit == expected.audit
    assert actual.diagnostic == expected.diagnostic
    for name in expected.comparison.series:
        a, e = actual.comparison.series[name], expected.comparison.series[name]
        np.testing.assert_array_equal(a.heights, e.heights)
        np.testing.assert_array_equal(a.covariance, e.covariance)
    for a, e in zip(observations["frames"], original.frames, strict=True):
        if e is None:
            assert a is None
        else:
            np.testing.assert_array_equal(a, e)
    evaluation = load_evaluation(tmp_path)
    np.testing.assert_array_equal(evaluation["truth"], original.truth)
    np.testing.assert_array_equal(evaluation["reference_covariance"], original.reference_covariance)


def test_inference_replay_never_opens_scoring_evidence(tmp_path, monkeypatch):
    write_synthetic_bundle(tmp_path)
    expected = infer_measurements(**load_observations(tmp_path))
    original_read = Path.read_bytes
    def guarded_read(path):
        assert path.name != "evaluation.npz", "held-out evidence entered observation path"
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    (tmp_path / "evaluation.npz").write_bytes(b"unreadable scoring truth")
    actual = infer_measurements(**load_observations(tmp_path))
    np.testing.assert_array_equal(actual.comparison.series["combined"].heights,
                                  expected.comparison.series["combined"].heights)


def test_changed_archive_bytes_are_refused(tmp_path):
    write_synthetic_bundle(tmp_path)
    path = tmp_path / "observations.npz"
    path.write_bytes(path.read_bytes() + b"unexpected")
    with pytest.raises(ValueError, match="checksum"):
        load_observations(tmp_path)


@pytest.mark.parametrize("source_kind", ["generated", "measured", "unspecified"])
def test_cannot_relabel_fixture_as_measurement_or_generated_fallback(tmp_path, source_kind):
    write_synthetic_bundle(tmp_path)
    rewrite_manifest(tmp_path, lambda data: data.update(source_kind=source_kind))
    with pytest.raises(ValueError, match="synthetic protocol"):
        load_observations(tmp_path)


def test_frame_hash_mismatch_is_refused(tmp_path):
    write_synthetic_bundle(tmp_path)
    rewrite_manifest(tmp_path, lambda data: data["frame_hashes"].__setitem__(0, "wrong"))
    with pytest.raises(ValueError, match="content hashes"):
        load_observations(tmp_path)


def test_changed_unit_declaration_is_not_silently_interpreted_as_metres(tmp_path):
    write_synthetic_bundle(tmp_path)
    rewrite_manifest(tmp_path, lambda data: data["units"].update(gauge_values="cm"))
    with pytest.raises(ValueError, match="units"):
        load_observations(tmp_path)


def test_pickle_objects_are_not_loaded_even_with_matching_checksum(tmp_path):
    write_synthetic_bundle(tmp_path)
    path = tmp_path / "observations.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["frames"] = np.array([object()], dtype=object)
    np.savez_compressed(path, **arrays)
    rewrite_manifest(tmp_path, lambda data: data["sha256"].update(
        {"observations.npz": hashlib.sha256(path.read_bytes()).hexdigest()}))
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        load_observations(tmp_path)


def test_scoring_archive_has_its_own_integrity_check(tmp_path):
    write_synthetic_bundle(tmp_path)
    (tmp_path / "evaluation.npz").write_bytes(b"changed")
    load_observations(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        load_evaluation(tmp_path)
