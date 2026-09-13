"""Integration evidence for the declared rendered-camera protocol, not field accuracy."""
from dataclasses import replace
import json

import numpy as np
import pytest

from set_lcm.camera import detect_level
from set_lcm.experiments import camera_baseline as experiment
from set_lcm.experiments.compare import compare, reproduction_failure
from set_lcm.experiments.provenance import REPO_ROOT
from set_lcm.frame_quality import frame_content_hash


def test_separate_anchor_frames_extract_exact_design_and_keep_shared_uncertainty():
    trial = experiment.build_case(10000, "healthy")
    evidence = trial.calibration_evidence
    rows = [detect_level(frame, water_columns=experiment.WATER_COLUMNS,
                         **experiment.DETECTOR_OPTIONS).row for frame in evidence.frames]
    np.testing.assert_allclose(rows, np.linspace(12, 40, 8), atol=1e-13)
    assert set(experiment.CALIBRATION_SEEDS).isdisjoint(experiment.EVALUATION_SEEDS)
    assert evidence.reference_covariance[0, 1] == experiment.SHARED_REFERENCE_SD_M**2
    analysis = experiment.analyze(trial)
    assert analysis.calibration.slope_m_per_pixel != experiment.TRUE_SLOPE_M_PER_PIXEL
    assert np.linalg.eigvalsh(analysis.calibration.parameter_covariance).min() > 0
    assert analysis.camera.covariance[0, 1] > 0
    assert analysis.comparison.raw_covariance[0, experiment.N_FRAMES] == experiment.SHARED_REFERENCE_SD_M**2


def test_rendered_frames_use_a_fixed_intensity_grid_and_stable_payload_hashes():
    trial = experiment.build_case(10000, "healthy")
    for frame in trial.frames:
        np.testing.assert_array_equal(frame,
                                      np.rint(frame / experiment.INTENSITY_GRID) * experiment.INTENSITY_GRID)
    a = experiment.render_frame(24.1234567, 8.0345678)
    b = experiment.render_frame(24.1234567 + 1e-13, 8.0345678 - 1e-13)
    assert frame_content_hash(a) == frame_content_hash(b)


def test_inference_never_reads_truth_independent_reference_or_scenario():
    trial = experiment.build_case(10000, "healthy")
    class MeasurementsOnly:
        def __getattr__(self, name):
            if name in {"truth", "reference_values", "reference_covariance", "seed", "scenario"}:
                raise AssertionError(f"inference accessed scoring-only field {name}")
            return getattr(trial, name)
    actual = experiment.analyze(MeasurementsOnly())
    expected = experiment.analyze(trial)
    np.testing.assert_array_equal(actual.comparison.series["combined"].heights,
                                  expected.comparison.series["combined"].heights)
    assert actual.diagnostic == expected.diagnostic
    changed_reference = replace(trial, reference_values=trial.reference_values + 1.0,
                                 truth=trial.truth - 1.0)
    np.testing.assert_array_equal(experiment.analyze(changed_reference).comparison.series["combined"].heights,
                                  expected.comparison.series["combined"].heights)
    assert experiment.score_trial(changed_reference, actual)["methods"]["combined"]["truth_rmse_m"] > .9


def test_gauge_drift_remains_ambiguous_and_reference_uncertainty_is_added():
    healthy = experiment.build_case(10000, "healthy")
    trial = experiment.build_case(10000, "gauge_drift")
    np.testing.assert_array_equal(healthy.truth, trial.truth)
    for a, b in zip(healthy.frames, trial.frames, strict=True):
        np.testing.assert_array_equal(a, b)
    analysis = experiment.analyze(trial)
    assert analysis.diagnostic["status"] == "ambiguous"
    assert set(analysis.diagnostic["candidates"]) == {"gauge_drift", "camera_drift"}
    fits = analysis.diagnostic["fits"]
    assert fits["gauge_drift"]["amplitude"] == -fits["camera_drift"]["amplitude"]
    score = experiment.score_trial(trial, analysis)
    for method in experiment.METHODS:
        result = score["methods"][method]
        assert result["reference_joint_dimension"] == result["common_samples"] == experiment.N_FRAMES
        assert result["mean_reference_difference_variance_m2"] == pytest.approx(
            result["mean_estimate_variance_m2"] + experiment.REFERENCE_SD_M**2)
        assert 0 <= result["truth_interval_covered"] <= result["common_samples"]


def test_shared_calibration_bias_is_blind_to_the_difference_but_visible_to_reference():
    healthy = experiment.build_case(10000, "healthy")
    shifted = experiment.build_case(10000, "shared_calibration_offset")
    a, b = experiment.analyze(healthy), experiment.analyze(shifted)
    np.testing.assert_array_equal(healthy.truth, shifted.truth)
    np.testing.assert_array_equal(healthy.reference_values, shifted.reference_values)
    np.testing.assert_allclose(shifted.gauge_values - healthy.gauge_values,
                               experiment.SHARED_OFFSET_M, atol=1e-15)
    np.testing.assert_allclose(a.comparison.difference.heights, b.comparison.difference.heights, atol=1e-13)
    assert a.diagnostic["status"] == b.diagnostic["status"] == "consistent"
    score = experiment.score_trial(shifted, b)
    assert score["methods"]["combined"]["reference_discrepancy_rmse_m"] > .025
    assert score["methods"]["combined"]["truth_interval_covered"] == 0


def test_motion_is_measured_from_the_fiducial_and_corrected_with_matching_support():
    trial = experiment.build_case(10000, "camera_vertical_drift")
    analysis = experiment.analyze(trial)
    score = experiment.score_trial(trial, analysis)["registration_comparison"]
    assert analysis.marker_displacements[-1] == pytest.approx(experiment.CAMERA_TRANSLATION_PIXELS, abs=.1)
    assert score["registered"]["common_samples"] == score["unregistered"]["common_samples"] == experiment.N_FRAMES
    assert score["registered"]["truth_rmse_m"] < score["unregistered"]["truth_rmse_m"] / 10
    np.testing.assert_allclose(analysis.camera.corrected_rows,
                               analysis.water_rows - analysis.marker_displacements)
    assert analysis.camera.covariance[0, 0] > analysis.unregistered_camera.covariance[0, 0]


@pytest.mark.parametrize("scenario,excluded", [("camera_obstruction", set(range(12, 18))),
                                                ("duplicate_frame", {16}),
                                                ("dropped_frame", {10, 20})])
def test_quality_gates_retain_time_and_fallback_with_fair_common_support(scenario, excluded):
    trial = experiment.build_case(10000, scenario)
    analysis = experiment.analyze(trial)
    assert set(range(experiment.N_FRAMES)) - set(analysis.camera.observed_indices) == excluded
    np.testing.assert_array_equal(analysis.audit.capture_times, trial.times)
    np.testing.assert_array_equal(analysis.comparison.series["combined"].heights[list(excluded)],
                                  trial.gauge_values[list(excluded)])
    scores = experiment.score_trial(trial, analysis)["methods"]
    assert {score["common_samples"] for score in scores.values()} == {experiment.N_FRAMES - len(excluded)}
    assert scores["gauge_only"]["available_samples"] == scores["combined"]["available_samples"] == experiment.N_FRAMES
    assert any(window["status"] == "refused" for window in analysis.spectral_windows)
    if scenario == "duplicate_frame":
        assert analysis.audit.content_hashes[15] == analysis.audit.content_hashes[16]
        assert "repeated_content" in analysis.audit.flags[16]
    if scenario == "camera_obstruction":
        assert all(analysis.water_statuses[i] == "low_contrast" for i in excluded)


def test_actual_jittered_times_remain_usable_but_have_no_uniform_window_spectra():
    trial = experiment.build_case(10000, "time_jitter")
    analysis = experiment.analyze(trial)
    assert np.ptp(np.diff(trial.times)) > .01
    np.testing.assert_array_equal(trial.times, trial.gauge_times)
    np.testing.assert_array_equal(analysis.audit.capture_times, trial.times)
    assert len(analysis.camera.observed_indices) == experiment.N_FRAMES
    assert all(analysis.audit.admitted)
    assert {window["status"] for window in analysis.spectral_windows} == {"refused"}
    assert all("uniform capture clock" in window["reason"] for window in analysis.spectral_windows)


def test_unmatched_camera_time_is_excluded_without_changing_either_clock():
    trial = experiment.build_case(10000, "healthy")
    times = trial.times.copy()
    times[5] += .05
    analysis = experiment.analyze(replace(trial, times=times))
    assert analysis.water_statuses[5] == "time_not_aligned"
    assert 5 not in analysis.camera.observed_indices
    assert analysis.audit.capture_times[5] == times[5]
    assert analysis.comparison.series["combined"].times[5] == trial.gauge_times[5]
    assert analysis.comparison.series["combined"].heights[5] == trial.gauge_values[5]


@pytest.mark.parametrize("clock", ["times", "gauge_times"])
def test_complex_clock_is_refused_without_losing_its_imaginary_part(clock):
    trial = experiment.build_case(10000, "healthy")
    with pytest.raises(ValueError, match="real-valued"):
        experiment.analyze(replace(trial, **{clock: getattr(trial, clock) + .1j}))


def test_all_missing_camera_abstains_and_keeps_gauge_availability():
    trial = experiment.build_case(10000, "healthy")
    trial = replace(trial, frames=(None,) * experiment.N_FRAMES)
    analysis = experiment.analyze(trial)
    assert analysis.diagnostic["status"] == "insufficient_evidence"
    assert analysis.diagnostic["null_statistic"] is None
    assert analysis.diagnostic["fits"] == {}
    np.testing.assert_array_equal(analysis.comparison.series["combined"].heights, trial.gauge_values)
    scores = experiment.score_trial(trial, analysis)["methods"]
    assert {score["common_samples"] for score in scores.values()} == {0}
    assert scores["combined"]["available_samples"] == experiment.N_FRAMES
    assert scores["camera_only"]["available_samples"] == 0
    assert all(window["status"] == "refused" for window in analysis.spectral_windows)
    aggregate = experiment._aggregate([experiment.score_trial(trial, analysis)])
    assert aggregate["record_null_rejections"]["denominator"] == 0


def test_no_inplace_mutation_of_observations_or_calibration():
    trial = experiment.build_case(10000, "camera_vertical_drift")
    before_frames = [frame.copy() for frame in trial.frames]
    arrays = (trial.times, trial.gauge_times, trial.gauge_values, trial.gauge_covariance,
              trial.calibration_evidence.reference_heights, trial.calibration_evidence.reference_covariance)
    copies = [array.copy() for array in arrays]
    experiment.analyze(trial)
    for original, copy in zip(arrays, copies, strict=True):
        np.testing.assert_array_equal(original, copy)
    for original, copy in zip(trial.frames, before_frames, strict=True):
        np.testing.assert_array_equal(original, copy)


@pytest.mark.parametrize("seed", [True, -1, 2.5, np.nan])
def test_invalid_simulation_seed_refused(seed):
    with pytest.raises(ValueError, match="seed"):
        experiment.build_case(seed, "healthy")


@pytest.fixture(scope="module")
def full_report():
    return experiment.run_experiment()


def test_default_report_retains_outcomes_and_denominators(full_report):
    assert len(full_report["design"]["evaluation_seeds"]) == 16
    assert set(full_report["design"]["calibration_development_seeds"]).isdisjoint(
        full_report["design"]["evaluation_seeds"])
    assert set(full_report["cases"]) == set(experiment.SCENARIOS)
    for case in full_report["cases"].values():
        aggregate = case["aggregate"]
        assert aggregate["records"] == len(case["per_seed"]) == 16
        assert sum(aggregate["diagnostic_status_counts"].values()) == 16
        assert aggregate["diagnostic_status_counts"]["identified"] == 0
        for result in aggregate["methods"].values():
            assert result["truth_interval_coverage"]["denominator"] == result["common_samples"]
            assert result["reference_interval_coverage"]["denominator"] == result["common_samples"]
            assert result["availability"]["denominator"] == 16 * experiment.N_FRAMES
        for record in case["per_seed"]:
            assert len(record["trace"]["times_seconds"]) == experiment.N_FRAMES
            assert len(record["frames"]["content_hashes"]) == experiment.N_FRAMES
    json.dumps(full_report, allow_nan=False)


def test_generator_roundtrip_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "EVALUATION_SEEDS", (10000,))
    first, second = tmp_path / "first", tmp_path / "second"
    assert experiment.main(first, quiet=True) == 0
    assert experiment.main(second, quiet=True) == 0
    a = json.loads((first / "camera_baseline.json").read_text(encoding="utf-8"))
    b = json.loads((second / "camera_baseline.json").read_text(encoding="utf-8"))
    assert compare(a, b, rel_tol=0.0) == []
    assert (first / "camera_baseline.md").read_text(encoding="utf-8") == experiment.render(a)


def test_committed_camera_report_reproduces(full_report):
    path = REPO_ROOT / "results" / "camera_baseline.json"
    committed = json.loads(path.read_text(encoding="utf-8"))
    failure = reproduction_failure("camera_baseline.json", full_report, committed)
    assert failure is None, failure
    assert path.with_suffix(".md").read_text(encoding="utf-8") == experiment.render(committed)


def test_the_declared_collinearity_of_the_catalogue_is_measured_not_only_asserted(full_report):
    """The report says the two candidates are collinear; the artifact now shows it.

    A gauge drift of +max(t-12,0) and a camera-level drift of -max(t-12,0) differ only
    in sign, so no amplitude and no covariance separates them. Every diagnostic in the
    report therefore carries min_separation 0.0 and no isolation amplification: the
    prose claim and the computed geometry have to agree.
    """
    separations, amplifications = set(), set()

    def walk(node):
        if isinstance(node, dict):
            if "min_separation" in node:
                separations.add(node["min_separation"])
                for fit in node.get("fits", {}).values():
                    if fit.get("nearest") is not None:
                        amplifications.add(fit["nearest_isolation_amplification"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(full_report)
    assert separations == {0.0}, separations
    assert amplifications == {None}, amplifications
    declared = json.dumps(full_report)
    assert "They are collinear: evidence can reject consistency without choosing" in declared
