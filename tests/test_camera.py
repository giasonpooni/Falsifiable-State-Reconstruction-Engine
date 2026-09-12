"""Known image geometry, independent GLS, and declared covariance camera checks."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from set_lcm.camera import (LevelSeries, LinearLevelCalibration, MarkerRegistration,
                            compensate_level, detect_level, fit_linear_calibration,
                            register_vertical_marker)


def _image(row=27.3, marker_row=14.2, *, bright=False, height=64, width=96):
    # Independent area coverage: each pixel is an integral over its unit-height band.
    centers = np.arange(height)
    water = np.minimum(np.maximum(centers + .5 - row, 0), 1)
    marker = np.minimum(np.maximum(centers + .5 - marker_row, 0), 1)
    frame = np.full((height, width), .5)
    frame[:, 8:64] = (.2 + .6*water if bright else .8 - .6*water)[:, None]
    frame[:, 76:90] = (.9 - .8*marker)[:, None]
    return frame


def _options(**changes):
    result = dict(water_columns=(8, 64), expected_polarity="dark_below",
                  min_contrast=.3, max_column_spread=.15)
    result.update(changes)
    return result


@pytest.mark.parametrize("row", [8.0, 12.1, 23.25, 31.5, 47.9, 55.0])
@pytest.mark.parametrize("bright", [False, True])
def test_area_antialias_edge_returns_actual_subpixel_boundary(row, bright):
    result = detect_level(_image(row, bright=bright), **_options(
        expected_polarity="bright_below" if bright else "dark_below"))
    assert result.status == "ok"
    assert result.row == pytest.approx(row, abs=1e-12)
    assert result.column_spread == pytest.approx(0, abs=1e-12)
    assert result.contrast == pytest.approx(.6)
    assert not hasattr(result, "variance")  # quality is not an uncertainty estimate


def test_declared_roi_retains_global_rows_despite_distracting_other_edges():
    result = detect_level(_image(32.7, 14.2), **_options(row_range=(20, 50)))
    assert result.row == pytest.approx(32.7, abs=1e-12)
    marker = register_vertical_marker(_image(32.7, 14.2), marker_columns=(76, 90),
        reference_row=12, expected_polarity="dark_below", min_contrast=.3, max_column_spread=.15)
    assert marker.row == pytest.approx(14.2)
    assert marker.displacement == pytest.approx(2.2)


def test_column_offsets_contrasts_and_small_intensity_noise_do_not_invent_a_large_shift():
    frame = _image()
    frame[:, 8:64] *= np.linspace(.8, 1.1, 56)
    frame[:, 8:64] += np.linspace(-.03, .04, 56)
    frame[:, 8:64] += np.random.default_rng(77).normal(0, .0002, (64, 56))
    result = detect_level(frame, **_options())
    assert result.status == "ok"
    assert abs(result.row - 27.3) < .005
    assert result.column_spread < .03


def test_partial_occlusion_and_multiple_edges_are_quality_refusals():
    frame = _image()
    frame[:, 20] = .5
    result = detect_level(frame, **_options())
    assert result.row is None and result.status == "low_contrast"
    frame = _image()
    frame[8:12, 8:64] = .2
    result = detect_level(frame, **_options())
    assert result.row is None and result.status == "invalid_edge"


def test_nonhorizontal_edge_is_refused_by_column_spread():
    frame = _image()
    rows = np.linspace(25, 30, 56)
    frame[:, 8:64] = .8 - .6*np.clip(np.arange(64)[:, None] + .5 - rows, 0, 1)
    result = detect_level(frame, **_options())
    assert result.row is None and result.status == "inconsistent_columns"
    assert result.column_spread == pytest.approx(5)


def test_marker_translation_is_removed_and_marker_failure_never_uses_raw_level():
    for motion in (-4.1, 0, 3.7):
        frame = _image(27.3 + motion, 14.2 + motion)
        raw = detect_level(frame, **_options())
        marker = register_vertical_marker(frame, marker_columns=(76, 90), reference_row=14.2,
            expected_polarity="dark_below", min_contrast=.3, max_column_spread=.15)
        corrected = compensate_level(raw, marker)
        assert marker.displacement == pytest.approx(motion, abs=1e-12)
        assert corrected.row == pytest.approx(27.3, abs=1e-12)
    frame[:, 76:90] = .5
    failed = register_vertical_marker(frame, marker_columns=(76, 90), reference_row=14.2,
        expected_polarity="dark_below", min_contrast=.3, max_column_spread=.15)
    corrected = compensate_level(raw, failed)
    assert failed.displacement is None
    assert corrected.row is None and corrected.status == "missing_marker"


def test_fixed_anchor_gls_matches_independent_information_form_with_correlated_reference():
    pixels = np.array([8., 17., 26., 39., 51.])
    heights = .95 - .02*pixels + np.array([.002, -.001, .003, -.002, .001])
    R = np.diag([1, 2, 1.5, 2.5, 1]) * .002**2 + .004**2*np.ones((5, 5))
    X = np.column_stack((pixels, np.ones(5)))
    precision = X.T @ np.linalg.solve(R, X)
    expected_cov = np.linalg.inv(precision)
    expected_mean = np.linalg.solve(precision, X.T @ np.linalg.solve(R, heights))
    result = fit_linear_calibration(pixels, heights, R, calibration_ids=("anchor-session", "reference"))
    np.testing.assert_allclose([result.slope_m_per_pixel, result.intercept_m], expected_mean, rtol=1e-11)
    np.testing.assert_allclose(result.parameter_covariance, expected_cov, rtol=1e-11, atol=1e-15)
    assert result.valid_pixel_range == (8., 51.)
    assert result.calibration_ids == ("anchor-session", "reference")


def test_calibration_anchor_images_are_separate_exact_design_not_noisy_pixel_fits():
    pixels = np.array([10., 20., 30., 40., 50.])
    extracted = [detect_level(_image(p), **_options()).row for p in pixels]
    np.testing.assert_allclose(extracted, pixels, atol=1e-12)
    calibration = fit_linear_calibration(extracted, .95-.02*pixels, np.full(5, .002**2))
    mapped = calibration.map_rows([17.2, 32.8], localization_covariance=[.03**2, .03**2])
    np.testing.assert_allclose(mapped.heights, [.606, .294], atol=1e-12)
    assert any("fixed and exact" in statement for statement in mapped.assumptions)


def test_shared_calibration_uncertainty_creates_cross_time_covariance():
    calibration = LinearLevelCalibration(-.02, .95, [[0, 0], [0, .004**2]], (5, 55), ("shared-ref",))
    rows = np.array([10., 20., 30.])
    result = calibration.map_rows(rows, localization_covariance=[.01, .04, .09])
    expected = .004**2*np.ones((3, 3)) + .02**2*np.diag([.01, .04, .09])
    np.testing.assert_allclose(result.covariance, expected, rtol=0, atol=1e-18)
    assert result.covariance[0, 2] == pytest.approx(.004**2)


def test_marker_joint_covariance_subtracts_cross_terms_and_shared_reference_cancels():
    calibration = LinearLevelCalibration(-.02, .95, np.zeros((2, 2)), (5, 55))
    local = np.diag([.01, .02, .03]) + .04*np.ones((3, 3))
    marker = np.diag([.02, .01, .04]) + .04*np.ones((3, 3))
    cross = .04*np.ones((3, 3))
    result = calibration.map_rows([12, 25, 37], localization_covariance=local,
        marker_displacement=[2, 5, 7], marker_covariance=marker, localization_marker_covariance=cross)
    np.testing.assert_allclose(result.corrected_rows, [10, 20, 30])
    np.testing.assert_allclose(result.covariance, .02**2*np.diag([.03, .03, .07]), atol=1e-18)


def test_full_joint_error_simulation_matches_first_order_propagation():
    rng = np.random.default_rng(982)
    n = 3
    loadings = rng.normal(size=(2*n, 2*n)) * .03
    joint = loadings @ loadings.T + .002*np.eye(2*n)
    parameter_cov = np.array([[1e-8, -2e-7], [-2e-7, 1e-5]])
    calibration = LinearLevelCalibration(-.02, .95, parameter_cov, (5, 55))
    pixels = np.array([12., 27., 41.])
    displacement = np.array([2., 7., 11.])
    result = calibration.map_rows(pixels, localization_covariance=joint[:n, :n],
        marker_displacement=displacement, marker_covariance=joint[n:, n:],
        localization_marker_covariance=joint[:n, n:])
    count = 60_000
    errors = rng.normal(size=(count, 2*n)) @ np.linalg.cholesky(joint).T
    parameters = rng.normal(size=(count, 2)) @ np.linalg.cholesky(parameter_cov).T
    J = np.column_stack((pixels-displacement, np.ones(n)))
    linearized_errors = parameters @ J.T - .02*(errors[:, :n]-errors[:, n:])
    standardized = np.linalg.solve(np.linalg.cholesky(result.covariance), linearized_errors.T)
    np.testing.assert_allclose(np.cov(standardized), np.eye(n), atol=.025, rtol=0)


def test_mapping_is_explicitly_first_order_not_exact_product_covariance():
    calibration = LinearLevelCalibration(.1, .2, np.diag([.01, .02]), (5, 30))
    rows, Cp = np.array([10., 20.]), np.diag([.3, .4])
    result = calibration.map_rows(rows, localization_covariance=Cp)
    J = np.column_stack((rows, np.ones(2)))
    first_order = J @ calibration.parameter_covariance @ J.T + .1**2*Cp
    np.testing.assert_allclose(result.covariance, first_order)
    exact_independent_product = first_order + .01*Cp
    assert np.linalg.norm(exact_independent_product-result.covariance) > .004
    assert any("first order" in statement for statement in result.assumptions)


def test_missing_marker_and_range_failures_keep_original_order_and_valid_submatrix():
    calibration = LinearLevelCalibration(-.02, .95, [[0, 0], [0, .01]], (5, 55))
    result = calibration.map_rows([10, np.nan, 30, 100, 40], localization_covariance=np.ones(5),
        marker_displacement=[1, 2, np.nan, 0, 3], marker_covariance=np.ones(5)*2)
    assert result.statuses == ("ok", "missing_level", "missing_marker", "outside_calibration_range", "ok")
    assert result.observed_indices == (0, 4)
    np.testing.assert_allclose(result.corrected_rows, [9, np.nan, np.nan, np.nan, 37], equal_nan=True)
    np.testing.assert_allclose(result.covariance, .01*np.ones((2, 2)) + .02**2*3*np.eye(2))
    missing = calibration.map_rows([np.nan], localization_covariance=[1])
    assert missing.observed_indices == () and missing.covariance.shape == (0, 0)
    empty = calibration.map_rows([], localization_covariance=[])
    assert empty.heights.shape == (0,) and empty.covariance.shape == (0, 0)


def test_record_buffers_and_scalar_results_are_immutable_and_own_their_values():
    covariance = np.eye(2)*.001
    calibration = LinearLevelCalibration(-.02, .95, covariance, (5, 55))
    covariance[:] = 0
    np.testing.assert_array_equal(calibration.parameter_covariance, np.eye(2)*.001)
    result = calibration.map_rows([10, 20], localization_covariance=[1, 1])
    for array in (calibration.parameter_covariance, result.heights, result.covariance, result.corrected_rows):
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array.flat[0] = 0
    with pytest.raises(FrozenInstanceError):
        calibration.intercept_m = 0
    detection = detect_level(_image(), **_options())
    with pytest.raises(FrozenInstanceError):
        detection.row = 0


@pytest.mark.parametrize("field,value", [
    ("water_columns", (8, 100)), ("water_columns", (8, 8)), ("water_columns", (8., 64.)),
    ("expected_polarity", "automatic"), ("min_contrast", 0), ("min_contrast", np.nan),
    ("max_column_spread", -1), ("plateau_rows", 0), ("plateau_rows", 2.5),
    ("row_range", (1, 5)), ("max_profile_deviation", .5),
])
def test_malformed_detection_arguments_are_refused(field, value):
    with pytest.raises(ValueError):
        detect_level(_image(), **_options(**{field: value}))


@pytest.mark.parametrize("frame", [np.zeros((64, 96, 3)), np.full((64, 96), np.nan),
                                    np.full((64, 96), np.inf), np.zeros((64, 96), dtype=complex)])
def test_malformed_images_are_refused(frame):
    with pytest.raises(ValueError):
        detect_level(frame, **_options())


@pytest.mark.parametrize("pixels,heights,covariance", [
    ([10], [.5], [[.01]]), ([10, 10], [.5, .4], np.eye(2)),
    ([10, 20], [.5], np.eye(2)), ([10, np.nan], [.5, .4], np.eye(2)),
    ([10j, 20], [.5, .4], np.eye(2)), ([10, 20], [.5, .4], np.zeros((2, 2))),
    ([10, 20], [.5, .4], [[1, 1], [1, 1]]), ([10, 20], [.5, .4], [[1, 2], [2, 1]]),
])
def test_invalid_calibration_fits_are_refused(pixels, heights, covariance):
    with pytest.raises(ValueError):
        fit_linear_calibration(pixels, heights, covariance)


@pytest.mark.parametrize("field,value", [
    ("slope_m_per_pixel", 0), ("slope_m_per_pixel", np.inf), ("intercept_m", 1j),
    ("parameter_covariance", [[1e12, 0], [0, -1e-10]]),
    ("parameter_covariance", [[1e-20, 2], [2, 1e20]]),
    ("parameter_covariance", [[0, 1e-30], [1e-30, 1]]),
    ("parameter_covariance", [[1, .1], [0, 1]]),
    ("valid_pixel_range", (20, 10)), ("calibration_ids", "one-id"),
])
def test_invalid_calibration_declarations_are_refused(field, value):
    args = dict(slope_m_per_pixel=-.02, intercept_m=.95, parameter_covariance=np.eye(2)*.01,
                valid_pixel_range=(5, 55))
    args[field] = value
    with pytest.raises(ValueError):
        LinearLevelCalibration(**args)


@pytest.mark.parametrize("changes", [
    {"localization_covariance": None}, {"localization_covariance": [-1, 1]},
    {"localization_covariance": [[1, 2], [2, 1]]},
    {"localization_covariance": np.eye(2, dtype=complex)},
    {"marker_displacement": [0, 0]}, {"marker_covariance": [1, 1]},
    {"marker_displacement": [0], "marker_covariance": [1]},
    {"marker_displacement": [0, 0], "marker_covariance": [1, 1],
     "localization_marker_covariance": [[2, 0], [0, 2]]},
])
def test_unknown_uncertainty_and_invalid_joint_declarations_are_refused(changes):
    calibration = LinearLevelCalibration(-.02, .95, np.zeros((2, 2)), (5, 55))
    args = dict(localization_covariance=[1, 1])
    args.update(changes)
    with pytest.raises(ValueError):
        calibration.map_rows([10, 20], **args)


@pytest.mark.parametrize("rows", [[10j, 20], [10, np.inf], [[10, 20]]])
def test_invalid_row_values_are_refused(rows):
    calibration = LinearLevelCalibration(-.02, .95, np.zeros((2, 2)), (5, 55))
    with pytest.raises(ValueError):
        calibration.map_rows(rows, localization_covariance=[1, 1])


def test_covariance_scaling_preserves_representable_results_without_squaring_slope_first():
    calibration = LinearLevelCalibration(1e200, 0, np.zeros((2, 2)), (-1, 1))
    result = calibration.map_rows([0], localization_covariance=[1e-300])
    assert result.covariance[0, 0] == pytest.approx(1e100)
    with pytest.raises(ValueError):
        calibration.map_rows([0], localization_covariance=[1.0])


@pytest.mark.parametrize("indices", [(0.0,), (False,)])
def test_level_series_indices_are_integers_before_downstream_array_indexing(indices):
    with pytest.raises(ValueError):
        LevelSeries([1], [[1]], [10], ("ok",), indices, (), ())


def test_marker_registration_does_not_accept_inconsistent_displacement_metadata():
    with pytest.raises(ValueError):
        MarkerRegistration(12, 3, 10, "ok", "inconsistent")
