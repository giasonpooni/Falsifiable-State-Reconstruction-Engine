"""Independent uncertainty and availability checks for camera/gauge comparison."""
import numpy as np
import pytest

from set_lcm.camera import LevelSeries
from set_lcm.camera_fusion import compare_level_sources


def camera(values, covariance):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    return LevelSeries(values, covariance, values, tuple("ok" if v else "missing_level" for v in valid),
                       tuple(int(i) for i in np.flatnonzero(valid)), ("calibration:test",), ())


def test_single_time_matches_independent_information_form():
    C = np.array([[.04, .006], [.006, .09]])
    values = np.array([1.2, .9])
    result = compare_level_sources([0], values[:1], C[:1, :1], [0], camera(values[1:], C[1:, 1:]),
                                   cross_covariance=C[:1, 1:])
    inverse = np.linalg.inv(C)
    variance = 1 / (np.ones(2) @ inverse @ np.ones(2))
    mean = variance * np.ones(2) @ inverse @ values
    combined = result.series["combined"]
    np.testing.assert_allclose(combined.heights, [mean], rtol=1e-13)
    np.testing.assert_allclose(combined.covariance, [[variance]], rtol=1e-13)
    np.testing.assert_allclose(result.difference.heights, [.3], atol=1e-15)
    np.testing.assert_allclose(result.difference.covariance, [[.118]], rtol=1e-13)


def test_shared_reference_cannot_be_averaged_away_and_cancels_in_disagreement():
    n = 4
    shared = .16 * np.ones((n, n))
    result = compare_level_sources(np.arange(n), np.ones(n), shared + .04*np.eye(n), np.arange(n),
                                   camera(np.ones(n), shared + .01*np.eye(n)), cross_covariance=shared)
    np.testing.assert_allclose(result.camera_weight, .8, atol=1e-14)
    np.testing.assert_allclose(result.series["combined"].covariance, shared + .008*np.eye(n), atol=1e-14)
    np.testing.assert_allclose(result.difference.covariance, .05*np.eye(n), atol=1e-14)
    # The mean over time retains the shared .16 variance even with four paired samples.
    P = result.series["combined"].covariance
    assert np.ones(n) @ P @ np.ones(n) / n**2 == pytest.approx(.16 + .008/n)


def test_missing_frames_preserve_axis_and_combined_uses_available_source():
    times = np.arange(4, dtype=float)
    result = compare_level_sources(times, [1, np.nan, 3, np.nan], np.ones(4),
                                   [0, 1, np.nan, np.nan], camera([2, 2, np.nan, np.nan], np.eye(2)),
                                   cross_covariance=None)
    assert result.series["gauge_only"].observed_indices == (0, 2)
    assert result.series["camera_only"].observed_indices == (0, 1)
    assert result.series["combined"].observed_indices == (0, 1, 2)
    assert result.difference.observed_indices == (0,)
    np.testing.assert_allclose(result.series["combined"].heights[:3], [1.5, 2, 3])
    np.testing.assert_allclose(result.camera_weight[:3], [.5, 1, 0])
    assert np.isnan(result.camera_weight[3]) and np.isnan(result.series["combined"].heights[3])
    assert result.raw_order == (("gauge", 0), ("gauge", 2), ("camera", 0), ("camera", 1))


def test_correlated_sources_can_have_negative_blue_weight_without_clipping():
    result = compare_level_sources([0], [2], [[1]], [0], camera([4], [[4]]),
                                   cross_covariance=[[1.5]])
    np.testing.assert_allclose(result.camera_weight, [-.25])
    np.testing.assert_allclose(result.series["combined"].heights, [1.5])
    np.testing.assert_allclose(result.series["combined"].covariance, [[.875]])


def test_large_finite_covariance_does_not_overflow_weight_calculation():
    result = compare_level_sources([0], [2], [[1e308]], [0], camera([1], [[1e308]]),
                                   cross_covariance=[[8e307]])
    np.testing.assert_allclose(result.camera_weight, [.5], rtol=1e-14)
    np.testing.assert_allclose(result.series["combined"].heights, [1.5], rtol=1e-14)
    np.testing.assert_allclose(result.series["combined"].covariance / 1e307, [[9]], rtol=1e-14)
    np.testing.assert_allclose(result.difference.covariance / 1e307, [[4]], rtol=1e-14)


def test_empirical_joint_errors_match_full_predicted_covariance():
    rng = np.random.default_rng(67)
    latent = rng.normal(size=(6, 6)) / 10
    C = latent @ latent.T + np.eye(6) / 10
    result = compare_level_sources([0, 1, 2], [1, 2, 3], C[:3, :3], [0, 1, 2],
                                   camera([1, 2, 3], C[3:, 3:]), cross_covariance=C[:3, 3:])
    noise = rng.normal(size=(24000, 6)) @ np.linalg.cholesky(C).T
    transformed = noise @ result.series["combined"].operator.T
    np.testing.assert_allclose(np.cov(transformed, rowvar=False), result.series["combined"].covariance,
                               rtol=.055, atol=.003)


def test_exact_shared_noise_is_preserved_but_conflicting_exact_sources_are_refused():
    result = compare_level_sources([0], [1], [[.1]], [0], camera([1], [[.1]]),
                                   cross_covariance=[[.1]])
    np.testing.assert_allclose(result.series["combined"].covariance, [[.1]])
    np.testing.assert_array_equal(result.difference.covariance, [[0]])
    with pytest.raises(ValueError, match="exact agreement"):
        compare_level_sources([0], [1], [[.1]], [0], camera([2], [[.1]]), cross_covariance=[[.1]])


def test_no_data_is_explicitly_empty():
    result = compare_level_sources([0], [np.nan], [1], [np.nan], camera([np.nan], np.empty((0, 0))),
                                   cross_covariance=None)
    for estimate in [*result.series.values(), result.difference]:
        assert estimate.observed_indices == () and estimate.covariance.shape == (0, 0)
        assert np.isnan(estimate.heights[0])


def test_no_implicit_timestamp_matching():
    with pytest.raises(ValueError, match="capture times"):
        compare_level_sources([0, 1], [1, 1], [1, 1], [0, 1.01], camera([1, 1], np.eye(2)),
                              cross_covariance=None)
    with pytest.raises(ValueError, match="capture times"):
        compare_level_sources([0], [1], [1], [np.nan], camera([1], [[1]]), cross_covariance=None)


@pytest.mark.parametrize("times,values,covariance,cross", [
    ([0, 0], [1, 1], [1, 1], None),
    ([1, 0], [1, 1], [1, 1], None),
    ([0, 1], [1, np.inf], [1, 1], None),
    ([0, 1], [1, 1j], [1, 1], None),
    ([0, 1], [1, 1], [-1, 1], None),
    ([0, 1], [1, 1], [1, 1], [[2, 0], [0, 2]]),
    ([0, 1], [1, 1], [1, 1], [[0]]),
])
def test_invalid_input_or_joint_covariance_is_refused(times, values, covariance, cross):
    with pytest.raises(ValueError):
        compare_level_sources(times, values, covariance, [0, 1], camera([1, 1], np.eye(2)),
                              cross_covariance=cross)


def test_output_arrays_and_mode_mapping_are_immutable():
    result = compare_level_sources([0], [1], [1], [0], camera([1], [[1]]), cross_covariance=None)
    with pytest.raises(TypeError):
        result.series["new"] = result.series["combined"]
    for array in (result.camera_weight, result.raw_covariance, result.difference.covariance,
                  result.series["combined"].heights, result.series["combined"].operator):
        with pytest.raises(ValueError):
            array.setflags(write=True)
