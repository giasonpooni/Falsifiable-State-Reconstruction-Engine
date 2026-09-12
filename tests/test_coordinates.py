"""Coordinate contracts checked through complete prediction/conditioning paths."""
import numpy as np
import pytest

from set_lcm.coordinates import AffineCoordinates
from set_lcm.invariant import GaussianState, invariant_error, predict, update


@pytest.mark.parametrize("matrix,offset", [
    ([[0, 1], [1, 0]], [0, 0]),
    ([[1, 1], [1, -1]], [9, -3]),
    ([[1000, 0], [0, .01]], [120, -2]),
    ([[1, .3], [-.4, 2]], [-5, 13]),
])
def test_complete_affine_record_equivariance(matrix, offset):
    chart = AffineCoordinates(matrix, offset)
    state = GaussianState([40, 60], [[2, .3], [.3, 1]])
    F = np.array([[.96, .01], [.04, .99]])
    Q = np.array([[.03, -.01], [-.01, .02]])
    H = np.array([[1, .1], [.2, .9], [1, -1]])
    a = np.array([2, -1, 3])
    R = np.array([[.2, .03, .02], [.03, .3, -.04], [.02, -.04, .4]])
    z = np.array([45, 63, -12.])
    transformed = chart.transform_state(state)
    for k in range(5):
        d = np.array([-1, 1]) * (k + 1) / 4
        state = predict(state, F, d, Q)
        transformed = predict(transformed, *chart.transform_dynamics(F, d, Q))
        mask = np.array([True, k % 2 == 0, k != 2])
        observation = z + k
        observation[~mask] = np.nan
        result = update(state, observation, H, R, a, mask)
        H_chart, a_chart = chart.transform_observation(H, a)
        result_chart = update(transformed, observation, H_chart, R, a_chart, mask)
        state, transformed = result.posterior, result_chart.posterior
        restored = chart.restore_state(transformed)
        np.testing.assert_allclose(restored.mean, state.mean, rtol=1e-12, atol=1e-11)
        np.testing.assert_allclose(restored.covariance, state.covariance, rtol=1e-11, atol=1e-13)
        np.testing.assert_allclose(result_chart.statistic, result.statistic, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(result_chart.gain, chart.matrix @ result.gain, rtol=1e-10, atol=1e-10)


def test_affine_origin_transports_error_and_constraint():
    chart = AffineCoordinates([[2, .4], [0, 3]], [40, -9])
    first, second = np.array([2., 3.]), np.array([3., 5.])
    np.testing.assert_allclose(
        invariant_error(chart.matrix @ first + chart.offset, chart.matrix @ second + chart.offset),
        chart.matrix @ invariant_error(first, second))
    A, b = np.array([[1., 1.]]), np.array([5.])
    mapped_A = np.linalg.solve(chart.matrix.T, A.T).T
    mapped_b = b + mapped_A @ chart.offset
    np.testing.assert_allclose(mapped_A @ (chart.matrix @ second + chart.offset) - mapped_b,
                               A @ second - b)


def test_exact_zero_covariance_roundtrip_and_empty_observations():
    chart = AffineCoordinates([[2, 1], [0, 3]], [4, -2])
    state = GaussianState([2, 8], np.zeros((2, 2)))
    restored = chart.restore_state(chart.transform_state(state))
    np.testing.assert_allclose(restored.mean, state.mean)
    np.testing.assert_array_equal(restored.covariance, state.covariance)
    H, a = chart.transform_observation(np.empty((0, 2)), [])
    assert H.shape == (0, 2) and a.shape == (0,)


def test_chart_and_transformed_arrays_have_immutable_owned_buffers():
    matrix, offset = np.eye(2), np.zeros(2)
    chart = AffineCoordinates(matrix, offset)
    matrix[0, 0], offset[0] = 99, 99
    np.testing.assert_array_equal(chart.matrix, np.eye(2))
    np.testing.assert_array_equal(chart.offset, np.zeros(2))
    arrays = [chart.matrix, chart.offset, *chart.transform_dynamics(np.eye(2), [0, 0], np.eye(2)),
              *chart.transform_observation(np.eye(2), [0, 0])]
    for array in arrays:
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize("matrix,offset", [
    ([[1, 1], [1, 1]], [0, 0]), ([[1, 0], [0, 1e-14]], [0, 0]),
    ([[1, 0, 0], [0, 1, 0]], [0, 0]), ([], []),
    ([[1, 0], [0, np.nan]], [0, 0]), ([[1, 0], [0, 1]], [0, np.inf]),
    ([[1, 0], [0, 1j]], [0, 0]), ([[1, 0], [0, 1]], [0, 1j]),
])
def test_invalid_charts_are_refused(matrix, offset):
    with pytest.raises(ValueError):
        AffineCoordinates(matrix, offset)


def test_invalid_dimensions_models_and_covariance_are_refused():
    chart = AffineCoordinates(np.eye(2), [0, 0])
    with pytest.raises(ValueError, match="dimension"):
        chart.transform_state(GaussianState([1], [[1]]))
    with pytest.raises(TypeError):
        chart.restore_state([1, 2])
    with pytest.raises(ValueError):
        chart.transform_dynamics(np.eye(2), [0, 0], [[1, 2], [2, 1]])
    with pytest.raises(ValueError):
        chart.transform_dynamics(np.eye(3), [0, 0], np.eye(2))
    with pytest.raises(ValueError):
        chart.transform_observation(np.eye(2), [0])
    with pytest.raises(ValueError):
        chart.transform_observation([[1, np.nan]], [0])


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_scalar_chart_refuses_covariance_underflow_or_overflow_despite_condition_one(scale):
    chart = AffineCoordinates([[scale]], [0])
    with pytest.raises(ValueError):
        chart.transform_state(GaussianState([1], [[1]]))
    with pytest.raises(ValueError):
        chart.restore_state(GaussianState([1], [[1]]))
    with pytest.raises(ValueError):
        chart.transform_dynamics([[1]], [0], [[1]])


def test_dense_chart_refuses_material_covariance_loss_below_condition_cap():
    chart = AffineCoordinates([[1, 1], [1, 1 + 1e-7]], [0, 0])
    assert np.linalg.cond(chart.matrix) < 1e12
    # In binary arithmetic P=I previously round-tripped to diagonal entries .976996.
    with pytest.raises(ValueError, match="round trip"):
        chart.transform_state(GaussianState([1, 2], np.eye(2)))
    with pytest.raises(ValueError, match="round trip"):
        chart.transform_dynamics(np.eye(2), [0, 0], np.eye(2))


def test_huge_origin_cannot_erase_mean_or_drift():
    chart = AffineCoordinates([[1]], [1e20])
    with pytest.raises(ValueError, match="state mean.*round trip"):
        chart.transform_state(GaussianState([1], [[1]]))
    with pytest.raises(ValueError, match="dynamics drift.*round trip"):
        chart.transform_dynamics([[1]], [1], [[1]])


def test_exact_zero_scale_cannot_borrow_tolerance_from_another_component():
    chart = AffineCoordinates([[1, .3], [0, 1]], [1e20, 0])
    # The first deterministic component is exactly zero; recovering -0.3 is unacceptable.
    with pytest.raises(ValueError, match="state mean.*round trip"):
        chart.transform_state(GaussianState([0, 1], np.zeros((2, 2))))


def test_moderate_dense_chart_symmetrizes_roundoff_and_preserves_covariance_to_declared_precision():
    chart = AffineCoordinates([[1, 1], [1, 1.001]], [0, 0])
    state = GaussianState([1, 2], np.eye(2))
    restored = chart.restore_state(chart.transform_state(state))
    np.testing.assert_array_equal(restored.covariance, restored.covariance.T)
    np.testing.assert_allclose(restored.mean, state.mean, rtol=1e-8, atol=0)
    np.testing.assert_allclose(restored.covariance, state.covariance, rtol=0, atol=1e-8)


def test_dense_chart_with_exact_closed_process_noise_is_accurate_or_explicitly_refused():
    Q = .01 * np.array([[1., -1.], [-1., 1.]])
    chart = AffineCoordinates([[1000, 1000], [1000, -1000]], [0, 100])
    try:
        _, _, transformed = chart.transform_dynamics(np.eye(2), [-1, 1], Q)
    except ValueError as exc:
        # BLAS rounding may leave a zero variance beside tiny nonzero cross-covariance.
        # Refusal is supported; silently repairing the declared covariance is not.
        assert "covariance" in str(exc) or "Q" in str(exc)
    else:
        restored = chart.restore_state(GaussianState([0, 100], transformed))
        np.testing.assert_allclose(restored.covariance, Q, rtol=0, atol=1e-10)
        assert np.linalg.eigvalsh(transformed).min() >= 0
        zero = np.diag(transformed) == 0
        assert np.all(transformed[zero, :] == 0) and np.all(transformed[:, zero] == 0)


def test_diagonal_unit_chart_supports_the_same_exact_closed_process_noise():
    Q = .01 * np.array([[1., -1.], [-1., 1.]])
    chart = AffineCoordinates(np.eye(2) * 1000, [0, 100])
    _, _, transformed = chart.transform_dynamics(np.eye(2), [-1, 1], Q)
    np.testing.assert_allclose(transformed, Q * 1e6, rtol=1e-12, atol=0)
    restored = chart.restore_state(GaussianState([0, 100], transformed))
    np.testing.assert_allclose(restored.covariance, Q, rtol=1e-12, atol=0)
