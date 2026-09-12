"""Independent Gaussian and coordinate checks for the additive-group invariant layer.

These tests establish the affine Gaussian special case, not a nonlinear IEKF advantage.
The multistep reference conditions one joint Gaussian record without filter recursion.
"""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from set_lcm.invariant import GaussianState, invariant_error, predict, retract, update
from set_lcm.schema import Observation
from set_lcm.testbed.estimators import B, KFConfig, KalmanFilter


def _fixture():
    state = GaussianState(np.array([2.0, -1.0]), np.array([[2.0, 0.4], [0.4, 0.8]]))
    H = np.array([[1.0, 0.5], [-0.3, 1.0], [0.7, -0.2]])
    R = np.array([[0.5, 0.12, -0.03], [0.12, 0.7, 0.08], [-0.03, 0.08, 0.4]])
    y = np.array([2.3, -0.2, 1.5])
    offset = np.array([0.1, -0.2, 0.3])
    return state, y, H, R, offset


def _assert_state(actual, mean, covariance, *, rtol=2e-11, atol=2e-12):
    np.testing.assert_allclose(actual.mean, mean, rtol=rtol, atol=atol)
    np.testing.assert_allclose(actual.covariance, covariance, rtol=rtol, atol=atol)


def test_error_retraction_is_additive_and_translation_invariant():
    reference = np.array([1.5, -4.0, 8.0])
    state = np.array([-3.0, 2.0, 0.5])
    error = invariant_error(reference, state)
    np.testing.assert_array_equal(error, [-4.5, 6.0, -7.5])
    np.testing.assert_array_equal(retract(reference, error), state)
    translation = np.array([100.0, -2.0, 0.25])
    np.testing.assert_array_equal(invariant_error(reference + translation, state + translation), error)
    np.testing.assert_array_equal(retract(reference + translation, error), state + translation)


def test_update_matches_independent_information_form_conditioning():
    state, y, H, R, offset = _fixture()
    # Product of prior and likelihood in canonical Gaussian coordinates; no gain/Joseph update.
    prior_precision = np.linalg.inv(state.covariance)
    noise_precision = np.linalg.inv(R)
    posterior_precision = prior_precision + H.T @ noise_precision @ H
    posterior_covariance = np.linalg.inv(posterior_precision)
    posterior_mean = np.linalg.solve(
        posterior_precision, prior_precision @ state.mean + H.T @ noise_precision @ (y - offset))
    result = update(state, y, H, R, offset)
    _assert_state(result.posterior, posterior_mean, posterior_covariance)
    np.testing.assert_array_equal(result.prior.mean, state.mean)
    np.testing.assert_allclose(retract(result.prior.mean, result.correction), result.posterior.mean)
    np.testing.assert_allclose(result.gain @ result.innovation, result.correction)
    assert result.status == "updated" and result.dof == 3
    whitened = np.linalg.solve(np.linalg.cholesky(H @ state.covariance @ H.T + R), y - H @ state.mean - offset)
    assert result.statistic == pytest.approx(float(whitened @ whitened), rel=2e-12)


def test_multistep_filter_matches_one_joint_gaussian_conditioning():
    """Construct all states/observations from independent latent Gaussian coordinates."""
    mean0 = np.array([0.4, -1.2])
    P0 = np.array([[1.4, -0.2], [-0.2, 0.6]])
    transitions = [np.array([[1.0, 0.3], [0.0, 0.9]]), np.array([[0.8, -0.2], [0.1, 1.0]])]
    drifts = [np.array([0.2, -0.1]), np.array([-0.3, 0.4])]
    process = [np.array([[0.1, 0.02], [0.02, 0.2]]), np.array([[0.3, -0.04], [-0.04, 0.15]])]
    Hs = [np.array([[1.0, -0.2]]), np.array([[0.4, 1.0], [1.0, 0.1]]), np.array([[0.7, -0.8]])]
    Rs = [np.array([[0.4]]), np.array([[0.5, 0.1], [0.1, 0.3]]), np.array([[0.2]])]
    offsets = [np.array([0.2]), np.array([-0.1, 0.4]), np.array([0.0])]
    ys = [np.array([0.8]), np.array([-0.7, 0.9]), np.array([1.5])]
    latent_cov = np.zeros((10, 10))  # x0, w0, w1, v0, v1, v2
    for start, block in ((0, P0), (2, process[0]), (4, process[1]), (6, Rs[0]), (7, Rs[1]), (9, Rs[2])):
        latent_cov[start:start + len(block), start:start + len(block)] = block
    state_maps = [np.pad(np.eye(2), ((0, 0), (0, 8)))]
    state_means = [mean0]
    for k, F in enumerate(transitions):
        mapping = F @ state_maps[-1]
        mapping[:, 2 + 2*k:4 + 2*k] += np.eye(2)
        state_maps.append(mapping)
        state_means.append(F @ state_means[-1] + drifts[k])
    observation_maps, observation_means = [], []
    for k, start in enumerate((6, 7, 9)):
        mapping = Hs[k] @ state_maps[k]
        mapping[:, start:start + len(Rs[k])] += np.eye(len(Rs[k]))
        observation_maps.append(mapping)
        observation_means.append(Hs[k] @ state_means[k] + offsets[k])
    C = np.vstack(observation_maps)
    covariance_y = C @ latent_cov @ C.T
    cross = state_maps[-1] @ latent_cov @ C.T
    difference = np.concatenate(ys) - np.concatenate(observation_means)
    expected_mean = state_means[-1] + cross @ np.linalg.solve(covariance_y, difference)
    expected_cov = state_maps[-1] @ latent_cov @ state_maps[-1].T - cross @ np.linalg.solve(covariance_y, cross.T)
    state = GaussianState(mean0, P0)
    for k in range(3):
        if k:
            state = predict(state, transitions[k-1], drifts[k-1], process[k-1])
        state = update(state, ys[k], Hs[k], Rs[k], offsets[k]).posterior
    _assert_state(state, expected_mean, expected_cov)


def test_same_public_information_reproduces_existing_linear_kf():
    u = np.array([0.2, -0.4, 0.1, 0.0])
    cfg = KFConfig(sigma_w=0.15)
    legacy = KalmanFilter((2.0, 4.0), 0.8, 0.5, u, cfg)
    state = GaussianState([2.0, 4.0], np.eye(2) * 0.8**2)
    ys = [[2.1, 3.6], [1.9, 3.9], [2.3, 3.7], [2.0, 4.1]]
    masks = [[True, True], [True, False], [False, False], [False, True]]
    for j, (y, mask) in enumerate(zip(ys, masks, strict=True)):
        R = np.array([[0.2, 0.04], [0.04, 0.3]]) * (1 + 0.1*j)
        legacy.ingest(Observation(j * 0.5, j * 0.5, np.array(y), R, np.array(mask), ("a", "b")), j)
        if j:
            state = predict(state, np.eye(2), B * u[j-1] * 0.5, np.eye(2) * cfg.sigma_w**2)
        state = update(state, y, np.eye(2), R, mask=np.array(mask)).posterior
        _assert_state(state, *legacy.report(j))


@pytest.mark.parametrize("T,origin", [
    (np.array([[1.2, 0.3], [-0.4, 0.8]]), np.array([8.0, -3.0])),
    (np.diag([1e5, 1e-4]), np.array([0.0, 0.0])),
    (np.array([[0.0, -1.0], [1.0, 0.0]]), np.array([-2.0, 5.0])),
])
def test_affine_state_basis_units_and_origin_preserve_prediction_and_update(T, origin):
    state, y, H, R, offset = _fixture()
    F = np.array([[1.0, 0.2], [-0.1, 0.9]])
    drift, Q = np.array([0.3, -0.4]), np.array([[0.2, 0.02], [0.02, 0.1]])
    inverse = np.linalg.inv(T)
    transformed_F = T @ F @ inverse
    transformed = GaussianState(T @ state.mean + origin, T @ state.covariance @ T.T)
    a = predict(state, F, drift, Q)
    b = predict(transformed, transformed_F, T @ drift + origin - transformed_F @ origin, T @ Q @ T.T)
    _assert_state(b, T @ a.mean + origin, T @ a.covariance @ T.T, rtol=2e-10, atol=2e-10)
    plain = update(a, y, H, R, offset)
    changed_H = H @ inverse
    changed = update(b, y, changed_H, R, offset - changed_H @ origin)
    _assert_state(changed.posterior, T @ plain.posterior.mean + origin,
                  T @ plain.posterior.covariance @ T.T, rtol=3e-10, atol=3e-10)
    # Compare in original units too: absolute tolerances in tiny transformed units
    # could otherwise hide a materially incorrect small covariance component.
    np.testing.assert_allclose(inverse @ changed.posterior.covariance @ inverse.T,
                               plain.posterior.covariance, rtol=3e-10, atol=3e-12)
    np.testing.assert_allclose(inverse @ (changed.posterior.mean - origin), plain.posterior.mean,
                               rtol=3e-10, atol=3e-12)
    assert changed.statistic == pytest.approx(plain.statistic, rel=2e-10)
    np.testing.assert_allclose(changed.correction, T @ plain.correction, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(invariant_error(transformed.mean, b.mean), T @ invariant_error(state.mean, a.mean),
                               rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize("M", [
    np.array([[1.0, 0.2, -0.1], [0.3, 1.4, 0.1], [0.0, -0.2, 0.8]]),
    np.eye(3)[[2, 0, 1]],
    np.diag([1e-5, 1e4, -2.0]),
])
def test_invertible_measurement_changes_preserve_posterior_and_nis(M):
    state, y, H, R, offset = _fixture()
    reference = update(state, y, H, R, offset)
    transformed = update(state, M @ y, M @ H, M @ R @ M.T, M @ offset)
    _assert_state(transformed.posterior, reference.posterior.mean, reference.posterior.covariance,
                  rtol=2e-10, atol=2e-10)
    assert transformed.statistic == pytest.approx(reference.statistic, rel=2e-10)
    np.testing.assert_allclose(transformed.innovation, M @ reference.innovation, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(transformed.gain @ M, reference.gain, rtol=2e-10, atol=2e-10)


def test_mask_selects_correlated_covariance_submatrix_and_is_permutation_equivariant():
    state, y, H, R, offset = _fixture()
    mask = np.array([True, False, True])
    masked = update(state, y, H, R, offset, mask)
    explicit = update(state, y[mask], H[mask], R[np.ix_(mask, mask)], offset[mask])
    _assert_state(masked.posterior, explicit.posterior.mean, explicit.posterior.covariance)
    assert masked.statistic == pytest.approx(explicit.statistic) and masked.dof == 2
    assert masked.observed_indices == (0, 2)
    order = np.array([2, 0, 1])
    permuted = update(state, y[order], H[order], R[np.ix_(order, order)], offset[order], mask[order])
    _assert_state(permuted.posterior, masked.posterior.mean, masked.posterior.covariance)
    assert permuted.statistic == pytest.approx(masked.statistic)
    diagonalized = update(state, y[mask], H[mask], np.diag(np.diag(R)[mask]), offset[mask])
    assert np.linalg.norm(diagonalized.posterior.mean - masked.posterior.mean) > 1e-3


def test_psd_prior_and_noise_preserve_deterministic_state_direction():
    state = GaussianState([3.0, 1.0], [[0.0, 0.0], [0.0, 2.0]])
    predicted = predict(state, np.eye(2), [1.0, 0.0], [[0.0, 0.0], [0.0, 0.5]])
    result = update(predicted, [4.5, 2.0], np.eye(2), np.diag([1.0, 0.0]))
    _assert_state(result.posterior, [4.0, 2.0], np.zeros((2, 2)))
    assert result.dof == 2 and result.statistic == pytest.approx(0.25 + 1/2.5)


def test_non_axis_aligned_singular_prior_conditions_as_one_latent_variable():
    # x = [1, -2] + [1, 2] * a, a ~ N(0, 3), observed y = x_2 + N(0, 4).
    direction = np.array([1.0, 2.0])
    mean = np.array([1.0, -2.0])
    state = GaussianState(mean, 3 * np.outer(direction, direction))
    result = update(state, [2.0], [[0.0, 1.0]], [[4.0]])
    latent_variance = 1 / (1/3 + 2**2/4)
    latent_mean = latent_variance * (2/4) * (2 - mean[1])
    _assert_state(result.posterior, mean + direction * latent_mean,
                  latent_variance * np.outer(direction, direction))
    assert np.linalg.norm(result.posterior.covariance @ np.array([2.0, -1.0])) < 1e-12


def test_dependent_exact_observations_refuse_singular_innovation_with_positive_diagonal():
    state = GaussianState([0.0, 0.0], np.eye(2))
    with pytest.raises(ValueError):
        update(state, [1.0, 1.0], [[1.0, 0.0], [1.0, 0.0]], np.zeros((2, 2)))


def test_three_noiseless_observations_of_two_states_cannot_supply_three_innovation_dof():
    # Each variance is positive and rounding can let Cholesky succeed; rank is still at most two.
    H = np.random.default_rng(8).normal(size=(3, 2))
    state = GaussianState(np.zeros(2), np.eye(2))
    with pytest.raises(ValueError):
        update(state, np.zeros(3), H, np.zeros((3, 3)))


@pytest.mark.parametrize("mask", [None, np.array([True, True])])
def test_singular_innovation_covariance_is_refused_even_for_consistent_values(mask):
    state = GaussianState([1.0, 2.0], np.zeros((2, 2)))
    with pytest.raises(ValueError):
        update(state, [1.0, 2.0], np.eye(2), np.diag([1.0, 0.0]), mask=mask)


@pytest.mark.parametrize("empty_shape", [True, False])
def test_no_observations_preserve_state_without_a_statistic(empty_shape):
    state, y, H, R, offset = _fixture()
    if empty_shape:
        result = update(state, [], np.empty((0, 2)), np.empty((0, 0)))
    else:
        result = update(state, y, H, R, offset, np.zeros(3, dtype=bool))
    _assert_state(result.posterior, state.mean, state.covariance)
    assert result.status == "no_observations" and result.statistic is None and result.dof == 0
    assert result.observed_indices == ()
    assert result.innovation.shape == (0,) and result.innovation_covariance.shape == (0, 0)
    assert result.gain.shape == (2, 0)
    np.testing.assert_array_equal(result.correction, np.zeros(2))


def test_independent_gaussian_draws_calibrate_innovation_and_state_intervals():
    state = GaussianState([0.5, -0.2], [[0.8, 0.25], [0.25, 1.2]])
    H, R = np.array([[1.0, 0.3], [-0.4, 0.9]]), np.array([[0.4, -0.1], [-0.1, 0.6]])
    rng = np.random.default_rng(9864)
    count = 1600
    truths = state.mean + rng.standard_normal((count, 2)) @ np.linalg.cholesky(state.covariance).T
    ys = truths @ H.T + rng.standard_normal((count, 2)) @ np.linalg.cholesky(R).T
    statistics, errors = [], []
    for truth, y in zip(truths, ys, strict=True):
        result = update(state, y, H, R)
        statistics.append(result.statistic)
        errors.append((result.posterior.mean - truth) / np.sqrt(np.diag(result.posterior.covariance)))
    # For chi-square_2 the exact .95 quantile is -2 log(.05).
    rejection = np.mean(np.asarray(statistics) > -2*np.log(0.05))
    assert 0.03 < rejection < 0.07
    marginal_coverage = np.mean(np.abs(errors) <= 1.959963984540054, axis=0)
    assert np.all((0.93 < marginal_coverage) & (marginal_coverage < 0.97))


def test_inputs_are_defensively_copied_and_all_exposed_arrays_are_strongly_immutable():
    mean, covariance = np.array([1.0, 2.0]), np.eye(2)
    state = GaussianState(mean, covariance)
    mean[:] = -8
    covariance[:] = 0
    _assert_state(state, [1.0, 2.0], np.eye(2))
    result = update(state, [1.2, 1.8], np.eye(2), np.eye(2))
    arrays = (state.mean, state.covariance, result.posterior.mean, result.posterior.covariance,
              result.innovation, result.innovation_covariance, result.gain, result.correction,
              invariant_error([1.0], [2.0]), retract([1.0], [2.0]))
    for array in arrays:
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array.flat[0] = 9.0
    with pytest.raises(FrozenInstanceError):
        state.mean = np.zeros(2)
    with pytest.raises(FrozenInstanceError):
        result.statistic = 0.0


@pytest.mark.parametrize("mean,covariance", [
    ([], np.empty((0, 0))), ([[1.0]], [[1.0]]), ([1.0], [1.0]), ([1.0], np.eye(2)),
    ([np.nan], [[1.0]]), ([1.0], [[np.inf]]), ([1.0j], [[1.0]]), ([1.0], [[1.0+0j]]),
    ([1.0], [[-1e-30]]), ([1.0, 2.0], [[1.0, 0.3], [0.0, 1.0]]),
    ([1.0, 2.0], [[1.0, 2.0], [2.0, 1.0]]),
    ([1.0, 2.0], [[1e20, 1.2], [1.2, 1e-20]]),
])
def test_malformed_states_are_rejected(mean, covariance):
    with pytest.raises(ValueError):
        GaussianState(mean, covariance)


@pytest.mark.parametrize("arguments", [
    ([1.0], [1.0, 2.0]), ([[1.0]], [2.0]), ([np.nan], [1.0]), ([1j], [1.0]),
])
@pytest.mark.parametrize("operation", [invariant_error, retract])
def test_malformed_group_coordinates_are_rejected(operation, arguments):
    with pytest.raises(ValueError):
        operation(*arguments)


@pytest.mark.parametrize("field,value", [
    ("F", [1.0, 2.0]), ("F", np.ones((3, 2))), ("F", np.eye(2, dtype=complex)),
    ("drift", [[0.0, 0.0]]), ("drift", [0.0, np.nan]), ("drift", [0.0, 1j]),
    ("Q", np.diag([1.0, -1.0])), ("Q", np.ones((1, 1))),
    ("Q", [[1.0, 0.4], [0.0, 1.0]]), ("Q", np.eye(2, dtype=complex)),
])
def test_malformed_prediction_is_rejected(field, value):
    arguments = dict(F=np.eye(2), drift=np.zeros(2), Q=np.eye(2))
    arguments[field] = value
    with pytest.raises(ValueError):
        predict(GaussianState([0.0, 0.0], np.eye(2)), **arguments)


@pytest.mark.parametrize("field,value", [
    ("measurement", [[1.0, 2.0, 3.0]]), ("measurement", [1.0, 2.0]),
    ("measurement", [1.0, np.nan, 3.0]), ("measurement", [1.0, 2j, 3.0]),
    ("H", np.ones((3, 3))), ("H", np.ones(3)), ("H", np.ones((3, 2), dtype=complex)),
    ("R", np.eye(2)), ("R", np.diag([0.2, -0.1, 0.3])), ("R", np.eye(3, dtype=complex)),
    ("R", np.array([[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])),
    ("offset", [0.0]), ("offset", [0.0, 0.0, np.inf]),
    ("mask", [1, 0, 1]), ("mask", [True]), ("mask", [[True, True, False]]),
])
def test_malformed_update_is_rejected(field, value):
    state, y, H, R, offset = _fixture()
    arguments = dict(measurement=y, H=H, R=R, offset=offset)
    arguments[field] = value
    with pytest.raises(ValueError):
        update(state, **arguments)


def test_mask_allows_missing_nan_but_does_not_hide_invalid_noise_or_infinite_values():
    state, y, H, R, offset = _fixture()
    mask = np.array([True, False, True])
    R[1, 1] = -1.0
    with pytest.raises(ValueError):
        update(state, y, H, R, offset, mask)
    y[1] = np.nan
    missing = update(state, y, H, np.eye(3), offset, mask)
    explicit = update(state, y[mask], H[mask], np.eye(2), offset[mask])
    _assert_state(missing.posterior, explicit.posterior.mean, explicit.posterior.covariance)
    y[1] = np.inf
    with pytest.raises(ValueError):
        update(state, y, H, np.eye(3), offset, mask)


def test_negative_noise_cannot_be_hidden_by_positive_prior_covariance():
    state = GaussianState([0.0], [[10.0]])
    with pytest.raises(ValueError):
        update(state, [0.0], [[1.0]], [[-1.0]])


def test_unrepresentable_results_refuse_instead_of_returning_infinity():
    state = GaussianState([1e200], [[1.0]])
    with pytest.raises(ValueError):
        predict(state, [[1e200]], [0.0], [[0.0]])
    with pytest.raises(ValueError):
        update(GaussianState([0.0], [[1.0]]), [1e200], [[1.0]], [[1.0]])
