"""Analytic measurement balances and uncertainty propagation through the same raw H."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from set_lcm.measurement import BalanceRecord, balance_residuals


def _arguments(storage_support="instant", flow_support="mean"):
    edges = np.array([0.0, 2.0, 5.0, 9.0])
    flows = np.array([[5.0, 1.0], [1.0, 3.0], [4.0, 1.0]])
    boundaries = np.array([10.0, 18.0, 12.0, 24.0])
    storage = boundaries if storage_support == "instant" else 0.5 * (boundaries[:-1] + boundaries[1:])
    if flow_support == "total":
        flows = flows * np.diff(edges)[:, None]
    return dict(edges=edges, storage=storage, flows=flows, flow_signs=np.array([1.0, -1.0]),
                covariance=np.ones(storage.size + flows.size), storage_support=storage_support,
                flow_support=flow_support, volume_unit="m3",
                flow_unit="m3/s" if flow_support == "mean" else "m3",
                within_interval_model=None if storage_support == "instant" else "constant_net_flow")


def test_instant_storage_uses_exact_integrals_over_unequal_intervals():
    record = BalanceRecord(**_arguments())
    result = balance_residuals(record)
    expected_H = np.array([
        [-1, 1, 0, 0, -2, 2, 0, 0, 0, 0],
        [0, -1, 1, 0, 0, 0, -3, 3, 0, 0],
        [0, 0, -1, 1, 0, 0, 0, 0, -4, 4],
    ], dtype=float)
    np.testing.assert_array_equal(result.operator, expected_H)
    np.testing.assert_array_equal(result.residual, np.zeros(3))
    np.testing.assert_array_equal(result.times, [2.0, 5.0, 9.0])
    np.testing.assert_array_equal(result.covariance, expected_H @ expected_H.T)


def test_mean_storage_alignment_uses_both_adjacent_intervals():
    record = BalanceRecord(**_arguments("mean"))
    result = balance_residuals(record)
    expected_H = np.array([
        [-1, 1, 0, -1, 1, -1.5, 1.5, 0, 0],
        [0, -1, 1, 0, 0, -1.5, 1.5, -2, 2],
    ])
    np.testing.assert_array_equal(result.operator, expected_H)
    np.testing.assert_array_equal(result.residual, np.zeros(2))
    np.testing.assert_array_equal(result.times, [3.5, 7.0])
    # Negative control: treating successive means as boundary readings invents imbalance.
    net_flow = record.flows @ record.flow_signs
    naive = np.diff(record.storage) - np.diff(record.edges)[:-1] * net_flow[:-1]
    np.testing.assert_array_equal(naive, [-7.0, 9.0])
    assert np.linalg.norm(naive) > 0.0


@pytest.mark.parametrize("support", ["instant", "mean"])
def test_total_flow_and_mean_flow_agree_after_explicit_covariance_conversion(support):
    means = _arguments(support)
    totals = _arguments(support, "total")
    # Unit conversion acts on uncertainty as well as the readings.
    scale = np.concatenate((np.ones(means["storage"].size), np.repeat(np.diff(means["edges"]), 2)))
    totals["covariance"] = means["covariance"] * scale ** 2
    a = balance_residuals(BalanceRecord(**means))
    b = balance_residuals(BalanceRecord(**totals))
    np.testing.assert_array_equal(a.residual, b.residual)
    np.testing.assert_allclose(a.covariance, b.covariance, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(a.operator, b.operator * scale)


@pytest.mark.parametrize("support", ["instant", "mean"])
def test_cumulative_balances_sum_operators_before_both_calculations(support):
    args = _arguments(support)
    args["storage"] = args["storage"] + np.arange(args["storage"].size) ** 2
    record = BalanceRecord(**args)
    local = balance_residuals(record)
    cumulative = balance_residuals(record, cumulative=True)
    L = np.tril(np.ones((local.residual.size, local.residual.size)))
    np.testing.assert_array_equal(cumulative.operator, L @ local.operator)
    np.testing.assert_array_equal(cumulative.residual, L @ local.residual)
    np.testing.assert_array_equal(cumulative.covariance, L @ local.covariance @ L.T)
    np.testing.assert_array_equal(cumulative.times, local.times)
    # Every cumulative row retains the shared first storage measurement only once.
    np.testing.assert_array_equal(cumulative.operator[:, 0], -np.ones(local.residual.size))
    np.testing.assert_array_equal(cumulative.operator[-1, 1:record.storage.size - 1], 0.0)


def test_shared_initial_storage_generates_offdiagonal_cumulative_covariance():
    args = _arguments()
    args["covariance"] = np.array([4.0, 9.0, 16.0, 25.0] + [0.0] * 6)
    record = BalanceRecord(**args)
    local = balance_residuals(record)
    cumulative = balance_residuals(record, cumulative=True)
    np.testing.assert_array_equal(local.covariance, [[13, -9, 0], [-9, 25, -16], [0, -16, 41]])
    np.testing.assert_array_equal(cumulative.covariance, [[13, 4, 4], [4, 20, 4], [4, 4, 29]])


def test_full_joint_covariance_matches_gaussian_matrix_simulation():
    args = _arguments()
    variances = np.array([4.0, 9.0, 16.0, 25.0] + [0.01] * 6)
    shared = np.array([2.0, 1.0, -1.0, 0.5, 0.2, 0.0, -0.1, 0.0, 0.3, 0.0])
    args["covariance"] = np.diag(variances) + np.outer(shared, shared)
    record = BalanceRecord(**args)
    result = balance_residuals(record, cumulative=True)
    rng = np.random.default_rng(2912)
    noise = rng.normal(size=(60_000, variances.size)) * np.sqrt(variances)
    noise += rng.normal(size=(60_000, 1)) * shared
    raw = np.concatenate((record.storage, record.flows.ravel()))
    residual_draws = (raw + noise) @ result.operator.T
    np.testing.assert_allclose(result.covariance,
                               result.operator @ args["covariance"] @ result.operator.T)
    # Tests the full joint matrix, not independent marginal interval widths.
    standardized = np.linalg.solve(np.linalg.cholesky(result.covariance), residual_draws.T)
    np.testing.assert_allclose(np.cov(standardized), np.eye(3), rtol=0, atol=0.025)
    np.testing.assert_allclose(np.mean(standardized, axis=1), np.zeros(3), rtol=0, atol=0.02)


@pytest.mark.parametrize("support", ["instant", "mean"])
def test_shared_storage_calibration_error_cancels_without_discarding_covariance(support):
    args = _arguments(support)
    independent = balance_residuals(BalanceRecord(**args), cumulative=True)
    common = np.zeros(args["covariance"].size)
    common[:args["storage"].size] = 7.0
    args["covariance"] = np.diag(args["covariance"]) + np.outer(common, common)
    args["storage"] = args["storage"] + 13.0
    shared = balance_residuals(BalanceRecord(**args), cumulative=True)
    np.testing.assert_array_equal(shared.operator @ common, np.zeros(shared.residual.size))
    np.testing.assert_array_equal(shared.residual, independent.residual)
    np.testing.assert_array_equal(shared.covariance, independent.covariance)


def test_diagonal_variances_and_full_joint_form_give_the_same_covariance():
    args = _arguments("mean")
    args["covariance"] = np.arange(1.0, args["covariance"].size + 1)
    diagonal_record = BalanceRecord(**args)
    args["covariance"] = np.diag(args["covariance"])
    joint_record = BalanceRecord(**args)
    assert diagonal_record.covariance.ndim == 1  # diagonal input stays a vector
    for cumulative in (False, True):
        diagonal = balance_residuals(diagonal_record, cumulative=cumulative)
        joint = balance_residuals(joint_record, cumulative=cumulative)
        np.testing.assert_array_equal(diagonal.residual, joint.residual)
        np.testing.assert_array_equal(diagonal.covariance, joint.covariance)


def test_multiple_evidence_ids_are_unioned_only_for_nonzero_raw_contributors():
    args = _arguments()
    args["evidence_ids"] = (
        ("s0", "reference"), "s1", ("s2", "reference"), "s3",
        ("q0in", "shared", "q0in"), "q0out", ("q1in", "shared"), "q1out", "q2in", "q2out",
    )
    record = BalanceRecord(**args)
    assert record.evidence_ids[1] == ("s1",)
    local = balance_residuals(record)
    assert local.evidence_ids[0] == ("s0", "reference", "s1", "q0in", "shared", "q0out")
    cumulative = balance_residuals(record, cumulative=True)
    assert cumulative.evidence_ids[1] == (
        "s0", "reference", "s2", "q0in", "shared", "q0out", "q1in", "q1out",
    )
    assert "s1" not in cumulative.evidence_ids[1]  # its coefficients canceled
    assert balance_residuals(BalanceRecord(**_arguments())).evidence_ids == ((), (), ())


def test_record_and_result_have_independent_immutable_numerical_copies():
    args = _arguments()
    record = BalanceRecord(**args)
    result = balance_residuals(record)
    before_storage = record.storage.copy()
    args["storage"][0] = 999.0
    args["covariance"][0] = 999.0
    np.testing.assert_array_equal(record.storage, before_storage)
    assert record.covariance[0] == 1.0
    for owner, names in ((record, ("edges", "storage", "flows", "flow_signs", "covariance")),
                         (result, ("residual", "covariance", "operator", "times"))):
        for name in names:
            array = getattr(owner, name)
            with pytest.raises(ValueError):
                array.flat[0] = 0.0
            with pytest.raises(ValueError):
                array.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        record.storage_support = "mean"


@pytest.mark.parametrize("field,value", [
    ("edges", [0.0, 2.0, 2.0, 9.0]), ("edges", [0.0, 5.0, 2.0, 9.0]),
    ("edges", [0.0, np.nan, 5.0, 9.0]), ("edges", [-1e308, 1e308]),
    ("storage", [1.0, 2.0]), ("storage", [[10.0, 18.0, 12.0, 24.0]]),
    ("storage", [10.0, np.nan, 12.0, 24.0]),
    ("flows", [[1.0, 2.0]]), ("flows", [1.0, 2.0, 3.0]), ("flows", np.empty((3, 0))),
    ("flows", [[np.inf, 1.0], [1.0, 3.0], [4.0, 1.0]]),
    ("flow_signs", [1.0]), ("flow_signs", [1.0, 0.0]),
    ("storage_support", "unknown"), ("flow_support", "instant"),
    ("within_interval_model", "linear_net_flow"), ("volume_unit", "L"),
    ("flow_unit", "m3/h"), ("covariance", None), ("covariance", np.ones(9)),
    ("covariance", np.full(10, np.nan)), ("covariance", np.full(10, -1.0)),
    ("evidence_ids", "one string"), ("evidence_ids", ("only_one",)),
    ("evidence_ids", (1,) * 10),
])
def test_invalid_or_incomplete_measurements_are_refused(field, value):
    args = _arguments()
    args[field] = value
    with pytest.raises(ValueError):
        BalanceRecord(**args)


@pytest.mark.parametrize("field", ["edges", "storage", "flows", "flow_signs", "covariance"])
def test_complex_inputs_are_refused_without_discarding_imaginary_components(field):
    args = _arguments()
    args[field] = args[field].astype(complex)
    args[field].flat[0] += 1j
    with pytest.raises(ValueError, match="real"):
        BalanceRecord(**args)


def test_complex_full_covariance_is_refused_even_if_its_real_part_is_valid():
    args = _arguments()
    args["covariance"] = np.eye(10, dtype=complex)
    args["covariance"][0, 1] = 1j
    args["covariance"][1, 0] = -1j
    with pytest.raises(ValueError, match="real"):
        BalanceRecord(**args)


@pytest.mark.parametrize("assumption", [None, "linear_net_flow", ""])
@pytest.mark.parametrize("flow_support", ["mean", "total"])
def test_mean_storage_requires_the_implemented_within_interval_assumption(assumption, flow_support):
    args = _arguments("mean", flow_support)
    args["within_interval_model"] = assumption
    with pytest.raises(ValueError, match="constant_net_flow"):
        BalanceRecord(**args)


def test_units_and_covariance_must_be_explicit_and_flow_support_must_agree():
    for missing in ("volume_unit", "flow_unit", "covariance"):
        args = _arguments()
        del args[missing]
        with pytest.raises(TypeError):
            BalanceRecord(**args)
    args = _arguments(flow_support="total")
    args["flow_unit"] = "m3/s"
    with pytest.raises(ValueError, match="supported units"):
        BalanceRecord(**args)


@pytest.mark.parametrize("kind", ["negative_small_variance", "small_indefinite_block", "small_asymmetry",
                                   "zero_variance_cross_covariance"])
def test_joint_covariance_validation_is_not_hidden_by_mixed_component_scales(kind):
    args = _arguments()
    covariance = np.eye(10)
    covariance[0, 0] = 1e12
    if kind == "negative_small_variance":
        covariance[1, 1] = -1e-3
    elif kind == "small_indefinite_block":
        covariance[1:3, 1:3] = [[1e-9, 2e-9], [2e-9, 1e-9]]
    elif kind == "small_asymmetry":
        covariance[1:3, 1:3] = [[1e-15, 5e-16], [0.0, 1e-15]]
    else:
        covariance[1, 1] = 0.0
        covariance[1, 2] = covariance[2, 1] = 1e-30
    args["covariance"] = covariance
    with pytest.raises(ValueError):
        BalanceRecord(**args)


def test_joint_covariance_accepts_shared_factor_psd_with_exact_zero_components():
    args = _arguments()
    loadings = np.zeros(10)
    loadings[:4] = [1e6, 1e-6, -1e-3, 2.0]
    args["covariance"] = np.outer(loadings, loadings)
    record = BalanceRecord(**args)
    result = balance_residuals(record)
    projected = result.operator @ loadings
    np.testing.assert_allclose(result.covariance, np.outer(projected, projected), rtol=1e-12, atol=1e-12)
    zero = _arguments()
    zero["covariance"] = np.zeros((10, 10))
    np.testing.assert_array_equal(balance_residuals(BalanceRecord(**zero)).covariance, np.zeros((3, 3)))


def test_single_interval_instant_record_works_but_one_mean_is_not_a_balance():
    args = dict(edges=[0.0, 2.0], storage=[10.0, 16.0], flows=[[3.0]], flow_signs=[1.0],
                covariance=[1.0, 1.0, 0.25], storage_support="instant", volume_unit="m3", flow_unit="m3/s")
    result = balance_residuals(BalanceRecord(**args))
    np.testing.assert_array_equal(result.residual, [0.0])
    np.testing.assert_array_equal(result.covariance, [[3.0]])
    args.update(storage=[13.0], covariance=[1.0, 0.25], storage_support="mean",
                within_interval_model="constant_net_flow")
    with pytest.raises(ValueError, match="at least two intervals"):
        BalanceRecord(**args)


def test_overflow_fails_instead_of_returning_nonfinite_residuals():
    args = _arguments()
    args["flows"] = np.full((3, 2), 1e308)
    args["flow_signs"] = [1.0, 1.0]
    with pytest.raises(ValueError, match="finite"):
        balance_residuals(BalanceRecord(**args))


def test_balance_residuals_refuses_wrong_object_or_nonboolean_cumulative():
    with pytest.raises(TypeError, match="BalanceRecord"):
        balance_residuals({})
    with pytest.raises(ValueError, match="boolean"):
        balance_residuals(BalanceRecord(**_arguments()), cumulative="yes")
