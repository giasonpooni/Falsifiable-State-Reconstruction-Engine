"""Conservation, explicit ambiguity and reproducibility of the synthetic benchmark."""
from dataclasses import replace
import json

import numpy as np
import pytest

from set_lcm.diagnostics import diagnose
from set_lcm.experiments import fluid_baseline as fb
from set_lcm.experiments.compare import compare, reproduction_failure
from set_lcm.experiments.provenance import REPO_ROOT
from set_lcm.measurement import balance_residuals


@pytest.mark.parametrize("field,value", [
    ("intervals", True), ("intervals", 12.0), ("intervals", 7),
    ("onset_interval", False), ("onset_interval", 3.5), ("onset_interval", 0),
    ("onset_interval", 21), ("interval_seconds", 0), ("interval_seconds", float("inf")),
    ("storage_sd_m3", -0.12), ("storage_sd_m3", 0), ("storage_sd_m3", True),
    ("flow_sd_m3_per_s", -0.025), ("flow_sd_m3_per_s", float("nan")),
    ("reference_storage_m3", float("nan")), ("reference_inflow_m3_per_s", float("inf")),
    ("weak_storage_m3", -1), ("strong_storage_m3", float("nan")),
    ("weak_rate_m3_per_s", float("nan")), ("strong_rate_m3_per_s", -1),
    ("weak_gain", float("nan")), ("strong_gain", float("inf")),
    ("alpha", 0), ("alpha", 1), ("alpha", True), ("alpha", np.nextafter(0.0, 1.0)),
    ("interval_level", 0), ("interval_level", 1),
    ("interval_level", np.nextafter(1.0, 0.0)), ("interval_level", np.nextafter(0.0, 1.0)),
])
def test_invalid_configuration_is_rejected_before_simulation(field, value):
    with pytest.raises(ValueError):
        replace(fb.DEFAULT_CONFIG, **{field: value})


def test_reference_sensitivities_may_be_signed_and_fault_amplitudes_zero():
    config = replace(fb.DEFAULT_CONFIG, reference_storage_m3=-0.08, weak_storage_m3=0.0,
                     intervals=np.int64(24), onset_interval=np.int64(6))
    case = fb.build_case("storage_step", "weak", config=config)
    assert case.magnitude == 0.0
    assert np.linalg.eigvalsh(case.record.covariance).min() > 0.0
    json.dumps(fb.asdict(config), allow_nan=False)


@pytest.mark.parametrize("argument,value", [
    ("n_eval", True), ("n_eval", 2.5), ("n_eval", 0), ("n_eval", -1),
    ("n_development", False), ("n_development", 1.0), ("n_development", 0),
    ("n_development", fb.EVALUATION_SEED_START),
])
def test_seed_counts_are_validated_before_range(argument, value):
    with pytest.raises(ValueError):
        fb.run_benchmark(**{argument: value})


def test_mean_storage_and_legitimate_flow_change_conserve_exactly():
    config = replace(fb.DEFAULT_CONFIG, interval_seconds=2.5)
    case = fb.build_case("legitimate_flow_change", config=config)
    series = balance_residuals(case.record)
    np.testing.assert_allclose(series.residual, 0.0, atol=3e-14)
    np.testing.assert_allclose(series.operator @ case.raw_intervention, 0.0, atol=3e-14)
    assert case.record.storage_support == "mean"
    assert case.record.within_interval_model == "constant_net_flow"
    assert case.record.covariance[0, config.intervals] != 0  # shared reference crosses channels
    independent = np.diag(np.r_[np.full(config.intervals, config.storage_sd_m3 ** 2),
                               np.full(2 * config.intervals, config.flow_sd_m3_per_s ** 2)])
    shared_residual = series.covariance - series.operator @ independent @ series.operator.T
    # The storage-reference component cancels; unequal flow sensitivities remain.
    expected = (config.interval_seconds * (config.reference_inflow_m3_per_s -
                                           config.reference_outflow_m3_per_s)) ** 2
    np.testing.assert_allclose(shared_residual, expected, atol=1e-16)


def test_raw_templates_preserve_the_exact_temporal_ambiguities():
    case = fb.build_case("storage_drift")
    n, onset = fb.DEFAULT_CONFIG.intervals, fb.DEFAULT_CONFIG.onset_interval
    expected_step = np.zeros(n - 1)
    expected_step[onset - 1] = 1.0
    np.testing.assert_allclose(case.hypotheses["storage_step"], expected_step, atol=1e-15)
    expected_rate = np.zeros(n - 1)
    expected_rate[onset - 1] = 0.5
    expected_rate[onset:] = 1.0
    np.testing.assert_allclose(case.hypotheses["storage_drift"], expected_rate, atol=1e-15)
    np.testing.assert_allclose(case.hypotheses["inflow_offset"], -expected_rate, atol=1e-15)
    constant = fb.build_case("constant_flow_gain")
    np.testing.assert_allclose(constant.hypotheses["inflow_gain"],
                               3 * constant.hypotheses["inflow_offset"], atol=1e-15)
    varying = np.column_stack((case.hypotheses["inflow_gain"], case.hypotheses["inflow_offset"]))
    assert np.linalg.matrix_rank(varying) == 2  # temporal variation adds information


@pytest.mark.parametrize("name,expected", [
    ("storage_step", "identified"),
    ("storage_drift", "ambiguous"),
    ("inflow_offset", "ambiguous"),
    ("constant_flow_gain", "ambiguous"),
    ("storage_common_offset", "consistent"),
    ("legitimate_flow_change", "consistent"),
    ("storage_drift_with_nuisance", "consistent"),
    ("omitted_flow_with_nuisance", "consistent"),
    ("storage_step_unrestricted_nuisance", "insufficient_evidence"),
])
@pytest.mark.parametrize("sign", [-1, 1])
def test_noiseless_controls_and_classification_follow_declared_geometry(name, expected, sign):
    case = fb.build_case(name)
    series = balance_residuals(case.record)
    residual = series.residual + sign * case.magnitude * series.operator @ case.raw_intervention
    result = diagnose(residual, series.covariance, case.hypotheses, nuisance=case.nuisance,
                      alpha=fb.DEFAULT_CONFIG.alpha)
    assert result.status == expected
    if expected == "identified":
        assert result.candidates == (case.true_hypothesis,)
    if name == "storage_drift_with_nuisance":
        assert not next(f for f in result.fits if f.name == "storage_drift").observable
    if name in ("storage_drift", "inflow_offset", "constant_flow_gain"):
        assert {"storage_drift", "inflow_offset"} <= set(result.candidates)


def test_small_run_keeps_disjoint_seeds_and_explicit_scoring_denominators():
    report = fb.run_benchmark(2, n_development=2)
    assert set(report["design"]["development_seeds"]).isdisjoint(report["design"]["evaluation_seeds"])
    for case in report["cases"].values():
        rows = case["evaluation"]["per_seed"]
        a = case["evaluation"]["aggregate"]
        assert len(rows) == a["n_records"] == 2
        assert sum(a["status_counts"].values()) == 2
        assert a["wrong_attribution_given_identified"]["denominator"] == a["status_counts"]["identified"]
        assert a["interval_coverage_given_correct_identification"]["denominator"] <= a["status_counts"]["identified"]
        assert a["instrument_fault_detection"]["denominator"] == (2 if case["instrument_fault"] else 0)
        if case["instrument_fault"]:
            assert a["instrument_fault_detection"]["count"] + a["instrument_fault_misses"] == 2
        if case["nuisance_columns"] == 0:
            assert all(r["diagnostic_rejected_no_fault"] == r["balance_only_rejected"] for r in rows)
    # Serialize strictly: missing intervals are null, never NaN or Infinity.
    json.dumps(report, allow_nan=False)


def test_generator_is_deterministic_and_markdown_renders_its_json(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    assert fb.main(first, n_eval=2, n_development=1, quiet=True) == 0
    assert fb.main(second, n_eval=2, n_development=1, quiet=True) == 0
    a = json.loads((first / "fluid_baseline.json").read_text(encoding="utf-8"))
    b = json.loads((second / "fluid_baseline.json").read_text(encoding="utf-8"))
    assert compare(a, b, rel_tol=0.0) == []
    assert (first / "fluid_baseline.md").read_text(encoding="utf-8") == fb.render(a)


@pytest.fixture(scope="module")
def full_benchmark():
    return fb.run_benchmark()


def test_default_benchmark_uses_at_least_100_independent_evaluation_records(full_benchmark):
    seeds = full_benchmark["design"]["evaluation_seeds"]
    assert len(seeds) == len(set(seeds)) >= 100
    assert {case["level"] for case in full_benchmark["cases"].values()} == {"weak", "strong"}


def test_committed_fluid_baseline_reproduces(full_benchmark):
    path = REPO_ROOT / "results" / "fluid_baseline.json"
    committed = json.loads(path.read_text(encoding="utf-8"))
    failure = reproduction_failure("fluid_baseline.json", full_benchmark, committed)
    assert failure is None, failure
    assert (path.with_suffix(".md")).read_text(encoding="utf-8") == fb.render(committed)
