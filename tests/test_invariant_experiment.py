"""The invariant experiment tests equivalence, not accuracy against hidden truth."""
import json

import numpy as np
import pytest

from set_lcm.experiments import invariant_layer as experiment
from set_lcm.experiments.compare import compare, reproduction_failure
from set_lcm.experiments.provenance import REPO_ROOT


def test_model_exercises_drift_offsets_correlations_and_missing_observations():
    model = experiment.model()
    assert not np.array_equal(model["F"], np.eye(2))
    np.testing.assert_allclose(np.ones(2) @ model["F"], np.ones(2), atol=1e-15)
    assert np.linalg.norm(model["drift"]) > 0 and model["drift"].sum() == 0
    np.testing.assert_allclose(model["Q"] @ np.ones(2), 0, atol=1e-15)
    assert np.all(model["observation_offset"] != 0)
    assert model["R"][0, 1] != 0 and np.linalg.eigvalsh(model["R"]).min() > 0
    assert set(experiment.measurement_mask().sum(axis=1)) == {0, 1, 2}
    charts = experiment.state_charts()
    assert np.linalg.cond(charts["unit_scale"].matrix) > 1e5
    assert np.any(charts["datum_translation"].offset != 0)
    assert not np.allclose(charts["total_difference"].matrix @
                           charts["total_difference"].matrix.T, np.eye(2))


def test_scenarios_share_physics_and_noise_but_change_only_declared_sensor_readings():
    clean, truth = experiment.simulate(2, "clean")
    biased, biased_truth = experiment.simulate(2, "biased_gauge")
    np.testing.assert_array_equal(truth, biased_truth)
    assert np.ptp(truth.sum(axis=1)) < 1e-12
    np.testing.assert_array_equal(clean.mask, biased.mask)
    assert np.isnan(clean.measurements[~clean.mask]).all()
    delta = biased.measurements - clean.measurements
    expected = np.zeros_like(delta)
    expected[experiment.BIAS_ONSET:, 0] = experiment.GAUGE_BIAS_KG
    np.testing.assert_allclose(delta[clean.mask], expected[clean.mask], atol=1e-14)


def test_reference_does_not_call_the_invariant_filter(monkeypatch):
    record, _ = experiment.simulate(0, "clean")
    def forbidden(*args, **kwargs):
        raise AssertionError("the independent reference must not call the implementation")
    monkeypatch.setattr(experiment, "predict", forbidden)
    monkeypatch.setattr(experiment, "update", forbidden)
    trace = experiment.ordinary_reference(record)
    assert trace["means"].shape == (experiment.N_STEPS, 2)
    assert set(trace["statuses"]) == {"updated", "no_observations"}


def test_experiment_never_reads_simulated_truth(monkeypatch):
    simulate = experiment.simulate
    class ForbiddenTruth:
        def __getattribute__(self, name):
            raise AssertionError("inference or scoring attempted to read physical truth")
        def __bool__(self):
            raise AssertionError("inference or scoring attempted to read physical truth")
    def without_truth(seed, scenario):
        record, _ = simulate(seed, scenario)
        return record, ForbiddenTruth()
    monkeypatch.setattr(experiment, "simulate", without_truth)
    monkeypatch.setattr(experiment, "SEEDS", (0,))
    monkeypatch.setattr(experiment, "SCENARIOS", ("clean",))
    report = experiment.run_experiment()
    assert report["summary"]["all_equivalent"]


@pytest.fixture(scope="module")
def full_report():
    return experiment.run_experiment()


def test_all_declared_charts_match_the_independent_kf(full_report):
    assert len(full_report["records"]) == 16
    assert full_report["summary"]["compared_runs"] == 16 * 5 * 4
    assert full_report["summary"]["equivalent_runs"] == full_report["summary"]["compared_runs"]
    assert full_report["summary"]["all_equivalent"]
    for record in full_report["records"]:
        assert set(record["reference_statuses"]) == {"updated", "no_observations"}
        for outcome in record["runs"].values():
            assert outcome["statuses_match"] and outcome["dofs_match"]
            assert outcome["statistic_presence_matches"]
            for key in experiment.METRICS:
                assert np.isfinite(outcome[key]) and outcome[key] <= experiment.ABSOLUTE_TOLERANCE


def test_generator_roundtrip_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "SEEDS", (0,))
    first, second = tmp_path / "first", tmp_path / "second"
    assert experiment.main(first, quiet=True) == 0
    assert experiment.main(second, quiet=True) == 0
    a = json.loads((first / "invariant_layer.json").read_text(encoding="utf-8"))
    b = json.loads((second / "invariant_layer.json").read_text(encoding="utf-8"))
    assert compare(a, b, rel_tol=0.0) == []
    assert (first / "invariant_layer.md").read_text(encoding="utf-8") == experiment.render(a)


def test_committed_equivalence_report_reproduces(full_report):
    path = REPO_ROOT / "results" / "invariant_layer.json"
    committed = json.loads(path.read_text(encoding="utf-8"))
    failure = reproduction_failure("invariant_layer.json", full_report, committed)
    assert failure is None, failure
    assert path.with_suffix(".md").read_text(encoding="utf-8") == experiment.render(committed)
