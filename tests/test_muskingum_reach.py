"""The two-reach Muskingum design: the simulator, the rows, and the predictions being tested.

The design study asserted from algebra what this topology could separate. These tests pin the
simulator to the physics it claims, the rows to the relations they claim, and the experiment's
findings to the predictions they were meant to check.
"""
import json

import numpy as np
import pytest

from set_lcm.experiments import muskingum_reach as experiment
from set_lcm.experiments.compare import reproduction_failure
from set_lcm.experiments.provenance import REPO_ROOT
from set_lcm.fdi import isolability
from set_lcm.schema import ConstraintSet
from set_lcm.testbed.muskingum import (
    Fault, MuskingumConfig, constraint_rows, design_matrix, flood_wave, observe,
    reading_names, route, routing_coefficients, simulate,
)

RESULTS = REPO_ROOT / "results"
CONFIG = MuskingumConfig()


# ---------------------------------------------------------------------------
# the simulator is the physics it says it is
# ---------------------------------------------------------------------------

def test_routing_coefficients_sum_to_one():
    """Muskingum conserves the routed volume; the coefficients summing to one is that."""
    assert sum(routing_coefficients(CONFIG)) == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize("bad,match", [
    ({"K": 0.0}, "positive"), ({"x": 0.7}, r"\[0, 0.5\]"), ({"dt": 5.0}, "stability window"),
    ({"K": 1.0, "x": 0.45, "dt": 0.5}, "stability window"),
])
def test_parameters_outside_the_stability_window_are_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        MuskingumConfig(**{**{"K": 1.0, "x": 0.2, "dt": 1.0}, **bad})


def test_the_noiseless_river_satisfies_both_declared_relations():
    """If it did not, every null in this experiment would be false and everything would reject."""
    truth = simulate(CONFIG, 21)
    mean = lambda s: 0.5 * (s[1:] + s[:-1])
    readings = {"storage_1": truth.storage_1, "storage_2": truth.storage_2,
                "inflow_gauge": mean(truth.inflow), "middle_gauge": mean(truth.middle),
                "outflow_gauge": mean(truth.outflow)}
    y = np.array([readings[sensor][k] for sensor, k in reading_names(21)])
    M = design_matrix(CONFIG, 21, with_routing=True)
    scale = max(truth.storage_1.max(), truth.inflow.max())
    assert np.abs(M @ y).max() < 1e-12 * scale


def test_a_sensor_fault_leaves_the_river_alone_and_a_physical_one_does_not():
    healthy = simulate(CONFIG, 21)
    biased = simulate(CONFIG, 21, Fault("sensor_bias", 5.0, 6, ("inflow_gauge",)))
    lateral = simulate(CONFIG, 21, Fault("lateral_inflow", 5.0, 6, (), physical=True))
    np.testing.assert_array_equal(healthy.middle, biased.middle)     # the gauge lied, not the river
    assert np.any(lateral.middle > healthy.middle)                   # the river changed
    assert np.all(lateral.lateral[:6] == 0.0) and np.all(lateral.lateral[6:] == 5.0)


def test_the_ungauged_inflow_is_ungauged():
    """The point of the case: the water arrives and the inflow gauge never sees it."""
    lateral = simulate(CONFIG, 21, Fault("lateral_inflow", 5.0, 6, (), physical=True))
    readings = observe(lateral, experiment.SIGMA, np.random.default_rng(0))
    clean = 0.5 * (lateral.inflow[1:] + lateral.inflow[:-1])
    assert np.abs(readings.values["inflow_gauge"] - clean).max() < 5.0 * experiment.SIGMA["inflow_gauge"]


def test_the_hydrograph_is_deterministic_so_a_seed_changes_only_the_readings():
    a, b = simulate(CONFIG, 21), simulate(CONFIG, 21)
    np.testing.assert_array_equal(a.inflow, b.inflow)
    np.testing.assert_array_equal(flood_wave(21), a.inflow)
    first = observe(a, experiment.SIGMA, np.random.default_rng(1)).values["inflow_gauge"]
    second = observe(a, experiment.SIGMA, np.random.default_rng(2)).values["inflow_gauge"]
    assert not np.array_equal(first, second)


# ---------------------------------------------------------------------------
# the rows are the relations they claim, and the study's predictions hold on them
# ---------------------------------------------------------------------------

def test_continuity_differences_the_storages_and_routing_averages_them():
    """The whole reason routing sees a persistent storage bias and continuity cannot."""
    rows = constraint_rows(CONFIG, with_routing=True)
    np.testing.assert_allclose(rows[0][:2], [-1.0, 1.0])       # continuity 1: a difference
    np.testing.assert_allclose(rows[2][:2], [0.5, 0.5])        # routing 1: an average


def test_the_design_studys_prediction_holds_on_the_rows_that_were_built():
    """5 of 7 isolable with routing, 1 confounded pair, and that pair is the physical one."""
    faults = {
        "storage 1 sensor bias": [1, 1, 0, 0, 0, 0, 0],
        "storage 2 sensor bias": [0, 0, 1, 1, 0, 0, 0],
        "inflow gauge bias": [0, 0, 0, 0, 1, 0, 0],
        "middle gauge bias": [0, 0, 0, 0, 0, 1, 0],
        "outflow gauge bias": [0, 0, 0, 0, 0, 0, 1],
        "common flow-gauge drift": [0, 0, 0, 0, 1, 1, 1],
        "ungauged lateral inflow": [0, 0, 0, 0, 1, 0, 0],
    }
    rows = constraint_rows(CONFIG, with_routing=True)
    result = isolability(faults, np.eye(7), ConstraintSet("m", rows, np.zeros(4), "m"))
    assert result.residual_rank == 4
    assert len(result.isolable) == 5
    assert result.invisible == []
    confounded = [sorted([p.a, p.b]) for p in result.pairs if p.orthogonal_fraction == 0.0]
    assert confounded == [sorted(["inflow gauge bias", "ungauged lateral inflow"])]


def test_the_coordinated_offset_is_exactly_invisible():
    """Named because every topology has a blind spot and this one's should not be a surprise."""
    rows = constraint_rows(CONFIG, with_routing=True)
    K = CONFIG.K
    direction = np.array([K, K, K, K, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(rows @ direction, np.zeros(4))


def test_adjacent_intervals_share_a_storage_reading():
    """Why Cov(r) is not block diagonal, and why the experiment forms M R M^T rather than
    assuming the intervals are independent."""
    M = design_matrix(CONFIG, 21, with_routing=True)
    index = {name: position for position, name in enumerate(reading_names(21))}
    column = index[("storage_1", 5)]
    touching = np.flatnonzero(M[:, column])
    assert touching.size > 0
    rows_per_interval = 4
    intervals = {int(row) // rows_per_interval for row in touching}
    assert intervals == {4, 5}, intervals      # the reading ends one interval and starts the next


# ---------------------------------------------------------------------------
# the experiment's findings
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def report():
    return experiment.compute()


def test_routing_recovers_every_declared_instrument_fault_and_continuity_does_not(report):
    strong = f" @ {max(experiment.MAGNITUDE_MULTIPLES):g} sigma"
    evaluation = report["evaluation"]
    for name in experiment.INSTRUMENT_CANDIDATES:
        routed = evaluation["continuity + routing"][name + strong]
        assert routed["outcomes"].get("identified correctly", 0) == routed["n_seeds"], name
    for name in ("storage 1 sensor bias", "storage 2 sensor bias", "common flow-gauge drift"):
        bare = evaluation["continuity only"][name + strong]
        assert bare["outcomes"].get("identified correctly", 0) == 0, name


def test_an_undeclared_physical_cause_is_confidently_blamed_on_an_instrument(report):
    """Not a bug in the rule: the directions are identical and the catalogue offered no other
    answer. Declaring the physical explanation turns the confident answer into ambiguity."""
    strong = f" @ {max(experiment.MAGNITUDE_MULTIPLES):g} sigma"
    case = experiment.PHYSICAL_CANDIDATE + strong
    without = report["evaluation"]["continuity + routing"][case]
    with_it = report["evaluation"]["continuity + routing, physical candidate declared"][case]
    assert without["outcomes"].get("identified WRONG", 0) > 0.5 * without["n_seeds"]
    assert without["outcomes"].get("identified correctly", 0) == 0
    assert with_it["outcomes"].get("ambiguous, correct among them", 0) > 0.5 * with_it["n_seeds"]


def test_parameter_error_reads_as_a_fault_until_its_uncertainty_is_declared(report):
    """Stage 3b's A_var on the first topology that needed it."""
    exact = report["evaluation"]["routing with parameter error, A treated as exact"]["healthy"]
    declared = report["evaluation"]["routing with parameter error, A_var declared"]["healthy"]
    assert exact["outcomes"].get("false alarm", 0) > 0.5 * exact["n_seeds"]
    assert declared["outcomes"].get("false alarm", 0) < exact["outcomes"].get("false alarm", 0)


def test_the_healthy_river_is_quiet_when_the_model_is_right(report):
    for variant in ("continuity only", "continuity + routing"):
        healthy = report["evaluation"][variant]["healthy"]
        assert healthy["outcomes"].get("false alarm", 0) <= 0.05 * healthy["n_seeds"], variant


def test_development_and_evaluation_seeds_are_disjoint(report):
    declared = report["declared"]
    assert not set(declared["development_seeds"]) & set(declared["evaluation_seeds"])
    assert report["development"].keys() == report["evaluation"].keys()


def test_the_weak_magnitude_is_weak_enough_to_be_informative(report):
    """A benchmark whose every cell saturates measures nothing about sensitivity."""
    weak = f" @ {min(experiment.MAGNITUDE_MULTIPLES):g} sigma"
    routed = report["evaluation"]["continuity + routing"]
    rates = [routed[name + weak]["outcomes"].get("identified correctly", 0) / routed[name + weak]["n_seeds"]
             for name in experiment.INSTRUMENT_CANDIDATES]
    assert min(rates) < 0.5 and max(rates) < 1.0, rates


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def test_every_qualitative_sentence_is_a_computed_condition(report):
    assert report["claims"] and all(report["claims"].values()), \
        [k for k, v in report["claims"].items() if not v]
    broken = dict(report, claims=dict(report["claims"], **{
        "routing recovers every declared instrument fault at the strong magnitude": False}))
    with pytest.raises(RuntimeError, match="no longer true of the numbers"):
        experiment.render(broken)


def test_the_report_says_it_is_synthetic(report):
    text = experiment.render(report)
    assert "Synthetic throughout" in text and "no field validation" in text
    assert "not field validation" in json.dumps(report["limitations"])


def test_the_report_reproduces(report):
    committed = json.loads((RESULTS / "muskingum_reach.json").read_text(encoding="utf-8"))
    failure = reproduction_failure("muskingum_reach.json", report, committed)
    assert failure is None, failure
    assert experiment.render(committed) == (RESULTS / "muskingum_reach.md").read_text(encoding="utf-8")
