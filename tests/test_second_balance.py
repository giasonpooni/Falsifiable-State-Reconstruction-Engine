"""The second-balance design study: what it computes, and what it must not be read as.

No evidence, no estimator and no record is involved. These tests pin the study's structural
findings to the algebra they come from, and pin the report to refusing its own prose when a
finding stops holding.
"""
import json

import numpy as np
import pytest

from set_lcm.experiments import second_balance as study
from set_lcm.experiments.compare import reproduction_failure
from set_lcm.experiments.provenance import REPO_ROOT
from set_lcm.fdi import isolability
from set_lcm.schema import ConstraintSet

RESULTS = REPO_ROOT / "results"


@pytest.fixture(scope="module")
def report():
    return study.compute()


def variants(report):
    return {t["key"]: {v["variant"]: v for v in t["variants"]} for t in report["topologies"]}


# ---------------------------------------------------------------------------
# the premise
# ---------------------------------------------------------------------------

def test_the_repository_today_isolates_nothing_and_that_is_the_reason_for_the_study(report):
    today = variants(report)["ridgway_today"]["one closure"]
    assert today["residual_rank"] == 1
    assert today["isolable"] == []
    # a scalar residual makes every visible pair collinear, so nothing is separated
    assert all(pair for pair in today["confounded_pairs"])


def test_conservation_alone_isolates_almost_nothing_in_either_candidate(report):
    v = variants(report)
    assert v["muskingum_two_reach"]["continuity only"]["n_isolable"] <= 1
    assert v["cooling_loop_mass_energy"]["mass + energy only"]["n_isolable"] <= 1


def test_the_constitutive_row_is_what_separates_sensors(report):
    """The study's main result: a second conservation row is not the same as information."""
    v = variants(report)
    for key, bare, full in (("muskingum_two_reach", "continuity only",
                             "continuity + declared routing"),
                            ("cooling_loop_mass_energy", "mass + energy only",
                             "mass + energy + declared duty")):
        assert v[key][full]["n_isolable"] > v[key][bare]["n_isolable"], key
        assert len(v[key][full]["confounded_pairs"]) < len(v[key][bare]["confounded_pairs"]), key


def test_the_river_isolates_more_and_its_residual_confound_is_physical(report):
    v = variants(report)
    river = v["muskingum_two_reach"]["continuity + declared routing"]
    loop = v["cooling_loop_mass_energy"]["mass + energy + declared duty"]
    assert river["n_isolable"] > loop["n_isolable"]
    # what the river cannot separate is a physical change against an instrument, which no
    # topology fixes; what the loop cannot separate is two instruments, which one might
    assert [sorted(p) for p in river["confounded_pairs"]] == [
        sorted(["inflow gauge bias", "ungauged lateral inflow to reach 1"])]
    assert all("bias" in p[0] and "bias" in p[1] for p in loop["confounded_pairs"])


def test_routing_makes_the_common_flow_drift_visible(report):
    v = variants(report)["muskingum_two_reach"]
    assert "common drift of all three flow gauges" in v["continuity only"]["invisible"]
    assert "common drift of all three flow gauges" not in v["continuity + declared routing"]["invisible"]
    assert "common drift of all three flow gauges" in v["continuity + declared routing"]["isolable"]


# ---------------------------------------------------------------------------
# the algebra behind it, recomputed independently of the experiment module
# ---------------------------------------------------------------------------

def test_the_muskingum_result_is_the_algebra_and_not_the_bookkeeping():
    """Rebuilt here from the relation itself, so a typo in the module cannot pass unnoticed."""
    k, x = study.MUSKINGUM_K, study.MUSKINGUM_X
    rows = np.array([[1, 0, -1, 1, 0], [0, 1, 0, -1, 1],
                     [1, 0, -k * x, -k * (1 - x), 0], [0, 1, 0, -k * x, -k * (1 - x)]], float)
    cs = ConstraintSet("check", rows, np.zeros(4), "continuity and routing")
    faults = study._muskingum().faults
    result = isolability(faults, np.eye(5), cs)
    assert result.residual_rank == 4
    assert len(result.isolable) == 5
    # continuity alone cannot reach rank 4, whatever the covariance
    assert np.linalg.matrix_rank(rows[:2]) == 2


def test_one_energy_equation_gives_one_residual_direction():
    """Why mass + energy alone confounds every energy-side fault: they share a row."""
    topology = study._cooling_loop()
    rows, _ = topology.variants["mass + energy only"]
    energy_side = ("inlet temperature bias", "outlet temperature bias",
                   "stored-energy sensor bias", "heat-duty error")
    residuals = np.array([rows @ np.array(topology.faults[name], float) for name in energy_side])
    assert np.allclose(residuals[:, 0], 0.0)          # none of them moves the mass row
    assert np.linalg.matrix_rank(residuals) == 1      # so all four share one direction


def test_an_unrestricted_signed_amplitude_makes_a_direction_and_its_negation_one_line():
    """The study says a sign convention cannot rescue a confounded pair; this is why."""
    cs = ConstraintSet("one row", np.array([[1.0, -1.0]]), np.zeros(1), "a closure")
    result = isolability({"up": [1.0, 0.0], "down": [-1.0, 0.0]}, np.eye(2), cs)
    assert result.pairs[0].orthogonal_fraction == 0.0
    assert result.isolable == []


# ---------------------------------------------------------------------------
# the declared inputs, swept rather than chosen
# ---------------------------------------------------------------------------

def test_the_finding_survives_the_declared_parameter_sweep(report):
    sensitivity = report["parameter_sensitivity"]
    river = variants(report)["muskingum_two_reach"]["continuity + declared routing"]
    counts = [cell["n_isolable"] for cell in sensitivity["muskingum_routing"]]
    assert min(counts) >= river["n_isolable"] - 1, counts
    assert len({cell["n_isolable"] for cell in sensitivity["cooling_loop_duty"]}) == 1


def test_structural_isolability_does_not_move_with_the_declared_prior(report):
    for topology in report["topologies"]:
        for variant in topology["variants"]:
            for cell in variant["by_prior"]:
                if not cell["refused"]:
                    assert cell["isolable"] == variant["isolable"], (topology["key"], variant["variant"])


def test_a_prior_the_kernel_refuses_is_reported_not_tuned_around(report):
    """The cooling loop's declared kg/J prior is near check_spd's tolerance. That is a
    measured property of the design, and the argument for nondimensionalising it."""
    loop = variants(report)["cooling_loop_mass_energy"]
    assert all(v["n_priors_refused"] > 0 for v in loop.values())
    refusals = [cell for v in loop.values() for cell in v["by_prior"] if cell["refused"]]
    assert refusals and all("positive definite" in cell["refusal"] for cell in refusals)
    river = variants(report)["muskingum_two_reach"]
    assert all(v["n_priors_refused"] == 0 for v in river.values())


def test_every_separation_reported_is_actionable_not_merely_structural(report):
    """An earlier draft whitened joules against kilograms and reported 204,258x. Units are
    not a neutral choice, and the study declares each topology's prior in its own."""
    for topology in report["topologies"]:
        for variant in topology["variants"]:
            span = variant["amplification_range"]
            if span is not None:
                assert span[1] < 100.0, (topology["key"], variant["variant"], span)
    assert study._cooling_loop().prior_units.startswith("kg")


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def test_every_qualitative_sentence_is_a_computed_condition(report):
    assert report["claims"] and all(report["claims"].values()), \
        [k for k, v in report["claims"].items() if not v]
    broken = dict(report, claims=dict(report["claims"],
                                      the_river_design_isolates_more_than_the_cooling_loop=False))
    with pytest.raises(RuntimeError, match="no longer true of the numbers"):
        study.render(broken)


def test_the_report_says_it_is_not_a_measurement(report):
    text = study.render(report)
    assert "No instrument, record or estimate appears" in text
    assert "nothing below is a measurement" in text
    for phrase in ("declared parameter", "none on `A`", "units error"):
        assert phrase in text, phrase


def test_the_report_reproduces(report):
    committed = json.loads((RESULTS / "second_balance.json").read_text(encoding="utf-8"))
    failure = reproduction_failure("second_balance.json", report, committed)
    assert failure is None, failure
    assert study.render(committed) == (RESULTS / "second_balance.md").read_text(encoding="utf-8")


def test_confounding_is_read_from_the_structural_label_not_from_a_float():
    """A pair the geometry calls confounded whose whitened angle is small but not zero.

    `evaluate` selected confounded pairs on `orthogonal_fraction == 0.0`. `fdi` calls a pair
    confounded at `COLLINEAR_COS`, a tolerance, so a NEARLY collinear pair is labelled
    confounded while its whitened angle stays a small positive float. Those pairs then read as
    separated -- at an enormous amplification -- and one became the reported tightest
    separated pair. That is what made the committed cooling-loop report understate its
    confounds and name a confounded pair as its tightest separated one.
    """
    rows = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
    cs = ConstraintSet("near", rows, np.zeros(2), "two rows")
    directions = {"a": [1.0, 0.0, 0.0], "nearly a": [1.0, 1e-11, 0.0], "b": [0.0, 0.0, 1.0]}
    near = next(p for p in isolability(directions, np.eye(3), cs).as_dict()["pairs"]
                if {p["a"], p["b"]} == {"a", "nearly a"})
    assert not near["distinguishable"], "within COLLINEAR_COS, so the geometry says confounded"
    assert 0.0 < near["orthogonal_fraction"] < 1e-9, "and the float is not exactly zero"
    topology = study.Topology(
        key="near", label="near", states=("x", "y", "z"), faults=directions,
        variants={"two rows": (rows, "two rows")}, stored_states=(2,),
        reference_prior_sd=(1.0, 1.0, 1.0), prior_units="same", note="")
    cell = study.evaluate(topology, "two rows", 1.0)
    assert ["a", "nearly a"] in [sorted(pair) for pair in cell["confounded_pairs"]]
    tightest = cell["tightest_separated_pair"]
    assert tightest is None or {tightest["a"], tightest["b"]} != {"a", "nearly a"}


def test_a_declared_A_var_does_not_turn_a_confounded_pair_into_a_separated_one():
    """The same guard on the path that exposed it: Cov(E x) widening S."""
    rows = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
    width = 3
    directions = {"a": [1.0, 0.0, 0.0], "nearly a": [1.0, 1e-11, 0.0], "b": [0.0, 0.0, 1.0]}
    A_var = np.zeros((2 * width, 2 * width))
    A_var[0, 0] = A_var[width, width] = 0.04
    topology = study.Topology(
        key="uncertain", label="uncertain", states=("x", "y", "z"), faults=directions,
        variants={"two rows": (rows, "two rows")}, stored_states=(2,),
        reference_prior_sd=(1.0, 1.0, 1.0), prior_units="same", note="",
        a_var={"two rows": A_var}, operating_point=(1.0, 1.0, 1.0))
    cell = study.evaluate(topology, "two rows", 1.0, declare_A_var=True)
    assert cell["A_declared_uncertain"] is True
    assert ["a", "nearly a"] in [sorted(pair) for pair in cell["confounded_pairs"]]
    assert "a" not in cell["isolable"] and "nearly a" not in cell["isolable"]
