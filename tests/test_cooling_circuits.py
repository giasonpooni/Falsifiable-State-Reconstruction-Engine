"""The parallel-circuit cooling design study: what it computes, and what it must not be read as.

No evidence, no estimator and no record is involved. These tests pin the study's findings to
the algebra they come from -- above all to the two places an earlier draft got the MODELLING
wrong and produced a clean-looking answer anyway: a flow meter decoupled from the enthalpy it
computes, and a temperature datum leaking into the whitening.
"""
import json

import numpy as np
import pytest

from set_lcm.experiments import cooling_circuits as study
from set_lcm.experiments.compare import reproduction_failure
from set_lcm.experiments.provenance import REPO_ROOT
from set_lcm.experiments.second_balance import PRIOR_SWEEP, evaluate
from set_lcm.fdi import isolability, residual_covariance
from set_lcm.lcm import detectability
from set_lcm.schema import ConstraintSet

RESULTS = REPO_ROOT / "results"
FULL = study.FULL


@pytest.fixture(scope="module")
def report():
    return study.compute()


def by_variant(report, circuits, variant):
    return next(e for e in report["sweep"] if e["circuits"] == circuits)["by_variant"][variant]


# ---------------------------------------------------------------------------
# the physics the rows and the catalogue encode
# ---------------------------------------------------------------------------

def test_a_flow_meter_bias_moves_the_enthalpy_it_computes_not_only_the_mass():
    """The defect that made an earlier draft's answer meaningless.

    One meter reading is what computes both the mass and the enthalpy a circuit carries, so a
    bias moves two states. Declaring it as mass alone makes it invisible to an energy balance,
    which then reported every fault isolable for a reason that was arithmetic, not physics.
    """
    top = study.topology(2)
    f = np.array(top.faults["circuit 1 flow-meter bias"])
    assert f[study._slot(0, 0)] == 1.0
    assert f[study._slot(0, 1)] == pytest.approx(study.EPS)
    energy = top.variants["energy only"][0]
    assert energy[0] @ f != 0.0, "a flow-meter bias must be visible to its own energy balance"


def test_fouling_conserves_energy_and_violates_only_the_duty_relation():
    """A fouling circuit returns less enthalpy because it absorbed less heat."""
    top = study.topology(3)
    f = np.array(top.faults["circuit 2 fouling"])
    energy, duty = top.variants["energy only"][0], top.variants["energy + duty"][0]
    assert all(row @ f == 0.0 for row in energy), "fouling must satisfy every energy balance"
    assert duty[3 + 1] @ f != 0.0, "fouling must violate its own duty row"


def test_a_return_thermocouple_bias_moves_the_returned_enthalpy_alone():
    top = study.topology(2)
    f = np.array(top.faults["circuit 2 return-thermocouple bias"])
    assert f[study._slot(1, 1)] == 1.0 and np.count_nonzero(f) == 1


def test_every_declared_fault_belongs_to_a_circuit_or_to_the_header():
    top = study.topology(4)
    assert len(top.faults) == 3 * 4 + 1
    assert sum(study._circuit_of(name) is None for name in top.faults) == 1


# ---------------------------------------------------------------------------
# the findings
# ---------------------------------------------------------------------------

def test_conservation_alone_cannot_see_a_fouling_circuit_at_any_size(report):
    for entry in report["sweep"]:
        bare = entry["by_variant"]["energy only"]
        for i in range(entry["circuits"]):
            assert f"circuit {i + 1} fouling" in bare["invisible"]
        assert bare["n_isolable"] == 0


def test_the_duty_row_is_what_buys_isolation(report):
    for entry in report["sweep"]:
        duty = entry["by_variant"]["energy + duty"]
        assert duty["n_isolable"] == 3 * entry["circuits"]
        assert duty["invisible"] == ["header flow-meter bias"]


def test_metering_the_header_closes_the_catalogue(report):
    for entry in report["sweep"]:
        full = entry["by_variant"][FULL]
        assert full["n_isolable"] == entry["n_faults"]
        assert full["invisible"] == [] and full["confounded_pairs"] == []


def test_the_structural_count_is_not_the_finding(report):
    """Every fault isolable, and the hardest pair still bounded away from 1."""
    for entry in report["sweep"]:
        full = entry["by_variant"][FULL]
        assert full["n_isolable"] == entry["n_faults"]
        assert full["amplification_at_reference_prior"] > 1.0


def test_extra_circuits_buy_coverage_and_not_conditioning(report):
    """Without a header meter the hardest pair is identical at every circuit count."""
    amplifications = {by_variant(report, n, "energy + duty")["amplification_at_reference_prior"]
                      for n in study.CIRCUIT_SWEEP}
    assert max(amplifications) - min(amplifications) < 1e-9
    for n in study.CIRCUIT_SWEEP:
        hardest = by_variant(report, n, "energy + duty")["tightest_separated_pair"]
        assert hardest["kind"] == "within one circuit"
        assert hardest["every_tied_pair_involves_fouling"]


def test_the_header_metered_curve_hands_over_from_between_meters_to_within_a_circuit(report):
    first = by_variant(report, study.CIRCUIT_SWEEP[0], FULL)
    last = by_variant(report, study.CIRCUIT_SWEEP[-1], FULL)
    assert first["tightest_separated_pair"]["kind"] == "against the header meter"
    assert last["tightest_separated_pair"]["kind"] == "within one circuit"
    assert last["tightest_separated_pair"]["every_tied_pair_involves_fouling"]
    amplifications = [by_variant(report, n, FULL)["amplification_at_reference_prior"]
                      for n in study.CIRCUIT_SWEEP]
    assert all(a > b for a, b in zip(amplifications, amplifications[1:]))
    floor = by_variant(report, study.CIRCUIT_SWEEP[-1], "energy + duty")[
        "amplification_at_reference_prior"]
    assert floor < amplifications[-1] < floor * 1.05


def test_the_binding_pair_is_a_family_and_the_report_says_how_large(report):
    """A manifold is symmetric under relabelling circuits, so the minimum is attained once
    per circuit. Naming one member would present an arbitrary choice among equals."""
    for entry in report["sweep"]:
        hardest = entry["by_variant"][FULL]["tightest_separated_pair"]
        assert hardest["n_tied"] == entry["circuits"]
        assert hardest["n_tied"] < hardest["n_separated_pairs"] or entry["circuits"] == 1
        assert hardest["kind"] in ("within one circuit", "against the header meter")


def test_which_family_binds_is_conditional_on_the_declared_heat_load(report):
    """The handover is not a property of the circuit count, and the report must not imply it is."""
    def kind(entry, scale):
        return next(p["binding_pair_kind"]
                    for p in entry["by_variant"][FULL]["amplification_by_prior"]
                    if p["prior_scale"] == scale)

    tight = min(report["declared"]["prior_scales_swept"])
    loose = max(report["declared"]["prior_scales_swept"])
    assert all(kind(e, tight) == "against the header meter" for e in report["sweep"]), (
        "at a tight heat-load prior the handover does not happen anywhere in this sweep")
    handovers = {}
    for scale in (tight, 1.0, loose):
        handovers[scale] = next(
            (e["circuits"] for e in report["sweep"] if kind(e, scale) == "within one circuit"), None)
    assert handovers[tight] is None
    assert handovers[loose] is not None and handovers[1.0] is not None
    assert handovers[loose] < handovers[1.0], "a looser heat load moves the handover earlier"


def test_the_declared_rows_are_independent_and_the_operating_point_satisfies_them(report):
    """Two silent-failure guards.

    A dependent row set would make `declared_rows` something other than the residual
    dimension the report describes, and `reduced()` would quietly SVD-reduce it on the exact
    path. An operating point off the declared relation would poison every `Cov(E x)` number
    without any of them looking wrong, because that covariance is a quadratic form evaluated
    exactly there.
    """
    for entry in report["sweep"]:
        for variant, cell in entry["by_variant"].items():
            assert cell["declared_rows_are_independent"], (entry["circuits"], variant)
            assert cell["residual_rank"] == cell["declared_rows"]
            assert cell["operating_point_row_residual"] <= 1e-12, (entry["circuits"], variant)


def test_conservation_alone_separates_circuits_perfectly_or_not_at_all(report):
    """Energy balances are one row per circuit, so the geometry is binary: everything inside
    a circuit collapses into that row, and everything across circuits is orthogonal."""
    for entry in report["sweep"]:
        hardest = entry["by_variant"]["energy only"]["tightest_separated_pair"]
        if hardest is None:          # one circuit: a single row separates nothing at all
            assert entry["circuits"] == 1
            continue
        assert hardest["every_separated_pair_is_orthogonal"]
        assert hardest["isolation_amplification"] == pytest.approx(1.0)
        assert hardest["kind"] == "between circuits"


def test_the_declared_heat_load_binds_an_order_of_magnitude_harder_than_the_meter_count(report):
    over_prior = max(p["isolation_amplification"] for e in report["sweep"]
                     for p in e["by_variant"][FULL]["amplification_by_prior"]
                     if p["isolation_amplification"] is not None)
    over_circuits = max(e["by_variant"][FULL]["amplification_at_reference_prior"]
                        for e in report["sweep"])
    assert over_prior > 10.0 * over_circuits


def test_nothing_is_refused_which_is_what_nondimensionalising_bought(report):
    """`second_balance` had its cooling loop refused at 2 of 3 swept priors, in kg and J."""
    assert all(not cell["refused"] for e in report["sweep"] for cell in e["by_variant"].values())
    assert len(PRIOR_SWEEP) == 3


# ---------------------------------------------------------------------------
# the declared coefficient uncertainty
# ---------------------------------------------------------------------------

def test_one_supply_thermocouple_is_declared_as_a_dependence_between_rows():
    """The per-row form cannot say that one error moves every energy row together."""
    cov = study.supply_theta_cov(3, rows=7)
    width = 1 + 3 * 3
    for i in range(3):
        for j in range(3):
            block = cov[i * width:(i + 1) * width, j * width:(j + 1) * width]
            assert block[study._slot(i, 0), study._slot(j, 0)] == study.SUPPLY_THETA_SD ** 2
    assert np.linalg.matrix_rank(cov) == 1, "one instrument is one direction of error"


def test_the_supply_thermocouple_is_not_the_binding_instrument(report):
    for entry in report["sweep"]:
        for cell in entry["by_variant"].values():
            declared = cell["with_supply_temperature_declared_uncertain"]
            assert declared["isolable"] == cell["isolable"]
            if cell["amplification_at_reference_prior"] is not None:
                shift = abs(declared["isolation_amplification"]
                            - cell["amplification_at_reference_prior"])
                assert shift / cell["amplification_at_reference_prior"] < 0.01


def test_declaring_A_uncertain_reaches_the_geometry_rather_than_being_dropped():
    """The gap this study closed: fdi used to compute S without Cov(E x) whatever was declared."""
    top = study.topology(2, supply_sd=0.8)
    rows, _ = top.variants[FULL]
    x = np.array(top.operating_point)
    P = np.diag(np.array(top.reference_prior_sd) ** 2)
    exact = ConstraintSet("exact", rows, np.zeros(rows.shape[0]), "exact")
    uncertain = ConstraintSet("uncertain", rows, np.zeros(rows.shape[0]), "uncertain",
                              A_var=top.a_var[FULL])
    _, S_exact = residual_covariance(P, exact)
    _, S_uncertain = residual_covariance(P, uncertain, x)
    assert not np.allclose(S_exact, S_uncertain)
    # A declared uncertainty can only widen the residual covariance.
    assert np.linalg.eigvalsh(S_uncertain - S_exact)[0] >= -1e-12


def test_the_signature_still_squares_to_detectability_under_a_declared_A_var():
    """The equality fdi's docstring claims, on the path that used to ignore A_var.

    It holds because residual_covariance now delegates to the kernel's own _residual_S, the
    same S detectability builds, rather than assembling a parallel one.
    """
    top = study.topology(2, supply_sd=0.6)
    rows, _ = top.variants[FULL]
    x = np.array(top.operating_point)
    P = np.diag(np.array(top.reference_prior_sd) ** 2)
    cs = ConstraintSet("uncertain", rows, np.zeros(rows.shape[0]), "u", A_var=top.a_var[FULL])
    result = isolability(top.faults, P, cs, x)
    for name, direction in top.faults.items():
        signature = np.array(result.signatures[name])
        assert signature @ signature == pytest.approx(
            detectability(direction, P, cs, x=x), rel=1e-9, abs=1e-12), name


def test_an_operating_point_is_required_exactly_when_A_is_declared_uncertain():
    top = study.topology(2)
    rows, _ = top.variants[FULL]
    P = np.diag(np.array(top.reference_prior_sd) ** 2)
    uncertain = ConstraintSet("uncertain", rows, np.zeros(rows.shape[0]), "u",
                              A_var=top.a_var[FULL])
    with pytest.raises(ValueError, match="quadratic form in the state"):
        isolability(top.faults, P, uncertain)
    exact = ConstraintSet("exact", rows, np.zeros(rows.shape[0]), "e")
    with pytest.raises(ValueError, match="declares A exact"):
        isolability(top.faults, P, exact, np.array(top.operating_point))


# ---------------------------------------------------------------------------
# the datum
# ---------------------------------------------------------------------------

def test_the_committed_rows_carry_no_temperature_datum():
    """There is no offset to leak, which is the whole point of writing the row on the rise."""
    top = study.topology(3)
    for i in range(3):
        assert top.variants["energy only"][0][i, study._slot(i, 0)] == 0.0


def test_the_rejected_datum_formulation_moves_with_an_arbitrary_datum(report):
    rows = report["sensitivity"]["rejected_datum_formulation"]
    amplifications = [row["isolation_amplification"] for row in rows]
    assert max(amplifications) / min(amplifications) > 1.2, (
        "the rejected formulation must visibly leak, or this check proves nothing")
    assert rows[0]["datum_below_supply"] == 0.0
    assert amplifications[0] == pytest.approx(
        by_variant(report, 4, FULL)["amplification_at_reference_prior"])


def test_the_datum_cancels_in_the_residual_which_is_why_the_defect_was_easy_to_miss():
    """A f is datum-free; A P A^T is not. Only the second is what the whitening uses."""
    plain, offset = study.topology(2), study.datum_offset_topology(2, 7.0)
    f = np.array(offset.faults["circuit 1 flow-meter bias"])
    g = np.array(plain.faults["circuit 1 flow-meter bias"])
    assert offset.variants["energy only"][0][0] @ f == pytest.approx(
        plain.variants["energy only"][0][0] @ g)
    P = np.diag(np.array(plain.reference_prior_sd) ** 2)
    A_plain, A_offset = plain.variants[FULL][0], offset.variants[FULL][0]
    assert not np.allclose(A_plain @ P @ A_plain.T, A_offset @ P @ A_offset.T)


# ---------------------------------------------------------------------------
# the report refuses its own prose
# ---------------------------------------------------------------------------

def test_every_claim_holds(report):
    assert all(report["claims"].values()), [k for k, v in report["claims"].items() if not v]


def test_render_refuses_when_a_claim_stops_holding(report):
    broken = json.loads(json.dumps(report))
    broken["claims"]["the_duty_row_is_what_buys_isolation"] = False
    with pytest.raises(RuntimeError, match="no longer true of the numbers"):
        study.render(broken)


def test_the_report_reproduces(report):
    committed = json.loads((RESULTS / "cooling_circuits.json").read_text(encoding="utf-8"))
    failure = reproduction_failure("cooling_circuits.json", report, committed)
    assert failure is None, failure
    assert study.render(committed) == (RESULTS / "cooling_circuits.md").read_text(encoding="utf-8")


def test_the_study_states_it_is_not_a_measurement(report):
    assert "No record appears" in report["scope"]
    assert any("design study" in line.lower() for line in report["limitations"])
