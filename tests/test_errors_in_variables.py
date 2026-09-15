"""Declared uncertainty on A: the schema contract, the kernel's use of it, and its calibration.

The kernel assumed the relation exact and let only b carry error. Every constitutive relation
the roadmap needs next has measured coefficients, so that assumption fails exactly where the
project is going. These tests pin what A_var means, what the kernel does with it, and the
measurement that chose the correction.
"""
import json

import numpy as np
import pytest

from set_lcm.experiments import errors_in_variables as eiv
from set_lcm.experiments.compare import reproduction_failure
from set_lcm.experiments.provenance import REPO_ROOT
from set_lcm.lcm import chi2_quantile, consistency_stat, detectability, matrix_uncertainty, reduced
from set_lcm.schema import ConstraintSet

RESULTS = REPO_ROOT / "results"
A = np.array([[1.0, -1.0, 0.0], [0.0, 0.6, -1.0]])
P = np.diag([0.40, 0.25, 0.10])
STATE = np.array([2.0, 5.0, -1.5])
PER_ROW = np.array([np.diag([0.0, 0.02, 0.0]), np.diag([0.0, 0.03, 0.0])])


def a_set(**kwargs):
    return ConstraintSet("eiv", A, np.zeros(2), "one exact row and one measured row", **kwargs)


# ---------------------------------------------------------------------------
# the declaration
# ---------------------------------------------------------------------------

def test_a_matrix_is_exact_unless_declared_otherwise():
    cs = a_set()
    assert cs.A_var is None and cs.matrix_exact
    with pytest.raises(ValueError, match="declares A exact"):
        cs.A_block(0, 0)


def test_a_per_row_stack_expands_to_a_block_diagonal_vec_covariance():
    cs = a_set(A_var=PER_ROW)
    assert cs.A_var.shape == (6, 6)
    np.testing.assert_allclose(cs.A_block(0, 0), PER_ROW[0])
    np.testing.assert_allclose(cs.A_block(1, 1), PER_ROW[1])
    # a per-row declaration says the rows are independent, and says it in the off-diagonal
    np.testing.assert_allclose(cs.A_block(0, 1), np.zeros((3, 3)))


def test_a_full_declaration_can_carry_dependence_a_per_row_one_cannot():
    full = np.zeros((6, 6))
    full[1, 1], full[4, 4] = 0.02, 0.03
    full[1, 4] = full[4, 1] = 0.015
    cs = a_set(A_var=full)
    assert cs.A_block(0, 1)[1, 1] == pytest.approx(0.015)


@pytest.mark.parametrize("bad,match", [
    (np.ones((2, 2)), "stack of per-row covariances"),
    (np.full((2, 3, 3), np.nan), "finite"),
    (np.array([[[1.0, 2.0], [0.0, 1.0]]] * 2), "stack of per-row covariances"),
    (np.array([np.diag([-1.0, 1.0, 1.0]), np.eye(3)]), "positive semidefinite"),
])
def test_a_malformed_declaration_is_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        a_set(A_var=bad)


def test_dependent_rows_with_a_declared_A_var_are_refused_as_b_var_already_is():
    """The SVD reduction would discard what the declaration says; merge the rows first."""
    rows = np.array([[1.0, -1.0], [2.0, -2.0]])
    cs = ConstraintSet("dependent", rows, np.zeros(2), "the second row is the first, doubled",
                       A_var=np.array([np.diag([0.01, 0.0]), np.diag([0.04, 0.0])]))
    with pytest.raises(ValueError, match="declared A_var"):
        reduced(cs)


# ---------------------------------------------------------------------------
# what the kernel does with it
# ---------------------------------------------------------------------------

def test_the_contribution_is_the_quadratic_form_the_docstring_states():
    cs = a_set(A_var=PER_ROW)
    Q = matrix_uncertainty(cs, STATE, P)
    for i in range(2):
        block = cs.A_block(i, i)
        assert Q[i, i] == pytest.approx(STATE @ block @ STATE + np.trace(block @ P))
    assert Q[0, 1] == pytest.approx(0.0)            # declared independent
    np.testing.assert_allclose(Q, Q.T)


def test_the_contribution_is_the_covariance_it_claims_to_be_when_sampled():
    """Against DRAWS, not against the formula again.

    Every other check of `matrix_uncertainty` re-evaluates `x^T Sigma_ij x + tr(Sigma_ij P)`
    and compares it with itself, so a wrong formula would satisfy all of them: drop the trace
    term and the assertions above still pass. This draws the two errors the derivation assumes
    -- one shared coefficient error, one state error from P, independent of each other -- forms
    E x, and requires the sample covariance to match.

    The system is two rows sharing ONE coefficient, placed at different states, so the check
    covers the off-diagonal a per-row declaration cannot express as well as the diagonal.
    Measured at this seed: 0.600% worst relative error, against 8.4% for the same formula
    with the trace term dropped. The 2% bound sits between them with room on both sides.
    """
    n, rows, sd = 3, 2, 0.3
    A = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
    shared = ((0, 0), (1, 2))                       # (row, state) the one coefficient sits at
    cov = np.zeros((rows * n, rows * n))
    for i, si in shared:
        for j, sj in shared:
            cov[i * n + si, j * n + sj] = sd ** 2
    cs = ConstraintSet("shared", A, np.zeros(rows), "one coefficient in two rows", A_var=cov)
    x, prior = np.array([3.0, 2.0, 1.0]), np.diag([0.5, 0.2, 0.8])
    analytic = matrix_uncertainty(cs, x, prior)

    draws = 60_000
    rng = np.random.default_rng(7)
    theta = rng.normal(0.0, sd, size=draws)         # ONE instrument, so one error per draw
    delta = np.linalg.cholesky(prior) @ rng.normal(size=(n, draws))
    Ex = np.stack([theta * (x[state] + delta[state]) for _, state in shared])
    sampled = (Ex @ Ex.T) / draws

    peak = np.abs(analytic).max()
    assert np.abs(analytic - sampled).max() < 0.02 * peak
    # and the check would fail for the formula without its second term
    first_term_only = np.array([[x @ cs.A_block(i, j) @ x for j in range(rows)]
                                for i in range(rows)])
    assert np.abs(analytic - first_term_only).max() > 0.05 * peak


def test_the_contribution_grows_with_the_state_because_it_is_a_quadratic_form():
    """Unlike b_var this is not a property of the set: it moves with the operating point."""
    cs = a_set(A_var=PER_ROW)
    small = matrix_uncertainty(cs, STATE * 0.1, P)
    large = matrix_uncertainty(cs, STATE * 10.0, P)
    assert np.all(np.diag(large) > np.diag(small))
    assert consistency_stat(STATE, P, cs) < consistency_stat(STATE, P, a_set(b_var=np.array([1e-9, 1e-9])))


def test_declaring_A_uncertain_can_only_widen_the_residual_covariance():
    """More declared uncertainty cannot make a fault more visible."""
    exact = a_set(b_var=np.array([0.05, 0.02]))
    uncertain = a_set(b_var=np.array([0.05, 0.02]), A_var=PER_ROW)
    fault = np.array([1.0, 0.0, 0.0])
    assert detectability(fault, P, uncertain, x=STATE) < detectability(fault, P, exact)
    assert consistency_stat(STATE + 1.0, P, uncertain) < consistency_stat(STATE + 1.0, P, exact)


def test_detectability_refuses_to_answer_without_a_state_when_A_is_uncertain():
    """d(f) stops being a property of the set, so a default operating point would be a fiction."""
    cs = a_set(A_var=PER_ROW)
    with pytest.raises(ValueError, match="quadratic form in the state"):
        detectability(np.array([1.0, 0.0, 0.0]), P, cs)


def test_a_zero_declaration_reproduces_the_exact_matrix_answer():
    zero = np.zeros((2, 3, 3))
    exact = a_set(b_var=np.array([0.05, 0.02]))
    declared_zero = a_set(b_var=np.array([0.05, 0.02]), A_var=zero)
    assert consistency_stat(STATE + 0.3, P, declared_zero) == pytest.approx(
        consistency_stat(STATE + 0.3, P, exact), rel=1e-12)


def test_an_exact_set_is_untouched_by_this_change():
    """Every set in the repository declares A exact; none of their arithmetic may move."""
    plain = ConstraintSet("sum", np.array([[1.0, 1.0]]), np.array([100.0]), "a declared total")
    assert plain.matrix_exact
    assert consistency_stat(np.array([52.0, 46.0]), np.eye(2) * 4.0, plain) == pytest.approx(
        (52.0 + 46.0 - 100.0) ** 2 / 8.0)


# ---------------------------------------------------------------------------
# the calibration, which is what chose the correction
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def report():
    return eiv.compute()


def test_treating_an_uncertain_relation_as_exact_rejects_a_healthy_system(report):
    for experiment in report["experiments"]:
        cell = experiment["by_declaration"]["A treated as exact"]
        assert cell["rejection_rate_over_nominal"] > 5.0, experiment["shared_row_covariance"]
        assert cell["mean"] > 2.0 * experiment["rank"]


def test_the_full_declaration_is_calibrated_at_every_swept_dependence(report):
    for experiment in report["experiments"]:
        cell = experiment["by_declaration"]["A_var, full vec(A) covariance"]
        assert cell["mean"] == pytest.approx(experiment["rank"], abs=0.1)
        assert cell["rejection_rate_over_nominal"] < 1.5


def test_a_per_row_declaration_degrades_exactly_where_the_roadmap_says_it_would(report):
    """"Shared evidence can correlate the rows"; independent rows is then the wrong claim."""
    independent = [e for e in report["experiments"] if e["shared_row_covariance"] == 0.0]
    dependent = sorted((e for e in report["experiments"] if e["shared_row_covariance"] > 0.0),
                       key=lambda e: e["shared_row_covariance"])
    assert independent and dependent
    for experiment in independent:
        assert experiment["by_declaration"]["A_var, rows declared independent"][
            "rejection_rate_over_nominal"] < 1.5
    rates = [e["by_declaration"]["A_var, rows declared independent"]["rejection_rate"]
             for e in dependent]
    assert rates == sorted(rates), rates            # worse as the rows become more dependent
    for experiment in dependent:
        per_row = experiment["by_declaration"]["A_var, rows declared independent"]
        exact = experiment["by_declaration"]["A treated as exact"]
        assert per_row["rejection_rate"] < exact["rejection_rate"]


def test_the_first_order_residual_shrinks_as_the_approximation_improves(report):
    """Measured rather than assumed small: if it did not shrink the correction would be wrong."""
    quality = report["approximation_quality"]
    assert quality[0]["coefficient_scale"] > quality[-1]["coefficient_scale"]
    assert quality[-1]["rejection_rate_over_nominal"] < quality[0]["rejection_rate_over_nominal"]
    assert quality[-1]["rejection_rate_over_nominal"] == pytest.approx(1.0, abs=0.25)


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def test_every_qualitative_sentence_is_a_computed_condition(report):
    assert report["claims"] and all(report["claims"].values()), \
        [k for k, v in report["claims"].items() if not v]
    broken = dict(report, claims=dict(report["claims"],
                                      the_full_declaration_is_calibrated_everywhere=False))
    with pytest.raises(RuntimeError, match="no longer true of the numbers"):
        eiv.render(broken)


def test_the_report_says_what_the_correction_does_not_do(report):
    text = eiv.render(report)
    limitations = json.dumps(report["limitations"])
    assert "The projection is unchanged" not in limitations, (
        "the projection now follows the declaration; this limitation was closed by "
        "results/eiv_projection and must not be re-asserted")
    assert "results/eiv_projection" in limitations
    assert "true by construction" in text
    for phrase in ("not conservative", "first order", "quadratic form in the state"):
        assert phrase in text, phrase


def test_the_report_reproduces(report):
    committed = json.loads((RESULTS / "errors_in_variables.json").read_text(encoding="utf-8"))
    failure = reproduction_failure("errors_in_variables.json", report, committed)
    assert failure is None, failure
    assert eiv.render(committed) == (RESULTS / "errors_in_variables.md").read_text(encoding="utf-8")
