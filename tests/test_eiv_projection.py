"""The projection under a declared A_var: the kernel path, and the calibration that priced it.

A constraint set could declare uncertainty on its own coefficients, and `consistency_stat` and
`detectability` both widened S by Cov(E x) when it did -- while `project_hard` and
`project_soft` branched on `b_var` alone and reconciled against A_bar as though the relation
were exact. Silently: no refusal, no warning, in the same module as two functions that refuse
to answer without a state. These tests pin the closed version and the measurement of it.
"""
import json

import numpy as np
import pytest

from set_lcm.experiments import eiv_projection as ep
from set_lcm.experiments.compare import reproduction_failure
from set_lcm.experiments.provenance import REPO_ROOT
from set_lcm.lcm import matrix_uncertainty, project_hard, project_soft, reconcile
from set_lcm.schema import ConstraintSet, Status

RESULTS = REPO_ROOT / "results"

A = np.array([[1.0, -1.0, 0.0], [0.0, 0.6, -1.0]])
P = np.diag([0.40, 0.25, 0.10])
STATE = np.array([2.0, 5.0, -1.5])
B_VAR = np.array([0.05, 0.02])
B = A @ STATE + np.array([0.30, -0.20])          # a state that does not satisfy the relation

FULL = np.zeros((6, 6))
FULL[1, 1], FULL[4, 4] = 0.02, 0.03
FULL[1, 4] = FULL[4, 1] = 0.01                   # the two rows share a measured coefficient


def a_set(**kwargs):
    return ConstraintSet("proj", A, B, "one exact row and one measured row", **kwargs)


# ---------------------------------------------------------------------------
# the kernel path
# ---------------------------------------------------------------------------

def test_a_declared_A_var_changes_the_projection_at_all():
    """The regression guard. Before this, these two returned the same thing."""
    exact_x, exact_P = project_hard(STATE, P, a_set(b_var=B_VAR))
    eiv_x, eiv_P = project_hard(STATE, P, a_set(b_var=B_VAR, A_var=FULL))
    assert not np.allclose(exact_x, eiv_x)
    assert not np.allclose(exact_P, eiv_P)


def test_declaring_the_relation_uncertain_shrinks_the_correction_and_widens_the_report():
    exact_x, exact_P = project_hard(STATE, P, a_set(b_var=B_VAR))
    eiv_x, eiv_P = project_hard(STATE, P, a_set(b_var=B_VAR, A_var=FULL))
    assert np.linalg.norm(eiv_x - STATE) < np.linalg.norm(exact_x - STATE)
    assert np.trace(eiv_P) > np.trace(exact_P)
    # and never past the prior it started from: an uncertain relation can say nothing,
    # not less than nothing.
    assert np.trace(eiv_P) <= np.trace(P) + 1e-12


def test_the_update_is_the_hand_computed_errors_in_variables_gain():
    cs = a_set(b_var=B_VAR, A_var=FULL)
    Sigma = np.diag(B_VAR) + matrix_uncertainty(cs, STATE, P)
    K = P @ A.T @ np.linalg.inv(A @ P @ A.T + Sigma)
    I_KA = np.eye(3) - K @ A
    np.testing.assert_allclose(project_hard(STATE, P, cs)[0], STATE - K @ (A @ STATE - B),
                               rtol=1e-12, atol=0)
    np.testing.assert_allclose(project_hard(STATE, P, cs)[1],
                               I_KA @ P @ I_KA.T + K @ Sigma @ K.T, rtol=1e-12, atol=0)


def test_cov_of_the_residual_gain_is_the_declared_quadratic_form():
    """Q_ij = x^T Sigma_ij x + tr(Sigma_ij P), by hand, on the shared coefficient."""
    Q = matrix_uncertainty(a_set(b_var=B_VAR, A_var=FULL), STATE, P)
    assert Q[0, 0] == pytest.approx(STATE[1] ** 2 * 0.02 + 0.02 * P[1, 1])
    assert Q[1, 1] == pytest.approx(STATE[1] ** 2 * 0.03 + 0.03 * P[1, 1])
    assert Q[0, 1] == pytest.approx(STATE[1] ** 2 * 0.01 + 0.01 * P[1, 1])


def test_a_declared_but_zero_A_var_is_bit_identical_to_declaring_A_exact():
    """Zero uncertainty IS exactness, and must not trip the singular-S guard."""
    for extra in ({}, {"b_var": B_VAR}):
        plain = project_hard(STATE, P, a_set(**extra))
        zeroed = project_hard(STATE, P, a_set(**extra, A_var=np.zeros((6, 6))))
        assert np.array_equal(plain[0], zeroed[0]) and np.array_equal(plain[1], zeroed[1])


def test_the_posterior_covariance_stays_positive_definite():
    """Joseph form: the gain is taken at the incoming state, so it is not always optimal."""
    for scale in (0.1, 1.0, 10.0, 100.0):
        _, P_star = project_hard(STATE * scale, P, a_set(b_var=B_VAR, A_var=FULL))
        assert np.linalg.eigvalsh(P_star).min() > 0.0
        np.testing.assert_allclose(P_star, P_star.T, rtol=0, atol=0)


def test_an_uncertain_relation_stops_correcting_where_it_can_say_nothing():
    """Cov(E x) is quadratic in the state, so the gain shrinks as the operating point grows."""
    corrections = []
    for scale in (1.0, 2.0, 4.0, 8.0):
        x = STATE * scale
        cs = ConstraintSet("proj", A, A @ x - np.array([0.30, -0.20]), "same residual, bigger x",
                           b_var=B_VAR, A_var=FULL)
        corrections.append(float(np.linalg.norm(project_hard(x, P, cs)[0] - x)))
    assert all(a > b for a, b in zip(corrections, corrections[1:])), corrections
    assert corrections[-1] < 0.1 * corrections[0]


def test_soft_projection_carries_the_declaration_too():
    hard = project_hard(STATE, P, a_set(b_var=B_VAR, A_var=FULL))
    infinite = project_soft(STATE, P, a_set(b_var=B_VAR, A_var=FULL), float("inf"))
    assert np.array_equal(hard[0], infinite[0]) and np.array_equal(hard[1], infinite[1])

    slack = project_soft(STATE, P, a_set(b_var=B_VAR, A_var=FULL), 4.0)
    assert np.linalg.norm(slack[0] - STATE) < np.linalg.norm(hard[0] - STATE)
    assert np.trace(slack[1]) > np.trace(hard[1])

    tight = project_soft(STATE, P, a_set(b_var=B_VAR, A_var=FULL), 1e9)
    np.testing.assert_allclose(tight[0], hard[0], rtol=1e-6, atol=1e-9)


def test_soft_projection_without_b_var_still_carries_a_declared_A_var():
    plain = project_soft(STATE, P, a_set(), 4.0)
    declared = project_soft(STATE, P, a_set(A_var=FULL), 4.0)
    assert not np.allclose(plain[0], declared[0])
    assert np.linalg.norm(declared[0] - STATE) < np.linalg.norm(plain[0] - STATE)


def test_reconcile_picks_the_declaration_up_in_both_modes():
    cs = a_set(b_var=B_VAR, A_var=FULL)
    for mode, args in (("hard", {}), ("soft", {"lam": 4.0})):
        got = reconcile(STATE, P, cs, mode=mode, **args)
        expected = (project_hard(STATE, P, cs) if mode == "hard"
                    else project_soft(STATE, P, cs, 4.0))
        assert got.status is Status.OK
        assert np.array_equal(got.x, expected[0]) and np.array_equal(got.P, expected[1])


def test_dependent_rows_with_a_declared_A_var_are_still_refused_by_the_projection():
    doubled = np.vstack([A, 2.0 * A[0]])
    cs = ConstraintSet("dep", doubled, np.append(B, 2.0 * B[0]), "a dependent row",
                       A_var=np.zeros((9, 9)) + np.eye(9) * 0.01)
    with pytest.raises(ValueError, match="dependent rows"):
        project_hard(STATE, P, cs)


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def report():
    return ep.compute()


def test_every_claim_the_report_makes_holds(report):
    failed = [name for name, held in report["claims"].items() if not held]
    assert not failed, failed


def test_render_refuses_to_print_a_sentence_that_stopped_being_true(report):
    broken = dict(report, claims=dict(report["claims"],
                                      the_errors_in_variables_projection_is_calibrated_at_every_scale=False))
    with pytest.raises(RuntimeError, match="no longer true of the numbers"):
        ep.render(broken)


def test_the_overconfidence_is_quadratic_in_the_operating_point(report):
    fit = report["quadratic_fit"]
    assert fit["worst_relative_error"] < 0.05
    for ratio in fit["excess_ratio_per_doubling"]:
        assert ratio == pytest.approx(4.0, abs=0.5)


def test_treating_the_relation_as_exact_can_be_worse_than_not_projecting(report):
    worst = max(report["experiments"], key=lambda e: e["state_scale"])
    assert worst["by_arm"][ep.EXACT]["rms_error"] > worst["by_arm"][ep.NO_PROJECTION]["rms_error"]
    assert worst["by_arm"][ep.FULL]["rms_error"] <= worst["by_arm"][ep.NO_PROJECTION]["rms_error"]


def test_the_one_step_gain_is_priced_rather_than_assumed(report):
    for e in report["experiments"]:
        assert abs(e["one_step_gain"]["reiterated_nees_change"]) < 0.1 * e["dimension"]
    text = ep.render(report)
    for phrase in ("one step, not iterated to a fixed point", "quadratic form in the state"):
        assert phrase in text, phrase


def test_the_report_says_what_it_does_not_fix(report):
    limitations = json.dumps(report["limitations"])
    assert "true by construction" in limitations
    assert "one step rather than iterated" in limitations


def test_the_report_is_deterministic_and_json_safe(report):
    again = ep.compute()
    again["provenance"] = report["provenance"]
    assert json.dumps(again, allow_nan=False) == json.dumps(report, allow_nan=False)


def test_the_report_reproduces(report):
    committed = json.loads((RESULTS / "eiv_projection.json").read_text(encoding="utf-8"))
    failure = reproduction_failure("eiv_projection.json", report, committed)
    assert failure is None, failure
    assert ep.render(committed) == (RESULTS / "eiv_projection.md").read_text(encoding="utf-8")
