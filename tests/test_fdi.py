"""Static single-fault residual-vector geometry, distinct from scalar detection power."""
import numpy as np
import pytest

from set_lcm.experiments import real_water_balance as wb
from set_lcm.fdi import (
    COLLINEAR_COS, VISIBLE_D, isolability, residual_covariance, whitened_signature,
)
from set_lcm.lcm import detectability
from set_lcm.schema import ConstraintSet

SUM2 = ConstraintSet(version="sum-v1", A=np.array([[1.0, 1.0]]), b=np.array([100.0]),
                     description="the two-reservoir total")
P2 = np.diag([4.0, 9.0])


def test_the_signature_squared_is_exactly_the_kernels_detectability():
    """The whitened signature is not a parallel definition: its squared norm IS d(f), so a
    statement about signatures is a statement about the statistic actually reported."""
    for f in ([1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0], [0.3, -2.7]):
        s = whitened_signature(f, P2, SUM2)
        assert float(s @ s) == pytest.approx(detectability(f, P2, SUM2), rel=1e-12, abs=1e-15)


def test_the_residual_covariance_is_the_one_the_kernel_uses():
    A, S = residual_covariance(P2, SUM2)
    assert A.shape == (1, 2)
    assert S == pytest.approx(A @ P2 @ A.T)
    # with b_var declared, S carries it -- the same S consistency_stat forms
    uncertain = ConstraintSet(version="sum-bv", A=SUM2.A, b=SUM2.b, description="d",
                              b_var=np.array([25.0]))
    _, S2 = residual_covariance(P2, uncertain)
    assert S2 == pytest.approx(S + 25.0)
    assert detectability([1.0, 1.0], P2, uncertain) < detectability([1.0, 1.0], P2, SUM2)


def test_the_null_direction_is_invisible_and_confusable_with_nothing():
    r = isolability({"transfer": [1.0, -1.0], "loss": [1.0, 1.0]}, P2, SUM2)
    assert r.invisible == ["transfer"] and r.visible == ["loss"]
    assert r.d["transfer"] <= VISIBLE_D
    pair = r.pairs[0]
    assert not pair.distinguishable and "invisible" in pair.why
    assert np.isnan(pair.cos)


def test_rank_one_makes_every_visible_pair_confusable():
    """THE bound. rank(A) = 1 gives a scalar residual, so every visible signature lies on one
    axis for the static map with unrestricted signed unknown fault amplitude."""
    dirs = {"m1_bias": [1.0, 0.0], "m2_bias": [0.0, 1.0], "both": [1.0, 1.0], "transfer": [1.0, -1.0]}
    # a wildly anisotropic P and a strongly correlated one: the result is about rank(A),
    # not about the covariance, so neither changes it
    for P in (P2, np.diag([1e-3, 1e3]), np.array([[4.0, 5.9], [5.9, 9.0]])):
        r = isolability(dirs, P, SUM2)
        assert r.residual_rank == 1
        assert r.isolable == []
        assert "no visible candidate is isolatable from another" in r.note.lower()
        for p in r.pairs:
            if p.a in r.visible and p.b in r.visible:
                assert abs(p.cos) >= COLLINEAR_COS, (p.a, p.b, p.cos)
                assert not p.distinguishable


def test_rank_two_can_isolate_when_the_signatures_are_not_collinear():
    """The converse, so the rank-1 result is a finding and not a bug: give the residual a
    second dimension and independent directions separate."""
    cs = ConstraintSet(version="two-row", A=np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
                       b=np.array([10.0, 20.0]), description="two nodes sharing a stream")
    P = np.diag([1.0, 1.0, 1.0])
    r = isolability({"a": [1.0, 0.0, 0.0], "c": [0.0, 0.0, 1.0]}, P, cs)
    assert r.residual_rank == 2
    assert set(r.isolable) == {"a", "c"}
    assert r.pairs[0].distinguishable and abs(r.pairs[0].cos) < COLLINEAR_COS
    # and a direction that IS collinear with another is still not isolable
    r2 = isolability({"a": [1.0, 0.0, 0.0], "a_twice": [2.0, 0.0, 0.0], "c": [0.0, 0.0, 1.0]}, P, cs)
    assert "a" not in r2.isolable and "a_twice" not in r2.isolable
    assert "c" in r2.isolable


def test_a_rank_two_residual_does_not_by_itself_buy_isolation():
    """Room in the residual is necessary, not sufficient: two faults that happen to push the
    same way stay confusable even at rank 2."""
    cs = ConstraintSet(version="two-row", A=np.array([[1.0, 0.0], [0.0, 1.0]]),
                       b=np.array([0.0, 0.0]), description="two independent rows")
    r = isolability({"f": [1.0, 1.0], "g": [2.0, 2.0]}, np.eye(2), cs)
    assert r.residual_rank == 2 and r.isolable == []
    assert "not collinear" in r.note or "not merely a residual" in r.note


# ---------------------------------------------------------------------------
# the real topologies
# ---------------------------------------------------------------------------

def test_every_constraint_in_this_repository_has_a_rank_one_residual():
    """So the bound above applies to all of them, today."""
    sets = {
        "two-reservoir sum": SUM2,
        "ridgway closure (wb_open)": wb.constraint_open(60000.0, 200.0 ** 2),
        "ridgway closure augmented (wb_aug)": wb.constraint_aug(60000.0, 200.0 ** 2),
    }
    for name, cs in sets.items():
        assert cs.rank == 1, name
        A, _ = residual_covariance(np.eye(np.atleast_2d(cs.A).shape[1]), cs)
        assert A.shape[0] == 1, name


def test_the_ridgway_balance_cannot_isolate_a_gauge():
    """On the topology as built: storage-only and cumulative-inflow-only faults are both
    visible, both move the one residual, and nothing in the constraint separates them."""
    cs = wb.constraint_open(60000.0, wb.STORAGE_SIGMA_BASE ** 2)
    P = np.diag([wb.STORAGE_SIGMA_BASE ** 2, 100.0 ** 2])
    r = isolability({
        "storage_reading_high": [1.0, 0.0],
        "gauged_inflow_over_reported": [0.0, 1.0],
        "both_together": [1.0, 1.0] / np.sqrt(2.0),
    }, P, cs)
    assert r.residual_rank == 1
    assert set(r.visible) >= {"storage_reading_high", "gauged_inflow_over_reported"}
    assert r.isolable == []
    assert "testbed.cusum" in r.note          # a separate channel, not a guaranteed localiser


def test_structural_visibility_does_not_depend_on_covariance_or_fault_magnitude():
    directions = {"first": [1.0, 0.0], "tiny": [1e-20, 0.0], "underflow": [1e-200, 0.0],
                  "null": [1.0, -1.0], "tiny_null": [1e-20, -1e-20], "zero": [0.0, 0.0]}
    baseline = isolability(directions, np.eye(2), SUM2)
    uncertain = isolability(directions, 1e12 * np.eye(2), SUM2)
    assert baseline.visible == uncertain.visible == ["first", "tiny", "underflow"]
    assert baseline.invisible == uncertain.invisible == ["null", "tiny_null", "zero"]
    assert uncertain.d["first"] == pytest.approx(baseline.d["first"] / 1e12)
    assert baseline.d["tiny"] == pytest.approx(baseline.d["first"] * 1e-40, rel=1e-12, abs=0)
    assert uncertain.d["first"] < VISIBLE_D       # weak power, still structurally visible
    assert baseline.d["underflow"] == 0.0         # d can underflow without erasing geometry
    for result in (baseline, uncertain):
        for pair in result.pairs:
            if pair.a in result.visible and pair.b in result.visible:
                assert np.isfinite(pair.cos) and not pair.distinguishable


def test_declared_constraint_uncertainty_cannot_create_structural_blindness():
    cs = ConstraintSet(version="large-bvar", A=SUM2.A, b=SUM2.b, description="uncertain total",
                       b_var=np.array([1e15]))
    r = isolability({"loss": [1.0, 0.0], "transfer": [1.0, -1.0]}, np.eye(2), cs)
    assert r.visible == ["loss"] and r.invisible == ["transfer"]
    assert 0 < r.d["loss"] < VISIBLE_D


def test_covariance_can_reduce_angular_separation_without_changing_structural_collinearity():
    cs = ConstraintSet(version="identity", A=np.eye(2), b=np.zeros(2), description="two rows")
    directions = {"f": [1.0, 1.0], "g": [1.0, -1.0]}
    ordinary = isolability(directions, np.eye(2), cs)
    stretched = isolability(directions, np.diag([1.0, 1e10]), cs)
    assert ordinary.isolable == stretched.isolable == ["f", "g"]
    assert ordinary.pairs[0].distinguishable and stretched.pairs[0].distinguishable
    # Reported cos is whitened geometry: practical separation can become very poor.
    assert abs(stretched.pairs[0].cos) > COLLINEAR_COS


def test_vector_isolation_does_not_imply_scalar_statistic_isolation():
    cs = ConstraintSet(version="identity", A=np.eye(2), b=np.zeros(2), description="two rows")
    r = isolability({"x": [1.0, 0.0], "y": [0.0, 1.0]}, np.eye(2), cs)
    assert r.pairs[0].distinguishable
    # Orthogonal residual vectors give exactly the same global statistic T.
    assert r.d["x"] == r.d["y"] == 1.0
    assert "scalar global statistic T discards residual direction" in r.note
    assert "one active candidate fault" in r.note
    assert "unrestricted signed unknown amplitude" in r.note


def test_opposite_signatures_are_confusable_when_amplitude_can_have_either_sign():
    r = isolability({"positive": [1.0, 0.0], "negative": [-1.0, 0.0]}, P2, SUM2)
    assert r.pairs[0].cos == pytest.approx(-1.0)
    assert not r.pairs[0].distinguishable


def test_static_rank_one_result_does_not_rule_out_known_temporal_signatures():
    r = isolability({"step": [1.0, 0.0], "ramp": [0.0, 1.0]}, P2, SUM2)
    assert not r.pairs[0].distinguishable
    # Additional assumed profiles across time add independent information. A scalar
    # residual at each instant can separate these two fixed-amplitude histories.
    histories = np.column_stack([np.ones(4), np.arange(1.0, 5.0)])
    assert np.linalg.matrix_rank(histories) == 2
    assert "Known temporal fault profiles" in r.note and "not analyzed here" in r.note


def test_one_visible_candidate_does_not_get_a_contradictory_rank_one_note():
    r = isolability({"only": [1.0, 0.0]}, P2, SUM2)
    assert r.isolable == ["only"]
    assert "1 of 1 visible" in r.note


@pytest.mark.parametrize("direction", [[1.0], [1.0, float("nan")], [float("inf"), 0.0]])
def test_isolability_refuses_invalid_fault_directions(direction):
    with pytest.raises(ValueError, match="2 finite values"):
        isolability({"invalid": direction}, P2, SUM2)


@pytest.mark.parametrize("direction", [
    [1.0], [1.0, np.nan], [np.inf, 0.0], [[1.0], [0.0]], [],
])
def test_exported_whitening_refuses_nonfinite_or_mismatched_directions(direction):
    with pytest.raises(ValueError):
        whitened_signature(direction, P2, SUM2)


@pytest.mark.parametrize("P", [
    np.eye(3), np.diag([np.inf, 1.0]), np.diag([np.nan, 1.0]),
    np.array([[1.0, 1.0], [0.0, 1.0]]), np.diag([1.0, -1.0]), np.zeros((2, 2)),
])
def test_exported_whitening_and_covariance_share_kernel_covariance_guards(P):
    for call in (lambda: residual_covariance(P, SUM2),
                 lambda: whitened_signature([1.0, 0.0], P, SUM2)):
        with pytest.raises(ValueError):
            call()


def test_whitening_refuses_the_same_ill_conditioned_residual_as_the_kernel():
    A = np.array([[1.0, 0.0], [1.0, 1e-7]])
    exact = ConstraintSet("near", A, np.zeros(2), "near-dependent rows")
    for call in (lambda: detectability([1.0, 0.0], np.eye(2), exact),
                 lambda: whitened_signature([1.0, 0.0], np.eye(2), exact),
                 lambda: residual_covariance(np.eye(2), exact)):
        with pytest.raises(ValueError, match="numerically singular"):
            call()
    # A declared uncertainty can make the same residual covariance well-conditioned.
    uncertain = ConstraintSet("near-uncertain", A, np.zeros(2), "near", b_var=np.ones(2))
    signature = whitened_signature([1.0, 0.0], np.eye(2), uncertain)
    assert float(signature @ signature) == pytest.approx(
        detectability([1.0, 0.0], np.eye(2), uncertain), rel=1e-12)


def test_whitening_does_not_return_a_nonfinite_signature_after_overflow():
    cs = ConstraintSet("large", np.array([[2.0]]), np.zeros(1), "large")
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="finite"):
            whitened_signature([1e308], np.eye(1), cs)


def test_symmetrizing_a_large_finite_covariance_does_not_overflow():
    cs = ConstraintSet("large-P", np.ones((1, 1)), np.zeros(1), "one row")
    _, S = residual_covariance(np.array([[1e308]]), cs)
    assert S[0, 0] == 1e308
    signature = whitened_signature([1.0], np.array([[1e308]]), cs)
    assert signature[0] == pytest.approx(1e-154, rel=1e-12, abs=0.0)


def test_isolability_refuses_an_empty_declaration():
    with pytest.raises(ValueError, match="at least one"):
        isolability({}, P2, SUM2)


def test_a_covariance_with_no_uncertainty_along_the_constraint_is_refused():
    """A hard-projected P is rank-deficient by construction; whitening it would divide by
    zero and report a fault as infinitely visible."""
    singular = np.array([[1.0, -1.0], [-1.0, 1.0]])       # no uncertainty along [1, 1]
    with pytest.raises(ValueError):
        whitened_signature([1.0, 0.0], singular, SUM2)
