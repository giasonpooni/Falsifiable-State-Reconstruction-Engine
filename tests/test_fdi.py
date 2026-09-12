"""What the consistency statistic can tell apart, and the rank-1 result that bounds it.

The load-bearing test here is test_rank_one_makes_every_visible_pair_confusable: every
constraint in this repository today has rank 1, and that alone forbids isolation. It is
checked against the real topologies, not only a toy.
"""
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
    axis. No covariance, no record length and no b_var changes it."""
    dirs = {"m1_bias": [1.0, 0.0], "m2_bias": [0.0, 1.0], "both": [1.0, 1.0], "transfer": [1.0, -1.0]}
    # a wildly anisotropic P and a strongly correlated one: the result is about rank(A),
    # not about the covariance, so neither changes it
    for P in (P2, np.diag([1e-3, 1e3]), np.array([[4.0, 5.9], [5.9, 9.0]])):
        r = isolability(dirs, P, SUM2)
        assert r.residual_rank == 1
        assert r.isolable == []
        assert "no fault is isolatable from any other" in r.note.lower()
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


def test_a_separation_too_small_to_act_on_is_not_an_isolation():
    """A cosine gate alone would call these distinguishable. The orthogonal fraction says what
    that would cost: telling them apart needs a fault 1/sin times the size detecting one does,
    so the gate is a declared separation, not floating-point distance from 1."""
    cs = ConstraintSet(version="two-row", A=np.array([[1.0, 0.0], [0.0, 1.0]]),
                       b=np.array([0.0, 0.0]), description="two independent rows")
    near = np.array([1.0, 2e-3])                      # |cos| with [1,0] is 0.999998
    r = isolability({"f": [1.0, 0.0], "g": near}, np.eye(2), cs)
    pair = r.pairs[0]
    assert abs(pair.cos) > COLLINEAR_COS - 1e-5 or abs(pair.cos) > 0.99999
    assert pair.orthogonal_fraction == pytest.approx(2e-3, rel=1e-3)
    assert pair.isolation_amplification == pytest.approx(500.0, rel=1e-2)
    assert not pair.distinguishable and "numerical residue" in pair.why
    assert r.isolable == []
    # a caller with a real margin can declare it, and a genuinely separated pair still passes
    wide = isolability({"f": [1.0, 0.0], "h": [0.0, 1.0]}, np.eye(2), cs)
    assert wide.pairs[0].orthogonal_fraction == pytest.approx(1.0)
    assert wide.pairs[0].isolation_amplification == pytest.approx(1.0)
    assert set(wide.isolable) == {"f", "h"}
    # declaring a separation stricter than the pair achieves withdraws the isolation
    assert isolability({"f": [1.0, 0.0], "g": near}, np.eye(2), cs,
                       min_separation=1e-4).isolable == ["f", "g"]


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
    assert "testbed.cusum" in r.note          # the note names where localisation must come from


def test_isolability_refuses_an_empty_declaration():
    with pytest.raises(ValueError, match="at least one"):
        isolability({}, P2, SUM2)


def test_a_covariance_with_no_uncertainty_along_the_constraint_is_refused():
    """A hard-projected P is rank-deficient by construction; whitening it would divide by
    zero and report a fault as infinitely visible."""
    singular = np.array([[1.0, -1.0], [-1.0, 1.0]])       # no uncertainty along [1, 1]
    with pytest.raises(ValueError):
        whitened_signature([1.0, 0.0], singular, SUM2)
