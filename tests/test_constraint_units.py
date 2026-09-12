"""Optional constraint row-unit declarations preserve the numerical contract."""
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from set_lcm.lcm import consistency_stat, detectability, project_hard, project_soft
from set_lcm.schema import ConstraintSet


def _constraint(**kwargs):
    return ConstraintSet("balance-v1", np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
                         np.array([10.0, 20.0]), "two balance rows", **kwargs)


def test_existing_positional_construction_keeps_its_meaning():
    cs = _constraint()
    assert cs.row_units is None and cs.b_var is None
    fifth_argument = np.array([1.0, 4.0])
    uncertain = ConstraintSet(cs.version, cs.A, cs.b, cs.description, fifth_argument)
    assert uncertain.row_units is None
    np.testing.assert_array_equal(uncertain.b_var, fifth_argument)
    assert not uncertain.b_var.flags.writeable


def test_row_units_preserve_order_and_cannot_drift_with_the_callers_list():
    declared = ["acre-ft", "ft^3/s"]
    cs = _constraint(row_units=declared)
    declared[0] = "m^3"
    assert cs.row_units == ("acre-ft", "ft^3/s")
    with pytest.raises(FrozenInstanceError):
        cs.row_units = ("m^3", "m^3/s")
    with pytest.raises(TypeError):
        cs.row_units[0] = "m^3"


@pytest.mark.parametrize("units", [(), ("kg",), ("kg", "kg", "kg"),
                                   ("kg", ""), ("kg", " \t"), ("kg", None), ("kg", 1),
                                   "kg", b"kg", {"first": "kg", "second": "kg"}, {"kg", "m"}, 1])
def test_row_units_refuse_wrong_counts_and_invalid_declarations(units):
    with pytest.raises(ValueError, match="row_units"):
        _constraint(row_units=units)


def test_unit_count_uses_declared_rows_not_independent_rank():
    A, b = np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([100.0, 100.0])
    cs = ConstraintSet("duplicate", A, b, "two declared rows, one independent", row_units=("kg", "kg"))
    assert cs.dof == 2 and cs.rank == 1 and cs.row_units == ("kg", "kg")
    with pytest.raises(ValueError, match="A has 2 row"):
        replace(cs, row_units=("kg",))


def test_a_one_dimensional_A_has_one_row_unit():
    cs = ConstraintSet("sum", np.array([1.0, 1.0]), np.array([100.0]), "one row", row_units=["kg"])
    assert cs.row_units == ("kg",)


@pytest.mark.parametrize("b_var", [None, np.array([0.25, 1.0])])
def test_unit_declarations_are_numerically_inert(b_var):
    plain = _constraint(b_var=b_var)
    declared = replace(plain, row_units=("kg", "kg"))
    x, P, fault = np.array([4.0, 8.0, 10.0]), np.diag([1.0, 4.0, 9.0]), np.array([1.0, 0.0, 0.0])
    assert consistency_stat(x, P, plain) == consistency_stat(x, P, declared)
    assert detectability(fault, P, plain) == detectability(fault, P, declared)
    for project in (project_hard, lambda x, P, cs: project_soft(x, P, cs, lam=2.0)):
        before, after = project(x, P, plain), project(x, P, declared)
        for original, annotated in zip(before, after):
            np.testing.assert_array_equal(original, annotated)
