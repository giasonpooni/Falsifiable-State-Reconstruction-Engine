"""A site declared as data: what the format promises, and what it refuses.

Every ConstraintSet in this repository used to be built by a function inside one experiment
module, with the site's gauge ids, drainage areas, declared sigmas and citations as module
globals beside it. A second reservoir cost a second 570-line module, and every kernel
capability was out of reach of anyone not editing this codebase.

The format's value is review, not convenience, so most of these tests are refusals: a
declaration that guesses is worse than no declaration.
"""
import numpy as np
import pytest

from set_lcm.declaration import SCHEMA, Declaration, load, loads
from set_lcm.experiments import real_water_balance as wb
from set_lcm.experiments.provenance import REPO_ROOT

RIDGWAY = REPO_ROOT / "declarations" / "ridgway.toml"

MINIMAL = f"""
schema = "{SCHEMA}"
key = "toy"
label = "A two-state toy"
note = "Enough to exercise the loader."

[[state]]
name = "a"
unit = "m^3"
description = "first"

[[state]]
name = "b"
unit = "m^3"
description = "second"

[[variant]]
key = "only"
version = "toy-v1"
label = "The only variant"
description = "a minus b is zero"
states = ["a", "b"]

  [[variant.row]]
  name = "closure"
  unit = "m^3"
  description = "a minus b"
  coefficients = {{ a = 1.0, b = -1.0 }}
  b = {{ value = 0.0 }}
"""


COEFFS = "coefficients = { a = 1.0, b = -1.0 }"
STATES = 'states = ["a", "b"]'


def toy(*replacements: tuple[str, str]) -> str:
    """MINIMAL with exact substitutions. Plain str.replace, never str.format: TOML inline
    tables are full of braces and a format spec would eat them."""
    text = MINIMAL
    for old, new in replacements:
        assert text.count(old) == 1, (old, text.count(old))
        text = text.replace(old, new)
    return text


def sensor(extra: str, *, match: str = '{ station = "1", unit = "m" }',
           citation: str = '"somewhere"') -> str:
    return MINIMAL + (f'\n[[sensor]]\nrole = "s"\nsource_id = "x:1"\n'
                      f'extraction_method = "m"\nproperty = "p"\n'
                      f'match = {match}\ncitation = {citation}\n{extra}\n')


# ---------------------------------------------------------------------------
# what it builds
# ---------------------------------------------------------------------------

def test_the_shipped_declaration_loads():
    d = load(RIDGWAY)
    assert d.key == "ridgway"
    assert list(d.states) == ["storage", "cumulative_gauged", "cumulative_ungauged"]
    assert list(d.variants) == ["open", "augmented"]
    assert [s.role for s in d.sensors] == list(wb.COLUMNS)


def test_the_declaration_builds_the_constraint_sets_the_reports_use():
    d = load(RIDGWAY)
    s0, s0_var = 60000.0, 200.0 ** 2
    open_cs = d.constraint_set("open", readings={"closure": s0}, variances={"closure": s0_var})
    assert open_cs.version == "ridgway-closure-open-v1"
    np.testing.assert_array_equal(open_cs.A, [[1.0, -1.0]])
    np.testing.assert_array_equal(open_cs.b, [s0])
    np.testing.assert_array_equal(open_cs.b_var, [s0_var])
    assert open_cs.row_units == ("acre-ft",)

    aug = d.constraint_set("augmented", readings={"closure": s0}, variances={"closure": s0_var})
    assert aug.version == "ridgway-closure-augmented-v1"
    np.testing.assert_array_equal(aug.A, [[1.0, -1.0, -1.0]])


def test_an_omitted_coefficient_is_zero():
    d = loads(toy((STATES, STATES + '\nunconstrained = ["b"]'),
                  (COEFFS, "coefficients = { a = 1.0 }")))
    np.testing.assert_array_equal(d.variants["only"].matrix(), [[1.0, 0.0]])


def test_declared_strings_are_stripped_so_a_paragraph_equals_its_one_line_form():
    """TOML keeps the newline before a closing multi-line delimiter; a citation must not
    differ from itself by a character no reader can see."""
    d = loads(toy(('description = "a minus b is zero"',
                   'description = """\na minus b is zero\n"""')))
    assert d.variants["only"].description == "a minus b is zero"


def test_the_experiment_reads_its_constants_from_the_declaration_not_from_literals():
    """The point of the format is that the site stops living in Python. This fails the moment
    a constant is written back into the module."""
    d = load(RIDGWAY)
    assert wb.DECLARATION.key == d.key
    assert wb.GAUGED_SQ_MI == d.scalar("gauged_drainage_area")
    assert wb.TOTAL_SQ_MI == d.scalar("total_drainage_area")
    assert wb.PRIOR_STORAGE_STD == d.scalar("prior_storage_std")
    assert wb.STORAGE_SIGMA_BASE == d.sensor("storage").sigma
    assert wb.STORAGE_SIGMA_CITATION == d.sensor("storage").citation
    assert wb.FLOW_SIGMA_RELATIVE == d.sensor("outflow").relative
    assert wb.FLOW_SIGMA_FLOOR == d.sensor("outflow").sigma_floor
    assert wb.FLOW_SIGMA_CITATION == d.sensor("outflow").citation
    assert wb.CFG.q_storage == d.scalar("q_storage")
    assert wb.SOURCE_IDS == tuple(s.source_id for s in d.sensors)


def test_every_declared_sigma_reaches_the_bridge_with_its_citation():
    sigma = wb.declared_sigma(wb.STORAGE_SIGMA_BASE)
    assert set(sigma) == set(wb.SOURCE_IDS)
    for source_id, declared in sigma.items():
        assert declared.citation.strip()


# ---------------------------------------------------------------------------
# what it refuses
# ---------------------------------------------------------------------------

def test_an_unknown_key_raises_and_names_itself():
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['citaton'\]"):
        loads(toy(('description = "first"', 'description = "first"\ncitaton = "oops"')))


def test_a_wrong_schema_raises():
    with pytest.raises(ValueError, match="this loader reads"):
        loads(toy((f'schema = "{SCHEMA}"', 'schema = "fsre-declaration-v99"')))


def test_a_row_using_a_state_the_variant_does_not_declare_raises():
    with pytest.raises(ValueError, match="uses state 'c', which the variant does not declare"):
        loads(toy((COEFFS, "coefficients = { a = 1.0, c = -1.0 }")))


def test_a_state_no_row_constrains_raises_unless_it_is_declared_unconstrained():
    with pytest.raises(ValueError, match=r"declares state\(s\) \['b'\] that no row constrains"):
        loads(toy((COEFFS, "coefficients = { a = 1.0 }")))


def test_a_zero_coefficient_does_not_count_as_constraining():
    """Writing b = 0.0 explicitly must not buy past the orphan check; it is still zero."""
    with pytest.raises(ValueError, match="that no row constrains"):
        loads(toy((COEFFS, "coefficients = { a = 1.0, b = 0.0 }")))


def test_a_row_must_give_exactly_one_of_a_constant_or_a_reading():
    with pytest.raises(ValueError, match="exactly one of b.value"):
        loads(toy(("b = { value = 0.0 }", "b = { }")))
    with pytest.raises(ValueError, match="exactly one of b.value"):
        loads(toy(("b = { value = 0.0 }",
                   'b = { value = 0.0, from = "reading", sensor = "s", index = 0 }')))


def test_a_reading_row_naming_no_declared_sensor_raises():
    with pytest.raises(ValueError, match="which no \\[\\[sensor\\]\\] declares"):
        loads(toy(("b = { value = 0.0 }",
                   'b = { from = "reading", sensor = "storage", index = 0 }')))


def test_building_a_set_refuses_a_reading_the_row_did_not_ask_for():
    d = loads(MINIMAL)
    with pytest.raises(ValueError, match="declares a constant b; a reading was supplied"):
        d.constraint_set("only", readings={"closure": 1.0})


def test_building_a_set_requires_the_reading_a_row_did_ask_for():
    d = load(RIDGWAY)
    with pytest.raises(ValueError, match="takes b from reading 0 of sensor 'storage'"):
        d.constraint_set("open")
    with pytest.raises(ValueError, match="takes b's variance from its reading"):
        d.constraint_set("open", readings={"closure": 1.0})


def test_a_reading_supplied_for_a_row_that_does_not_exist_raises():
    d = load(RIDGWAY)
    with pytest.raises(ValueError, match=r"readings supplied for row\(s\) \['typo'\]"):
        d.constraint_set("open", readings={"closure": 1.0, "typo": 2.0},
                         variances={"closure": 1.0})


def test_an_unknown_variant_raises_and_lists_the_known_ones():
    with pytest.raises(KeyError, match="declares no variant"):
        load(RIDGWAY).constraint_set("closed")


def test_an_unknown_sensor_or_scalar_raises_and_lists_the_known_ones():
    d = load(RIDGWAY)
    with pytest.raises(KeyError, match="declares no sensor with role"):
        d.sensor("evaporation")
    with pytest.raises(KeyError, match="declares no scalar"):
        d.scalar("q_evaporation")


def test_a_sensor_must_pin_its_unit_so_two_units_cannot_join_one_column():
    with pytest.raises(ValueError, match="must pin a non-empty unit in match.unit"):
        loads(sensor("sigma = 1.0", match='{ station = "1" }'))


def test_a_sensor_declaring_no_uncertainty_or_two_kinds_raises():
    with pytest.raises(ValueError, match="exactly one of `sigma`"):
        loads(sensor(""))
    with pytest.raises(ValueError, match="exactly one of `sigma`"):
        loads(sensor("sigma = 1.0\nrelative = 0.05\nsigma_floor = 1.0"))
    with pytest.raises(ValueError, match="`relative` and `sigma_floor` go together"):
        loads(sensor("relative = 0.05"))


def test_a_declared_number_without_a_source_raises_where_one_is_required():
    """DeclaredSigma's own refusal, kept at the format boundary too."""
    with pytest.raises(ValueError, match="citation must be a non-empty string"):
        loads(sensor("sigma = 1.0", citation='"   "'))


def test_a_repeated_name_raises_rather_than_letting_the_last_one_win():
    with pytest.raises(ValueError, match="declares state 'a' twice"):
        loads(MINIMAL + '\n[[state]]\nname = "a"\nunit = "m^3"\ndescription = "again"\n')


def test_a_declaration_with_no_states_or_no_variants_raises():
    with pytest.raises(ValueError, match="declares no states"):
        loads(f'schema = "{SCHEMA}"\nkey = "k"\nlabel = "l"\nnote = "n"\nstate = []\nvariant = []\n')


def test_a_row_of_no_coefficients_raises():
    with pytest.raises(ValueError, match="declares no coefficients"):
        loads(toy((COEFFS, "coefficients = { }")))
