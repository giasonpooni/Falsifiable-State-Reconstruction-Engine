"""A site declared as data instead of as Python.

Every ConstraintSet in this repository was built by a function inside one experiment module,
with the site's constants -- gauge ids, drainage areas, declared sigmas and their citations --
as module globals beside it. That made a second reservoir cost a second experiment module, and
it put every capability the kernel has (diagnose, detectability, the isolability tables, the
design study, a declared A_var) out of reach of anyone not editing this codebase.

The abstraction is not invented here. `experiments.second_balance.Topology` already carries
states, faults, variants and declared priors, and already has three instances. This promotes
that shape out of one experiment and makes it loadable, so a site is a document a reviewer can
read rather than a module a reviewer must trace.

WHAT A DECLARATION IS FOR. Not convenience: review. A declared sigma with a citation, a row's
units, the drainage area a scaling rests on, the epoch a cumulative state is zero at -- these
are the assumptions the whole project exists to keep visible, and in a TOML table they are one
diff away from a reader instead of scattered through 570 lines beside the code that uses them.

WHAT IT REFUSES. The format is deliberately unforgiving, because a declaration that guesses is
worse than no declaration:

  - an unknown key anywhere raises, naming it. A misspelled `citaton` must not silently mean
    "no citation"
  - a state named in a row but not declared by that variant raises, naming both
  - a declared state that no row constrains raises unless the variant lists it under
    `unconstrained`, so an orphan state is a decision rather than an oversight
  - a coefficient omitted from a row IS zero, which is stated here and nowhere else; that is
    the one implicit thing the format allows, and the orphan check above is what makes it safe
  - a declared sigma without a citation raises, in DeclaredSigma's own words

WHAT IT DOES NOT DO. It carries no data and no estimator. A declaration says what the system
is, what evidence exists for it and what uncertainty the consumer declares; the record still
arrives through the DAF bridge and the filters are still configured in code. `b` is the one
place where the two meet: a row whose right-hand side is a READING cannot be resolved until
the record exists, so the declaration names the sensor and the index and `constraint_set()`
takes the value at build time.

NORMALISATION. Every declared string is stripped of leading and trailing whitespace, and
that is the only value this format alters. The reason is TOML's own multi-line string form,
which trims the newline after the opening delimiter but keeps the one before the closing
delimiter: a citation written as a paragraph would differ from the same citation written on
one line by a character no reader can see. Requiring a line-continuation backslash before
every closing delimiter would make that a permanent footgun with invisible diffs. Whitespace
around a name, a unit or a paragraph is never semantic, so normalising it is a stated rule
rather than a guess -- and it is what lets a declaration reproduce a Python constant exactly.

Units are declared and carried, never converted. That matches ConstraintSet.row_units: the
caller remains responsible for consistent units, and the declaration exists to make an
inconsistency visible to a reader rather than to fix one silently.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .schema import ConstraintSet

SCHEMA = "fsre-declaration-v1"

__all__ = ["SCHEMA", "Declaration", "State", "Row", "Variant", "Sensor", "Scalar", "Record",
           "load", "loads"]


def _str(value, what: str) -> str:
    """Every declared string, stripped. See NORMALISATION in the module docstring."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string, got {value!r}")
    return value.strip()


def _float(value, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a number, got {value!r}")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{what} must be finite, got {value!r}")
    return value


def _only(table: dict, allowed: set[str], what: str) -> None:
    """Refuse an unknown key rather than ignoring it: a typo must not read as a default."""
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ValueError(f"{what} has unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def _table(value, what: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{what} must be a table, got {type(value).__name__}")
    return value


def _tables(value, what: str) -> list[dict]:
    if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
        raise ValueError(f"{what} must be an array of tables")
    return value


@dataclass(frozen=True)
class State:
    """One component of the state vector, in the order its variant declares."""
    name: str
    unit: str
    description: str

    @staticmethod
    def parse(t: dict) -> State:
        _only(t, {"name", "unit", "description"}, "[[state]]")
        name = _str(t.get("name"), "[[state]].name")
        return State(name, _str(t.get("unit"), f"state {name!r} unit"),
                     _str(t.get("description"), f"state {name!r} description"))


@dataclass(frozen=True)
class Row:
    """One row of A x = b, with its coefficients keyed by state name.

    A state omitted from `coefficients` has coefficient zero. That is the format's single
    implicit rule; the variant's orphan-state check is what keeps it from hiding a mistake.

    `b_value` and `b_from` are exclusive. `b_from` is (sensor, index): the row's right-hand
    side is a reading, so it is resolved when the record exists, not when the file is read.
    """
    name: str
    unit: str
    description: str
    coefficients: dict[str, float]
    b_value: float | None = None
    b_from: tuple[str, int] | None = None
    b_var_value: float | None = None
    b_var_from_sensor: bool = False

    @staticmethod
    def parse(t: dict) -> Row:
        _only(t, {"name", "unit", "description", "coefficients", "b"}, "[[variant.row]]")
        name = _str(t.get("name"), "[[variant.row]].name")
        coefficients = _table(t.get("coefficients"), f"row {name!r} coefficients")
        coeffs = {_str(k, f"row {name!r} coefficient name"): _float(v, f"row {name!r} coefficient {k!r}")
                  for k, v in coefficients.items()}
        if not coeffs:
            raise ValueError(f"row {name!r} declares no coefficients; a row of zeros constrains nothing")

        b = _table(t.get("b", {}), f"row {name!r} b")
        _only(b, {"value", "from", "sensor", "index", "var", "var_from_sensor"}, f"row {name!r} b")
        given = [k for k in ("value", "from") if k in b]
        if len(given) != 1:
            raise ValueError(
                f"row {name!r} must give exactly one of b.value (a constant) or b.from "
                f"(a reading), got {given or 'neither'}")
        b_value = b_from = None
        if "value" in b:
            b_value = _float(b["value"], f"row {name!r} b.value")
        else:
            if _str(b["from"], f"row {name!r} b.from") != "reading":
                raise ValueError(f"row {name!r} b.from must be \"reading\"; nothing else is supported")
            b_from = (_str(b.get("sensor"), f"row {name!r} b.sensor"),
                      _index(b.get("index"), f"row {name!r} b.index"))
        var_from_sensor = bool(b.get("var_from_sensor", False))
        b_var_value = _float(b["var"], f"row {name!r} b.var") if "var" in b else None
        if var_from_sensor and b_var_value is not None:
            raise ValueError(f"row {name!r} declares both b.var and b.var_from_sensor; pick one")
        if var_from_sensor and b_from is None:
            raise ValueError(f"row {name!r} sets b.var_from_sensor but its b is not a reading")
        if b_var_value is not None and b_var_value < 0.0:
            raise ValueError(f"row {name!r} b.var must be nonnegative, got {b_var_value!r}")
        return Row(name, _str(t.get("unit"), f"row {name!r} unit"),
                   _str(t.get("description"), f"row {name!r} description"),
                   coeffs, b_value, b_from, b_var_value, var_from_sensor)


def _index(value, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{what} must be nonnegative, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class Variant:
    """One constraint set over one state ordering.

    A site has more than one because the declaration is a statement about what is BELIEVED,
    and the alternatives are what the reports compare: Ridgway's open balance says four gauges
    account for every drop, and its augmented balance adds an ungauged state that can absorb
    what they miss. Each is a ConstraintSet over its own states, which is why the state list
    belongs to the variant rather than to the site.
    """
    key: str
    version: str
    label: str
    description: str
    states: tuple[str, ...]
    rows: tuple[Row, ...]
    unconstrained: tuple[str, ...] = ()

    @staticmethod
    def parse(t: dict, declared: dict[str, State]) -> Variant:
        _only(t, {"key", "version", "label", "description", "states", "row", "unconstrained"},
              "[[variant]]")
        key = _str(t.get("key"), "[[variant]].key")
        names = t.get("states")
        if not isinstance(names, list) or not names:
            raise ValueError(f"variant {key!r} must declare a non-empty ordered `states` list")
        states = tuple(_str(n, f"variant {key!r} state name") for n in names)
        if len(set(states)) != len(states):
            raise ValueError(f"variant {key!r} repeats a state: {states}")
        for n in states:
            if n not in declared:
                raise ValueError(
                    f"variant {key!r} names state {n!r}, which no [[state]] declares; "
                    f"declared: {sorted(declared)}")
        rows = tuple(Row.parse(r) for r in _tables(t.get("row"), f"variant {key!r} [[variant.row]]"))
        if not rows:
            raise ValueError(f"variant {key!r} declares no rows")
        for row in rows:
            for name in row.coefficients:
                if name not in states:
                    raise ValueError(
                        f"variant {key!r} row {row.name!r} uses state {name!r}, which the variant "
                        f"does not declare; its states are {list(states)}")
        unconstrained = tuple(_str(n, f"variant {key!r} unconstrained name")
                              for n in t.get("unconstrained", []))
        for n in unconstrained:
            if n not in states:
                raise ValueError(f"variant {key!r} lists {n!r} as unconstrained but does not declare it")
        touched = {n for row in rows for n, c in row.coefficients.items() if c != 0.0}
        orphans = [n for n in states if n not in touched and n not in unconstrained]
        if orphans:
            raise ValueError(
                f"variant {key!r} declares state(s) {orphans} that no row constrains. An omitted "
                f"coefficient is zero, so this is what that rule looks like when it is a mistake; "
                f"if it is not, list them under `unconstrained` so the choice is declared")
        return Variant(key, _str(t.get("version"), f"variant {key!r} version"),
                       _str(t.get("label"), f"variant {key!r} label"),
                       _str(t.get("description"), f"variant {key!r} description"),
                       states, rows, unconstrained)

    def matrix(self) -> np.ndarray:
        A = np.zeros((len(self.rows), len(self.states)))
        index = {name: i for i, name in enumerate(self.states)}
        for r, row in enumerate(self.rows):
            for name, value in row.coefficients.items():
                A[r, index[name]] = value
        return A


@dataclass(frozen=True)
class Sensor:
    """One measured column: where the record comes from, and what uncertainty is declared for it.

    `sigma` / `relative` + `sigma_floor` mirror bridge.daf.DeclaredSigma exactly, including its
    refusal to accept a number without a citation. The selector fields are carried verbatim so
    the bridge can be built from the declaration without this module importing it, and `match`
    is normalised to sorted pairs for the same reason SeriesSelector does it.

    The unit is not a field of its own: `match` must pin it, because pinning is what stops
    records in two units from silently joining one column. Two places to state a unit would be
    one place for them to disagree.
    """
    role: str
    source_id: str
    extraction_method: str
    property: str
    match: tuple[tuple[str, str], ...]
    citation: str
    unit_field: str = "unit"
    sigma: float | None = None
    relative: float | None = None
    sigma_floor: float | None = None

    @property
    def unit(self) -> str:
        return dict(self.match)[self.unit_field]

    @staticmethod
    def parse(t: dict) -> Sensor:
        _only(t, {"role", "source_id", "extraction_method", "property", "match", "unit_field",
                  "citation", "sigma", "relative", "sigma_floor"}, "[[sensor]]")
        role = _str(t.get("role"), "[[sensor]].role")
        match = _table(t.get("match", {}), f"sensor {role!r} match")
        if not match:
            raise ValueError(
                f"sensor {role!r} matches on nothing; declare at least one content field "
                f"(the station or monitoring-location id, at minimum)")
        unit_field = _str(t.get("unit_field", "unit"), f"sensor {role!r} unit_field")
        if not str(match.get(unit_field, "")).strip():
            raise ValueError(
                f"sensor {role!r} must pin a non-empty unit in match.{unit_field}; without it "
                f"records in different units could join one column")
        given = [k for k in ("sigma", "relative") if k in t]
        if len(given) != 1:
            raise ValueError(
                f"sensor {role!r} must declare exactly one of `sigma` (absolute) or `relative` "
                f"(a fraction of |value|), got {given or 'neither'}")
        if ("relative" in t) != ("sigma_floor" in t):
            raise ValueError(
                f"sensor {role!r}: `relative` and `sigma_floor` go together. A rating stated as a "
                f"percentage says nothing at zero flow, and a zero sigma would call the reading exact")
        return Sensor(
            role=role,
            source_id=_str(t.get("source_id"), f"sensor {role!r} source_id"),
            extraction_method=_str(t.get("extraction_method"), f"sensor {role!r} extraction_method"),
            property=_str(t.get("property"), f"sensor {role!r} property"),
            match=tuple(sorted((_str(k, f"sensor {role!r} match key"),
                                _str(v, f"sensor {role!r} match {k!r}")) for k, v in match.items())),
            citation=_str(t.get("citation"), f"sensor {role!r} citation"),
            unit_field=unit_field,
            sigma=_float(t["sigma"], f"sensor {role!r} sigma") if "sigma" in t else None,
            relative=_float(t["relative"], f"sensor {role!r} relative") if "relative" in t else None,
            sigma_floor=(_float(t["sigma_floor"], f"sensor {role!r} sigma_floor")
                         if "sigma_floor" in t else None))


@dataclass(frozen=True)
class Scalar:
    """A declared number that is neither a coefficient nor a sigma: a prior width, a process
    noise scale, a drainage area. Carried with its unit and its reason so that a reader meets
    the justification in the same place as the value, and a citation where one exists."""
    name: str
    value: float
    unit: str
    description: str
    citation: str | None = None

    @staticmethod
    def parse(t: dict) -> Scalar:
        _only(t, {"name", "value", "unit", "description", "citation"}, "[[scalar]]")
        name = _str(t.get("name"), "[[scalar]].name")
        return Scalar(name, _float(t.get("value"), f"scalar {name!r} value"),
                      _str(t.get("unit"), f"scalar {name!r} unit"),
                      _str(t.get("description"), f"scalar {name!r} description"),
                      _str(t["citation"], f"scalar {name!r} citation") if "citation" in t else None)


@dataclass(frozen=True)
class Record:
    """Where this site's committed evidence lives, relative to the repository root.

    Optional: a declaration can describe a system that has no record yet, which is exactly
    what a design study does when it asks whether a topology could isolate a fault before
    anything is built. A site with a record names it here so that the last thing tying an
    experiment module to one reservoir stops being a filename in Python.
    """
    directory: str
    manifest: str
    observations: str

    @staticmethod
    def parse(t: dict) -> Record:
        _only(t, {"directory", "manifest", "observations"}, "[record]")
        return Record(_str(t.get("directory"), "[record].directory"),
                      _str(t.get("manifest"), "[record].manifest"),
                      _str(t.get("observations"), "[record].observations"))


@dataclass(frozen=True)
class Declaration:
    key: str
    label: str
    note: str
    states: dict[str, State]
    variants: dict[str, Variant]
    sensors: tuple[Sensor, ...] = ()
    scalars: dict[str, Scalar] = field(default_factory=dict)
    record: Record | None = None

    def sensor(self, role: str) -> Sensor:
        try:
            return next(s for s in self.sensors if s.role == role)
        except StopIteration:
            raise KeyError(
                f"{self.key!r} declares no sensor with role {role!r}; "
                f"declared: {[s.role for s in self.sensors]}") from None

    def scalar(self, name: str) -> float:
        if name not in self.scalars:
            raise KeyError(f"{self.key!r} declares no scalar {name!r}; "
                           f"declared: {sorted(self.scalars)}")
        return self.scalars[name].value

    def constraint_set(self, variant: str, *, readings: dict[str, float] | None = None,
                       variances: dict[str, float] | None = None) -> ConstraintSet:
        """Build the ConstraintSet this variant declares.

        `readings` supplies the value for every row whose b is a reading, keyed by the row's
        name; `variances` the same for every row that declared `b.var_from_sensor`. Both are
        refused when a row did not ask for them and required when it did, so a caller cannot
        quietly substitute a constant for a measurement or the reverse.
        """
        if variant not in self.variants:
            raise KeyError(f"{self.key!r} declares no variant {variant!r}; "
                           f"declared: {sorted(self.variants)}")
        v = self.variants[variant]
        readings, variances = dict(readings or {}), dict(variances or {})
        b, b_var = np.empty(len(v.rows)), np.empty(len(v.rows))
        declared_any = False
        for i, row in enumerate(v.rows):
            if row.b_from is None:
                if row.name in readings:
                    raise ValueError(f"row {row.name!r} declares a constant b; a reading was supplied")
                b[i] = row.b_value
            else:
                if row.name not in readings:
                    raise ValueError(
                        f"row {row.name!r} takes b from reading {row.b_from[1]} of sensor "
                        f"{row.b_from[0]!r}; supply it in `readings`")
                b[i] = _float(readings.pop(row.name), f"reading for row {row.name!r}")
            if row.b_var_from_sensor:
                if row.name not in variances:
                    raise ValueError(f"row {row.name!r} takes b's variance from its reading; "
                                     f"supply it in `variances`")
                b_var[i] = _float(variances.pop(row.name), f"variance for row {row.name!r}")
                declared_any = True
            elif row.b_var_value is not None:
                b_var[i] = row.b_var_value
                declared_any = True
            else:
                b_var[i] = 0.0
        if readings:
            raise ValueError(f"readings supplied for row(s) {sorted(readings)}, which take no reading")
        if variances:
            raise ValueError(f"variances supplied for row(s) {sorted(variances)}, which take none")
        return ConstraintSet(
            version=v.version, A=v.matrix(), b=b, description=v.description,
            b_var=b_var if declared_any else None,
            row_units=tuple(row.unit for row in v.rows))


def loads(text: str, *, source: str = "<string>") -> Declaration:
    t = tomllib.loads(text)
    _only(t, {"schema", "key", "label", "note", "state", "variant", "sensor", "scalar", "record"},
          f"{source} top level")
    schema = _str(t.get("schema"), f"{source} schema")
    if schema != SCHEMA:
        raise ValueError(f"{source} declares schema {schema!r}; this loader reads {SCHEMA!r}")
    states = [State.parse(s) for s in _tables(t.get("state"), f"{source} [[state]]")]
    if not states:
        raise ValueError(f"{source} declares no states")
    by_name: dict[str, State] = {}
    for s in states:
        if s.name in by_name:
            raise ValueError(f"{source} declares state {s.name!r} twice")
        by_name[s.name] = s
    variants = [Variant.parse(v, by_name) for v in _tables(t.get("variant"), f"{source} [[variant]]")]
    if not variants:
        raise ValueError(f"{source} declares no variants")
    by_key: dict[str, Variant] = {}
    for v in variants:
        if v.key in by_key:
            raise ValueError(f"{source} declares variant {v.key!r} twice")
        by_key[v.key] = v
    sensors = tuple(Sensor.parse(s) for s in _tables(t.get("sensor", []), f"{source} [[sensor]]"))
    roles = [s.role for s in sensors]
    if len(set(roles)) != len(roles):
        raise ValueError(f"{source} declares a sensor role twice: {roles}")
    for v in variants:
        for row in v.rows:
            if row.b_from is not None and row.b_from[0] not in roles:
                raise ValueError(
                    f"{source} variant {v.key!r} row {row.name!r} takes b from sensor "
                    f"{row.b_from[0]!r}, which no [[sensor]] declares; declared: {roles}")
    scalars: dict[str, Scalar] = {}
    for s in (Scalar.parse(x) for x in _tables(t.get("scalar", []), f"{source} [[scalar]]")):
        if s.name in scalars:
            raise ValueError(f"{source} declares scalar {s.name!r} twice")
        scalars[s.name] = s
    record = Record.parse(_table(t["record"], f"{source} [record]")) if "record" in t else None
    return Declaration(key=_str(t.get("key"), f"{source} key"),
                       label=_str(t.get("label"), f"{source} label"),
                       note=_str(t.get("note"), f"{source} note"),
                       states=by_name, variants=by_key, sensors=sensors, scalars=scalars,
                       record=record)


def load(path: Path | str) -> Declaration:
    path = Path(path)
    return loads(path.read_text(encoding="utf-8"), source=str(path))
