"""DAF bridge: serialized DAF observations -> Observation, with refusals and provenance.

DAF (the Data Acquisition Fabric) admits evidence; this module consumes it and owns
none of it. Its input is DAF observation dicts exactly as DAF's own
`daf.storage.serialization.observation_to_dict` writes them (data/daf/*.observations.json,
produced by tools/export_daf_fixtures.py from DAF's committed NOAA fixtures). It imports
nothing from DAF, SCOUT or the vendored evidence substrate at runtime; the one function
that does, `verify_ids`, is optional, imports DAF's code from a checkout the caller names,
and is not on the bridging path.

What a record must be. A DAF observation dict with an `id`, an `extraction_method` some
declared selector names, and a `content` carrying at least `property`, `value`,
`measurement_time`, the fields the selector matches on, and -- when the source stated one
-- `uncertainty` with `uncertainty_kind` "stated". Anything else is refused, not skipped.
The bridge was built for DAF's `NoaaWaterLevelMeasurementExtractor` (extraction method
"json:noaa_water_level_measurement_v1", property "water_level", content value / unit /
datum / station_id / measurement_time / sigma / conditions), which is still what
`noaa_water_level_series()` and the (station_id, datum, unit) shorthand declare; nothing
about NOAA is assumed anywhere else.

What the caller must declare (keyword-only):

    series           one `SeriesSelector` per sensor column, in column order -- the
                     extraction_method, the content property and the content fields whose
                     values identify the column. The NOAA (station_id, datum, unit) tuple
                     is accepted as shorthand for `noaa_water_level_series(...)`. A record
                     whose extraction_method some selector declares but which matches no
                     selector is ignored and counted; a record whose extraction_method NO
                     selector declares is refused, so a file of the wrong kind cannot be
                     silently dropped as "ignored". Each selector must pin its unit field.
                     Two selectors alike but for that unit are refused -- the same quantity
                     would enter twice, and the bridge converts no units.
    time_zone        the zone measurement_time is written in. DAF's content carries none
                     (the NOAA binding requests time_zone=gmt in its URL, recorded in
                     data/daf/PROVENANCE.md); "UTC"/"GMT" are supported, anything else
                     raises, and None -- naive parsing -- is refused.
    time_semantics   what a measurement_time names. "instant" (the default, and the only
                     behaviour before this argument existed) reads NOAA's zone-less
                     "YYYY-MM-DD HH:MM[:SS]" in the declared zone. "calendar_day" reads a
                     bare "YYYY-MM-DD", which names a day and not an instant, and so also
                     requires `day_anchor` ("start" / "midpoint" / "end" -- which instant
                     of the day stands for it) and `day_zone_citation` (the source stating
                     that the date is a day in the declared zone, or the assumption being
                     made in its absence, recorded verbatim). The two shapes are never
                     interchanged: a bare date under "instant", or a time under
                     "calendar_day", is refused rather than coerced, so the default can
                     mis-read nothing -- it can only refuse.
    cadence_s        the grid spacing (NOAA's 6-minute product: 360; a daily series:
                     86400). The grid runs from the first to the last reading of the
                     listed series; a reading more than GRID_TOLERANCE_S (1 s) off it is
                     refused, every such reading listed by evidence id; a grid point with
                     no reading is mask False, y NaN.
    arrival_policy   "replay" with an explicit latency_s (arrival_t = t + latency_s), or
                     "as_acquired" (arrival_t from each record's extracted_at, an
                     ISO-8601 instant with an explicit offset, on the same clock; refused
                     where absent or naive, or earlier than the measurement). An
                     observation built from several records arrives when the last of them
                     did; one with no reading arrives at its own t.
    conflict_policy  two records for the same (series, grid point) whose contents differ
                     in any field -- the value, the uncertainty that becomes R, or any other
                     field of Observation.content (extracted_at, confidence and the ids are
                     not content) -- are a conflict: "refuse" raises naming every id;
                     "report_and_keep_both" leaves the point missing (neither value used,
                     neither id carried) and lists the conflict, with the content fields
                     that differ, in provenance. Records with identical content (an
                     identical re-extraction, the same reading in an original and a revised
                     window) are deduplicated with every id kept: the reading enters once.
    daf_commit       the DAF commit the records came from (caller-supplied; recorded).

Uncertainty. R_ii = uncertainty**2 where uncertainty_kind == "stated" (NOAA's `s`: the
standard deviation of the 1-second samples behind the 6-minute value, as DAF records it).
A series with a reading that states none is refused unless the caller passes
`declared_sigma[series]` for it -- a `DeclaredSigma`, never a bare number, because a
`citation` is required: an R the source did not state is an assumption, and this bridge
does not carry an assumption without its source. A DeclaredSigma gives either an absolute
`sigma` or a `relative` fraction of |value| with a `sigma_floor` (a rating stated as a
percentage says nothing at zero flow, and a zero R would tell the filter the reading is
exact). A series is either all source-stated or all consumer-declared -- declaring a sigma
for a series whose readings state their own is refused, so the two are never mixed in one
series. A stated 0.0 (NOAA reports s to 1 mm) is passed through as R = 0 and counted in
provenance, never floored.

Output: BridgedSeries(inputs, observations, provenance, source_ids, component_evidence).
inputs.t is seconds from the declared epoch (the first grid point) and u_commanded is
zeros: nothing is commanded in a gauge record. Each Observation's source_ids are the
selectors' declared source_ids (for NOAA, "noaa:<station>:<datum>:<unit>"), its
evidence_ids the DAF ids behind each present component in column order
(component_evidence keeps them per component). Provenance: daf_commit, the selectors,
record counts (n_records_in = n_used + n_ignored + n_in_conflict), n_deduplicated,
conflicts, the grid check, time semantics (zone, and for a calendar day its anchor and the
citation), epoch, cadence, arrival policy, R_source per series -- with the citation where
it is consumer-declared -- and a sha256 over the sorted evidence ids used.

The DAF boundary this respects (docs/DAF_STATE_SPACE_BOUNDARY.md, sections 10-13, 18):
t is the source event time read out of Observation.content, never retrieved_at or
extracted_at; extracted_at is read only under "as_acquired", as an arrival clock, never as
identity or for deduplication, which compares whole contents; contradictory observations
are never averaged, never merged, and the later (revised) one never silently wins -- a
revision is not a state transition; nothing here needs a RawDocument, an adapter or a DAF
type; DAF's ids are never recomputed here (verify_ids asks DAF's own code to do that), only
compared: one id must name one (record_ids, extraction_method, content), the fields DAF's
Observation.id is computed from; and DAF's evidence ids are carried as provenance
(Observation.evidence_ids, read by no estimator), never as model identity.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..schema import Observation
from ..testbed.inputs import PublicInputs

__all__ = [
    "BRIDGE_VERSION", "DAF_NOAA_MEASUREMENT_METHOD", "GRID_TOLERANCE_S", "DAY_ANCHORS", "TIME_SEMANTICS",
    "BridgeRefusal", "BridgedSeries", "DeclaredSigma", "SeriesSelector", "bridge", "load_records",
    "noaa_water_level_series", "series_source_id", "strict_json_loads", "verify_ids",
]

BRIDGE_VERSION = "daf-measurement-bridge-v2"
DAF_NOAA_MEASUREMENT_METHOD = "json:noaa_water_level_measurement_v1"
PROPERTY = "water_level"
STATED = "stated"
GRID_TOLERANCE_S = 1.0
CONFLICT_POLICIES = ("refuse", "report_and_keep_both")
ARRIVAL_POLICIES = ("replay", "as_acquired")
TIME_SEMANTICS = ("instant", "calendar_day")
DAY_ANCHORS = ("start", "midpoint", "end")
_DAY_ANCHOR_SECONDS = {"start": 0.0, "midpoint": 43200.0, "end": 86400.0}
_ZONES = {"UTC": timezone.utc, "GMT": timezone.utc}
_MEASUREMENT_TIME = re.compile(r"\A(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})(?::(\d{2}))?\Z")
_CALENDAR_DAY = re.compile(r"\A(\d{4})-(\d{2})-(\d{2})\Z")

SeriesKey = tuple[str, str, str]   # (station_id, datum, unit): the NOAA shorthand's key


class BridgeRefusal(ValueError):
    """The bridge will not build an observation record from this input. `reason` is a short
    code (e.g. "off_grid", "missing_uncertainty", "conflict", "time_zone", "mixed_units");
    `evidence_ids` lists every record the refusal is about, when it is about records."""

    def __init__(self, reason: str, message: str, evidence_ids: Iterable[str] = ()):
        self.reason = reason
        self.evidence_ids = tuple(evidence_ids)
        super().__init__(f"[{reason}] {message}")


# ---------------------------------------------------------------------------
# reading DAF's files
# ---------------------------------------------------------------------------

def strict_json_loads(text: str) -> Any:
    """json.loads refusing what is not JSON: the bare NaN / Infinity / -Infinity that
    Python's json accepts (as DAF's own strict_json_loads refuses them) and, in addition,
    a key repeated within one object (which plain json.loads silently resolves to the
    last value)."""
    def refuse_constant(constant: str):
        raise BridgeRefusal("non_json", f"bare literal {constant!r} is a Python json extension, not JSON")

    def no_duplicate_keys(pairs):
        out: dict = {}
        for k, v in pairs:
            if k in out:
                raise BridgeRefusal("non_json", f"key {k!r} repeated within one JSON object")
            out[k] = v
        return out

    return json.loads(text, parse_constant=refuse_constant, object_pairs_hook=no_duplicate_keys)


def load_records(path: str | Path) -> list[dict]:
    """A data/daf/*.observations.json file: a JSON array of DAF observation dicts, read
    with the strict loader."""
    data = strict_json_loads(Path(path).read_bytes().decode("utf-8"))
    if not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
        raise BridgeRefusal("not_records", f"{path}: expected a JSON array of DAF observation objects")
    return data


def series_source_id(key: SeriesKey) -> str:
    station, datum, unit = key
    return f"noaa:{station}:{datum}:{unit}"


# ---------------------------------------------------------------------------
# what the caller declares
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeriesSelector:
    """One sensor column, declared in full: which DAF observations belong to it.

    A record joins this column when its `extraction_method` equals `extraction_method`,
    its `content["property"]` equals `property`, and every (field, value) pair in `match`
    equals the record's `content[field]`. Nothing is inferred: a record that matches no
    selector but whose extraction_method IS declared by some selector is ignored and
    counted; a record whose extraction_method is declared by no selector is refused, so a
    file of the wrong kind can never be silently dropped.

    `source_id` names the column in Observation.source_ids and in provenance. `unit_field`
    names the content field carrying the unit (NOAA and USGS both have one); `match` must
    pin its non-empty value. Units are not converted, only recorded and checked, so records
    in different units cannot silently join one column.
    """
    source_id: str
    extraction_method: str
    property: str
    match: tuple[tuple[str, str], ...]
    unit_field: str = "unit"

    def __post_init__(self):
        for name in ("source_id", "extraction_method", "property", "unit_field"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"SeriesSelector.{name} must be a non-empty string; got {v!r}")
        m = self.match
        if isinstance(m, Mapping):
            m = tuple(sorted((str(k), str(v)) for k, v in m.items()))
        else:
            try:
                m = tuple(sorted((str(k), str(v)) for k, v in m))
            except (TypeError, ValueError):
                raise ValueError(f"SeriesSelector.match must be a mapping or pairs of content field -> "
                                 f"required value; got {self.match!r}") from None
        if not m:
            raise ValueError(f"SeriesSelector {self.source_id!r} matches on nothing; declare at least one "
                             "content field (e.g. the station or monitoring-location id)")
        if len({k for k, _ in m}) != len(m):
            raise ValueError(f"SeriesSelector {self.source_id!r} names a match field twice: {m}")
        if not dict(m).get(self.unit_field, "").strip():
            raise ValueError(f"SeriesSelector {self.source_id!r} must pin a non-empty unit in "
                             f"match[{self.unit_field!r}]; the bridge converts no units")
        object.__setattr__(self, "match", m)

    @property
    def match_map(self) -> dict[str, str]:
        return dict(self.match)

    @property
    def unit(self) -> str:
        """The required unit this selector pins through `unit_field`."""
        return self.match_map[self.unit_field]

    def describe(self) -> dict:
        return {"source_id": self.source_id, "extraction_method": self.extraction_method,
                "property": self.property, "match": {k: v for k, v in self.match},
                "unit_field": self.unit_field}


def noaa_water_level_series(station_id: str, datum: str, unit: str) -> SeriesSelector:
    """The NOAA per-measurement water-level column the bridge was built for. A `series`
    entry given as the (station_id, datum, unit) tuple means exactly this."""
    return SeriesSelector(
        source_id=series_source_id((station_id, datum, unit)),
        extraction_method=DAF_NOAA_MEASUREMENT_METHOD, property=PROPERTY,
        match=(("datum", datum), ("station_id", station_id), ("unit", unit)))


@dataclass(frozen=True)
class DeclaredSigma:
    """Uncertainty the CONSUMER declares for a series whose records state none.

    Exactly one of `sigma` (absolute, in the series' own unit) or `relative` (a fraction of
    |value|, per reading) is given. `relative` also needs `sigma_floor`, the absolute value
    R falls back to where relative*|value| is smaller -- a rating stated as a percentage says
    nothing at zero flow, and a zero R would tell the filter the reading is exact.
    `citation` is required and must be non-empty: a consumer-declared R is an assumption, and
    an assumption without a source is the thing this repository exists to refuse. It is
    recorded verbatim in provenance and carried into every report.
    """
    citation: str
    sigma: float | None = None
    relative: float | None = None
    sigma_floor: float | None = None

    def __post_init__(self):
        if not isinstance(self.citation, str) or not self.citation.strip():
            raise ValueError("DeclaredSigma.citation is required: say where the number comes from")
        given = [n for n in ("sigma", "relative") if getattr(self, n) is not None]
        if len(given) != 1:
            raise ValueError(f"DeclaredSigma takes exactly one of sigma= or relative=; got {given or 'neither'}")
        for name in ("sigma", "relative", "sigma_floor"):
            v = getattr(self, name)
            if v is None:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) <= 0:
                raise ValueError(f"DeclaredSigma.{name} must be a finite positive number; got {v!r}")
            object.__setattr__(self, name, float(v))
        if self.relative is not None and self.sigma_floor is None:
            raise ValueError("DeclaredSigma(relative=...) needs sigma_floor: the absolute sigma to use where "
                             "relative * |value| would be smaller (a percentage rating says nothing at zero)")
        if self.relative is None and self.sigma_floor is not None:
            raise ValueError("DeclaredSigma.sigma_floor applies only to a relative declaration")

    def sigma_for(self, value: float) -> float:
        if self.sigma is not None:
            return self.sigma
        return max(self.relative * abs(value), self.sigma_floor)

    def describe(self) -> dict:
        d = {"kind": "consumer-declared", "citation": self.citation}
        if self.sigma is not None:
            d["sigma"] = self.sigma
            d["rule"] = "R_ii = sigma**2 for every reading of this series"
        else:
            d["relative"] = self.relative
            d["sigma_floor"] = self.sigma_floor
            d["rule"] = "R_ii = max(relative * |value|, sigma_floor)**2, per reading"
        return d


# ---------------------------------------------------------------------------
# the bridged record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BridgedSeries:
    """What the estimator side receives: public inputs, one Observation per grid point
    (indexed by sampling step on inputs.t, as runner.run() requires), and the provenance
    of how they were built. component_evidence[k][i] is the tuple of DAF evidence ids
    behind y[k, i] (() where missing)."""
    inputs: PublicInputs
    observations: list[Observation]
    provenance: dict
    source_ids: tuple[str, ...]
    component_evidence: tuple[tuple[tuple[str, ...], ...], ...]

    @property
    def evidence_ids_used(self) -> tuple[str, ...]:
        """Every DAF evidence id carried by an observation, sorted, each once."""
        return tuple(sorted({e for o in self.observations for e in o.evidence_ids}))


@dataclass
class _Reading:
    eid: str
    col: int
    measurement_time: str
    instant: datetime
    value: float
    stated_sd: float | None       # the source-stated uncertainty (a standard deviation), or None
    extracted_at: Any
    content_json: str             # canonical Observation.content: what deduplication compares
    idx: int = -1                 # grid index, set once the epoch is known


# ---------------------------------------------------------------------------
# argument checks
# ---------------------------------------------------------------------------

def _check_series(series) -> list[SeriesSelector]:
    """Normalise `series` to selectors. A (station_id, datum, unit) tuple is the NOAA
    shorthand; a SeriesSelector is taken as declared."""
    if series is None or isinstance(series, (str, bytes)) or not isinstance(series, Sequence) or not series:
        raise ValueError("series must be a non-empty list of SeriesSelector objects, or of the NOAA "
                         "(station_id, datum, unit) shorthand tuples")
    sels: list[SeriesSelector] = []
    for s in series:
        if isinstance(s, SeriesSelector):
            sels.append(s)
            continue
        if (isinstance(s, (str, bytes)) or not isinstance(s, Sequence) or len(s) != 3
                or not all(isinstance(x, str) and x.strip() for x in s)):
            raise ValueError(f"series entry {s!r} is neither a SeriesSelector nor a "
                             "(station_id, datum, unit) tuple of non-empty strings")
        sels.append(noaa_water_level_series(s[0], s[1], s[2]))
    ids = [sel.source_id for sel in sels]
    if len(set(ids)) != len(ids):
        raise ValueError(f"series lists a source_id twice: {ids}")
    keys = [(sel.extraction_method, sel.property, sel.match) for sel in sels]
    if len(set(keys)) != len(keys):
        raise ValueError("series lists the same (extraction_method, property, match) twice under different "
                         "source_ids: the same records would enter as two sensors")
    # one quantity in two units is never pooled: the same selector but for its unit field
    quantities: dict[tuple, set[str]] = defaultdict(set)
    for sel in sels:
        rest = tuple(sorted((k, v) for k, v in sel.match if k != sel.unit_field))
        quantities[(sel.extraction_method, sel.property, rest)].add(sel.unit)
    for (method, prop, rest), us in quantities.items():
        if len(us) > 1:
            where = ", ".join(f"{k}={v}" for k, v in rest) or prop
            raise BridgeRefusal(
                "mixed_units",
                f"{where} is listed in units {sorted(us)}: one quantity in two units would enter as two "
                "sensors of the same readings, and the bridge converts no units")
    return sels


def _check_time_semantics(time_semantics, time_zone, day_anchor, day_zone_citation) -> dict:
    """How a record's measurement_time becomes a point on the clock. Declared, never guessed:
    'instant' reads NOAA's zone-less 'YYYY-MM-DD HH:MM[:SS]' in the declared zone; 'calendar_day'
    reads a bare 'YYYY-MM-DD', which names a day and not an instant, and so additionally needs
    the anchor within the day and a citation for reading that date as a day in the declared
    zone -- the source that says so, or the assumption being made in its absence."""
    if time_semantics not in TIME_SEMANTICS:
        raise ValueError(f"time_semantics must be one of {TIME_SEMANTICS}; got {time_semantics!r}")
    zone = _check_zone(time_zone)
    if time_semantics == "instant":
        if day_anchor is not None:
            raise ValueError("day_anchor applies only to time_semantics 'calendar_day'")
        if day_zone_citation is not None:
            raise ValueError("day_zone_citation applies only to time_semantics 'calendar_day'")
        return {"time_semantics": "instant", "time_zone": zone, "time_zone_declared": time_zone,
                "rule": "measurement_time is 'YYYY-MM-DD HH:MM[:SS]', read in the declared zone"}
    if day_anchor not in DAY_ANCHORS:
        raise BridgeRefusal("day_anchor", f"time_semantics 'calendar_day' needs day_anchor, one of {DAY_ANCHORS}; "
                                          f"got {day_anchor!r}. A date names a day; which instant of it stands "
                                          "for the day is the caller's declaration, not the bridge's guess")
    if not isinstance(day_zone_citation, str) or not day_zone_citation.strip():
        raise BridgeRefusal(
            "day_zone_citation",
            "time_semantics 'calendar_day' needs day_zone_citation: a bare date carries no zone, so reading it "
            f"as a day in {zone} is an assumption. Cite the source that states the zone, or state the assumption "
            "being made in its absence; the bridge records it verbatim and refuses to assume silently")
    return {"time_semantics": "calendar_day", "time_zone": zone, "time_zone_declared": time_zone,
            "day_anchor": day_anchor, "day_anchor_offset_s": _DAY_ANCHOR_SECONDS[day_anchor],
            "day_zone_citation": day_zone_citation.strip(),
            "rule": f"measurement_time is a bare 'YYYY-MM-DD'; t is that day's {day_anchor} "
                    f"({_DAY_ANCHOR_SECONDS[day_anchor]:.0f} s after 00:00) in {zone}, on the cited assumption"}


def _check_zone(time_zone) -> str:
    if time_zone is None:
        raise BridgeRefusal("time_zone", "time_zone is required: DAF's measurement_time carries no zone, and "
                                         "naive parsing is refused")
    if not isinstance(time_zone, str) or time_zone.upper() not in _ZONES:
        raise BridgeRefusal("time_zone", f"unsupported time_zone {time_zone!r}; supported: {sorted(_ZONES)}")
    return "UTC"   # NOAA's time_zone=gmt is Greenwich time with no daylight saving: the UTC clock


def _check_cadence(cadence_s) -> float:
    if isinstance(cadence_s, bool) or not isinstance(cadence_s, (int, float)):
        raise ValueError(f"cadence_s must be a number of seconds; got {cadence_s!r}")
    c = float(cadence_s)
    if not math.isfinite(c) or c <= 2.0 * GRID_TOLERANCE_S:
        raise ValueError(f"cadence_s must be finite and above {2.0 * GRID_TOLERANCE_S} s (twice the grid "
                         f"tolerance); got {cadence_s!r}")
    return c


def _check_arrival(arrival_policy, latency_s) -> dict:
    if arrival_policy not in ARRIVAL_POLICIES:
        raise ValueError(f"arrival_policy must be one of {ARRIVAL_POLICIES}; got {arrival_policy!r}")
    if arrival_policy == "replay":
        if latency_s is None or isinstance(latency_s, bool) or not isinstance(latency_s, (int, float)):
            raise ValueError("arrival_policy 'replay' needs an explicit latency_s (seconds, >= 0)")
        lat = float(latency_s)
        if not math.isfinite(lat) or lat < 0.0:
            raise ValueError(f"latency_s must be finite and >= 0; got {latency_s!r}")
        return {"policy": "replay", "latency_s": lat, "rule": "arrival_t = t + latency_s"}
    if latency_s is not None:
        raise ValueError("arrival_policy 'as_acquired' takes its arrival times from extracted_at; "
                         "a latency_s would contradict it")
    return {"policy": "as_acquired", "clock": "extracted_at",
            "rule": "arrival_t = the latest extracted_at of the records behind the observation, in seconds from "
                    "the epoch (never before t); t where no reading"}


def _check_declared(declared_sigma, sels: list[SeriesSelector]) -> dict[int, DeclaredSigma]:
    """declared_sigma maps a series -- by source_id, by SeriesSelector, or by the NOAA
    (station_id, datum, unit) shorthand -- to a DeclaredSigma. A bare number is refused:
    a consumer-declared R has to say where it comes from."""
    if declared_sigma is None:
        return {}
    if not isinstance(declared_sigma, Mapping):
        raise ValueError("declared_sigma must map a series (its source_id, its SeriesSelector, or the NOAA "
                         "(station_id, datum, unit) shorthand) -> DeclaredSigma")
    col_of: dict[str, int] = {sel.source_id: i for i, sel in enumerate(sels)}
    out: dict[int, DeclaredSigma] = {}
    for k, v in declared_sigma.items():
        if isinstance(k, SeriesSelector):
            name = k.source_id
        elif isinstance(k, (list, tuple)) and len(k) == 3 and all(isinstance(x, str) for x in k):
            name = series_source_id((k[0], k[1], k[2]))
        elif isinstance(k, str):
            name = k
        else:
            raise ValueError(f"declared_sigma key {k!r} is not a source_id, a SeriesSelector, or a "
                             "(station_id, datum, unit) tuple")
        if name not in col_of:
            raise ValueError(f"declared_sigma names {k!r}, which is not a listed series")
        if not isinstance(v, DeclaredSigma):
            raise ValueError(
                f"declared_sigma[{k!r}] is {v!r}; a consumer-declared uncertainty must be a DeclaredSigma, which "
                "requires a citation -- an R the source did not state is an assumption, and this bridge does not "
                "carry an assumption without its source")
        if col_of[name] in out:
            raise ValueError(f"declared_sigma names series {name!r} twice")
        out[col_of[name]] = v
    return out


# ---------------------------------------------------------------------------
# record checks
# ---------------------------------------------------------------------------

def _finite_number(x) -> bool:
    return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(float(x))


def _record_key(r, sels: list[SeriesSelector], methods: frozenset[str]) -> tuple[str, Mapping, int | None]:
    """The id, the content, and the column of a record -- or None where it matches no declared
    selector. A record whose extraction_method no selector declares is refused, so a file of
    the wrong kind cannot be silently dropped as 'ignored'."""
    if not isinstance(r, Mapping):
        raise BridgeRefusal("malformed", f"a record is a {type(r).__name__}, not a DAF observation object")
    eid = r.get("id")
    if not isinstance(eid, str) or not eid:
        raise BridgeRefusal("malformed", f"a record has no evidence id: {dict(r)!r}"[:300])
    method = r.get("extraction_method")
    if method not in methods:
        reason = "not_noaa_measurement" if methods == frozenset({DAF_NOAA_MEASUREMENT_METHOD}) else "not_declared"
        raise BridgeRefusal(reason,
                            f"record {eid} has extraction_method {method!r}; the declared series read "
                            f"{', '.join(repr(m) for m in sorted(methods))} observations only", [eid])
    c = r.get("content")
    if not isinstance(c, Mapping):
        raise BridgeRefusal("malformed", f"record {eid} has no content object", [eid])
    for i, sel in enumerate(sels):
        if sel.extraction_method != method or c.get("property") != sel.property:
            continue
        if all(c.get(f) == v for f, v in sel.match):
            return eid, c, i
    # matched no column: the fields a selector for this method matches on must still be usable,
    # so a malformed record cannot hide in the ignored count
    for sel in sels:
        if sel.extraction_method != method:
            continue
        for f, _ in sel.match:
            if f in c and not (isinstance(c.get(f), str) and c.get(f)):
                raise BridgeRefusal("malformed", f"record {eid} has no usable content[{f!r}]", [eid])
    return eid, c, None


def _measurement_instant(eid: str, text, tz, semantics: Mapping) -> datetime:
    """The instant a record's measurement_time names, under the declared time semantics. The
    two shapes are never interchanged: a bare date under 'instant', or a time under
    'calendar_day', is refused rather than coerced."""
    if semantics["time_semantics"] == "calendar_day":
        m = _CALENDAR_DAY.match(text) if isinstance(text, str) else None
        if m is None:
            raise BridgeRefusal("measurement_time",
                                f"record {eid} has measurement_time {text!r}; time_semantics 'calendar_day' "
                                "expects a bare 'YYYY-MM-DD'", [eid])
        y, mo, d = (int(g) for g in m.groups())
        try:
            return datetime(y, mo, d, tzinfo=tz) + timedelta(seconds=semantics["day_anchor_offset_s"])
        except ValueError as exc:
            raise BridgeRefusal("measurement_time", f"record {eid}: measurement_time {text!r}: {exc}",
                                [eid]) from None
    m = _MEASUREMENT_TIME.match(text) if isinstance(text, str) else None
    if m is None:
        raise BridgeRefusal("measurement_time",
                            f"record {eid} has measurement_time {text!r}; expected NOAA's zone-less "
                            "'YYYY-MM-DD HH:MM[:SS]', to be read in the declared time_zone", [eid])
    y, mo, d, H, M, S = (int(g) if g is not None else 0 for g in m.groups())
    try:
        return datetime(y, mo, d, H, M, S, tzinfo=tz)
    except ValueError as exc:
        raise BridgeRefusal("measurement_time", f"record {eid}: measurement_time {text!r}: {exc}", [eid]) from None


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _ignored_key(r: Mapping, c: Mapping, sels: list[SeriesSelector]) -> tuple[tuple[str, Any], ...]:
    """How an unmatched record is counted: the values it carries in the fields some declared
    selector of its own extraction_method matches on, plus its property. Enough to say what
    was left out without listing every record."""
    method = r.get("extraction_method")
    fields = sorted({f for sel in sels if sel.extraction_method == method for f, _ in sel.match})
    return (("property", c.get("property")),) + tuple((f, c.get(f)) for f in fields)


def _reading(eid: str, c: Mapping, col: int, sel: SeriesSelector, extracted_at, tz, semantics: Mapping) -> _Reading:
    # the selector has already matched extraction_method, property and every match field
    cond = c.get("conditions")
    if cond is not None:
        if not isinstance(cond, Mapping):
            raise BridgeRefusal("malformed", f"record {eid}: conditions {cond!r} is not an object", [eid])
        # NOAA states the datum twice: content["datum"] and conditions["datum"]. Where the series is
        # selected on datum, a present conditions block must state it and agree (the original rule).
        # Every other selected field may be absent from conditions but must not contradict it.
        for f, v in sel.match:
            if (f == "datum" or f in cond) and cond.get(f) != v:
                raise BridgeRefusal("malformed",
                                    f"record {eid}: conditions {dict(cond)!r} disagree with "
                                    f"{f} {v!r}", [eid])
    if not _finite_number(c.get("value")):
        raise BridgeRefusal("malformed", f"record {eid} has a non-numeric or non-finite value {c.get('value')!r}",
                            [eid])
    stated = None
    if c.get("uncertainty_kind") == STATED:
        u = c.get("uncertainty")
        if not _finite_number(u) or float(u) < 0.0:
            raise BridgeRefusal("malformed", f"record {eid} states uncertainty {u!r}, not a finite value >= 0",
                                [eid])
        stated = float(u)
    return _Reading(
        eid=eid, col=col, measurement_time=c.get("measurement_time"),
        instant=_measurement_instant(eid, c.get("measurement_time"), tz, semantics), value=float(c["value"]),
        stated_sd=stated, extracted_at=extracted_at, content_json=_canonical(dict(c)),
    )


def _extracted_instant(eid: str, raw) -> datetime:
    if not isinstance(raw, str) or not raw:
        raise BridgeRefusal("arrival", f"record {eid} has no extracted_at; arrival_policy 'as_acquired' "
                                       "cannot place it on the clock", [eid])
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        raise BridgeRefusal("arrival", f"record {eid}: extracted_at {raw!r} is not ISO-8601", [eid]) from None
    if dt.tzinfo is None:
        raise BridgeRefusal("arrival", f"record {eid}: extracted_at {raw!r} names no offset; naive times are "
                                       "refused", [eid])
    return dt.astimezone(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# the bridge
# ---------------------------------------------------------------------------

def bridge(
    records: Iterable[Mapping],
    *,
    series: Sequence[SeriesSelector | SeriesKey],
    time_zone: str,
    cadence_s: float,
    arrival_policy: str,
    conflict_policy: str,
    daf_commit: str,
    time_semantics: str = "instant",
    day_anchor: str | None = None,
    day_zone_citation: str | None = None,
    latency_s: float | None = None,
    declared_sigma: Mapping[Any, DeclaredSigma] | None = None,
) -> BridgedSeries:
    """DAF observation dicts -> BridgedSeries. See the module docstring for the contract;
    every refusal raises BridgeRefusal (a ValueError) or ValueError, never a partial record."""
    sels = _check_series(series)
    semantics = _check_time_semantics(time_semantics, time_zone, day_anchor, day_zone_citation)
    zone = semantics["time_zone"]
    tz = _ZONES[zone]
    cadence = _check_cadence(cadence_s)
    arrival = _check_arrival(arrival_policy, latency_s)
    if conflict_policy not in CONFLICT_POLICIES:
        raise ValueError(f"conflict_policy must be one of {CONFLICT_POLICIES}; got {conflict_policy!r}")
    declared = _check_declared(declared_sigma, sels)
    if not isinstance(daf_commit, str) or not daf_commit.strip():
        raise ValueError("daf_commit must name the DAF commit the records came from")

    records = list(records)
    if not records:
        raise BridgeRefusal("empty", "no records")
    methods = frozenset(sel.extraction_method for sel in sels)
    source_ids = tuple(sel.source_id for sel in sels)

    # 1. classify: listed series or ignored (and counted). A content-addressed id names one
    # (record_ids, extraction_method, content) -- the fields DAF computes Observation.id from
    # -- wherever it appears: in one grid point, two, or a group that is ignored. The id is
    # compared, never recomputed.
    readings: list[_Reading] = []
    ignored: Counter = Counter()
    identity_of: dict[str, str] = {}
    for r in records:
        eid, c, col = _record_key(r, sels, methods)
        rids = r.get("record_ids")
        ident = _canonical({"record_ids": sorted(map(str, rids)) if isinstance(rids, (list, tuple)) else rids,
                            "extraction_method": r.get("extraction_method"), "content": dict(c)})
        if identity_of.setdefault(eid, ident) != ident:
            raise BridgeRefusal("id_reused", f"evidence id {eid} carries two different (record_ids, extraction_method, "
                                             "content); a content-addressed id cannot", [eid])
        if col is None:
            ignored[_ignored_key(r, c, sels)] += 1
            continue
        readings.append(_reading(eid, c, col, sels[col], r.get("extracted_at"), tz, semantics))

    # 2. uncertainty: each series is all source-stated or all consumer-declared
    by_col: dict[int, list[_Reading]] = defaultdict(list)
    for rd in readings:
        by_col[rd.col].append(rd)
    for i, sel in enumerate(sels):
        rs = by_col.get(i, [])
        if not rs:
            raise BridgeRefusal("empty_series", f"series {source_ids[i]} matched no record "
                                                f"({len(records)} in, {sum(ignored.values())} ignored)")
        stated = [rd.eid for rd in rs if rd.stated_sd is not None]
        unstated = [rd.eid for rd in rs if rd.stated_sd is None]
        if i in declared and stated:
            raise BridgeRefusal(
                "mixed_uncertainty",
                f"declared_sigma is given for {source_ids[i]}, but {len(stated)} of its readings state their own "
                "uncertainty; source-stated and consumer-declared R are never mixed in one series", stated)
        if i not in declared and unstated:
            raise BridgeRefusal(
                "missing_uncertainty",
                f"{len(unstated)} reading(s) of {source_ids[i]} state no uncertainty (uncertainty_kind != "
                f"{STATED!r}); pass declared_sigma for that series to declare one", unstated)

    # 3. the grid: from the first to the last reading of the listed series
    epoch = min(rd.instant for rd in readings)
    off_grid = []
    max_off = 0.0
    for rd in readings:
        dt = (rd.instant - epoch).total_seconds()
        rd.idx = int(round(dt / cadence))
        off = dt - rd.idx * cadence
        if abs(off) > GRID_TOLERANCE_S:
            off_grid.append((rd.eid, rd.measurement_time, off))
        else:
            max_off = max(max_off, abs(off))
    if off_grid:
        listing = "; ".join(f"{e} ({mt}, {off:+.1f} s)" for e, mt, off in off_grid[:10])
        raise BridgeRefusal(
            "off_grid",
            f"{len(off_grid)} reading(s) lie more than {GRID_TOLERANCE_S} s off the {cadence} s grid from "
            f"{_iso(epoch)}: {listing}{' ...' if len(off_grid) > 10 else ''}", [e for e, _, _ in off_grid])
    n_grid = max(rd.idx for rd in readings) + 1

    # 4. one value per (series, grid point): deduplicate records whose Observation.content is
    # identical, surface every other difference as a conflict. Whole contents are compared --
    # never stamps, and never only the fields the bridge reads -- so two observations whose
    # contents DAF keeps apart are never merged into one reading.
    groups: dict[tuple[int, int], list[_Reading]] = defaultdict(list)
    for rd in readings:
        groups[(rd.col, rd.idx)].append(rd)
    used: dict[tuple[int, int], tuple[float, float, tuple[str, ...], list[_Reading]]] = {}
    conflicts: list[dict] = []
    n_used = n_dedup = n_conflict = 0
    for (col, idx), rs in sorted(groups.items()):
        dec = declared.get(col)
        sd_of = (lambda rd, d=dec: d.sigma_for(rd.value)) if dec is not None else (lambda rd: rd.stated_sd)
        ids = tuple(sorted({rd.eid for rd in rs}))
        by_content: dict[str, list[_Reading]] = defaultdict(list)
        for rd in rs:
            by_content[rd.content_json].append(rd)
        if len(by_content) == 1:
            used[(col, idx)] = (rs[0].value, sd_of(rs[0]), ids, rs)
            n_used += len(rs)
            n_dedup += len(rs) - 1
            continue
        contents = [json.loads(cj) for cj in by_content]
        differs = sorted(f for f in set().union(*contents)
                         if len({_canonical(c.get(f)) for c in contents}) > 1)
        entry = {
            "series": source_ids[col], "grid_index": idx, "measurement_time": rs[0].measurement_time,
            "readings": [{"evidence_id": e, "value": v, "sigma": s}
                         for v, s, e in sorted({(rd.value, sd_of(rd), rd.eid) for rd in rs})],
            "differs_in": differs,
            "resolution": "point left missing; neither value used",
        }
        if conflict_policy == "refuse":
            listing = ", ".join(f"{x['evidence_id']} = {x['value']} (sigma {x['sigma']})" for x in entry["readings"])
            raise BridgeRefusal("conflict", f"{source_ids[col]} at {rs[0].measurement_time}: {listing}", ids)
        conflicts.append(entry)
        n_conflict += len(rs)

    # 5. arrival
    t_grid = np.arange(n_grid, dtype=float) * cadence
    arrival_s: dict[int, float] = {}
    if arrival["policy"] == "as_acquired":
        unusable: list[str] = []
        early: list[str] = []
        for (col, idx), (_, _, _, rs) in sorted(used.items()):
            for rd in rs:
                try:
                    ext = _extracted_instant(rd.eid, rd.extracted_at)
                except BridgeRefusal:
                    unusable.append(rd.eid)
                    continue
                if ext < rd.instant:
                    early.append(rd.eid)
                    continue
                arrival_s[idx] = max(arrival_s.get(idx, -math.inf), (ext - epoch).total_seconds())
        if unusable:
            raise BridgeRefusal("arrival", f"{len(unusable)} record(s) have no usable extracted_at (absent, not "
                                           "ISO-8601, or naive); 'as_acquired' cannot place them on the clock",
                                sorted(set(unusable)))
        if early:
            raise BridgeRefusal("arrival", f"{len(early)} record(s) were extracted before they were measured",
                                sorted(set(early)))

    # 6. the observation record
    n_s = len(sels)
    observations: list[Observation] = []
    component_evidence = []
    for k in range(n_grid):
        y = np.full(n_s, np.nan)
        var = np.full(n_s, np.nan)
        mask = np.zeros(n_s, dtype=bool)
        comp: list[tuple[str, ...]] = []
        for i in range(n_s):
            u = used.get((i, k))
            if u is None:
                comp.append(())
                continue
            value, sd, ids, _ = u
            y[i], var[i], mask[i] = value, sd ** 2, True
            comp.append(ids)
        tk = float(t_grid[k])
        if arrival["policy"] == "replay":
            arr = tk + arrival["latency_s"]
        else:
            arr = max(arrival_s.get(k, tk), tk)
        observations.append(Observation(
            t=tk, arrival_t=float(arr), y=y, R=np.diag(var), mask=mask, source_ids=source_ids,
            evidence_ids=tuple(e for ids in comp for e in ids)))
        component_evidence.append(tuple(comp))

    used_ids = sorted({e for (_, _, ids, _) in used.values() for e in ids})
    r_source = {}
    for i, sel in enumerate(sels):
        sds = [sd for (col, _), (_, sd, _, _) in used.items() if col == i]
        dec = declared.get(i)
        if dec is not None:
            r_source[source_ids[i]] = {
                **dec.describe(), "n_points": len(sds),
                "sigma_min": min(sds) if sds else None, "sigma_max": max(sds) if sds else None,
                "note": "declared by the caller (declared_sigma) for a series whose readings state no uncertainty"}
        else:
            r_source[source_ids[i]] = {
                "kind": "source-stated", "rule": "R_ii = uncertainty**2 where uncertainty_kind == 'stated'",
                "n_points": len(sds), "sigma_min": min(sds) if sds else None, "sigma_max": max(sds) if sds else None,
                "n_zero": sum(1 for s in sds if s == 0.0)}
    methods_sorted = sorted(methods)
    provenance = {
        "bridge": "set_lcm.bridge.daf",
        "bridge_version": BRIDGE_VERSION,
        "daf_commit": daf_commit,
        "extraction_method": methods_sorted[0] if len(methods_sorted) == 1 else methods_sorted,
        "series": list(source_ids),
        "selectors": [sel.describe() for sel in sels],
        "n_records_in": len(records),
        "n_used": n_used,
        "n_ignored": sum(ignored.values()),
        "ignored": [{"series_key": [v for _, v in k[1:]], "matched_on": {f: v for f, v in k}, "n": n}
                    for k, n in sorted(ignored.items(), key=lambda kv: _canonical(kv[0]))],
        "n_in_conflict": n_conflict,
        "n_deduplicated": n_dedup,
        "conflicts": conflicts,
        "conflict_policy": conflict_policy,
        "off_grid": {"tolerance_s": GRID_TOLERANCE_S, "n_refused": 0, "max_abs_offset_s": max_off,
                     "rule": "a reading further off the grid than tolerance_s is refused (bridge() raises, listing "
                             "every such evidence id)"},
        "time_zone": zone,
        "time_zone_declared": time_zone,
        "time": semantics,
        "epoch_iso": _iso(epoch),
        "cadence_s": cadence,
        "n_grid": n_grid,
        "n_missing": {source_ids[i]: n_grid - sum(1 for (col, _) in used if col == i) for i in range(n_s)},
        "arrival_policy": arrival,
        "R_source": r_source,
        "n_evidence_ids": len(used_ids),
        "evidence_ids_sha256": hashlib.sha256("\n".join(used_ids).encode("utf-8")).hexdigest(),
        "evidence_ids_sha256_rule": "sha256 of the sorted, distinct evidence ids used, joined by '\\n' (utf-8)",
    }
    inputs = PublicInputs(t=t_grid, u_commanded=np.zeros(n_grid))
    return BridgedSeries(inputs=inputs, observations=observations, provenance=provenance,
                         source_ids=source_ids, component_evidence=tuple(component_evidence))


# ---------------------------------------------------------------------------
# optional: DAF recomputes the ids
# ---------------------------------------------------------------------------

def verify_ids(records: Iterable[Mapping], daf_root: str | Path) -> int:
    """Recompute every record's id with DAF's own `observation_from_dict` (which rebuilds
    the Observation through the vendored `evidence.types.make_observation` and raises
    `ArtifactIdentityMismatch` when the recomputed id differs) and return how many were
    verified. Imports DAF and its vendored substrate from `daf_root` -- the only place in
    this package that does -- with bytecode writing off, so the checkout gains no files;
    sys.path is restored afterwards. Refuses to run against a DAF already imported from
    somewhere else."""
    root = Path(daf_root).resolve()
    vendor = root / "vendor" / "scout-retrieval-agent"
    if not (root / "daf" / "storage" / "serialization.py").is_file() or not (vendor / "evidence").is_dir():
        raise ValueError(f"{root} is not a DAF checkout with its vendored substrate initialised")
    saved_path, saved_dwb = list(sys.path), sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        sys.path[:0] = [str(vendor), str(root)]
        from daf.storage.serialization import observation_from_dict  # noqa: PLC0415 -- optional, DAF-side
        for mod, base in (("daf", root), ("evidence", vendor)):
            f = Path(sys.modules[mod].__file__).resolve()
            if base not in f.parents:
                raise ValueError(f"module {mod!r} is already imported from {f}, not from {base}")
        n = 0
        for r in records:
            rebuilt = observation_from_dict(dict(r))
            if rebuilt.id != r["id"]:   # observation_from_dict raises first; belt and braces
                raise ValueError(f"record {r['id']} re-hashes to {rebuilt.id}")
            n += 1
        return n
    finally:
        sys.path[:] = saved_path
        sys.dont_write_bytecode = saved_dwb
