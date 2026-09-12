"""DAF bridge: serialized DAF observations -> Observation, with refusals and provenance.

DAF (the Data Acquisition Fabric) admits evidence; this module consumes it and owns
none of it. Its input is DAF observation dicts exactly as DAF's own
`daf.storage.serialization.observation_to_dict` writes them (data/daf/*.observations.json,
produced by tools/export_daf_fixtures.py from DAF's committed NOAA fixtures). It imports
nothing from DAF, SCOUT or the vendored evidence substrate at runtime; the one function
that does, `verify_ids`, is optional, imports DAF's code from a checkout the caller names,
and is not on the bridging path.

What a record must be. A per-measurement NOAA water-level observation from DAF's
`NoaaWaterLevelMeasurementExtractor` (extraction method
"json:noaa_water_level_measurement_v1", content property "water_level") with an `id` and,
in `content`: value, unit, datum, station_id, measurement_time and, when the source
stated one, uncertainty with uncertainty_kind "stated" (DAF also carries sigma and
conditions). Anything else is refused, not skipped.

What the caller must declare (keyword-only, no defaults):

    series           the (station_id, datum, unit) groups, one sensor column each, in
                     column order. A record whose (station_id, datum, unit) is not listed
                     is ignored and counted. Two datums or two units are never pooled:
                     each listed key is its own column, and listing one (station, datum)
                     in two units is refused -- the same quantity would enter twice, and
                     the bridge converts no units.
    time_zone        the zone measurement_time is written in. DAF's content carries none
                     (the NOAA binding requests time_zone=gmt in its URL, recorded in
                     data/daf/PROVENANCE.md); "UTC"/"GMT" are supported, anything else
                     raises, and None -- naive parsing -- is refused.
    cadence_s        the grid spacing (NOAA's 6-minute product: 360). The grid runs from
                     the first to the last reading of the listed series; a reading more
                     than GRID_TOLERANCE_S (1 s) off it is refused, every such reading
                     listed by evidence id; a grid point with no reading is mask False,
                     y NaN.
    arrival_policy   "replay" with an explicit latency_s (arrival_t = t + latency_s), or
                     "as_acquired" (arrival_t from each record's extracted_at, an
                     ISO-8601 instant with an explicit offset, on the same clock; refused
                     where absent or naive, or earlier than the measurement). An
                     observation built from several records arrives when the last of them
                     did; one with no reading arrives at its own t.
    conflict_policy  two records for the same (series, grid point) that disagree -- in
                     value or in the uncertainty that becomes R -- are a conflict:
                     "refuse" raises naming every id; "report_and_keep_both" leaves the
                     point missing (neither value used) and lists the conflict in
                     provenance. Identical readings (an identical re-extraction, the same
                     reading in an original and a revised window) are deduplicated with
                     every id kept.
    daf_commit       the DAF commit the records came from (caller-supplied; recorded).

Uncertainty. R_ii = uncertainty**2 where uncertainty_kind == "stated" (NOAA's `s`: the
standard deviation of the 1-second samples behind the 6-minute value, as DAF records it).
A series with a reading that states none is refused unless the caller passes
`declared_sigma[key]` for that series, which is then recorded as consumer-declared; a
series is either all source-stated or all consumer-declared -- declaring a sigma for a
series whose readings state their own is refused, so the two are never mixed in one
series. A stated 0.0 (NOAA reports s to 1 mm) is passed through as R = 0 and counted in
provenance, never floored.

Output: BridgedSeries(inputs, observations, provenance, source_ids, component_evidence).
inputs.t is seconds from the declared epoch (the first grid point) and u_commanded is
zeros: nothing is commanded in a tide-gauge record. Each Observation's source_ids are the
series ids ("noaa:<station>:<datum>:<unit>"), its evidence_ids the DAF ids behind each
present component in column order (component_evidence keeps them per component).
Provenance: daf_commit, record counts (n_records_in = n_used + n_ignored + n_in_conflict),
n_deduplicated, conflicts, the grid check, time zone, epoch, cadence, arrival policy,
R_source per series and a sha256 over the sorted evidence ids used.

The DAF boundary this respects (docs/DAF_STATE_SPACE_BOUNDARY.md, sections 10-13, 18):
t is the source event time read out of Observation.content, never retrieved_at or
extracted_at; extracted_at is read only under "as_acquired", as an arrival clock, never as
identity or for deduplication; contradictory observations are never averaged and the
later (revised) one never silently wins -- a revision is not a state transition; nothing
here needs a RawDocument, an adapter or a DAF type; and DAF's evidence ids are carried as
provenance (Observation.evidence_ids, read by no estimator), never as model identity.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..schema import Observation
from ..testbed.inputs import PublicInputs

__all__ = [
    "BRIDGE_VERSION", "DAF_NOAA_MEASUREMENT_METHOD", "GRID_TOLERANCE_S", "BridgeRefusal", "BridgedSeries",
    "bridge", "load_records", "series_source_id", "strict_json_loads", "verify_ids",
]

BRIDGE_VERSION = "daf-noaa-measurement-bridge-v1"
DAF_NOAA_MEASUREMENT_METHOD = "json:noaa_water_level_measurement_v1"
PROPERTY = "water_level"
STATED = "stated"
GRID_TOLERANCE_S = 1.0
CONFLICT_POLICIES = ("refuse", "report_and_keep_both")
ARRIVAL_POLICIES = ("replay", "as_acquired")
_ZONES = {"UTC": timezone.utc, "GMT": timezone.utc}
_MEASUREMENT_TIME = re.compile(r"\A(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})(?::(\d{2}))?\Z")

SeriesKey = tuple[str, str, str]   # (station_id, datum, unit)


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
    content_json: str             # canonical content, to catch one id carrying two contents
    idx: int = -1                 # grid index, set once the epoch is known


# ---------------------------------------------------------------------------
# argument checks
# ---------------------------------------------------------------------------

def _check_series(series) -> list[SeriesKey]:
    if series is None or isinstance(series, (str, bytes)) or not isinstance(series, Sequence) or not series:
        raise ValueError("series must be a non-empty list of (station_id, datum, unit) tuples")
    keys: list[SeriesKey] = []
    for s in series:
        if (isinstance(s, (str, bytes)) or not isinstance(s, Sequence) or len(s) != 3
                or not all(isinstance(x, str) and x.strip() for x in s)):
            raise ValueError(f"series entry {s!r} is not a (station_id, datum, unit) tuple of non-empty strings")
        keys.append((s[0], s[1], s[2]))
    if len(set(keys)) != len(keys):
        raise ValueError(f"series lists a group twice: {keys}")
    units: dict[tuple[str, str], set[str]] = defaultdict(set)
    for station, datum, unit in keys:
        units[(station, datum)].add(unit)
    for (station, datum), us in units.items():
        if len(us) > 1:
            raise BridgeRefusal(
                "mixed_units",
                f"station {station} datum {datum} is listed in units {sorted(us)}: one quantity in two units "
                "would enter as two sensors of the same readings, and the bridge converts no units")
    return keys


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


def _check_declared(declared_sigma, keys: list[SeriesKey]) -> dict[SeriesKey, float]:
    if declared_sigma is None:
        return {}
    if not isinstance(declared_sigma, Mapping):
        raise ValueError("declared_sigma must map (station_id, datum, unit) -> sigma")
    out: dict[SeriesKey, float] = {}
    for k, v in declared_sigma.items():
        key = tuple(k) if isinstance(k, (list, tuple)) else k
        if key not in keys:
            raise ValueError(f"declared_sigma names {k!r}, which is not a listed series")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) <= 0.0:
            raise ValueError(f"declared_sigma[{k!r}] must be a finite positive standard deviation; got {v!r}")
        out[key] = float(v)
    return out


# ---------------------------------------------------------------------------
# record checks
# ---------------------------------------------------------------------------

def _finite_number(x) -> bool:
    return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(float(x))


def _record_key(r) -> tuple[str, Mapping, SeriesKey]:
    """The id, the content and the (station, datum, unit) of a record, refusing anything
    that is not a per-measurement NOAA water-level observation."""
    if not isinstance(r, Mapping):
        raise BridgeRefusal("malformed", f"a record is a {type(r).__name__}, not a DAF observation object")
    eid = r.get("id")
    if not isinstance(eid, str) or not eid:
        raise BridgeRefusal("malformed", f"a record has no evidence id: {dict(r)!r}"[:300])
    if r.get("extraction_method") != DAF_NOAA_MEASUREMENT_METHOD:
        raise BridgeRefusal("not_noaa_measurement",
                            f"record {eid} has extraction_method {r.get('extraction_method')!r}; the bridge reads "
                            f"{DAF_NOAA_MEASUREMENT_METHOD!r} observations only", [eid])
    c = r.get("content")
    if not isinstance(c, Mapping):
        raise BridgeRefusal("malformed", f"record {eid} has no content object", [eid])
    for f in ("station_id", "datum", "unit"):
        if not isinstance(c.get(f), str) or not c.get(f):
            raise BridgeRefusal("malformed", f"record {eid} has no usable content[{f!r}]", [eid])
    return eid, c, (c["station_id"], c["datum"], c["unit"])


def _measurement_instant(eid: str, text, tz) -> datetime:
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


def _reading(eid: str, c: Mapping, col: int, extracted_at, tz) -> _Reading:
    if c.get("property") != PROPERTY:
        raise BridgeRefusal("malformed", f"record {eid} has property {c.get('property')!r}, not {PROPERTY!r}", [eid])
    cond = c.get("conditions")
    if cond is not None and (not isinstance(cond, Mapping) or cond.get("datum") != c["datum"]):
        raise BridgeRefusal("malformed", f"record {eid}: conditions {cond!r} disagree with datum {c['datum']!r}", [eid])
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
        instant=_measurement_instant(eid, c.get("measurement_time"), tz), value=float(c["value"]),
        stated_sd=stated, extracted_at=extracted_at,
        content_json=json.dumps(dict(c), sort_keys=True, separators=(",", ":"), default=str),
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
    series: Sequence[SeriesKey],
    time_zone: str,
    cadence_s: float,
    arrival_policy: str,
    conflict_policy: str,
    daf_commit: str,
    latency_s: float | None = None,
    declared_sigma: Mapping[SeriesKey, float] | None = None,
) -> BridgedSeries:
    """DAF observation dicts -> BridgedSeries. See the module docstring for the contract;
    every refusal raises BridgeRefusal (a ValueError) or ValueError, never a partial record."""
    keys = _check_series(series)
    zone = _check_zone(time_zone)
    tz = _ZONES[zone]
    cadence = _check_cadence(cadence_s)
    arrival = _check_arrival(arrival_policy, latency_s)
    if conflict_policy not in CONFLICT_POLICIES:
        raise ValueError(f"conflict_policy must be one of {CONFLICT_POLICIES}; got {conflict_policy!r}")
    declared = _check_declared(declared_sigma, keys)
    if not isinstance(daf_commit, str) or not daf_commit.strip():
        raise ValueError("daf_commit must name the DAF commit the records came from")

    records = list(records)
    if not records:
        raise BridgeRefusal("empty", "no records")
    col_of = {k: i for i, k in enumerate(keys)}
    source_ids = tuple(series_source_id(k) for k in keys)

    # 1. classify: listed series or ignored (and counted)
    readings: list[_Reading] = []
    ignored: Counter = Counter()
    for r in records:
        eid, c, key = _record_key(r)
        if key not in col_of:
            ignored[key] += 1
            continue
        readings.append(_reading(eid, c, col_of[key], r.get("extracted_at"), tz))

    # 2. uncertainty: each series is all source-stated or all consumer-declared
    by_col: dict[int, list[_Reading]] = defaultdict(list)
    for rd in readings:
        by_col[rd.col].append(rd)
    for i, key in enumerate(keys):
        rs = by_col.get(i, [])
        if not rs:
            raise BridgeRefusal("empty_series", f"series {source_ids[i]} matched no record "
                                                f"({len(records)} in, {sum(ignored.values())} ignored)")
        stated = [rd.eid for rd in rs if rd.stated_sd is not None]
        unstated = [rd.eid for rd in rs if rd.stated_sd is None]
        if key in declared and stated:
            raise BridgeRefusal(
                "mixed_uncertainty",
                f"declared_sigma is given for {source_ids[i]}, but {len(stated)} of its readings state their own "
                "uncertainty; source-stated and consumer-declared R are never mixed in one series", stated)
        if key not in declared and unstated:
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

    # 4. one value per (series, grid point): deduplicate identical readings, surface conflicts
    groups: dict[tuple[int, int], list[_Reading]] = defaultdict(list)
    for rd in readings:
        groups[(rd.col, rd.idx)].append(rd)
    used: dict[tuple[int, int], tuple[float, float, tuple[str, ...], list[_Reading]]] = {}
    conflicts: list[dict] = []
    n_used = n_dedup = n_conflict = 0
    for (col, idx), rs in sorted(groups.items()):
        first: dict[str, _Reading] = {}
        for rd in rs:
            if rd.eid in first and first[rd.eid].content_json != rd.content_json:
                raise BridgeRefusal("id_reused", f"evidence id {rd.eid} carries two different contents; a "
                                                 "content-addressed id cannot", [rd.eid])
            first.setdefault(rd.eid, rd)
        sd_of = (lambda rd: declared[keys[col]]) if keys[col] in declared else (lambda rd: rd.stated_sd)
        distinct: dict[tuple[float, float], list[str]] = defaultdict(list)
        for rd in rs:
            distinct[(rd.value, sd_of(rd))].append(rd.eid)
        ids = tuple(sorted(first))
        if len(distinct) == 1:
            ((value, sd),) = distinct.keys()
            used[(col, idx)] = (value, sd, ids, rs)
            n_used += len(rs)
            n_dedup += len(rs) - 1
            continue
        entry = {
            "series": source_ids[col], "grid_index": idx, "measurement_time": rs[0].measurement_time,
            "readings": [{"evidence_id": e, "value": v, "sigma": s}
                         for (v, s), es in sorted(distinct.items()) for e in sorted(set(es))],
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
    n_s = len(keys)
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
    for i, key in enumerate(keys):
        sds = [sd for (col, _), (_, sd, _, _) in used.items() if col == i]
        if key in declared:
            r_source[source_ids[i]] = {
                "kind": "consumer-declared", "sigma": declared[key], "R": declared[key] ** 2, "n_points": len(sds),
                "note": "declared by the caller (declared_sigma) for a series whose readings state no uncertainty"}
        else:
            r_source[source_ids[i]] = {
                "kind": "source-stated", "rule": "R_ii = uncertainty**2 where uncertainty_kind == 'stated'",
                "n_points": len(sds), "sigma_min": min(sds) if sds else None, "sigma_max": max(sds) if sds else None,
                "n_zero": sum(1 for s in sds if s == 0.0)}
    provenance = {
        "bridge": "set_lcm.bridge.daf",
        "bridge_version": BRIDGE_VERSION,
        "daf_commit": daf_commit,
        "extraction_method": DAF_NOAA_MEASUREMENT_METHOD,
        "series": list(source_ids),
        "n_records_in": len(records),
        "n_used": n_used,
        "n_ignored": sum(ignored.values()),
        "ignored": [{"series_key": list(k), "n": n} for k, n in sorted(ignored.items())],
        "n_in_conflict": n_conflict,
        "n_deduplicated": n_dedup,
        "conflicts": conflicts,
        "conflict_policy": conflict_policy,
        "off_grid": {"tolerance_s": GRID_TOLERANCE_S, "n_refused": 0, "max_abs_offset_s": max_off,
                     "rule": "a reading further off the grid than tolerance_s is refused (bridge() raises, listing "
                             "every such evidence id)"},
        "time_zone": zone,
        "time_zone_declared": time_zone,
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
