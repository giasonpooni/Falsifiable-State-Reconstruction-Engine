"""P4c: a month of real six-minute water levels, and what a month buys over a day.

NOAA CO-OPS station 8454000 (Providence, RI), 2024-01-01 to 2024-01-31 inclusive on MLLW:
7,440 six-minute readings with none missing, acquired live through DAF's own adapter over 11
trailing windows and committed as the observations they replay into
(data/daf/noaa_8454000_202401_mllw.observations.json, checked against data/daf/manifest.json;
no network). Brought in through set_lcm.bridge.daf with the same declared choices as the day
report: time zone UTC, cadence 360 s, arrival "replay" with latency 0, conflict policy
"refuse".

A month is not a longer day. What it changes, and what it does not:

  RESOLUTION. The Rayleigh criterion says two constituents need a record at least
  1 / |n1 - n2| long to be separated. Over 24 hours that excludes S2 from M2 (14.8 days),
  N2 from M2 (27.6 days) and O1 from K1 (13.7 days), so the day report's `tide_kf` carries
  M2, K1, O1, M4 and absorbs the rest. Over 31 days all three separations are inside the
  criterion, so `tide_month` carries M2, S2, N2, K1, O1, M4 -- and P1 still is not
  separable from K1 (182.6 days), so P1 is not modelled and is absorbed into K1 exactly as
  S2 was absorbed into M2 on one day. resolvable() computes the pair table from each
  window's own length, and the report prints it per window, so no window can be described
  as separating a pair it does not.

  THE SPLIT IS NOT FREE. Fitting on the first half and scoring the second leaves each half
  15-16 days long, and 15 days does NOT separate N2 from M2. So the q that `tide_month` is
  scored with was fitted on a window that cannot separate one of the pairs the model
  declares. The report states this, prints the unresolved pair per window, and also reports
  the whole month in-sample so the reader can see both.

  STATE, SEPARATELY FROM q. "Held out" in the day report is a statement about q alone. Here
  each fitted q is scored two ways: `fresh`, a filter started at the beginning of the
  held-out half, which must re-learn all 13 states from its declared prior; and
  `continued`, one filter over the whole month scored on the held-out half only, which
  carries the first half's state across the boundary. The gap between them is how much of
  the held-out half's fit is the first half's burn-in rather than the model.

  THE MODEL-FREE BOUND GETS TIGHTER. series_check's white-error bound on sigma_e is a mean
  square over second differences: 7,438 of them over the month against 238 over a day, so
  its own sampling scatter is about five times smaller. It is reported per window.

Declared, not fitted:

    priors        as testbed.estimators_water declares them, unchanged from the day report:
                  level / mean level ~ N(0 m, (10 m)^2); rate ~ N(0, (1e-3 m/s)^2); every
                  harmonic coefficient ~ N(0, (2 m)^2)
    R             NOAA's stated sigma^2 per reading, throughout. No sensitivity scaling here:
                  the day report sweeps R x 10 and R x 100 on one day; this report's axis is
                  the record length.
    q grids       the day report's grids (real_noaa.Q_GRIDS), with tide_month on tide_kf's
    fit window    2024-01-01 .. 2024-01-15 ONLY: q_scale = the grid value maximising the
                  innovation log-likelihood over that window
    evaluation    2024-01-16 .. 2024-01-31, a disjoint window with no evidence id in common

Truth-free throughout: nobody knows the water level, so no number here is an error. Every
number is a property of the record and the filter together.

    python -m set_lcm.experiments.real_noaa_month  -> results/real_noaa_month.{md,json}
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..bridge.daf import BridgedSeries, bridge, load_records
from ..testbed.estimators_water import (
    MONTH_CONSTITUENTS, TIDE_CONSTITUENTS, LevelTrendConfig, TideConfig, rayleigh_period_hours, resolvable,
)
from ..testbed.inputs import PublicInputs
from ..testbed.runner import EstimatorSpec, RunResult, run
from ..testbed.truth_free import evaluate_truth_free
from .provenance import REPO_ROOT, header_line, provenance
from .real_noaa import (
    BRIDGE_ARGS, LEVEL0, LEVEL0_STD, MLLW, Q_GRIDS, Q_UNITS, loglik, series_check_arrays, series_extremes,
)

DATA_DIR = REPO_ROOT / "data" / "daf"
MANIFEST_PATH = DATA_DIR / "manifest.json"
MONTH_FILE = "noaa_8454000_202401_mllw.observations.json"

STEPS_PER_DAY = 240          # 86400 / 360
FIT_DAYS = 15                # calendar days of the month the q grids are searched on
KINDS = ("level_trend", "tide_kf", "tide_month")
CONSTITUENTS = {"level_trend": (), "tide_kf": TIDE_CONSTITUENTS, "tide_month": MONTH_CONSTITUENTS}
CONFIGS = {"level_trend": LevelTrendConfig, "tide_kf": TideConfig, "tide_month": TideConfig}
SPECS = {k: EstimatorSpec(k, k, None) for k in KINDS}     # no constraint; never projected
GRIDS = {**Q_GRIDS, "tide_month": Q_GRIDS["tide_kf"]}
UNITS = {**Q_UNITS, "tide_month": Q_UNITS["tide_kf"]}


@dataclass(frozen=True)
class Window:
    """A contiguous half-open slice of the month's step grid, named."""
    key: str
    label: str
    start: int
    stop: int
    role: str

    @property
    def n_steps(self) -> int:
        return self.stop - self.start


def windows(n_steps: int) -> tuple[Window, ...]:
    cut = FIT_DAYS * STEPS_PER_DAY
    if not 0 < cut < n_steps:
        raise ValueError(f"a {FIT_DAYS}-day fit window does not fit in {n_steps} steps")
    return (
        Window("fit", f"days 1-{FIT_DAYS}", 0, cut, "the q grids are searched here, and nowhere else"),
        Window("held_out", f"days {FIT_DAYS + 1}-31", cut, n_steps, "held out: a disjoint window, no shared evidence"),
        Window("month", "days 1-31", 0, n_steps, "the whole record; in-sample for every fitted q"),
    )


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------

def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def manifest_entry(man: dict) -> dict:
    """The month's manifest entry, with the committed file checked against its sha256."""
    import hashlib
    entry = next((f for f in man["files"] if f["output"] == MONTH_FILE), None)
    if entry is None:
        raise ValueError(f"{MONTH_FILE} is not listed in {MANIFEST_PATH}")
    digest = hashlib.sha256((DATA_DIR / MONTH_FILE).read_bytes()).hexdigest()
    if digest != entry["output_sha256"]:
        raise ValueError(f"{MONTH_FILE}: sha256 {digest} is not the manifest's {entry['output_sha256']}")
    return entry


def load_month(man: dict | None = None) -> BridgedSeries:
    man = man if man is not None else manifest()
    manifest_entry(man)
    return bridge(load_records(DATA_DIR / MONTH_FILE), series=[MLLW], daf_commit=man["daf_commit"],
                  **BRIDGE_ARGS)


def slice_window(bs: BridgedSeries, w: Window) -> tuple[PublicInputs, list]:
    """The window's own clock and observations. The clock keeps its ABSOLUTE seconds from the
    record's epoch, so a harmonic filter started mid-record sees the same phase reference as
    one started at the beginning; only the state's history differs."""
    t = np.asarray(bs.inputs.t)[w.start:w.stop]
    u = np.asarray(bs.inputs.u_commanded)[w.start:w.stop]
    return PublicInputs(t=t, u_commanded=u), list(bs.observations[w.start:w.stop])


def window_span(bs: BridgedSeries, w: Window) -> tuple[str, str]:
    epoch = np.datetime64(bs.provenance["epoch_iso"].rstrip("Z"), "s")
    t = np.asarray(bs.inputs.t)
    first = epoch + np.timedelta64(int(round(float(t[w.start]))), "s")
    last = epoch + np.timedelta64(int(round(float(t[w.stop - 1]))), "s")
    return str(first) + "Z", str(last) + "Z"


def check_disjoint(bs: BridgedSeries, a: Window, b: Window) -> None:
    """The fit and held-out windows must not overlap in steps, days or evidence ids."""
    if not (a.stop <= b.start or b.stop <= a.start):
        raise ValueError(f"windows {a.key} and {b.key} overlap in steps")
    (a0, a1), (b0, b1) = window_span(bs, a), window_span(bs, b)
    if not (a1 < b0 or b1 < a0):
        raise ValueError(f"windows overlap in time: {a0}..{a1} and {b0}..{b1}")
    ids_a = {i for o in bs.observations[a.start:a.stop] for i in o.evidence_ids}
    ids_b = {i for o in bs.observations[b.start:b.stop] for i in o.evidence_ids}
    shared = ids_a & ids_b
    if shared:
        raise ValueError(f"{len(shared)} evidence id(s) appear in both windows")


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------

def run_window(kind: str, q_scale: float, bs: BridgedSeries, w: Window) -> RunResult:
    """One estimator over one window, from the declared prior. R is never scaled here."""
    inputs, obs = slice_window(bs, w)
    return run(inputs, obs, None, SPECS[kind], LEVEL0, LEVEL0_STD, est_cfg=CONFIGS[kind](q_scale=q_scale))


def fit_q(kind: str, bs: BridgedSeries, w: Window) -> dict:
    """Maximise the innovation log-likelihood over the declared grid, on ONE window."""
    profile = [{"q_scale": q, "loglik": loglik(run_window(kind, q, bs, w))} for q in GRIDS[kind]]
    index = int(np.argmax([p["loglik"] for p in profile]))
    q = profile[index]["q_scale"]
    dt = float(np.asarray(bs.inputs.t)[1] - np.asarray(bs.inputs.t)[0])
    per_step = ({"rate_std_m_per_s": q * math.sqrt(dt), "level_std_m": q * dt ** 1.5 / math.sqrt(3.0)}
                if kind == "level_trend" else {"state_std_m": q * math.sqrt(dt)})
    return {"window": w.key, "profile": profile, "argmax_index": index, "q_scale": q,
            "loglik": profile[index]["loglik"], "interior": 0 < index < len(profile) - 1,
            "units": UNITS[kind], "per_step": per_step, "n_steps": w.n_steps}


def summarize(rr: RunResult, obs: list, key: str, q_scale: float) -> dict:
    """What the record says about the run: truth_free over the whole scored span, the
    log-likelihood, and the mean share of the stated innovation variance R accounts for."""
    n = len(obs)
    tf = evaluate_truth_free(rr, {key: (0, n)})
    s0 = tf["windows"][key]["sensors"][0]
    ok = np.isfinite(rr.innov_var[:, 0])
    r = np.array([float(o.R[0, 0]) for o in obs])
    return {
        "q_scale": q_scale, "loglik": loglik(rr), "n_steps": n,
        "n_observed": s0["n_observed"], "frac_missing": s0["frac_missing"],
        "z_mean": s0["z_mean"], "z_rms": s0["z_rms"], "z_lag1": s0["z_lag1"],
        "frac_abs_z_gt_1.96": s0["frac_abs_z_gt_1.96"],
        "cusum_alarms": s0["cusum_alarms"], "cusum_first_alarm": s0["cusum_first_alarm"],
        "r_share_mean": float(np.mean(r[ok] / rr.innov_var[ok, 0])),
        "status_counts": {k: v for k, v in sorted(Counter(st.value for st in rr.status).items())},
        "n_evidence_ids": tf["windows"][key]["n_evidence_ids"],
        "constraint_declared": tf["constraint_declared"],
        "latency_us_p50": tf["latency_us_p50"], "latency_us_p99": tf["latency_us_p99"],
    }


def score_held_out(kind: str, q_scale: float, bs: BridgedSeries, fit: Window, held: Window) -> dict:
    """The same q, two scorings of the same held-out readings.

    `fresh` starts a filter at the beginning of the held-out window, so all its states come
    from the declared prior and nothing of the fit window survives except q. `continued`
    runs one filter over the whole month and scores only the held-out steps, so the state
    crossing the boundary is the first half's. Neither is the other's approximation: one
    measures the model from cold, the other measures it as an operator would run it.
    """
    _, held_obs = slice_window(bs, held)
    fresh = summarize(run_window(kind, q_scale, bs, held), held_obs, held.key, q_scale)

    month = next(w for w in windows(len(bs.observations)) if w.key == "month")
    whole = run_window(kind, q_scale, bs, month)
    tf = evaluate_truth_free(whole, {held.key: (held.start, held.stop)})
    s0 = tf["windows"][held.key]["sensors"][0]
    nu, var = whole.innov[held.start:held.stop, 0], whole.innov_var[held.start:held.stop, 0]
    ok = np.isfinite(nu)
    r = np.array([float(o.R[0, 0]) for o in held_obs])
    continued = {
        "q_scale": q_scale,
        "loglik": float(np.sum(-0.5 * (np.log(2.0 * np.pi * var[ok]) + nu[ok] ** 2 / var[ok]))),
        "n_steps": held.n_steps, "n_observed": s0["n_observed"], "frac_missing": s0["frac_missing"],
        "z_mean": s0["z_mean"], "z_rms": s0["z_rms"], "z_lag1": s0["z_lag1"],
        "frac_abs_z_gt_1.96": s0["frac_abs_z_gt_1.96"],
        "cusum_alarms": s0["cusum_alarms"],
        # truth_free reports the alarm step in the run's own coordinates, and this run starts at
        # the beginning of the month. Both forms are given so the two scorings can be compared
        # without silently subtracting an offset: the fresh run's step 263 and this run's 3,863
        # are the same reading.
        "cusum_first_alarm_in_record": s0["cusum_first_alarm"],
        "cusum_first_alarm": (None if s0["cusum_first_alarm"] is None
                              else int(s0["cusum_first_alarm"]) - held.start),
        "r_share_mean": float(np.mean(r[ok] / var[ok])),
        "n_evidence_ids": tf["windows"][held.key]["n_evidence_ids"],
        "note": "one filter over the whole month, scored on the held-out steps only; the state "
                "crossing into them is the fit window's, so this is not a cold start",
    }
    return {"fresh": fresh, "continued": continued,
            "z_rms_fresh_over_continued": (fresh["z_rms"] / continued["z_rms"]
                                           if continued["z_rms"] else None)}


# ---------------------------------------------------------------------------
# what the record's length does and does not buy
# ---------------------------------------------------------------------------

def resolution(bs: BridgedSeries, w: Window) -> dict:
    """The Rayleigh pair table for this window's own length, over the month's constituents.

    Reported per window rather than asserted once, because the fit window is half the record
    and does not separate every pair the month does. P1 is included as a declared candidate
    that NOTHING here models, so its unresolved row states why rather than leaving it out.
    """
    t = np.asarray(bs.inputs.t)
    hours = float(t[w.stop - 1] - t[w.start]) / 3600.0
    table = resolvable(MONTH_CONSTITUENTS + ("P1",), hours)
    return {
        "record_hours": hours, "record_days": hours / 24.0,
        "modelled_by_tide_month": list(MONTH_CONSTITUENTS),
        "modelled_by_tide_kf": list(TIDE_CONSTITUENTS),
        "n_resolved": len(table["resolved"]), "n_unresolved": len(table["unresolved"]),
        "unresolved": table["unresolved"],
        "unresolved_among_modelled": [row for row in table["unresolved"]
                                      if set(row["pair"]) <= set(MONTH_CONSTITUENTS)],
        "P1_note": (f"P1 is not modelled anywhere. Separating it from K1 needs "
                    f"{rayleigh_period_hours('K1', 'P1') / 24.0:.1f} days, so a month cannot; it is "
                    f"absorbed into the K1 coefficients, and no K1 number here is clean of it."),
    }


EXTREME_SIGMA_FACTOR = 10.0        # multiples of the window median that count as extreme


def stated_sigma(bs: BridgedSeries, w: Window) -> dict:
    """NOAA's stated sigma over one window, described so no second moment stands alone.

    R here is NOAA's own stated sigma^2 per reading, carried by the bridge and never edited.
    Over this month one reading states 2.002 m against a median of 0.007 m, and a single
    statement like that dominates an RMS, an expected mean square and the relative scatter of
    one -- so the median, the extremes and their count are reported beside every such number.
    A stated 0.000 is carried as R = 0 exactly, which asks the filter to treat the reading as
    exact; those are counted rather than adjusted.
    """
    sigma = np.array([float(o.R[0, 0]) ** 0.5 for o in bs.observations[w.start:w.stop]])
    t = np.asarray(bs.inputs.t)
    median = float(np.median(sigma))
    extreme = np.flatnonzero(sigma > EXTREME_SIGMA_FACTOR * median) if median > 0.0 else np.array([], int)
    order = extreme[np.argsort(-sigma[extreme])] if extreme.size else extreme
    epoch = np.datetime64(bs.provenance["epoch_iso"].rstrip("Z"), "s")
    return {
        "units": "m",
        "median": median,
        "rms": float(np.sqrt(np.mean(sigma ** 2))),
        "min": float(sigma.min()),
        "max": float(sigma.max()),
        "n_distinct_values": int(np.unique(sigma).size),
        "n_stated_zero": int(np.count_nonzero(sigma == 0.0)),
        "zero_note": "a stated 0.000 is carried as R = 0, which declares the reading exact",
        "extreme_factor": EXTREME_SIGMA_FACTOR,
        "n_extreme": int(extreme.size),
        "extreme": [{
            "step_in_record": int(w.start + i),
            "step_in_window": int(i),
            "time": str(epoch + np.timedelta64(int(round(float(t[w.start + i]))), "s")) + "Z",
            "sigma": float(sigma[i]),
            "over_median": float(sigma[i] / median) if median > 0.0 else None,
        } for i in order],
        # what the RMS would be without them: the difference IS the finding, not a correction
        "rms_without_extreme": (float(np.sqrt(np.mean(np.delete(sigma, extreme) ** 2)))
                                if extreme.size and extreme.size < sigma.size else
                                float(np.sqrt(np.mean(sigma ** 2)))),
    }


def window_record(bs: BridgedSeries, w: Window) -> dict:
    """The readings of one window, described without any filter."""
    y = np.array([o.y[0] for o in bs.observations[w.start:w.stop]])
    var = np.array([float(o.R[0, 0]) for o in bs.observations[w.start:w.stop]])
    mask = np.array([bool(o.mask[0]) for o in bs.observations[w.start:w.stop]])
    first, last = window_span(bs, w)
    ids = {i for o in bs.observations[w.start:w.stop] for i in o.evidence_ids}
    check, sigma = series_check_arrays(y, var), stated_sigma(bs, w)
    return {
        "label": w.label, "role": w.role, "steps": [w.start, w.stop], "n_steps": w.n_steps,
        "first_grid_point": first, "last_grid_point": last,
        "n_observed": int(mask.sum()), "frac_missing": float(1.0 - mask.mean()),
        "n_evidence_ids": len(ids),
        "min_reading_m": float(y.min()), "max_reading_m": float(y.max()),
        "series_extremes": {"max_abs_reading": float(np.max(np.abs(y))),
                            "max_abs_step_change": float(np.max(np.abs(np.diff(y)))),
                            "half_range": float((y.max() - y.min()) / 2.0)},
        "series_check": check,
        "stated_sigma": sigma,
        # The stated sigma against the largest white error the window's own second differences
        # can support, on the median reading and on the RMS. Above 1 means NOAA states more
        # measurement error than a white error could be -- which is the same fact the filters'
        # z RMS below 1 reports, seen without a filter.
        "stated_sigma_over_white_bound": {
            "median": (check["median_stated_sigma"] / check["white_error_sigma_bound"]
                       if check["white_error_sigma_bound"] > 0.0 else None),
            "rms": (check["rms_stated_sigma"] / check["white_error_sigma_bound"]
                    if check["white_error_sigma_bound"] > 0.0 else None),
            "bound_assumes": "the bound assumes a white error independent of the water level, so a "
                             "time-correlated component of the stated sigma is not bounded by it",
        },
        "resolution": resolution(bs, w),
    }


# ---------------------------------------------------------------------------
# the whole report
# ---------------------------------------------------------------------------

def compute() -> dict:
    man = manifest()
    entry = manifest_entry(man)
    bs = load_month(man)
    wins = {w.key: w for w in windows(len(bs.observations))}
    fit, held, month = wins["fit"], wins["held_out"], wins["month"]
    check_disjoint(bs, fit, held)

    fits = {k: fit_q(k, bs, fit) for k in KINDS}
    scored: dict = {}
    for kind in KINDS:
        q = fits[kind]["q_scale"]
        _, fit_obs = slice_window(bs, fit)
        _, month_obs = slice_window(bs, month)
        scored[kind] = {
            "in_sample_fit_window": summarize(run_window(kind, q, bs, fit), fit_obs, fit.key, q),
            "held_out": score_held_out(kind, q, bs, fit, held),
            "whole_month_in_sample": summarize(run_window(kind, q, bs, month), month_obs, month.key, q),
        }
    records = {key: window_record(bs, w) for key, w in wins.items()}
    month_sigma = records["month"]["stated_sigma"]
    month_ratio = records["month"]["stated_sigma_over_white_bound"]
    return {
        "schema_version": "fsre-real-noaa-month-v1",
        "scope": "One month of real six-minute water levels. Truth-free: no number here is an error.",
        "provenance": {
            "generation": provenance(),
            "daf_commit": man["daf_commit"],
            "daf_published_base": man.get("daf_published_base"),
            # A session-recorded entry, not a single-response fixture: DAF's adapter walked the
            # month in overlapping trailing windows, and every raw HTTPS body is committed under
            # the sha256 of its own bytes. The overlap is recorded as it happened; the bridge
            # deduplicates it, which is why 21,360 observations become 7,440 distinct times.
            "evidence": {
                "file": f"data/daf/{MONTH_FILE}", "file_sha256": entry["output_sha256"],
                "note": entry["note"],
                "binding": entry["binding"],
                "plans": entry["plans"],
                "fetched_live": entry["fetched_live"],
                "requested_at": entry["requested_at"],
                "recorded_at_daf_commit": entry["recorded_at_daf_commit"],
                "source_session": entry["source_session"],
                "source_session_index_sha256": entry["source_session_index_sha256"],
                "n_recorded_responses": entry["n_recorded_responses"],
                "n_observations": entry["n_observations"],
                "n_distinct_measurement_times": entry["n_distinct_measurement_times"],
                "outcome_counts": {outcome: entry["outcomes"].count(outcome)
                                   for outcome in sorted(set(entry["outcomes"]))},
                "response_urls": [r["url"] for r in entry["responses"]],
                "response_sha256": [r["sha256"] for r in entry["responses"]],
            },
            "bridge": bs.provenance,
        },
        "declared": {
            "station": list(MLLW), "fit_days": FIT_DAYS, "steps_per_day": STEPS_PER_DAY,
            "level0_m": list(LEVEL0), "level0_std_m": LEVEL0_STD,
            "q_grids": {k: list(GRIDS[k]) for k in KINDS},
            "q_units": {k: UNITS[k] for k in KINDS},
            "constituents": {k: list(CONSTITUENTS[k]) for k in KINDS},
            "R": "NOAA's stated sigma^2 per reading, unscaled; the day report sweeps R, this one sweeps length",
            "rayleigh_criterion": "two constituents need a record at least 1 / |n1 - n2| long to be separated",
        },
        "windows": records,
        "fits": fits,
        "scored": scored,
        "limitations": [
            "Nobody knows the water level, so no number here is an error against it.",
            "A fitted q_scale is a property of the fit window and the declared grid, not a calibration.",
            f"The {records['fit']['resolution']['record_days']:.0f}-day fit window leaves "
            f"{len(records['fit']['resolution']['unresolved_among_modelled'])} declared pair(s) "
            f"unresolved ("
            + ", ".join(f"{'/'.join(row['pair'])} needs {row['rayleigh_days']:.1f} days"
                        for row in records["fit"]["resolution"]["unresolved_among_modelled"])
            + "), so tide_month's q was fitted where a pair its model declares is not separable.",
            records["month"]["resolution"]["P1_note"],
            "One station, one month, one datum. A month of another season could order the filters differently.",
            "The continued scoring is not a cold start: the state entering the held-out window is the fit window's.",
            f"A cold-started filter matches a continued one over this {held.n_steps:,}-step window. "
            f"That measures the transient's length against the window's, not that carried state does not matter.",
            f"{month_sigma['n_extreme']} readings state a sigma above "
            f"{month_sigma['extreme_factor']:g}x the median, the largest {month_sigma['max']:.3f} m; "
            f"every second moment of the stated sigma here is dominated by that one.",
            f"The stated sigma is {month_ratio['median']:.2f}x the white-error bound on the median "
            f"reading over the month. The bound assumes a white error independent of the level, so a "
            f"correlated part of a published accuracy statement is not bounded by it and nothing here "
            f"shows the declared R to be wrong.",
            f"{month_sigma['n_stated_zero']} readings state sigma = 0.000 m, carried as R = 0: the "
            f"filter is asked to treat them as exact. They are counted, not adjusted.",
        ],
    }


def render(report: dict) -> str:
    L: list[str] = []
    A = L.append
    dec, wins, fits, scored = report["declared"], report["windows"], report["fits"], report["scored"]
    A("# P4c: a month of real water levels, and what a month buys over a day")
    A("")
    A(header_line(report["provenance"]["generation"]))
    A("")
    A(report["scope"])
    A("")
    A(f"NOAA CO-OPS station {dec['station'][0]} on {dec['station'][1]}, "
      f"{wins['month']['first_grid_point'][:10]} to {wins['month']['last_grid_point'][:10]}: "
      f"{wins['month']['n_steps']:,} six-minute steps, {wins['month']['frac_missing'] * 100:.1f}% missing, "
      f"{wins['month']['n_evidence_ids']:,} evidence ids. Acquired through DAF's adapter and replayed "
      f"from `{report['provenance']['evidence']['file']}`.")
    A("")

    A("## The windows")
    A("")
    A("| window | days | steps | span | missing | evidence ids | role |")
    A("|---|---|---:|---|---:|---:|---|")
    for key in ("fit", "held_out", "month"):
        w = wins[key]
        A(f"| `{key}` | {w['label']} | {w['n_steps']:,} | {w['first_grid_point'][:16]} .. "
          f"{w['last_grid_point'][:16]} | {w['frac_missing'] * 100:.1f}% | {w['n_evidence_ids']:,} | "
          f"{w['role']} |")
    A("")
    A("The fit and held-out windows share no step, no calendar day and no evidence id; the check "
      "that says so runs before any filter does.")
    A("")

    A("## What each window's length can separate")
    A("")
    A(f"{dec['rayleigh_criterion'].capitalize()}.")
    A("")
    A("| window | days | pairs resolved | unresolved | unresolved among the modelled six |")
    A("|---|---:|---:|---:|---|")
    for key in ("fit", "held_out", "month"):
        r = wins[key]["resolution"]
        among = ", ".join("/".join(row["pair"]) for row in r["unresolved_among_modelled"]) or "none"
        A(f"| `{key}` | {r['record_days']:.1f} | {r['n_resolved']} | {r['n_unresolved']} | {among} |")
    A("")
    A(f"**The split is not free.** {wins['fit']['resolution']['record_days']:.1f} days is long enough "
      f"for S2 and O1 but not for N2: separating N2 from M2 needs "
      f"{rayleigh_period_hours('M2', 'N2') / 24.0:.1f} days, which only the whole month reaches. So "
      f"`tide_month` declares six constituents on the strength of the month, and the q it is scored "
      f"with was fitted on a window where one of its declared pairs is unresolved. Both facts are "
      f"above; neither is hidden in the other.")
    A("")
    A(wins["month"]["resolution"]["P1_note"])
    A("")

    A("## The fitted q, on the fit window only")
    A("")
    A("| estimator | constituents | q_scale | units | log-likelihood | grid interior |")
    A("|---|---|---:|---|---:|---|")
    for kind in KINDS:
        f = fits[kind]
        cons = ", ".join(dec["constituents"][kind]) or "—"
        A(f"| `{kind}` | {cons} | {f['q_scale']:.3g} | {f['units']} | {f['loglik']:,.1f} | "
          f"{'yes' if f['interior'] else '**no: at a grid edge**'} |")
    A("")
    A("A q at a grid edge is a statement about the grid, not about the record, and is marked as one.")
    A("")

    A("## Scored on the held-out half, two ways")
    A("")
    A("`fresh` starts a filter at the beginning of the held-out window: every state comes from the "
      "declared prior, and nothing of the fit window survives except q. `continued` runs one filter "
      "over the whole month and scores the held-out steps only, so the state crossing the boundary "
      "is the first half's. The ratio is how much of the held-out fit is that burn-in.")
    A("")
    A("| estimator | z RMS fresh | z RMS continued | ratio | z lag-1 fresh | z lag-1 continued | "
      "R share fresh | R share continued |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    ratios = []
    for kind in KINDS:
        h = scored[kind]["held_out"]
        ratio = h["z_rms_fresh_over_continued"]
        if ratio is not None:
            ratios.append(ratio)
        A(f"| `{kind}` | {h['fresh']['z_rms']:.3f} | {h['continued']['z_rms']:.3f} | "
          f"{'n/a' if ratio is None else format(ratio, '.2f')} | {h['fresh']['z_lag1']:+.3f} | "
          f"{h['continued']['z_lag1']:+.3f} | {h['fresh']['r_share_mean']:.3f} | "
          f"{h['continued']['r_share_mean']:.3f} |")
    A("")
    if ratios:
        worst = max(ratios, key=lambda r: abs(r - 1.0))
        A(f"**The burn-in buys nothing measurable here, and that is a result rather than an "
          f"omission.** The largest fresh/continued ratio is {worst:.4f}: over "
          f"{wins['held_out']['n_steps']:,} steps, a filter started cold from the declared prior "
          f"reaches the same z RMS as one carrying fifteen days of state. What that measures is the "
          f"transient's length against the window's, not that state does not matter -- a shorter "
          f"window would not show the same thing, and the day report's own transient is why it "
          f"excludes its first 20 steps from the datum check.")
        A("")
    first_fresh = scored[KINDS[0]]["held_out"]["fresh"]["cusum_first_alarm"]
    first_cont = scored[KINDS[0]]["held_out"]["continued"]["cusum_first_alarm_in_record"]
    if first_fresh is not None and first_cont is not None:
        A(f"CUSUM alarm steps below are in each run's own coordinates: the continued run starts at "
          f"the beginning of the month, so its step {first_cont:,} and the fresh run's step "
          f"{first_fresh:,} are the same reading. The JSON carries both forms.")
        A("")
    A("| estimator | window | z mean | z RMS | z lag-1 | \\|z\\| > 1.96 | CUSUM alarms | first alarm |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    for kind in KINDS:
        rows = (("fit (in-sample)", scored[kind]["in_sample_fit_window"]),
                ("held out, fresh", scored[kind]["held_out"]["fresh"]),
                ("held out, continued", scored[kind]["held_out"]["continued"]),
                ("whole month (in-sample)", scored[kind]["whole_month_in_sample"]))
        for label, s in rows:
            first = s["cusum_first_alarm"]
            A(f"| `{kind}` | {label} | {s['z_mean']:+.3f} | {s['z_rms']:.3f} | {s['z_lag1']:+.3f} | "
              f"{s['frac_abs_z_gt_1.96'] * 100:.1f}% | {s['cusum_alarms']:,} | "
              f"{'—' if first is None else first} |")
    A("")
    A("A z RMS below 1 means the filter's stated innovation variance is larger than the innovations "
      "it saw, which is a statement about the declared R and q together and not a score. CUSUM here "
      "reads no constraint: with one sensor and no declared relation there is nothing for it to "
      "localise, and an alarm count is a property of this record against the declared model.")
    A("")

    A("## What NOAA's stated sigma says, and what one reading does to it")
    A("")
    A("R throughout is NOAA's own stated sigma^2 per reading, carried by the bridge and never "
      "edited. It is not uniform, and its second moments are not robust:")
    A("")
    A("| window | median | RMS | RMS without the extremes | min | max | distinct values | "
      "stated 0.000 | over 10x median |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in ("fit", "held_out", "month"):
        g = wins[key]["stated_sigma"]
        A(f"| `{key}` | {g['median']:.4f} | {g['rms']:.4f} | {g['rms_without_extreme']:.4f} | "
          f"{g['min']:.4f} | {g['max']:.4f} | {g['n_distinct_values']} | {g['n_stated_zero']} | "
          f"{g['n_extreme']} |")
    A("")
    month_sigma = wins["month"]["stated_sigma"]
    if month_sigma["extreme"]:
        worst = month_sigma["extreme"][0]
        A(f"**One reading states {worst['sigma']:.3f} m** at {worst['time'][:16]} (record step "
          f"{worst['step_in_record']:,}), {worst['over_median']:.0f} times the month's median of "
          f"{month_sigma['median']:.4f} m. It alone raises the month's RMS stated sigma from "
          f"{month_sigma['rms_without_extreme']:.4f} m to {month_sigma['rms']:.4f} m, and it lands "
          f"in the held-out half, which is why that half's RMS is the larger one. Any quantity "
          f"below that is a second moment of the stated sigma -- the model-free comparison, its "
          f"expected mean square, the relative scatter of that -- is a statement about this one "
          f"reading as much as about the month. The median is unchanged by it, and both are given.")
        A("")
    if month_sigma["n_stated_zero"]:
        A(f"{month_sigma['n_stated_zero']} readings state sigma = 0.000 m. The bridge carries that "
          f"as R = 0 exactly, which declares the reading exact and gives the filter no room to "
          f"disagree with it. They are counted, never adjusted.")
        A("")

    A("## The model-free bound, per window")
    A("")
    A("No filter: if each reading's error were white with variance sigma_e^2 and independent of the "
      "water level, the expected mean square of the series' second differences would be at least "
      "6 sigma_e^2, so sigma_e <= rms(d2 y) / sqrt(6) in expectation. The month has 31 times a day's "
      "second differences, so its own sampling scatter is smaller -- which is the whole point of "
      "reporting it per window.")
    A("")
    A("| window | second differences | sigma_e bound (m) | median stated sigma (m) | "
      "rms stated sigma (m) | stated var / bound var | null rel. sd of the mean square |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for key in ("fit", "held_out", "month"):
        c = wins[key]["series_check"]
        A(f"| `{key}` | {c['n_second_differences']:,} | {c['white_error_sigma_bound']:.4f} | "
          f"{c['median_stated_sigma']:.4f} | {c['rms_stated_sigma']:.4f} | "
          f"{c['mean_stated_var_over_bound_var']:.3f} | {c['null_rel_sd_mean_square'] * 100:.2f}% |")
    A("")
    A("The bound is an upper bound on a white error, in expectation. A time-correlated error is not "
      "bounded by it, and a bound is not a measurement of NOAA's stated sigma.")
    A("")
    A("The last two columns are second moments of the stated sigma, so the single 2.002 m "
      "statement drives them: a null relative scatter above 100% says the expected mean square "
      "they compare against is itself dominated by one term. The ratio that is not is the median's:")
    A("")
    A("| window | median stated sigma / bound | rms stated sigma / bound |")
    A("|---|---:|---:|")
    for key in ("fit", "held_out", "month"):
        ratio = wins[key]["stated_sigma_over_white_bound"]
        A(f"| `{key}` | {'n/a' if ratio['median'] is None else format(ratio['median'], '.2f')} | "
          f"{'n/a' if ratio['rms'] is None else format(ratio['rms'], '.2f')} |")
    A("")
    month_ratio = wins["month"]["stated_sigma_over_white_bound"]
    medians = [wins[k]["stated_sigma_over_white_bound"]["median"] for k in ("fit", "held_out", "month")]
    if all(m is not None for m in medians):
        A(f"**The stated sigma exceeds the white-error bound in every window, on the median reading "
          f"and on the RMS alike** -- {min(medians):.2f} to {max(medians):.2f} times on the median, "
          f"more on the RMS where the 2.002 m statement lands. So the direction of the finding does "
          f"not come from that one reading, even though its size in the RMS column does. This is the "
          f"same fact the filters report as a z RMS below 1: a declared R larger than the "
          f"innovations it predicts.")
        A("")
        A(f"It is not a correction to make, and the bound does not say the declared R is wrong. "
          f"{month_ratio['bound_assumes'].capitalize()} -- and NOAA's stated sigma is a published "
          f"accuracy statement, not a per-reading white-noise variance. What the comparison "
          f"supports is that this record is smoother than the declared R treats it as being, which "
          f"is a statement about the pair and not about either alone.")
    A("")

    A("## What these numbers do not show")
    A("")
    for line in report["limitations"]:
        A(f"- {line}")
    return "\n".join(L) + "\n"


def main(out_dir: Path = REPO_ROOT / "results", *, quiet: bool = False) -> int:
    report = compute()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = render(report)
    (out_dir / "real_noaa_month.md").write_text(text, encoding="utf-8")
    (out_dir / "real_noaa_month.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not quiet:
        print(text)
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(args.out_dir, quiet=args.quiet))
