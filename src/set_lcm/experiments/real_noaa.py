"""P4: the first real observations through the same runner, and a truth-free report.

NOAA CO-OPS station 8454000 (Providence, RI), six-minute water levels, as DAF admitted them
from its committed fixtures (data/daf/, exported at the DAF commit in data/daf/manifest.json;
no network). Each file is checked against the manifest's sha256 and brought in through
set_lcm.bridge.daf with every choice declared: time zone UTC (the NOAA binding's
time_zone=gmt), cadence 360 s, arrival "replay" with latency 0, conflict policy "refuse".
Two estimators from testbed.estimators_water run through runner.run() exactly as the
simulated ones do -- one sensor, n_report = 1 (the water level), no constraint declared:

    level_trend   [level, rate], continuous white-noise acceleration, q_scale in m s^-3/2
    tide_kf       [mean level, (a, b) for M2, K1, O1, M4], random walk on every state,
                  q_scale in m s^-1/2

Declared, not fitted, and not changed after the first run of this report:

    priors        level / mean level ~ N(0 m, (10 m)^2); rate ~ N(0, (1e-3 m/s)^2);
                  every harmonic coefficient ~ N(0, (2 m)^2)          (estimators_water)
    R             NOAA's stated sigma^2 per reading (the bridge's R), throughout; a stated
                  0.000 stays R = 0. Sensitivity: the same runs with R x 10 and R x 100 --
                  declared alternatives, not fits.
    q grids       13 values, half a decade apart: level_trend 1e-9 ... 1e-3 m s^-3/2,
                  tide_kf 1e-7 ... 1e-1 m s^-1/2 (Q_GRIDS)
    fit window    2024-01-15, MLLW datum, ONLY: q_scale = the grid value that maximises the
                  innovation log-likelihood  sum_j -1/2 (log(2 pi S_j) + nu_j^2 / S_j)  over
                  every observed step of that day, under the declared prior
    evaluation    2026-08-23 (preliminary, q=p), held out -- a different day, different
                  evidence ids; and 2024-01-15 again, labelled in-sample
    datum check   2024-01-15 on MLLW and on STND with the same q_scale: max |z_MLLW - z_STND|
                  after the first BURN_IN = 20 steps, and the offset the level (level_trend)
                  or mean-level (tide_kf) states converge to, against the offset in the data

Per estimator and day the report gives the fitted q_scale and log-likelihood and, from
testbed.truth_free (the RunResult alone): z mean, z RMS, lag-1 autocorrelation of z, fraction
|z| > 1.96, CUSUM alarms (count, first step) and the missing fraction; and the mean share of
the stated innovation variance that R accounts for, R_j / S_j. A model-free series check adds
what a white measurement error would imply for the day's second differences.

What no number here can do: score an estimate against the water level, which nobody knows.
The report states what each number can and cannot validate (render()).

    python -m set_lcm.experiments.real_noaa   -> results/real_noaa.{md,json}
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..bridge.daf import BridgedSeries, bridge, load_records
from ..testbed.cusum import CusumConfig
from ..testbed.estimators_water import (
    CONSTITUENT_SPEED_DEG_PER_HOUR, TIDE_CONSTITUENTS, LevelTrendConfig, LevelTrendKF, TideConfig, TideKF,
    rayleigh_period_hours,
)
from ..testbed.runner import EstimatorSpec, RunResult, run
from ..testbed.truth_free import evaluate_truth_free
from .provenance import REPO_ROOT

DATA_DIR = REPO_ROOT / "data" / "daf"
MANIFEST_PATH = DATA_DIR / "manifest.json"

MLLW = ("8454000", "MLLW", "m")
STND = ("8454000", "STND", "m")


@dataclass(frozen=True)
class Day:
    key: str
    label: str
    file: str
    series: tuple[str, str, str]
    role: str


FIT = Day("fit", "2024-01-15 MLLW", "noaa_live_8454000_20240115_mllw.observations.json", MLLW,
          "fit window; in-sample when evaluated")
HELD_OUT = Day("held_out", "2026-08-23 MLLW (preliminary)", "noaa_live_8454000_preliminary.observations.json", MLLW,
               "held out")
DATUM = Day("stnd", "2024-01-15 STND", "noaa_live_8454000_20240115_stnd.observations.json", STND,
            "datum check only")
DAYS = (FIT, HELD_OUT, DATUM)

BRIDGE_ARGS = dict(time_zone="UTC", cadence_s=360, arrival_policy="replay", latency_s=0.0, conflict_policy="refuse")

LEVEL0 = (0.0,)       # m: declared prior mean of the level (level_trend) / mean level (tide_kf)
LEVEL0_STD = 10.0     # m
KINDS = ("level_trend", "tide_kf")
SPECS = {k: EstimatorSpec(k, k, None) for k in KINDS}       # no constraint, never projected; CUSUM on (k 0.5, h 8)
CONFIGS = {"level_trend": LevelTrendConfig, "tide_kf": TideConfig}
Q_UNITS = {"level_trend": "m s^-3/2", "tide_kf": "m s^-1/2"}
Q_GRIDS = {
    "level_trend": tuple(float(q) for q in 10.0 ** np.linspace(-9.0, -3.0, 13)),
    "tide_kf": tuple(float(q) for q in 10.0 ** np.linspace(-7.0, -1.0, 13)),
}
R_SCALES = (1.0, 10.0, 100.0)    # 1 = NOAA's stated sigma^2; 10 and 100 are declared alternatives
BURN_IN = 20                     # steps excluded from the datum check's max |dz|
WINDOW = "day"


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------

def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _manifest_entry(man: dict, name: str) -> dict:
    entry = next((f for f in man["files"] if f["output"] == name), None)
    if entry is None:
        raise ValueError(f"{name} is not listed in {MANIFEST_PATH}")
    raw = (DATA_DIR / name).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != entry["output_sha256"]:
        raise ValueError(f"{name}: sha256 {digest} is not the manifest's {entry['output_sha256']}")
    return entry


def load_day(day: Day, man: dict | None = None) -> BridgedSeries:
    """One committed DAF file through the bridge, with every choice declared (BRIDGE_ARGS)."""
    man = man if man is not None else manifest()
    _manifest_entry(man, day.file)
    return bridge(load_records(DATA_DIR / day.file), series=[day.series], daf_commit=man["daf_commit"],
                  **BRIDGE_ARGS)


def day_span(bs: BridgedSeries) -> tuple[str, str]:
    """[first grid point, last grid point] as ISO-8601 UTC."""
    epoch = np.datetime64(bs.provenance["epoch_iso"].rstrip("Z"), "s")
    last = epoch + np.timedelta64(int(round(float(bs.inputs.t[-1]))), "s")
    return bs.provenance["epoch_iso"], str(last) + "Z"


def check_disjoint(a: BridgedSeries, b: BridgedSeries) -> None:
    """The fit window and the evaluation window: different calendar days, non-overlapping
    spans, no evidence id in common. Raises otherwise."""
    (a0, a1), (b0, b1) = day_span(a), day_span(b)
    if a0[:10] == b0[:10] or not (a1 < b0 or b1 < a0):
        raise ValueError(f"windows overlap: {a0}..{a1} and {b0}..{b1}")
    if set(a.evidence_ids_used) & set(b.evidence_ids_used):
        raise ValueError("the fit window and the evaluation window share evidence ids")


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------

def run_one(kind: str, q_scale: float, bs: BridgedSeries, r_scale: float = 1.0) -> RunResult:
    """One estimator over one bridged day. r_scale != 1 hands run() NEW observations whose R
    is scaled; the bridged ones are never edited."""
    obs = bs.observations if r_scale == 1.0 else [replace(o, R=o.R * r_scale) for o in bs.observations]
    return run(bs.inputs, obs, None, SPECS[kind], LEVEL0, LEVEL0_STD, est_cfg=CONFIGS[kind](q_scale=q_scale))


def loglik(rr: RunResult) -> float:
    """Innovation (prediction-error) log-likelihood of the observed samples under the filter:
    sum over finite innovations of -1/2 (log(2 pi S) + nu^2 / S). Reads the RunResult only."""
    nu, s = rr.innov[:, 0], rr.innov_var[:, 0]
    ok = np.isfinite(nu)
    return float(np.sum(-0.5 * (np.log(2.0 * np.pi * s[ok]) + nu[ok] ** 2 / s[ok])))


def fit_q(kind: str, bs: BridgedSeries) -> dict:
    """Maximise the innovation log-likelihood over Q_GRIDS[kind] on ONE bridged day."""
    profile = [{"q_scale": q, "loglik": loglik(run_one(kind, q, bs))} for q in Q_GRIDS[kind]]
    i = int(np.argmax([p["loglik"] for p in profile]))
    q = profile[i]["q_scale"]
    dt = float(bs.inputs.t[1] - bs.inputs.t[0])
    if kind == "level_trend":   # what the fitted q means per six-minute step
        per_step = {"rate_std_m_per_s": q * math.sqrt(dt), "level_std_m": q * dt ** 1.5 / math.sqrt(3.0)}
    else:
        per_step = {"state_std_m": q * math.sqrt(dt)}
    return {"evidence_ids_sha256": bs.provenance["evidence_ids_sha256"], "profile": profile,
            "argmax_index": i, "q_scale": q, "loglik": profile[i]["loglik"], "interior": 0 < i < len(profile) - 1,
            "units": Q_UNITS[kind], "per_step": per_step}


def summarize(rr: RunResult, bs: BridgedSeries, r_scale: float, q_scale: float) -> dict:
    """What the record says about the run (truth_free.evaluate_truth_free over the whole day),
    the log-likelihood, and the mean share of the stated innovation variance R accounts for."""
    n = len(bs.observations)
    tf = evaluate_truth_free(rr, {WINDOW: (0, n)})
    s0 = tf["windows"][WINDOW]["sensors"][0]
    ok = np.isfinite(rr.innov_var[:, 0])
    r = np.array([float(o.R[0, 0]) * r_scale for o in bs.observations])
    return {
        "q_scale": q_scale,
        "r_scale": r_scale,
        "loglik": loglik(rr),
        "n_steps": n,
        "n_observed": s0["n_observed"],
        "frac_missing": s0["frac_missing"],
        "z_mean": s0["z_mean"],
        "z_rms": s0["z_rms"],
        "z_lag1": s0["z_lag1"],
        "frac_abs_z_gt_1.96": s0["frac_abs_z_gt_1.96"],
        "cusum_alarms": s0["cusum_alarms"],
        "cusum_first_alarm": s0["cusum_first_alarm"],
        "r_share_mean": float(np.mean(r[ok] / rr.innov_var[ok, 0])),
        "status_counts": {k: v for k, v in sorted(Counter(st.value for st in rr.status).items())},
        "n_evidence_ids": tf["windows"][WINDOW]["n_evidence_ids"],
        "constraint_declared": tf["constraint_declared"],
        "latency_us_p50": tf["latency_us_p50"],
        "latency_us_p99": tf["latency_us_p99"],
    }


def _level_state(kind: str, rr: RunResult) -> np.ndarray:
    """The state a datum offset should end up in: the level (level_trend), the mean level (tide_kf)."""
    return rr.x[:, 0] if kind == "level_trend" else rr.extra["mean_level_hat"]


def datum_check(kind: str, q_scale: float, bs_mllw: BridgedSeries, bs_stnd: BridgedSeries) -> dict:
    """The same water surface on two datums, the same filter, the same declared prior."""
    rm, rs = run_one(kind, q_scale, bs_mllw), run_one(kind, q_scale, bs_stnd)
    dz = np.abs(rm.innov_z[:, 0] - rs.innov_z[:, 0])
    off = _level_state(kind, rs) - _level_state(kind, rm)
    y_off = np.array([o.y[0] for o in bs_stnd.observations]) - np.array([o.y[0] for o in bs_mllw.observations])
    late = dz[BURN_IN:]
    return {
        "q_scale": q_scale,
        "state": "level" if kind == "level_trend" else "mean_level",
        "burn_in_steps": BURN_IN,
        "max_abs_dz_after_burn_in": float(late.max()),
        "argmax_step_after_burn_in": int(BURN_IN + np.argmax(late)),
        "max_abs_dz_all_steps": float(dz.max()),
        "abs_dz_at_step_0": float(dz[0]),
        "max_abs_dz_from_step_1": float(dz[1:].max()),
        "offset_final": float(off[-1]),
        "offset_min_after_burn_in": float(off[BURN_IN:].min()),
        "offset_max_after_burn_in": float(off[BURN_IN:].max()),
        "reported_level_offset_final": float(rs.x[-1, 0] - rm.x[-1, 0]),
        "data_offset_min": float(y_off.min()),
        "data_offset_max": float(y_off.max()),
    }


def series_check(bs: BridgedSeries) -> dict:
    """Model-free: if the error of each six-minute value were white with variance sigma_e^2
    and independent of the water level, the mean square of the series' second differences
    would be at least 6 sigma_e^2 (in expectation), so sigma_e <= rms(d2 y) / sqrt(6). Set
    against NOAA's stated sigma. A time-correlated error is not bounded by this."""
    y = np.array([o.y[0] for o in bs.observations])
    var = np.array([float(o.R[0, 0]) for o in bs.observations])
    if not np.all(np.isfinite(y)):
        raise ValueError("series_check needs a complete day")
    d2 = np.diff(y, 2)
    rms_d2 = float(np.sqrt(np.mean(d2 ** 2)))
    bound = rms_d2 / math.sqrt(6.0)
    rms_sigma = float(np.sqrt(np.mean(var)))
    return {
        "n_second_differences": int(d2.size),
        "rms_second_difference": rms_d2,
        "white_error_sigma_bound": bound,
        "rms_stated_sigma": rms_sigma,
        "median_stated_sigma": float(np.median(np.sqrt(var))),
        "mean_stated_var_over_bound_var": float(np.mean(var) / bound ** 2),
    }


# ---------------------------------------------------------------------------
# the whole report
# ---------------------------------------------------------------------------

def compute() -> dict:
    man = manifest()
    bs = {d.key: load_day(d, man) for d in DAYS}
    check_disjoint(bs[FIT.key], bs[HELD_OUT.key])
    fits = {k: {"day": FIT.key, **fit_q(k, bs[FIT.key])} for k in KINDS}
    runs: dict = {}
    for k in KINDS:
        q = fits[k]["q_scale"]
        runs[k] = {}
        for d in (FIT, HELD_OUT):
            runs[k][d.key] = {f"R x{s:g}": summarize(run_one(k, q, bs[d.key], s), bs[d.key], s, q) for s in R_SCALES}
    datum = {k: datum_check(k, fits[k]["q_scale"], bs[FIT.key], bs[DATUM.key]) for k in KINDS}
    days = {}
    for d in DAYS:
        entry = _manifest_entry(man, d.file)
        start, end = day_span(bs[d.key])
        days[d.key] = {
            "label": d.label, "role": d.role, "file": f"data/daf/{d.file}", "series": list(d.series),
            "first_grid_point": start, "last_grid_point": end, "n_grid": bs[d.key].provenance["n_grid"],
            "evidence_ids_sha256": bs[d.key].provenance["evidence_ids_sha256"],
            "series_check": series_check(bs[d.key]) if d is not DATUM else None,
            "file_sha256": entry["output_sha256"], "daf_version_id": entry["daf_version_id"],
            "request_url": entry["request_url"], "note": entry["note"],
        }
    out = {
        "provenance": {
            "daf_commit": man["daf_commit"],
            "daf_repository": man["daf_repository"],
            "vendored_substrate": man["vendored_substrate"],
            "bridge": {d.key: bs[d.key].provenance for d in DAYS},
        },
        "declared": {
            "bridge": dict(BRIDGE_ARGS),
            "prior": {"level0_m": LEVEL0[0], "level0_std_m": LEVEL0_STD,
                      "rate0_m_per_s": LevelTrendConfig.rate0, "rate0_std_m_per_s": LevelTrendConfig.rate0_std,
                      "coef0_std_m": TideConfig.coef0_std},
            "models": {"level_trend": LevelTrendKF.model_version, "tide_kf": TideKF.model_version},
            "constituents_deg_per_hour": {c: CONSTITUENT_SPEED_DEG_PER_HOUR[c] for c in TIDE_CONSTITUENTS},
            "rayleigh_period_hours": {
                f"{a}-{b}": rayleigh_period_hours(a, b)
                for a, b in (("S2", "M2"), ("N2", "M2"), ("K1", "O1"), ("M2", "K1"), ("M2", "O1"))},
            "q_grids": {k: list(v) for k, v in Q_GRIDS.items()},
            "q_units": dict(Q_UNITS),
            "r_scales": list(R_SCALES),
            "cusum": {"k": CusumConfig().k, "h": CusumConfig().h},
            "burn_in_steps": BURN_IN,
            "fit_day": FIT.key,
            "evaluation_days": [HELD_OUT.key, FIT.key],
            "loglik": "sum over observed steps of -1/2 (log(2 pi S_j) + nu_j^2 / S_j), declared prior included",
        },
        "days": days,
        "fit": fits,
        "runs": runs,
        "datum_invariance": datum,
    }
    out["claims"] = claims(out)
    return out


def claims(r: dict) -> dict:
    """Every qualitative sentence render() prints, as a computed condition. If one fails the
    report is not written: the text would no longer be true of the numbers."""
    prim = [r["runs"][k][d]["R x1"] for k in KINDS for d in (FIT.key, HELD_OUT.key)]
    c = {
        "fits_interior": all(r["fit"][k]["interior"] for k in KINDS),
        "r_is_minority_of_s": all(p["r_share_mean"] < 0.5 for p in prim),
        "innovations_autocorrelated": all(p["z_lag1"] is not None and p["z_lag1"] > 0.3 for p in prim),
        "no_missing": all(p["frac_missing"] == 0.0 for p in prim),
        "loglik_falls_with_r_scale": all(
            r["runs"][k][d]["R x1"]["loglik"] > r["runs"][k][d]["R x10"]["loglik"] > r["runs"][k][d]["R x100"]["loglik"]
            for k in KINDS for d in (FIT.key, HELD_OUT.key)),
        "stated_var_exceeds_white_bound": all(
            r["days"][d]["series_check"]["mean_stated_var_over_bound_var"] > 1.0 for d in (FIT.key, HELD_OUT.key)),
        "no_constraint_declared": all(not p["constraint_declared"] and set(p["status_counts"]) == {"skipped"}
                                      for k in KINDS for d in (FIT.key, HELD_OUT.key)
                                      for p in r["runs"][k][d].values()),
        "level_trend_datum_offset_absorbed_at_first_reading": (
            r["datum_invariance"]["level_trend"]["max_abs_dz_from_step_1"] < 1e-4
            and r["datum_invariance"]["level_trend"]["max_abs_dz_after_burn_in"] < 1e-6
            and abs(r["datum_invariance"]["level_trend"]["offset_final"]
                    - r["datum_invariance"]["level_trend"]["data_offset_max"]) < 1e-9),
        "tide_kf_datum_offset_absorbed_slowly": (
            r["datum_invariance"]["tide_kf"]["offset_min_after_burn_in"]
            < r["datum_invariance"]["tide_kf"]["data_offset_min"] - 0.01
            and abs(r["datum_invariance"]["tide_kf"]["offset_final"]
                    - r["datum_invariance"]["tide_kf"]["data_offset_max"]) < 1e-4),
        "data_offset_constant": all(
            r["datum_invariance"][k]["data_offset_max"] - r["datum_invariance"][k]["data_offset_min"] < 1e-9
            for k in KINDS),
        "level_trend_predicts_both_days_better": all(
            r["runs"]["level_trend"][d]["R x1"]["loglik"] > r["runs"]["tide_kf"][d]["R x1"]["loglik"]
            for d in (FIT.key, HELD_OUT.key)),
        "tide_one_day_limits": (r["declared"]["rayleigh_period_hours"]["S2-M2"] > 24.0
                                and r["declared"]["rayleigh_period_hours"]["N2-M2"] > 24.0
                                and r["declared"]["rayleigh_period_hours"]["K1-O1"] > 24.0),
    }
    return c


def _alarms(p: dict) -> str:
    return f"{p['cusum_alarms']}" + (f" (step {p['cusum_first_alarm']})" if p["cusum_first_alarm"] is not None else "")


def _step_time(r: dict, day_key: str, step: int) -> str:
    """Report step -> HH:MM UTC on that day's grid (its first grid point plus step x cadence)."""
    t0 = np.datetime64(r["days"][day_key]["first_grid_point"].rstrip("Z"), "s")
    t = t0 + np.timedelta64(int(round(step * r["declared"]["bridge"]["cadence_s"])), "s")
    return str(t)[11:16] + " UTC"


def _alarm_phrase(r: dict, day_key: str, p: dict) -> str:
    n = p["cusum_alarms"]
    if not n:
        return "none"
    first = p["cusum_first_alarm"]
    return f"{n} (first at step {first}, {_step_time(r, day_key, first)})"


def render(r: dict) -> str:
    c = r["claims"]
    failed = [k for k, v in c.items() if not v]
    if failed:
        raise RuntimeError(f"report text no longer true of the numbers: {failed}; revise render() before writing")
    fit, held = r["days"][FIT.key], r["days"][HELD_OUT.key]
    bp = r["provenance"]["bridge"]
    L = ["# P4: first real observations — NOAA 8454000 water levels through the same runner", ""]
    L += [
        "Truth-free: nobody knows the water level these readings measured, so no number below is an error. "
        "Every number is computed from the run's own record (`testbed.truth_free`), the bridged observations, "
        "or the committed evidence.", "",
        f"Evidence: DAF `{r['provenance']['daf_commit'][:12]}` ({r['provenance']['daf_repository']}), committed "
        "fixtures replayed by DAF's own NOAA binding into `data/daf/` — no network. Bridge: "
        + ", ".join(f"{k} {v}" for k, v in r["declared"]["bridge"].items()) + ".", "",
        "| day | role | file | first – last grid point (UTC) | readings | stated σ min–max [m], zeros | evidence-id sha256 |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in (FIT, HELD_OUT, DATUM):
        x = r["days"][d.key]
        rs = bp[d.key]["R_source"][f"noaa:{d.series[0]}:{d.series[1]}:{d.series[2]}"]
        L.append(f"| {x['label']} | {x['role']} | `{Path(x['file']).name}` | {x['first_grid_point']} – "
                 f"{x['last_grid_point']} | {bp[d.key]['n_used']} of {bp[d.key]['n_records_in']} | "
                 f"{rs['sigma_min']:.3f}–{rs['sigma_max']:.3f}, {rs['n_zero']} | `{x['evidence_ids_sha256'][:16]}` |")
    pr = r["declared"]["prior"]
    L += ["", "## Declared, not fitted", "",
          f"- `level_trend` ({r['declared']['models']['level_trend']}): x = [level, rate], F = [[1, dt], [0, 1]], "
          "Q = q² [[dt³/3, dt²/2], [dt²/2, dt]] (continuous white-noise acceleration), H = [1, 0], dt = 360 s "
          "from the grid.",
          f"- `tide_kf` ({r['declared']['models']['tide_kf']}): x = [mean level, (a, b) for "
          + ", ".join(f"{k} {v} °/h" for k, v in r["declared"]["constituents_deg_per_hour"].items())
          + "] (Schureman 1958, SP 98), H_k = [1, cos ω t_k, sin ω t_k, …], F = I, Q = q² dt I; it reports the "
          "level H_k x_k.",
          f"- Priors: level / mean level N({pr['level0_m']:g} m, ({pr['level0_std_m']:g} m)²), rate "
          f"N({pr['rate0_m_per_s']:g}, ({pr['rate0_std_m_per_s']:g} m/s)²), every harmonic coefficient "
          f"N(0, ({pr['coef0_std_m']:g} m)²). One sensor, n_report = 1, no constraint, CUSUM k = "
          f"{r['declared']['cusum']['k']:g}, h = {r['declared']['cusum']['h']:g}.",
          "- R = NOAA's stated σ² per reading (a stated 0.000 stays R = 0); R × 10 and R × 100 are declared "
          "alternatives, not fits.",
          "- q_scale: the grid value that maximises the innovation log-likelihood "
          "Σ −½ (log 2πS_j + ν_j²/S_j) over every observed step of **2024-01-15 MLLW only** (declared prior "
          "included). Grids: 13 values half a decade apart, "
          + "; ".join(f"`{k}` {r['declared']['q_grids'][k][0]:.0e} … {r['declared']['q_grids'][k][-1]:.0e} "
                      f"{r['declared']['q_units'][k]}" for k in KINDS) + ".",
          f"- Evaluation: the held-out {held['label']} day (a different day, no evidence id in common) and "
          f"{fit['label']} again, labelled in-sample.", ""]
    L += ["## Identification on 2024-01-15 MLLW", "",
          "| i | level_trend q [m s^-3/2] | log-lik | tide_kf q [m s^-1/2] | log-lik |", "|---|---|---|---|---|"]
    for i in range(len(Q_GRIDS["level_trend"])):
        cells = []
        for k in KINDS:
            p = r["fit"][k]["profile"][i]
            mark = "**" if i == r["fit"][k]["argmax_index"] else ""
            cells += [f"{mark}{p['q_scale']:.2e}{mark}", f"{mark}{p['loglik']:.1f}{mark}"]
        L.append(f"| {i} | " + " | ".join(cells) + " |")
    L += ["", "Bold: the fitted value, interior to its grid for both estimators "
          f"(`level_trend` q = {r['fit']['level_trend']['q_scale']:.2e} {Q_UNITS['level_trend']}, "
          f"`tide_kf` q = {r['fit']['tide_kf']['q_scale']:.2e} {Q_UNITS['tide_kf']}). A half-decade grid "
          "resolves q to that factor and no finer. Per six-minute step the fitted values mean a rate random walk of "
          f"{r['fit']['level_trend']['per_step']['rate_std_m_per_s']:.1e} m/s for `level_trend`, and a random walk of "
          f"{r['fit']['tide_kf']['per_step']['state_std_m'] * 1000:.1f} mm on each of `tide_kf`'s nine states: its "
          "harmonic basis is re-fitted as the day goes, not held.", ""]
    L += ["## Per estimator and day (fitted q, R = stated σ²)", "",
          "| estimator | day | q_scale | log-lik | z mean | z RMS | z lag-1 | \\|z\\| > 1.96 | CUSUM alarms (first step) "
          "| missing | mean R/S |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in KINDS:
        for d in (HELD_OUT, FIT):
            p = r["runs"][k][d.key]["R x1"]
            tag = "held out" if d is HELD_OUT else "in-sample (fit day)"
            L.append(f"| {k} | {r['days'][d.key]['label']}, {tag} | {p['q_scale']:.2e} | {p['loglik']:.1f} | "
                     f"{p['z_mean']:+.3f} | {p['z_rms']:.3f} | {p['z_lag1']:+.3f} | {p['frac_abs_z_gt_1.96']:.3f} | "
                     f"{_alarms(p)} | {p['frac_missing']:.3f} | {p['r_share_mean']:.3f} |")
    L += ["", "## R sensitivity at the fitted q (declared alternatives, not fits)", "",
          "| estimator | day | R scale | log-lik | z RMS | z lag-1 | \\|z\\| > 1.96 | CUSUM alarms (first step) | mean R/S |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k in KINDS:
        for d in (HELD_OUT, FIT):
            for s in R_SCALES:
                p = r["runs"][k][d.key][f"R x{s:g}"]
                L.append(f"| {k} | {r['days'][d.key]['label']} | × {s:g} | {p['loglik']:.1f} | {p['z_rms']:.3f} | "
                         f"{p['z_lag1']:+.3f} | {p['frac_abs_z_gt_1.96']:.3f} | {_alarms(p)} | {p['r_share_mean']:.3f} |")
    L += ["", "## Datum invariance (computed, not assumed)", "",
          f"2024-01-15 on MLLW and on STND, the same q_scale and the same declared prior (0 ± 10 m in both datums). "
          "The data differ by a constant; once the prior is forgotten the innovations should not.", "",
          f"| estimator | state | max \\|z_MLLW − z_STND\\|, steps ≥ {BURN_IN} (at step) | all steps | STND − MLLW state, last "
          f"step [m] | range, steps ≥ {BURN_IN} [m] | STND − MLLW data [m] |", "|---|---|---|---|---|---|---|"]
    for k in KINDS:
        x = r["datum_invariance"][k]
        L.append(f"| {k} | {x['state']} | {x['max_abs_dz_after_burn_in']:.2e} ({x['argmax_step_after_burn_in']}) | "
                 f"{x['max_abs_dz_all_steps']:.4f} | {x['offset_final']:.6f} | {x['offset_min_after_burn_in']:.4f} – "
                 f"{x['offset_max_after_burn_in']:.4f} | {x['data_offset_min']:.6f} – {x['data_offset_max']:.6f} |")
    dl, dt = r["datum_invariance"]["level_trend"], r["datum_invariance"]["tide_kf"]
    L += ["", "By linearity, the STND − MLLW difference between the two runs is the filter's own response to a "
          "constant offset read from a prior mean of 0; both models represent a constant exactly (level or mean level "
          "equal to it, every other state 0), so the difference tends to it, and the table measures how fast. "
          f"`level_trend` absorbs it at the first reading (its 10 m prior std dwarfs √R): its innovations agree to "
          f"{dl['max_abs_dz_from_step_1']:.1e} from the second reading on and to "
          f"{dl['max_abs_dz_after_burn_in']:.1e} after step {BURN_IN}, and its level states differ by the data's "
          "offset. `tide_kf` absorbs it slowly: the first reading splits the offset between the mean level "
          "and the harmonic coefficients in proportion to their prior variances, and the coefficients hand it back "
          f"over hours — its mean-level states differ by {dt['offset_min_after_burn_in']:.4f}–"
          f"{dt['offset_max_after_burn_in']:.4f} m after step {BURN_IN} and by {dt['offset_final']:.6f} m at the "
          f"last step, and its innovations by up to {dt['max_abs_dz_after_burn_in']:.4f} in z after step {BURN_IN} "
          f"(at step {dt['argmax_step_after_burn_in']}). The datum does not change what either filter says about "
          "the water once the prior is gone; for `tide_kf` one day is not long enough for it to be entirely gone.",
          ""]
    L += ["## Model-free series check", "",
          "If each six-minute value's error were white with variance σ_e² and independent of the water level, the "
          "mean square of the day's second differences would be at least 6 σ_e², so σ_e ≤ rms(Δ²y)/√6. A "
          "time-correlated error is not bounded by this.", "",
          "| day | rms Δ²y [m] | white-error bound [m] | rms stated σ [m] | median stated σ [m] | mean σ² / bound² |",
          "|---|---|---|---|---|---|"]
    for d in (FIT, HELD_OUT):
        s = r["days"][d.key]["series_check"]
        L.append(f"| {r['days'][d.key]['label']} | {s['rms_second_difference']:.4f} | {s['white_error_sigma_bound']:.4f} | "
                 f"{s['rms_stated_sigma']:.4f} | {s['median_stated_sigma']:.4f} | {s['mean_stated_var_over_bound_var']:.1f} |")
    lt_f, lt_h = r["runs"]["level_trend"][FIT.key]["R x1"], r["runs"]["level_trend"][HELD_OUT.key]["R x1"]
    td_f, td_h = r["runs"]["tide_kf"][FIT.key]["R x1"], r["runs"]["tide_kf"][HELD_OUT.key]["R x1"]
    prim = [lt_f, lt_h, td_f, td_h]
    sc_f, sc_h = fit["series_check"], held["series_check"]
    rp = r["declared"]["rayleigh_period_hours"]
    L += ["", "## What these numbers can and cannot validate", "",
          "- **NOAA's σ is not the error of the six-minute value.** It is the standard deviation of the one-second "
          "samples behind it — waves and seiche included — reported to 1 mm; R = σ² passes the source's statement "
          "through. The series check says a white error that large is incompatible with these series: the "
          f"second differences allow at most {sc_f['white_error_sigma_bound']:.4f} m on {fit['label']} and "
          f"{sc_h['white_error_sigma_bound']:.4f} m on the held-out day, where the stated σ has an rms of "
          f"{sc_f['rms_stated_sigma']:.4f} and {sc_h['rms_stated_sigma']:.4f} m (mean σ² "
          f"{sc_f['mean_stated_var_over_bound_var']:.1f} and {sc_h['mean_stated_var_over_bound_var']:.1f} times the "
          "bound). A correlated error could be that large; nothing here can tell.",
          f"- **z RMS with R = σ² does not say σ² is a calibrated measurement variance for these filters.** R is "
          f"{min(p['r_share_mean'] for p in prim):.2f}–{max(p['r_share_mean'] for p in prim):.2f} of the stated "
          "innovation variance on average (mean R/S), so the innovations are mostly the filters' own prediction "
          f"uncertainty; and they are not white (lag-1 {min(p['z_lag1'] for p in prim):+.2f} to "
          f"{max(p['z_lag1'] for p in prim):+.2f}, where {lt_h['n_observed']} white samples would give 0 ± "
          f"{1.0 / math.sqrt(lt_h['n_observed']):.3f}), so neither model is right about the dynamics. z RMS is "
          f"{lt_h['z_rms']:.3f} for `level_trend` on the held-out day and {lt_f['z_rms']:.3f} in-sample, "
          f"{td_h['z_rms']:.3f} and {td_f['z_rms']:.3f} for `tide_kf`. Below 1 the filter states more innovation "
          f"variance than it sees; a value near 1 next to a lag-1 of {lt_h['z_lag1']:+.2f} is not calibration "
          "either.",
          "- **R and Q are not separated here, and one gauge cannot separate them without trusting the model.** "
          "Only q is fitted; R is held at "
          "σ², so the fitted q absorbs whatever σ² does not explain. At the fitted q the log-likelihood falls when R "
          "is scaled by 10 and by 100 on both days and for both filters: those alternatives predict worse *at this "
          "q*, which does not make σ² right. A joint fit would split the variance only through the model's own "
          "assumptions (a white measurement error, these dynamics); telling sensor error from unmodelled water "
          "motion needs an independent measurement of the same water surface, which one gauge does not provide.",
          f"- **The CUSUM alarms cannot be classified.** `level_trend`: {_alarm_phrase(r, FIT.key, lt_f)} on "
          f"{fit['label']}, {_alarm_phrase(r, HELD_OUT.key, lt_h)} on the held-out day; `tide_kf`: "
          f"{_alarm_phrase(r, FIT.key, td_f)} and {_alarm_phrase(r, HELD_OUT.key, td_h)}. On a simulated record an "
          "alarm is scored against a known onset; here there is none, and whether an alarm is a real change in the "
          "water, an instrument event or the model's own misfit needs evidence independent of this series. The "
          "null behind h = 8 was measured on simulated nominal records (results/calibration.md); these innovations "
          "are strongly autocorrelated (the lag-1 column), and a positively autocorrelated z drifts further before "
          "it turns, so h = 8's false-alarm rate on this record is unknown.",
          "- **No constraint is applied, so the consistency channel is not exercised on real data yet.** A single "
          "series has no conservation relation to declare; every step's status is `skipped`, and the χ² test, the "
          "guard and the projection — the reconciliation stage this repository is about — never ran on these "
          "records.",
          "- **The harmonic coefficients are not a tidal analysis.** One day of data cannot separate S2 or N2 from "
          f"M2 (Rayleigh periods {rp['S2-M2'] / 24:.1f} and {rp['N2-M2'] / 24:.1f} days), nor K1 from O1 "
          f"({rp['K1-O1'] / 24:.1f} days), and M2 sits at the limit against K1 ({rp['M2-K1']:.1f} h) and O1 "
          f"({rp['M2-O1']:.1f} h). The coefficients are nuisance states for the next six-minute prediction; "
          "nothing reads them as amplitudes.",
          "- **One held-out day is one day.** By the one-step log-likelihood `level_trend` predicts the held-out "
          f"day better than `tide_kf` ({lt_h['loglik']:.1f} against {td_h['loglik']:.1f}) and the fit day too "
          f"({lt_f['loglik']:.1f} against {td_f['loglik']:.1f}); that orders two misspecified filters on two days, "
          "and a day of another tidal range, season or weather could order them differently. The held-out day is "
          "preliminary (q=p): a revision by NOAA would arrive as a new DAF observation with a new id, and the "
          "bridge reports or refuses a disagreement, it does not follow it.",
          "- **Nothing here is an error against the water level.** Every number is a property of the record and the "
          "filter together; a filter can be confidently wrong with white, unit-variance innovations if the "
          "evidence is wrong in a way its model explains.", ""]
    return "\n".join(L)


def main(out_dir: Path, *, quiet: bool = False) -> int:
    from .provenance import header_line, provenance

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    prov = provenance()
    result = compute()
    result["provenance"] = {"generation": prov, **result["provenance"]}
    text = render(result).replace("\n\n", "\n\n" + header_line(prov) + "\n\n", 1)
    (out_dir / "real_noaa.md").write_text(text, encoding="utf-8")
    (out_dir / "real_noaa.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    if not quiet:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(Path.cwd() / "results"))
