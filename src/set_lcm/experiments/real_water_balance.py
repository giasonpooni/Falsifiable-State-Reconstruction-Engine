"""P4b: a real reservoir water balance through the reconciliation kernel.

The first time this repository's constraint machinery runs on a constraint that is a
physical law over evidence from independent instruments. Ridgway Reservoir on the
Uncompahgre River, Colorado, water years 2023-2025 (2022-10-01 to 2025-09-30, 1,096 days),
four USGS daily-mean series admitted by DAF and committed to `data/daf/`:

    storage   USGS-09147022  00054 acre-ft     daily mean
    outflow   USGS-09147025  00060 ft^3/s      daily mean, below the dam
    inflow    USGS-09146200  00060 ft^3/s      Uncompahgre River near Ridgway, 149 sq mi
    inflow    USGS-09147000  00060 ft^3/s      Dallas Creek near Ridgway, 97.2 sq mi

Gauged drainage 246.2 of 265 sq mi = 0.929. The missing 7.1% -- Beaver Creek and the
lake-local area -- plus evaporation, precipitation on the lake surface, and any gauge bias,
all land in one unmeasured term. The declared constraint is that the gauges close the
balance; the consistency statistic is what tests it. It is expected to FAIL, and the
question this report answers is whether it fails for the right reason and by the right
amount.

WHAT IS DECLARED, AND WHERE EACH NUMBER COMES FROM

USGS states NO per-value uncertainty anywhere in the daily-values API -- DAF's extractor
emits none and defaults none (docs/PHASE_45_USGS_DAILY_VALUES.md) -- so every R here is
CONSUMER-DECLARED and carries its citation through the bridge into this report.

    flows     sigma = 5% of the reading, floored at 1.0 ft^3/s. USGS rates a daily
              discharge record "Good" when about 95% of daily values are within 10% of
              their true value; read as a two-sided 95% normal interval that is 2 sigma,
              so sigma = 5%. Two assumptions are made in that sentence and neither is
              USGS's: that the record is "Good" (the site-and-period rating is published
              in USGS's annual data reports, which this repository did not acquire), and
              that the error is normal. The floor exists because a percentage says nothing
              at zero flow and a zero R would tell the filter the reading is exact.

    storage   sigma is NOT declared once. Reservoir storage is computed from a measured
              lake elevation through a stage-capacity table, and the uncertainty of that
              table at this reservoir is not something this repository has a source for.
              Inventing one number and reporting conclusions from it would be exactly the
              failure this repository exists to refuse. So the storage sigma is a DECLARED
              SWEEP (STORAGE_SIGMA_SWEEP), every conclusion is reported at each value, and
              the report says which conclusions survive the whole range and which do not.

The closure residual below needs no sigma: it is arithmetic on the readings under a
declared daily-mean time pairing, and it is the number to read first.

    python -m set_lcm.experiments.real_water_balance   -> results/real_water_balance.{md,json}
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..bridge.daf import BridgedSeries, DeclaredSigma, SeriesSelector, bridge, load_records
from ..lcm import constraint_bases, detectability
from ..declaration import load as load_declaration
from ..schema import ConstraintSet
from ..testbed.cusum import CusumConfig
from ..testbed.estimators_balance import CFS_DAY_TO_ACRE_FT, BalanceConfig
from ..testbed.runner import EstimatorSpec, RunResult, run
from ..testbed.truth_free import evaluate_truth_free
from .provenance import REPO_ROOT

# ---------------------------------------------------------------------------
# what the caller declares
# ---------------------------------------------------------------------------

DAY_ZONE_CITATION = (
    "The USGS Water Data OGC API defines a daily item's `time` only as the date the observation "
    "represents and states no time zone (measured: every value is a bare 'YYYY-MM-DD'). Legacy NWIS "
    "practice is the site's local standard-time day, which this API does not say. Read here as a UTC "
    "day anchored at its midpoint -- an assumption of this consumer, not a statement of the source. "
    "It shifts every series by the same amount, so it cannot change a same-day difference between "
    "them; it would matter to an alignment against a sub-daily series, and none is used."
)

# Everything below is read from declarations/ridgway.toml rather than written here. The
# names are kept because the rest of this module and its tests use them, but each is now a
# view of one reviewable document instead of a constant beside the code that consumes it.
DECLARATION_PATH = REPO_ROOT / "declarations" / "ridgway.toml"
DECLARATION = load_declaration(DECLARATION_PATH)

# Which record, and where: the last thing that tied this module to one reservoir.
DATA_DIR = REPO_ROOT / DECLARATION.record.directory
MANIFEST_PATH = DATA_DIR / DECLARATION.record.manifest
OBSERVATIONS = DECLARATION.record.observations

# The unit every volume in this report is quoted in, taken from the state that declares it
# rather than retyped into each string.
VOLUME_UNIT = DECLARATION.states["storage"].unit

# The declared sensor order IS the column order the estimators in testbed.estimators_balance
# expect, which used to be a comment next to a hand-written tuple and is now a property of the
# document: reorder the [[sensor]] tables and the columns move with them.
COLUMNS = tuple(sen.role for sen in DECLARATION.sensors)
SELECTORS = tuple(SeriesSelector(source_id=sen.source_id, extraction_method=sen.extraction_method,
                                 property=sen.property, match=sen.match)
                  for sen in DECLARATION.sensors)
SOURCE_IDS = tuple(sel.source_id for sel in SELECTORS)

GAUGED_SQ_MI = DECLARATION.scalar("gauged_drainage_area")
TOTAL_SQ_MI = DECLARATION.scalar("total_drainage_area")

FLOW_SIGMA_RELATIVE = DECLARATION.sensor("outflow").relative
FLOW_SIGMA_FLOOR = DECLARATION.sensor("outflow").sigma_floor
FLOW_SIGMA_CITATION = DECLARATION.sensor("outflow").citation
STORAGE_SIGMA_BASE = DECLARATION.sensor("storage").sigma
STORAGE_SIGMA_CITATION = DECLARATION.sensor("storage").citation

# The sweep is an analysis choice, not a property of the site: the declaration says what this
# consumer declares for the storage series, and this says which neighbourhood of that value
# every conclusion is recomputed at.
STORAGE_SIGMA_SWEEP = (50.0, STORAGE_SIGMA_BASE, 800.0)     # acre-ft

CFG = BalanceConfig(
    q_storage=DECLARATION.scalar("q_storage"),
    q_flow=DECLARATION.scalar("q_flow"),
    q_ungauged=DECLARATION.scalar("q_ungauged"),
    cumulative0_std=DECLARATION.scalar("cumulative0_std"),
    flow0=DECLARATION.scalar("flow0"),
    flow0_std=DECLARATION.scalar("flow0_std"),
)
PRIOR_STORAGE_STD = DECLARATION.scalar("prior_storage_std")


def declared_sigma(storage_sigma: float) -> dict:
    """The consumer-declared R for each column. `storage_sigma` is the sweep's current point;
    every other number, and every citation, is the declaration's."""
    storage = DECLARATION.sensor("storage")
    out = {storage.source_id: DeclaredSigma(sigma=storage_sigma, citation=storage.citation)}
    for sen in DECLARATION.sensors:
        if sen.role != storage.role:
            out[sen.source_id] = DeclaredSigma(relative=sen.relative, sigma_floor=sen.sigma_floor,
                                               citation=sen.citation)
    return out


def load(storage_sigma: float) -> BridgedSeries:
    """The committed DAF observations, checked against the manifest, through the bridge with
    every choice declared."""
    man = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entry = next(f for f in man["files"] if f["output"] == OBSERVATIONS)
    import hashlib
    raw = (DATA_DIR / OBSERVATIONS).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != entry["output_sha256"]:
        raise ValueError(f"{OBSERVATIONS} hashes to {got}; the manifest says {entry['output_sha256']}")
    return bridge(
        load_records(DATA_DIR / OBSERVATIONS), series=list(SELECTORS),
        time_zone="UTC", time_semantics="calendar_day", day_anchor="midpoint",
        day_zone_citation=DAY_ZONE_CITATION, cadence_s=86400,
        arrival_policy="replay", latency_s=0.0, conflict_policy="refuse",
        daf_commit=man["daf_commit"], declared_sigma=declared_sigma(storage_sigma))


# ---------------------------------------------------------------------------
# closure arithmetic on the readings, before filtering or uncertainty weighting
# ---------------------------------------------------------------------------

def closure_residual(bs: BridgedSeries) -> dict:
    """r_k = (S_{k+1} - S_k) - c (qin1 + qin2 - qout)_k, in acre-ft per day.

    Every quantity is a reading. No estimator or declared uncertainty is needed, but
    the daily-mean time alignment is approximate: even perfect gauges can leave a residual.
    The residual can include unmeasured flux, gauge error and time-aggregation error.
    The daily-mean alignment is stated
    rather than assumed away: storage here is a daily MEAN, so S_{k+1} - S_k is the change
    between two daily means, which is centred on the boundary between the days and pairs
    with the mean flow of day k only to the extent that flows vary smoothly.
    """
    Y = np.array([o.y for o in bs.observations], dtype=float)
    S, qout, qin1, qin2 = Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3]
    net_cfs = qin1 + qin2 - qout
    dS = np.diff(S)                                   # acre-ft, day k -> k+1
    inflow_volume = CFS_DAY_TO_ACRE_FT * net_cfs[:-1]  # acre-ft over day k
    r = dS - inflow_volume
    finite = np.isfinite(r)
    total_in = CFS_DAY_TO_ACRE_FT * np.nansum(qin1 + qin2)
    cum = float(np.nansum(r))
    return {
        "n_days": int(finite.sum()),
        "units": f"{VOLUME_UNIT} per day",
        "mean": float(np.nanmean(r)),
        "sd": float(np.nanstd(r, ddof=1)),
        "median": float(np.nanmedian(r)),
        "p05": float(np.nanpercentile(r, 5)),
        "p95": float(np.nanpercentile(r, 95)),
        "min": float(np.nanmin(r)),
        "max": float(np.nanmax(r)),
        "cumulative": cum,
        "cumulative_as_fraction_of_gauged_inflow": float(cum / total_in) if total_in else float("nan"),
        "gauged_inflow_volume": float(total_in),
        "mean_as_cfs": float(np.nanmean(r) / CFS_DAY_TO_ACRE_FT),
        "sd_as_cfs": float(np.nanstd(r, ddof=1) / CFS_DAY_TO_ACRE_FT),
        "mean_throughput_cfs": float(np.nanmean(qin1 + qin2)),
        "lag1_autocorr": _lag1(r[finite]),
        **_cumulative_uncertainty(cum, float(np.nanstd(r, ddof=1)), int(finite.sum()), _lag1(r[finite])),
        "series": r.tolist(),
    }


def alignment_comparison(bs: BridgedSeries) -> dict:
    """Storage here is a daily MEAN, so it is not obvious which day's mean flow its change
    should be paired with. Rather than assume, all three pairings are computed and the
    scatter of each is reported -- the one that closes best is evidence about the alignment,
    not a preference:

        same_day   dS(k -> k+1) against the mean flow of day k
        centred    dS(k -> k+1) against (flow_k + flow_{k+1}) / 2, which is what a change
                   between two daily MEANS is centred on
        next_day   dS(k -> k+1) against the mean flow of day k+1

    This settles nothing about the true alignment on its own: a smaller residual sd could
    also come from averaging two noisy flow readings, which the centred pairing does and
    the other two do not. The report says so.
    """
    Y = np.array([o.y for o in bs.observations], dtype=float)
    S, qout, qin1, qin2 = Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3]
    net = qin1 + qin2 - qout
    dS = np.diff(S)
    out = {}
    for name, flow in (("same_day", net[:-1]),
                       ("centred", 0.5 * (net[:-1] + net[1:])),
                       ("next_day", net[1:])):
        r = dS - CFS_DAY_TO_ACRE_FT * flow
        out[name] = {"mean": float(np.nanmean(r)), "sd": float(np.nanstd(r, ddof=1)),
                     "mean_as_cfs": float(np.nanmean(r) / CFS_DAY_TO_ACRE_FT),
                     "sd_as_cfs": float(np.nanstd(r, ddof=1) / CFS_DAY_TO_ACRE_FT),
                     "lag1_autocorr": _lag1(r[np.isfinite(r)])}
    best = min(out, key=lambda k: out[k]["sd"])
    out["smallest_sd"] = best
    out["note"] = ("the centred pairing also averages two flow readings, which reduces the flow's own "
                   "noise contribution by sqrt(2) whether or not the alignment is right; a smaller sd "
                   "there is therefore not by itself evidence of the alignment")
    return out


def _cumulative_uncertainty(cum: float, sd: float, n: int, lag1: float) -> dict:
    """How far the cumulative residual is from zero, in its own standard errors.

    The cumulative is a SUM of n daily residuals, so its scale grows as sqrt(n) even
    when the gauges close exactly: a large-looking total is what independent daily
    scatter produces on its own. Under independence the sum's standard error is
    sd * sqrt(n). The daily residuals here are not independent, so the sum's variance
    is also reported under an AR(1) model with the measured lag-1 rho, exactly for
    finite n rather than by the large-n limit (1 + rho) / (1 - rho):

        Var(sum) = sd^2 * (n + 2 * sum_{k=1}^{n-1} (n - k) rho^k)

    That is a model, not a measurement: AR(1) is the simplest correlation structure
    consistent with a single reported lag-1, and a longer-memory process would widen
    the interval further. Both are reported so neither is taken on faith. Nothing here
    is a test of the water balance -- it says only whether the cumulative is
    distinguishable from zero, which is the reading its magnitude invites.

    Returns None for every field when n < 2 or the inputs are not finite, rather than
    NaN, because the result dict is serialized with allow_nan=False.
    """
    none = {"cumulative_se": None, "cumulative_se_ar1": None, "cumulative_in_se": None,
            "cumulative_in_se_ar1": None, "cumulative_ci95_ar1": None}
    if n < 2 or not (np.isfinite(cum) and np.isfinite(sd)) or sd <= 0.0:
        return none
    se = sd * math.sqrt(n)
    if np.isfinite(lag1) and abs(lag1) < 1.0:
        k = np.arange(1, n, dtype=float)
        factor = float(n + 2.0 * np.sum((n - k) * lag1 ** k))
        se_ar1 = sd * math.sqrt(factor) if factor > 0.0 else None
    else:
        se_ar1 = None
    out = {"cumulative_se": float(se), "cumulative_in_se": float(cum / se)}
    if se_ar1 is None:
        out.update({"cumulative_se_ar1": None, "cumulative_in_se_ar1": None,
                    "cumulative_ci95_ar1": None})
    else:
        out.update({
            "cumulative_se_ar1": float(se_ar1),
            "cumulative_in_se_ar1": float(cum / se_ar1),
            # 1.96 is the normal quantile: a sum of ~1,000 terms, not a t interval.
            "cumulative_ci95_ar1": [float(cum - 1.96 * se_ar1), float(cum + 1.96 * se_ar1)],
        })
    return out


def _lag1(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    if v.size < 3:
        return float("nan")
    d = v - v.mean()
    den = float(d @ d)
    return float((d[:-1] @ d[1:]) / den) if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# the declared constraint
# ---------------------------------------------------------------------------

def _closure(variant: str, s0: float, s0_var: float) -> ConstraintSet:
    """One of the declaration's variants, with b resolved against the record.

    b is the day-0 storage READING, which is why the declaration cannot name a constant: it
    names the sensor and the index, and the value arrives here. Passing a variance the row did
    not ask for, or omitting one it did, raises rather than quietly substituting a constant for
    a measurement."""
    return DECLARATION.constraint_set(variant, readings={"closure": s0},
                                      variances={"closure": s0_var})


def constraint_open(s0: float, s0_var: float) -> ConstraintSet:
    """S - G = S0: the gauges close the balance."""
    return _closure("open", s0, s0_var)


def constraint_aug(s0: float, s0_var: float) -> ConstraintSet:
    """S - G - U = S0 with an ungauged state."""
    return _closure("augmented", s0, s0_var)


SPECS = (
    EstimatorSpec("wb_open", "wb_open", None, cusum=CusumConfig()),
    EstimatorSpec("wb_open+hard", "wb_open", "hard", cusum=CusumConfig()),
    EstimatorSpec("wb_open+hard+guard", "wb_open", "hard", guard=True, cusum=CusumConfig()),
    EstimatorSpec("wb_aug", "wb_aug", None, cusum=CusumConfig()),
    EstimatorSpec("wb_aug+hard", "wb_aug", "hard", cusum=CusumConfig()),
    # the only run in which U is ESTIMATED rather than reported: U is observed by nothing but
    # the constraint, so without feedback the filter's own U never leaves its prior. Labelled,
    # and read alongside the others: a constraint fed back is absorbed as if it were evidence.
    EstimatorSpec("wb_aug+hard+feedback", "wb_aug", "hard", feedback=True, cusum=CusumConfig()),
    EstimatorSpec("wb_closed", "wb_closed", None, cusum=CusumConfig()),
)


def run_one(spec: EstimatorSpec, bs: BridgedSeries, s0: float, s0_var: float) -> RunResult:
    if spec.kind == "wb_closed":
        cs = None
    elif spec.kind == "wb_aug":
        cs = constraint_aug(s0, s0_var)
    else:
        cs = constraint_open(s0, s0_var)
    return run(bs.inputs, bs.observations, cs, spec, (s0,), PRIOR_STORAGE_STD, est_cfg=CFG)


# ---------------------------------------------------------------------------
# what the constraint is structurally blind to
# ---------------------------------------------------------------------------

def blind_directions(s0: float, s0_var: float) -> dict:
    """d(f) for the open constraint, per named direction, plus the one the state space
    cannot express at all."""
    cs = constraint_open(s0, s0_var)
    P = np.diag([STORAGE_SIGMA_BASE ** 2, 100.0 ** 2])       # a representative reported covariance
    row, null = constraint_bases(cs)
    named = {
        "storage_only": np.array([1.0, 0.0]),
        "cumulative_inflow_only": np.array([0.0, 1.0]),
        "row_space (S - G)": row[0],
        "null_space (S + G)": null[0],
    }
    return {
        "A": cs.A.tolist(),
        "P_used": P.tolist(),
        "P_note": "a representative reported covariance: the declared storage sigma squared, and "
                  "(100 acre-ft)^2 on the cumulative gauged volume. d(f) scales with P, so these are "
                  "for comparing directions with each other, not an absolute sensitivity.",
        "d": {name: float(detectability(f, P, cs)) for name, f in named.items()},
        "row_space_basis": row.tolist(),
        "null_space_basis": null.tolist(),
        "structurally_invisible": {
            "direction": null[0].tolist(),
            "meaning": "storage and cumulative gauged inflow rising together by the same volume. "
                       "d(f) = 0 exactly: no covariance makes this visible to the constraint.",
        },
        "not_even_a_direction": {
            "meaning": "an equal bias on one inflow gauge and the outflow gauge. It cancels inside "
                       "(qin1 + qin2 - qout) before it reaches any state, so it never perturbs x at "
                       "all: it is invisible to the balance AND outside what d(f) can score, because "
                       "d(f) measures directions in the reported state space. The individual flow "
                       "channels may still disagree with their predictions; attribution needs "
                       "additional information beyond this balance.",
        },
    }


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def compute() -> dict:
    base = load(STORAGE_SIGMA_BASE)
    Y = np.array([o.y for o in base.observations], dtype=float)
    s0 = float(Y[0, 0])
    closure = closure_residual(base)

    sweep: dict[str, dict] = {}
    for sigma in STORAGE_SIGMA_SWEEP:
        bs = load(sigma)
        s0_var = sigma ** 2
        per_spec: dict[str, dict] = {}
        for spec in SPECS:
            r = run_one(spec, bs, s0, s0_var)
            n = len(r.x)
            ev = evaluate_truth_free(r, {"all": (0, n)})
            entry = {
                "n_steps": n,
                "flagged_steps": int(np.sum(r.flag)) if r.flag is not None else 0,
                "status_counts": {k: int(v) for k, v in
                                  sorted(__import__("collections").Counter(s.value for s in r.status).items())},
                "truth_free": ev,
            }
            if r.stat is not None and bool(np.isfinite(r.stat).any()):
                st = r.stat[np.isfinite(r.stat)]
                entry["consistency_stat"] = {
                    "mean": float(st.mean()), "median": float(np.median(st)),
                    "max": float(st.max()), "threshold": r.threshold,
                    "fraction_over_threshold": float(np.mean(st > r.threshold)) if r.threshold else None,
                }
            if r.res_pre is not None and bool(np.isfinite(r.res_pre).any()):
                rp = r.res_pre[np.isfinite(r.res_pre[:, 0]), 0]
                entry["residual_pre"] = {"mean": float(rp.mean()), "sd": float(rp.std(ddof=1)),
                                         "min": float(rp.min()), "max": float(rp.max()),
                                         "final": float(rp[-1]), "units": VOLUME_UNIT}
            if r.corr is not None and bool(np.isfinite(r.corr).any()):
                cn = np.linalg.norm(r.corr[np.isfinite(r.corr).all(axis=1)], axis=1)
                if cn.size:
                    entry["correction_norm"] = {"mean": float(cn.mean()), "max": float(cn.max()),
                                                "n_applied": int(cn.size), "units": "acre-ft"}
            if r.res_post is not None and bool(np.isfinite(r.res_post).any()):
                rq = r.res_post[np.isfinite(r.res_post[:, 0]), 0]
                entry["residual_post"] = {"mean_abs": float(np.abs(rq).mean()), "max_abs": float(np.abs(rq).max()),
                                          "units": "acre-ft"}
            if spec.kind == "wb_aug":
                U = r.x[:, 2]      # the REPORTED U: post-projection where a mode was applied
                final_sd = float(np.sqrt(r.P[-1, 2, 2]))
                entry["ungauged_cumulative"] = {
                    "final": float(U[-1]), "min": float(U.min()), "max": float(U.max()),
                    "final_sd": final_sd,
                    # The filter's own sd on its own final U: how far that U is from zero on
                    # the filter's own terms. Reported so the magnitude is never read alone.
                    "final_in_sd": float(U[-1] / final_sd) if final_sd > 0.0 else None,
                    "final_as_fraction_of_gauged_inflow":
                        float(U[-1] / closure["gauged_inflow_volume"]),
                    "units": "acre-ft",
                }
            per_spec[spec.name] = entry
        sweep[f"{sigma:g}"] = {"storage_sigma": sigma, "specs": per_spec}

    out = {
        "site": {
            "reservoir": "Ridgway Reservoir, Uncompahgre River, Colorado",
            "series": {sen.role: {"monitoring_location_id": dict(sen.match)["monitoring_location_id"],
                                  "parameter_code": dict(sen.match)["parameter_code"],
                                  "unit": sen.unit, "statistic_id": dict(sen.match)["statistic_id"],
                                  "source_id": sen.source_id} for sen in DECLARATION.sensors},
            "gauged_sq_mi": GAUGED_SQ_MI, "total_sq_mi": TOTAL_SQ_MI,
            "gauged_fraction": GAUGED_SQ_MI / TOTAL_SQ_MI,
        },
        "declared": {
            "flow_sigma_relative": FLOW_SIGMA_RELATIVE,
            "flow_sigma_floor": FLOW_SIGMA_FLOOR,
            "flow_sigma_citation": FLOW_SIGMA_CITATION,
            "storage_sigma_sweep": list(STORAGE_SIGMA_SWEEP),
            "storage_sigma_base": STORAGE_SIGMA_BASE,
            "storage_sigma_citation": STORAGE_SIGMA_CITATION,
            "day_zone_citation": DAY_ZONE_CITATION,
            "config": {k: getattr(CFG, k) for k in
                       ("q_storage", "q_flow", "q_ungauged", "cumulative0_std", "flow0", "flow0_std")},
            "prior_storage_std": PRIOR_STORAGE_STD,
            "cfs_day_to_acre_ft": CFS_DAY_TO_ACRE_FT,
        },
        "record": {
            "n_days": base.provenance["n_grid"],
            "first_day": base.provenance["epoch_iso"],
            "storage_acre_ft": {"first": s0, "min": float(np.nanmin(Y[:, 0])), "max": float(np.nanmax(Y[:, 0]))},
            "outflow_cfs": {"min": float(np.nanmin(Y[:, 1])), "max": float(np.nanmax(Y[:, 1]))},
            "inflow_cfs": {"min": float(np.nanmin(Y[:, 2] + Y[:, 3])), "max": float(np.nanmax(Y[:, 2] + Y[:, 3]))},
            "n_missing": base.provenance["n_missing"],
        },
        "closure_free": {k: v for k, v in closure.items() if k != "series"},
        "alignment": alignment_comparison(base),
        "closure_series": closure["series"],
        "blind": blind_directions(s0, STORAGE_SIGMA_BASE ** 2),
        "sweep": sweep,
        "provenance": {"bridge": base.provenance, "daf_commit": base.provenance["daf_commit"]},
    }
    out["claims"] = claims(out)
    return out


def claims(r: dict) -> dict:
    """Every qualitative sentence the report prints, as a computed condition.

    If one fails the report is not written, because the text would no longer be true of the
    numbers -- the guard real_noaa.claims() applies to the day report. The sentences that
    need it here are the ones about DISTANCE FROM ZERO: the report says the three-year
    cumulative is indistinguishable from zero and that the two routes agreeing near it is not
    evidence of an imbalance. Both are properties of this record, not of the arithmetic, and a
    record where they failed would make the text false rather than merely dull.
    """
    closure = r["closure_free"]
    interval = closure["cumulative_ci95_ar1"]
    augmented = [entry["ungauged_cumulative"] for cell in r["sweep"].values()
                 for entry in cell["specs"].values() if entry.get("ungauged_cumulative")]
    return {
        "the_cumulative_interval_contains_zero": (
            interval is not None and interval[0] < 0.0 < interval[1]),
        "the_cumulative_is_under_two_standard_errors_from_zero": (
            closure["cumulative_in_se_ar1"] is not None
            and abs(closure["cumulative_in_se_ar1"]) < 1.96),
        "a_positive_lag1_widens_the_standard_error": (
            closure["lag1_autocorr"] > 0.0 and closure["cumulative_se_ar1"] > closure["cumulative_se"] > 0.0),
        "the_daily_scatter_dwarfs_the_daily_mean": closure["sd"] > 10.0 * abs(closure["mean"]),
        "every_augmented_run_reports_its_own_sd_ratio": all(
            a["final_in_sd"] is not None for a in augmented),
        "feedback_reports_the_largest_sd_ratio": all(
            max(cell["specs"], key=lambda name: (
                cell["specs"][name]["ungauged_cumulative"]["final_in_sd"]
                if cell["specs"][name].get("ungauged_cumulative") else float("-inf"))
            ) == "wb_aug+hard+feedback" for cell in r["sweep"].values()),
        "a_wider_declared_sigma_never_rejects_more": (
            lambda rates: rates == sorted(rates, reverse=True)
        )([r["sweep"][f"{sigma:g}"]["specs"]["wb_open"]["consistency_stat"]["fraction_over_threshold"]
           for sigma in sorted(STORAGE_SIGMA_SWEEP)]),
    }


def main(out_dir: Path, *, quiet: bool = False) -> int:
    from .provenance import header_line, provenance

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    prov = provenance()
    result = compute()
    result["provenance"] = {"generation": prov, **result["provenance"]}
    text = render(result).replace("\n\n", "\n\n" + header_line(prov) + "\n\n", 1)
    (out_dir / "real_water_balance.md").write_text(text, encoding="utf-8")
    (out_dir / "real_water_balance.json").write_text(json.dumps(result, indent=2, allow_nan=False),
                                                     encoding="utf-8")
    if not quiet:
        print(text)
    return 0


def render(r: dict) -> str:
    from .real_water_balance_report import render_report
    return render_report(r)


if __name__ == "__main__":
    sys.exit(main(REPO_ROOT / "results"))
