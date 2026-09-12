"""Evaluation with no truth: what a run's own record says about the run.

A real record has no hidden truth, so nothing that scores an estimate against
one can be computed on it. evaluate_truth_free(run, windows) reads NOTHING but
the RunResult -- it takes no truth argument, and this module imports nothing
that could supply one -- and reports, per window:

  per sensor, on the SAMPLING clock (sample j is the observation taken at step j,
  whenever it arrived):
    n_samples            steps in the window
    n_observed           samples present and ingested within the run (RunResult.observed)
    frac_missing         1 - n_observed / n_samples (missing from the record, or never arrived)
    n_z                  samples with a finite normalised innovation z = nu / sqrt(S_ii)
                         (fewer than n_observed for an estimator that has no prediction)
    z_mean, z_rms        mean and RMS of z. Under the filter's own model z is N(0, 1)
                         and white: z_mean ~ 0 and z_rms ~ 1. z_rms < 1 means the filter
                         states more innovation variance than it sees; > 1, less.
    z_lag1               lag-1 autocorrelation of z over adjacent sampling steps where
                         both are finite (gaps are not closed up), normalised by the
                         variance over all finite samples; ~ 0 for a white sequence
    frac_abs_z_gt_1.96   fraction of finite z with |z| > 1.96 (0.05 under the model)
  per sensor, on the REPORT clock (where the grid stamps the evidence-side channel):
    cusum_alarms         CUSUM alarms raised at report steps in the window
    cusum_first_alarm    the report step of the first of them, or None
                         (both None where the channel is off or has no innovation)
  when a constraint was declared (RunResult.threshold is not None), on the report clock:
    flag_steps, first_flag_step    the debounced consistency flag
    stat_mean                      mean consistency statistic (chi2(rank A) mean under
                                   the joint hypothesis: rank A)
    frac_stat_above_threshold      per-step exceedance before debounce
    status_counts                  reconciliation statuses reported in the window
  for an estimator with extra-state flags (RunResult.extra), on the report clock:
    extra_flags[name]              flag_steps and first_flag_step for each flag_<name>
  n_evidence_ids                   evidence ids ingested at report steps in the window

and, over the whole run, latency p50 / p99 in microseconds (this machine).

Windows are (lo, hi) step ranges, half-open, 0 <= lo < hi <= N. A sampling-clock
window and a report-clock window with the same bounds cover the same steps of
the clock; under arrival delay a sample taken in the window may be ingested --
and so alarm -- after it.

What these numbers cannot say: whether the estimate is right. A filter can be
confidently wrong with white, unit-variance innovations if the evidence itself
is wrong in a way the model explains (an undeclared sensor bias absorbed into
the state), and a filter whose stated uncertainty is far too large can read
z_rms ~ 1 when the measurement noise dominates the innovation variance. They are
the checks that remain when no truth exists, not a substitute for it.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .runner import RunResult

Z_TWO_SIDED_95 = 1.96


def evaluate_truth_free(run: RunResult, windows: dict[str, tuple[int, int]]) -> dict:
    n = len(run.x)
    n_sensors = int(run.innov_z.shape[1])
    wins = _check_windows(windows, n)
    cusum_on = bool(run.spec.cusum is not None and np.isfinite(run.innov_z).any())
    constraint = run.threshold is not None
    flag_keys = sorted(k for k in (run.extra or {}) if k.startswith("flag_"))

    out: dict = {
        "n_steps": n,
        "n_sensors": n_sensors,
        "n_report": int(run.x.shape[1]),
        "cusum_applicable": cusum_on,
        "constraint_declared": constraint,
        "threshold": run.threshold,
        "latency_us_p50": float(np.percentile(run.latency_s, 50) * 1e6),
        "latency_us_p99": float(np.percentile(run.latency_s, 99) * 1e6),
        "windows": {},
    }
    for name, (lo, hi) in wins.items():
        w: dict = {"range": [lo, hi], "sensors": []}
        for i in range(n_sensors):
            z = run.innov_z[lo:hi, i]
            ok = np.isfinite(z)
            zf = z[ok]
            n_obs = int(run.observed[lo:hi, i].sum())
            rec = {
                "n_samples": hi - lo,
                "n_observed": n_obs,
                "frac_missing": 1.0 - n_obs / (hi - lo),
                "n_z": int(ok.sum()),
                "z_mean": float(np.nanmean(z)) if zf.size else None,   # the arithmetic of evaluate()'s z̄
                "z_rms": float(np.sqrt(np.mean(zf ** 2))) if zf.size else None,
                "z_lag1": _lag1(z),
                "frac_abs_z_gt_1.96": float(np.mean(np.abs(zf) > Z_TWO_SIDED_95)) if zf.size else None,
                "cusum_alarms": None,
                "cusum_first_alarm": None,
            }
            if cusum_on:
                a = np.flatnonzero(run.cusum_alarm[lo:hi, i])
                rec["cusum_alarms"] = int(a.size)
                rec["cusum_first_alarm"] = int(lo + a[0]) if a.size else None
            w["sensors"].append(rec)
        if constraint:
            f = np.flatnonzero(run.flag[lo:hi])
            st = run.stat[lo:hi]
            st = st[np.isfinite(st)]
            w["constraint"] = {
                "flag_steps": int(f.size),
                "first_flag_step": int(lo + f[0]) if f.size else None,
                "stat_mean": float(st.mean()) if st.size else None,
                "frac_stat_above_threshold": float(np.mean(st > run.threshold)) if st.size else None,
                "status_counts": dict(Counter(s.value for s in run.status[lo:hi])),
            }
        else:
            w["constraint"] = None
        if flag_keys:
            w["extra_flags"] = {}
            for key in flag_keys:
                f = np.flatnonzero(run.extra[key][lo:hi])
                w["extra_flags"][key[len("flag_"):]] = {
                    "flag_steps": int(f.size), "first_flag_step": int(lo + f[0]) if f.size else None}
        else:
            w["extra_flags"] = None
        w["n_evidence_ids"] = int(sum(len(ids) for ids in run.ingested_evidence[lo:hi]))
        out["windows"][name] = w
    return out


def _check_windows(windows: dict[str, tuple[int, int]], n: int) -> dict[str, tuple[int, int]]:
    out = {}
    for name, (lo, hi) in windows.items():
        lo, hi = int(lo), int(hi)
        if not 0 <= lo < hi <= n:
            raise ValueError(f"window {name!r} = ({lo}, {hi}) is not a non-empty range inside [0, {n}]")
        out[name] = (lo, hi)
    return out


def _lag1(z: np.ndarray) -> float | None:
    """Lag-1 autocorrelation over adjacent steps where both samples are finite; None
    with fewer than two such pairs or no variance."""
    ok = np.isfinite(z)
    if ok.sum() < 3:
        return None
    d = np.where(ok, z - z[ok].mean(), np.nan)
    var = float(np.mean(d[ok] ** 2))
    pair = ok[:-1] & ok[1:]
    if pair.sum() < 2 or var == 0.0:
        return None
    return float(np.mean(d[:-1][pair] * d[1:][pair]) / var)
