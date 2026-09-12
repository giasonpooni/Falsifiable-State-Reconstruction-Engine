"""Evaluation against hidden truth: the module that scores a run. Besides it, the
only code that reads Truth.m, Truth.leak or Truth.u_actual is the observation
operator (degrade.observe measures Truth.m) and the experiment code that routes
the oracle bound's hidden inputs (experiments.phase1.oracle_inputs_for); nothing
on the estimator side does, and runner.run() cannot receive a Truth. For a record
that has no truth, truth_free.evaluate_truth_free reads the RunResult alone.

Per run it reports: reconstruction error (overall, per window, and decomposed
along row(A) / null(A)); calibration as 95 % interval coverage and as the RMS
normalised error e_i / sigma_i (target 1.0), both per window; constraint
residuals pre/post; correction magnitude; the detectability of the scenario's
declared fault direction; false alarms and (censorable) detection delay for
the constraint flag and, per sensor, for the innovation CUSUM; solver
failures; latency. For an augmented estimator (RunResult.extra) it adds the
error of the pump scale alpha against u_actual / u_commanded over the steps
where the pump is commanded on, the error of the boundary flux L against the
hidden leak, and the same false-alarm / censored-delay bookkeeping for the
alpha and L flags.

NEES (e^T P^-1 e) is deliberately absent: after a hard projection P is
rank-deficient along the constraint and a pseudo-inverse NEES silently drops
that direction, which is exactly the one that matters.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from ..lcm import constraint_bases, detectability
from ..schema import ConstraintSet, Status
from .runner import RunResult
from .simulator import Truth


def _rms(a: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=float)
    if a.size == 0 or np.all(np.isnan(a)):
        return None
    return float(np.sqrt(np.nanmean(a ** 2)))


def _by_window(a: np.ndarray, windows: dict[str, tuple[int, int]]) -> dict[str, float | None]:
    return {name: _rms(a[lo:hi]) for name, (lo, hi) in windows.items()}


def _normalised(err: np.ndarray, P: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.stack([np.diag(p) for p in P]))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, err / sd, np.nan)


def evaluate(
    run: RunResult,
    truth: Truth,
    fault_onset: int | None,
    windows: dict[str, tuple[int, int]],
    cs: ConstraintSet | None = None,
    fault_direction: tuple[float, ...] | None = None,
) -> dict:
    n = len(run.x)
    err = run.x - truth.m
    err_u = run.x_unproj - truth.m
    nz = _normalised(err, run.P)
    nz_u = _normalised(err_u, run.P_unproj)
    covered = np.abs(err) <= 1.96 * np.sqrt(np.stack([np.diag(p) for p in run.P]))

    out: dict = {
        "rmse_all": _rms(err),
        "rmse_by_window": _by_window(err, windows),
        "coverage95": float(covered.mean()),
        "coverage95_by_window": {name: float(covered[lo:hi].mean()) for name, (lo, hi) in windows.items()},
        "nz_rms_all": _rms(nz),
        "nz_rms_by_window": _by_window(nz, windows),
        "nz_rms_unproj_by_window": _by_window(nz_u, windows),
        "mean_abs_res_pre": _nanmean_abs(run.res_pre),
        "mean_abs_res_post": _nanmean_abs(run.res_post),
        "mean_correction_norm": _nanmean_norm(run.corr),
        "status_counts": dict(Counter(s.value for s in run.status)),
        "solver_failures": int(sum(s in (Status.INFEASIBLE, Status.NOT_CONVERGED) for s in run.status)),
        "latency_us_p50": float(np.percentile(run.latency_s, 50) * 1e6),
        "latency_us_p99": float(np.percentile(run.latency_s, 99) * 1e6),
    }

    if cs is not None:
        v_row, v_null = constraint_bases(cs)
        out["rms_err_by_direction"] = {
            name: {"row": _rms(err[lo:hi] @ v_row.T), "null": _rms(err[lo:hi] @ v_null.T)}
            for name, (lo, hi) in windows.items()
        }
        out["rms_err_unproj_by_direction"] = {
            name: {"row": _rms(err_u[lo:hi] @ v_row.T), "null": _rms(err_u[lo:hi] @ v_null.T)}
            for name, (lo, hi) in windows.items()
        }
        if fault_direction is not None and fault_onset is not None:
            # evaluate d(f) on the unprojected P at the last step of the first window
            # that starts at or after the onset (the last window if none does)
            after = [(name, lo, hi) for name, (lo, hi) in windows.items() if lo >= fault_onset]
            if after:
                name, lo, hi = min(after, key=lambda w: w[1])
            else:
                name, (lo, hi) = list(windows.items())[-1]
            step = hi - 1
            out["detectability"] = {
                "value": detectability(fault_direction, run.P_unproj[step], cs),
                "window": name, "step": int(step), "direction": list(map(float, fault_direction)),
            }

    # fault_onset: first step at which the joint hypothesis "constraint AND model AND
    # calibrated uncertainty" stops being true. Flags before it are false alarms;
    # flags after it are detections. None means it holds for the whole run.
    fa, rate, delay = _flag_bookkeeping(run.flag, fault_onset)
    out["false_alarms"] = fa
    out["fa_rate"] = rate
    out["detection_delay_steps"] = delay
    out["detection_censor_steps"] = None if fault_onset is None else int(n - fault_onset)   # never-flagged seeds are censored here

    # Evidence-side channel: the same bookkeeping per sensor for the innovation CUSUM.
    # It shares detection_censor_steps. Not applicable when the estimator records no
    # innovation (hold-last) or the channel is switched off. The mean normalised
    # innovation per window (over the window's SAMPLING steps) is what the CUSUM has to
    # work with: a shift below k cannot accumulate.
    alarms = run.cusum_alarm
    out["cusum_applicable"] = bool(run.spec.cusum is not None and np.isfinite(run.innov_z).any())
    out["innov_z_mean_by_window"] = {name: _nanmean_cols(run.innov_z[lo:hi]) for name, (lo, hi) in windows.items()}
    out["cusum_false_alarms"] = []
    out["cusum_fa_rate"] = []
    out["cusum_detection_delay_steps"] = []
    for i in range(alarms.shape[1]):
        fa, rate, delay = _flag_bookkeeping(alarms[:, i], fault_onset)
        out["cusum_false_alarms"].append(fa)
        out["cusum_fa_rate"].append(rate)
        out["cusum_detection_delay_steps"].append(delay)

    # Augmented-state outputs: alpha against the hidden parameter ratio over the steps
    # where the pump is commanded on (alpha is unobservable elsewhere), L against the
    # hidden leak everywhere, and the flags with the same onset rule as above. The flag
    # names come from the record, so a flag that fires in a scenario whose fault is
    # something else (a sensor bias the model can only read as a pump-scale change) is
    # counted as a detection of the broken conjunction, exactly as the constraint flag
    # is; the table says which flag fired.
    out["aug"] = _evaluate_aug(run, truth, fault_onset, windows) if run.extra is not None else None
    return out


def _flag_bookkeeping(flags: np.ndarray, fault_onset: int | None) -> tuple[int, float, int | None]:
    """(false alarms, false-alarm rate per step, delay of the first flag after onset or None).
    Without an onset every flag is a false alarm and there is no delay."""
    flags = np.asarray(flags, dtype=bool)
    if fault_onset is None:
        return int(flags.sum()), float(flags.mean()), None
    pre = flags[:fault_onset]
    later = np.flatnonzero(flags[fault_onset:])
    return int(pre.sum()), float(pre.mean()) if pre.size else 0.0, int(later[0]) if later.size else None


def _evaluate_aug(run: RunResult, truth: Truth, fault_onset: int | None, windows: dict[str, tuple[int, int]]) -> dict:
    ex = run.extra
    u = np.asarray(truth.u_commanded, dtype=float)
    on = u != 0.0
    alpha_true = np.full(u.shape, np.nan)
    alpha_true[on] = np.asarray(truth.u_actual, dtype=float)[on] / u[on]
    e_alpha = np.where(on, ex["alpha_hat"] - alpha_true, np.nan)     # NaN where the pump is off
    e_L = ex["L_hat"] - np.asarray(truth.leak, dtype=float)
    out = {
        "alpha_rmse_pump_on": _rms(e_alpha),
        "alpha_rmse_pump_on_by_window": _by_window(e_alpha, windows),
        "L_rmse": _rms(e_L),
        "L_rmse_by_window": _by_window(e_L, windows),
        "alpha_hat_end": float(ex["alpha_hat"][-1]), "alpha_sd_end": float(ex["alpha_sd"][-1]),
        "L_hat_end": float(ex["L_hat"][-1]), "L_sd_end": float(ex["L_sd"][-1]),
        "flags": {},
    }
    for key in ex:
        if key.startswith("flag_"):
            fa, rate, delay = _flag_bookkeeping(ex[key], fault_onset)
            out["flags"][key[len("flag_"):]] = {"false_alarms": fa, "fa_rate": rate, "detection_delay_steps": delay}
    return out


def _nanmean_abs(a: np.ndarray) -> float | None:
    return None if np.all(np.isnan(a)) else float(np.nanmean(np.abs(a)))


def _nanmean_cols(a: np.ndarray) -> list[float | None]:
    """Column means ignoring NaN; None for an all-NaN column (sensor dark or no innovation)."""
    return [None if np.all(np.isnan(col)) else float(np.nanmean(col)) for col in np.asarray(a, dtype=float).T]


def _nanmean_norm(a: np.ndarray) -> float | None:
    if np.all(np.isnan(a)):
        return None
    ok = ~np.isnan(a).any(axis=1)
    return float(np.mean(np.linalg.norm(a[ok], axis=1)))
