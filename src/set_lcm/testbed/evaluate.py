"""Evaluation against hidden truth. This is the only module that reads Truth.m.

Per run it reports: reconstruction error (overall, per window, and decomposed
along row(A) / null(A)); calibration as 95 % interval coverage and as the RMS
normalised error e_i / sigma_i (target 1.0), both per window; constraint
residuals pre/post; correction magnitude; the detectability of the scenario's
declared fault direction; false alarms and (censorable) detection delay; solver
failures; latency.

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
    flags = run.flag
    if fault_onset is None:
        out["false_alarms"] = int(flags.sum())
        out["fa_rate"] = float(flags.mean())
        out["detection_delay_steps"] = None
        out["detection_censor_steps"] = None
    else:
        pre = flags[:fault_onset]
        out["false_alarms"] = int(pre.sum())
        out["fa_rate"] = float(pre.mean()) if pre.size else 0.0
        later = np.flatnonzero(flags[fault_onset:])
        out["detection_delay_steps"] = int(later[0]) if later.size else None
        out["detection_censor_steps"] = int(n - fault_onset)   # a never-flagged seed is censored here
    return out


def _nanmean_abs(a: np.ndarray) -> float | None:
    return None if np.all(np.isnan(a)) else float(np.nanmean(np.abs(a)))


def _nanmean_norm(a: np.ndarray) -> float | None:
    if np.all(np.isnan(a)):
        return None
    ok = ~np.isnan(a).any(axis=1)
    return float(np.mean(np.linalg.norm(a[ok], axis=1)))
