"""Evaluation against hidden truth. This is the only module that reads Truth.m.

Reports the dimensions both reviews asked for: reconstruction error, constraint
residuals (pre and post), uncertainty calibration, fault detection (false alarms
and detection delay), correction magnitude, solver failures, and latency.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from ..schema import Status
from .runner import RunResult
from .simulator import Truth


def _rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err ** 2)))


def evaluate(
    run: RunResult,
    truth: Truth,
    fault_onset: int | None,
    windows: dict[str, tuple[int, int]],
) -> dict:
    err = run.x - truth.m
    sd = np.sqrt(np.stack([np.diag(p) for p in run.P]))
    covered = np.abs(err) <= 1.96 * sd

    out: dict = {
        "rmse_all": _rmse(err),
        "rmse_by_window": {name: _rmse(err[a:b]) for name, (a, b) in windows.items()},
        "coverage95": float(covered.mean()),
        "coverage95_by_window": {name: float(covered[a:b].mean()) for name, (a, b) in windows.items()},
        "mean_abs_res_pre": _nanmean_abs(run.res_pre),
        "mean_abs_res_post": _nanmean_abs(run.res_post),
        "mean_correction_norm": _nanmean_norm(run.corr),
        "status_counts": dict(Counter(s.value for s in run.status)),
        "solver_failures": int(sum(s in (Status.INFEASIBLE, Status.NOT_CONVERGED) for s in run.status)),
        "latency_us_p50": float(np.percentile(run.latency_s, 50) * 1e6),
        "latency_us_p99": float(np.percentile(run.latency_s, 99) * 1e6),
    }

    # fault_onset: first step at which the joint hypothesis "constraint AND model AND
    # calibrated uncertainty" stops being true. Flags before it are false alarms;
    # flags after it are detections. None means it holds for the whole run.
    flags = run.flag
    if fault_onset is None:
        out["false_alarms"] = int(flags.sum())
        out["fa_rate"] = float(flags.mean())
        out["detection_delay_steps"] = None
    else:
        pre = flags[:fault_onset]
        out["false_alarms"] = int(pre.sum())
        out["fa_rate"] = float(pre.mean()) if pre.size else 0.0
        after = np.flatnonzero(flags[fault_onset:])
        out["detection_delay_steps"] = int(after[0]) if after.size else None
    return out


def _nanmean_abs(a: np.ndarray) -> float | None:
    return None if np.all(np.isnan(a)) else float(np.nanmean(np.abs(a)))


def _nanmean_norm(a: np.ndarray) -> float | None:
    if np.all(np.isnan(a)):
        return None
    ok = ~np.isnan(a).any(axis=1)
    return float(np.mean(np.linalg.norm(a[ok], axis=1)))
