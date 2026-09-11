"""Runs one (scenario, estimator spec) pair step by step and records everything
the evaluator needs. The estimator never sees Truth.m; it sees observations,
the commanded input, the declared initial fill, and the declared constraint.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ..lcm import chi2_quantile, consistency_stat, reconcile
from ..schema import ConstraintSet, Observation, Status
from .estimators import ESTIMATORS, KFConfig
from .simulator import Truth


@dataclass(frozen=True)
class EstimatorSpec:
    name: str
    kind: str                 # key into ESTIMATORS
    mode: str | None          # None | "soft" | "hard"
    lam: float = 1.0          # soft only; 1/lam is the pseudo-measurement variance
    guard: bool = False       # if True, hold projection while the consistency stat is exceeded
    threshold_q: float = 0.999
    debounce: int = 3         # consecutive exceedances before the guard acts


@dataclass
class RunResult:
    spec: EstimatorSpec
    x: np.ndarray            # (N, 2) reported
    P: np.ndarray            # (N, 2, 2)
    x_unproj: np.ndarray     # (N, 2)
    P_unproj: np.ndarray
    status: list[Status]
    stat: np.ndarray         # (N,) consistency stat vs declared constraint (NaN if none)
    threshold: float | None
    flag: np.ndarray         # (N,) bool: debounced exceedance (computed for every spec)
    res_pre: np.ndarray      # (N, m)
    res_post: np.ndarray     # (N, m)  NaN where no projection applied
    corr: np.ndarray         # (N, 2)  NaN where no projection applied
    latency_s: np.ndarray    # (N,)


def run(
    truth: Truth,
    obs: list[Observation],
    cs: ConstraintSet | None,
    spec: EstimatorSpec,
    nominal_m0: tuple[float, float],
    delay_steps: int,
    kf_cfg: KFConfig = KFConfig(),
) -> RunResult:
    n = len(obs)
    dt = float(truth.t[1] - truth.t[0])
    est = ESTIMATORS[spec.kind](nominal_m0, dt, truth.u_commanded, kf_cfg)

    m = cs.dof if cs is not None else 1
    thr = chi2_quantile(cs.dof, spec.threshold_q) if cs is not None else None

    x = np.zeros((n, 2)); P = np.zeros((n, 2, 2))
    xu = np.zeros((n, 2)); Pu = np.zeros((n, 2, 2))
    status: list[Status] = []
    stat = np.full(n, np.nan)
    flag = np.zeros(n, dtype=bool)
    res_pre = np.full((n, m), np.nan)
    res_post = np.full((n, m), np.nan)
    corr = np.full((n, 2), np.nan)
    lat = np.zeros(n)

    streak = 0
    for k in range(n):
        t0 = perf_counter()
        est.ingest(obs[k], k)
        xr, Pr = est.report(k, delay_steps)

        s = consistency_stat(xr, Pr, cs) if cs is not None else np.nan
        exceed = cs is not None and s > thr
        streak = streak + 1 if exceed else 0
        f = streak >= spec.debounce
        hold = spec.guard and f

        se = reconcile(
            xr, Pr, cs, mode=spec.mode, lam=spec.lam, hold=hold, threshold=thr,
            stat=s if cs is not None else None, t=float(truth.t[k]),
            model_version=est.model_version,
        )
        lat[k] = perf_counter() - t0

        x[k], P[k] = se.x, se.P
        xu[k], Pu[k] = se.x_unprojected, se.P_unprojected
        status.append(se.status)
        stat[k] = s
        flag[k] = f
        if se.residual_pre is not None:
            res_pre[k] = se.residual_pre
        if se.residual_post is not None:
            res_post[k] = se.residual_post
        if se.correction is not None:
            corr[k] = se.correction

    return RunResult(spec, x, P, xu, Pu, status, stat, thr, flag, res_pre, res_post, corr, lat)
