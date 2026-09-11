"""Runs one (scenario, estimator spec) pair step by step and records everything
the evaluator needs.

What the estimator side is allowed to see: the observations (only once each
has arrived, per Observation.arrival_t), the commanded input, a DECLARED prior
for the initial state, and the declared constraint set. Truth.m is never read
here; Truth is passed only for its public fields (t, u_commanded).
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ..lcm import chi2_quantile, consistency_stat, is_feasible, reconcile
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
    res_pre: np.ndarray      # (N, rows)
    res_post: np.ndarray     # (N, rows)  NaN where no projection applied
    corr: np.ndarray         # (N, 2)     NaN where no projection applied
    latency_s: np.ndarray    # (N,)


def run(
    truth: Truth,
    obs: list[Observation],
    cs: ConstraintSet | None,
    spec: EstimatorSpec,
    declared_m0: tuple[float, float],
    declared_m0_std: float = 5.0,
    kf_cfg: KFConfig = KFConfig(),
) -> RunResult:
    n = len(obs)
    dt = float(truth.t[1] - truth.t[0])
    est = ESTIMATORS[spec.kind](declared_m0, declared_m0_std, dt, truth.u_commanded, kf_cfg)

    rows = cs.dof if cs is not None else 1
    feasible = cs is not None and is_feasible(cs)
    thr = chi2_quantile(cs.rank, spec.threshold_q) if cs is not None else None

    x = np.zeros((n, 2)); P = np.zeros((n, 2, 2))
    xu = np.zeros((n, 2)); Pu = np.zeros((n, 2, 2))
    status: list[Status] = []
    stat = np.full(n, np.nan)
    flag = np.zeros(n, dtype=bool)
    res_pre = np.full((n, rows), np.nan)
    res_post = np.full((n, rows), np.nan)
    corr = np.full((n, 2), np.nan)
    lat = np.zeros(n)

    streak = 0
    next_obs = 0   # index of the first observation that has not yet arrived
    for k in range(n):
        t0 = perf_counter()
        tk = float(truth.t[k])
        while next_obs < n and obs[next_obs].arrival_t <= tk:
            est.ingest(obs[next_obs], next_obs)
            next_obs += 1
        xr, Pr = est.report(k)

        s = consistency_stat(xr, Pr, cs) if feasible else np.nan
        exceed = feasible and s > thr
        streak = streak + 1 if exceed else 0
        f = streak >= spec.debounce
        hold = spec.guard and f

        se = reconcile(
            xr, Pr, cs, mode=spec.mode, lam=spec.lam, hold=hold, threshold=thr,
            stat=s if feasible else None, t=tk, model_version=est.model_version,
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
