"""Runs one (scenario, estimator spec) pair step by step and records everything
the evaluator needs.

What the estimator side is allowed to see: the observations (only once each
has arrived, per Observation.arrival_t), the commanded input, a DECLARED prior
for the initial state, and the declared constraint set. Truth.m is never read
here; Truth is passed only for its public fields (t, u_commanded).

THE ONE EXCEPTION, deliberately visible: for spec.kind == "oracle" -- and for
no other kind -- run() hands the estimator constructor the hidden per-step
arrays (truth.u_actual, truth.leak) through the keyword `oracle_inputs`. The
oracle is a BOUND on what a perfect model of the inputs could do, labelled
"oracle (bound)" wherever it is tabulated, and never a candidate. Truth.m is
not read even there. This is the only place hidden truth crosses to the
estimator side; a test corrupts both arrays and checks that every other
estimator's record is bit-identical.

Feedback (EstimatorSpec.feedback): after a projection has been applied at
report step k (status ok), the projected mass marginal (x*, P*) is pushed
back into the estimator with est.set_state(). When the estimator is current
(its last ingested sampling step j equals k) that is literally the reported
(x*, P*); under arrival delay (j < k) the report is a prediction from the
state at j, so the projection of that state -- same mode, same constraint --
is what goes back, which keeps the feedback on the estimator's own clock. P*
is rank-deficient along the constraint; the next predict adds Q, so the
reported covariance the kernel sees at k + 1 is SPD again (asserted in
tests). While the guard holds (model_inconsistent) nothing is fed back.
RunResult.x_unproj for a fed-back estimator is what that estimator reported,
which already carries every earlier projection; nothing in the record is
overwritten. A consequence the results make measurable: a fed-back filter
carries A P A^T = A Q A^T after each predict and its residual shrinks with it,
so the consistency statistic on its own marginal no longer rejects a stale
constraint -- the filter has been told the constraint every step and no
longer disagrees with it.

Three detection channels run side by side and never talk to each other: the
constraint-side consistency flag (debounced chi-square exceedance, on the
report clock), the evidence-side per-sensor CUSUM on the estimator's
normalised innovation (updated on the ingest clock, alarms stamped at the
report step of ingestion), and, for an estimator that carries extra state
(kf_aug: pump scale alpha, boundary flux L), one debounced flag per extra
component on the report clock: |x_i - nominal_i| / sd_i > PARAM_FLAG_Z for
PARAM_FLAG_DEBOUNCE consecutive reports. The last are estimator outputs, not
reconciliation statuses; Status is unchanged. The reconciliation stage acts on
the mass marginal (x[:2], P[:2, :2]) only and reads none of the channels except
the first, through the guard.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ..lcm import chi2_quantile, consistency_stat, is_feasible, project_hard, project_soft, reconcile
from ..schema import ConstraintSet, Observation, Status
from .cusum import Cusum, CusumConfig
from .estimators import ESTIMATORS, AugConfig, ClosedQConfig, KFConfig, OracleConfig
from .simulator import Truth

# Flags on an augmented estimator's extra components: two-sided z test at the 0.001
# level (3.29 = Phi^-1(0.9995)) against the component's own reported sd, debounced
# like the constraint guard's default.
PARAM_FLAG_Z = 3.29
PARAM_FLAG_DEBOUNCE = 3


@dataclass(frozen=True)
class EstimatorSpec:
    name: str
    kind: str                 # key into ESTIMATORS
    mode: str | None          # None | "soft" | "hard"
    lam: float = 1.0          # soft only; 1/lam is the pseudo-measurement variance
    guard: bool = False       # if True, hold projection while the consistency stat is exceeded
    threshold_q: float = 0.999
    debounce: int = 3         # consecutive exceedances before the guard acts
    # Evidence-side channel: per-sensor two-sided CUSUM on the normalised innovation,
    # run on the ingest clock. None switches it off (cusum_stat NaN, no alarms).
    cusum: CusumConfig | None = CusumConfig()
    # Push the projected (x*, P*) back into the estimator after each applied projection
    # (see the module docstring). Needs a projection mode; the estimator must implement
    # set_state (kf and kf_closedq do; kf_aug and hold_last raise NotImplementedError).
    feedback: bool = False

    def __post_init__(self):
        if self.feedback and self.mode is None:
            raise ValueError(f"spec {self.name!r}: feedback requires a projection mode")


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
    # Innovation record, indexed by SAMPLING step j (NaN where the sensor was missing,
    # where the estimator has no prediction, or where the observation never arrived).
    innov: np.ndarray        # (N, 2)  nu = y - H x_pred
    innov_var: np.ndarray    # (N, 2)  S_ii
    innov_z: np.ndarray      # (N, 2)  nu / sqrt(S_ii)
    # CUSUM on z, indexed by REPORT step k: the statistic after the last observation
    # ingested at step k (before the reset an alarm applies; NaN if the channel is off),
    # and whether an alarm was raised while ingesting at step k. Because the alarm is
    # stamped at the report step, its delay includes the observation's arrival delay.
    cusum_stat: np.ndarray   # (N, 2)
    cusum_alarm: np.ndarray  # (N, 2) bool
    # Augmented estimators only (else None): for each name in the estimator's aug_names,
    # "<name>_hat" and "<name>_sd" (N,) on the report clock and "flag_<name>" (N,) bool,
    # the debounced |hat - nominal| / sd > PARAM_FLAG_Z. x / P above stay the mass marginal.
    extra: dict[str, np.ndarray] | None = None


def run(
    truth: Truth,
    obs: list[Observation],
    cs: ConstraintSet | None,
    spec: EstimatorSpec,
    declared_m0: tuple[float, float],
    declared_m0_std: float = 5.0,
    kf_cfg: KFConfig = KFConfig(),
    aug_cfg: AugConfig = AugConfig(),
    closedq_cfg: ClosedQConfig = ClosedQConfig(),
    oracle_cfg: OracleConfig = OracleConfig(),
) -> RunResult:
    n = len(obs)
    dt = float(truth.t[1] - truth.t[0])
    est_cls = ESTIMATORS[spec.kind]
    cfg = {KFConfig: kf_cfg, AugConfig: aug_cfg, ClosedQConfig: closedq_cfg, OracleConfig: oracle_cfg}[est_cls.config_cls]
    est_kwargs: dict = {}
    if spec.kind == "oracle":
        # THE ONE PLACE hidden truth crosses to the estimator side. The oracle (bound) gets
        # the hidden actual pump parameter rate and the hidden leak as known inputs. No
        # other kind receives them, and Truth.m is not read even here.
        est_kwargs["oracle_inputs"] = (truth.u_actual, truth.leak)
    est = est_cls(declared_m0, declared_m0_std, dt, truth.u_commanded, cfg, **est_kwargs)
    aug_names = tuple(est.aug_names)

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
    cusum = Cusum(spec.cusum) if spec.cusum is not None else None
    cusum_stat = np.full((n, 2), np.nan)
    cusum_alarm = np.zeros((n, 2), dtype=bool)
    extra: dict[str, np.ndarray] | None = None
    if aug_names:
        extra = {}
        for name in aug_names:
            extra[f"{name}_hat"] = np.full(n, np.nan)
            extra[f"{name}_sd"] = np.full(n, np.nan)
            extra[f"flag_{name}"] = np.zeros(n, dtype=bool)
    aug_streak = [0] * len(aug_names)

    streak = 0
    next_obs = 0   # index of the first observation that has not yet arrived
    for k in range(n):
        t0 = perf_counter()
        tk = float(truth.t[k])
        while next_obs < n and obs[next_obs].arrival_t <= tk:
            est.ingest(obs[next_obs], next_obs)
            next_obs += 1
            if cusum is not None:
                cusum_alarm[k] |= cusum.update(est.innov_z[-1])
        if cusum is not None:
            cusum_stat[k] = cusum.stat
        xr, Pr = est.report(k)
        if aug_names:
            # the reconciliation stage sees the mass marginal only; the rest is an output
            for i, (name, nominal) in enumerate(zip(aug_names, est.aug_nominal)):
                hat, sd = float(xr[2 + i]), float(np.sqrt(Pr[2 + i, 2 + i]))
                extra[f"{name}_hat"][k] = hat
                extra[f"{name}_sd"][k] = sd
                aug_streak[i] = aug_streak[i] + 1 if abs(hat - nominal) > PARAM_FLAG_Z * sd else 0
                extra[f"flag_{name}"][k] = aug_streak[i] >= PARAM_FLAG_DEBOUNCE
            xr, Pr = xr[:2], Pr[:2, :2]

        s = consistency_stat(xr, Pr, cs) if feasible else np.nan
        exceed = feasible and s > thr
        streak = streak + 1 if exceed else 0
        f = streak >= spec.debounce
        hold = spec.guard and f

        se = reconcile(
            xr, Pr, cs, mode=spec.mode, lam=spec.lam, hold=hold, threshold=thr,
            stat=s if feasible else None, t=tk, model_version=est.model_version,
        )
        if spec.feedback and se.status is Status.OK and next_obs > 0:
            _feed_back(est, se, cs, spec, k, j=next_obs - 1)
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

    # innovation record: sampling steps that never arrived within the run stay NaN
    innov = np.full((n, 2), np.nan); innov_var = np.full((n, 2), np.nan); innov_z = np.full((n, 2), np.nan)
    if est.innov:
        j = len(est.innov)
        innov[:j] = np.asarray(est.innov)
        innov_var[:j] = np.asarray(est.innov_var)
        innov_z[:j] = np.asarray(est.innov_z)

    return RunResult(spec, x, P, xu, Pu, status, stat, thr, flag, res_pre, res_post, corr, lat,
                     innov, innov_var, innov_z, cusum_stat, cusum_alarm, extra)


def _feed_back(est, se, cs: ConstraintSet, spec: EstimatorSpec, k: int, j: int) -> None:
    """Push the applied projection back into the estimator's state at its own last
    ingested sampling step j. When j == k the report at k IS the state at j, so the
    reported (x*, P*) goes back verbatim; under arrival delay (j < k) the state at j
    is projected with the same mode. Only the mass marginal is ever fed back."""
    if j == k:
        xs, Ps = se.x, se.P
    else:
        xj, Pj = est.report(j)
        xj, Pj = xj[:2], Pj[:2, :2]
        if spec.mode == "hard":
            xs, Ps = project_hard(xj, Pj, cs)
        elif spec.mode == "soft":
            xs, Ps = project_soft(xj, Pj, cs, spec.lam)
        else:   # unreachable: __post_init__ requires a mode
            raise ValueError(f"unknown mode {spec.mode!r}")
    est.set_state(xs, Ps)
