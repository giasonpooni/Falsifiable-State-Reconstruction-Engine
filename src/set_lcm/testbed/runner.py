"""Runs one (observation record, estimator spec) pair step by step and records
everything an evaluator needs.

What the estimator side is allowed to see: the observations (only once each
has arrived, per Observation.arrival_t), the public inputs (PublicInputs: the
step clock and the commanded input), a DECLARED prior for the initial state,
and the declared constraint set. run() takes nothing that could carry hidden
simulated state: a real record has none, and this module does not know that a
simulator exists. A simulated run hands it PublicInputs.from_truth(...), built
by experiment code.

THE ONE EXCEPTION, deliberately visible: the oracle bound needs the hidden
per-step actual pump parameter rate and leak. They arrive only through the
explicit keyword `oracle_inputs=(u_actual, leak)`, passed by the caller --
experiment code is the one place that reads them and routes them here -- and
run() forwards them to the estimator constructor for spec.kind == "oracle",
raises for any other kind, and raises if the oracle is run without them. The
oracle is a BOUND on what a perfect model of the inputs could do, labelled
"oracle (bound)" wherever it is tabulated, and never a candidate; a test
corrupts both hidden arrays and checks that every other estimator's record is
bit-identical.

Shapes come from the record and the estimator, not from the two-reservoir
problem: the number of sensors is obs[0].y.size (every observation must carry
that many), the reported-state dimension is the estimator's n_report, and the
reconciliation stage acts on the first n_report components of what the
estimator reports; components after those are the estimator's aug_names. For
every estimator in the tree n_report = 2 (the masses m1, m2) and there are two
sensors.

Provenance: RunResult.ingested_evidence[k] is the concatenation, in sampling
order, of Observation.evidence_ids over every observation ingested at report
step k (() where nothing arrived, and () throughout a simulated run). No
estimator reads them.

Feedback (EstimatorSpec.feedback): after a projection has been applied at
report step k (status ok), the projected mass marginal (x*, P*) is pushed
back into the estimator with est.set_state(). When the estimator is current
(its last ingested sampling step j equals k) that is literally the reported
(x*, P*); under arrival delay (j < k) the report is a prediction from the
state at j, so the projection of that state -- same mode, same constraint --
is what goes back, which keeps the feedback on the estimator's own clock. For
an exact constraint set P* is rank-deficient along the constraint; the next
predict adds Q, so the reported covariance the kernel sees at k + 1 is SPD
again (asserted in tests). With declared b_var > 0, P* is SPD already -- and
the filter is then told the same uncertain b as if it were a fresh, independent
pseudo-measurement at every step, which it is not. While the guard holds (model_inconsistent) nothing is fed back, and
each ingested step is fed back at most once: if the ingest clock stalls (a
late observation blocks the sampling order for several report steps) the
state at j already carries the projection and is not projected again. Only
the first n_report components are ever fed back.
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
PARAM_FLAG_DEBOUNCE consecutive reports (no flag for a component whose nominal
the estimator declares None). The last are estimator outputs, not
reconciliation statuses; Status is unchanged. The reconciliation stage acts on
the first n_report components (x[:n_report], P[:n_report, :n_report]; the mass
marginal for every estimator here) and reads none of the channels except the
first, through the guard.

Declared constraint uncertainty (ConstraintSet.b_var) is never read here; the
kernel reads it in consistency_stat, reconcile and the projections, so every
spec -- guard and feedback included -- honours whatever the set declares, and
the flag threshold stays chi2(rank A).
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ..lcm import chi2_quantile, consistency_stat, is_feasible, project_hard, project_soft, reconcile
from ..schema import ConstraintSet, Observation, Status
from .cusum import Cusum, CusumConfig
from .estimators import ESTIMATORS, AugConfig, ClosedQConfig, KFConfig, OracleConfig
from .inputs import PublicInputs

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
    """Everything one run produced. N report steps, n = the estimator's n_report,
    s = the number of sensors (obs[0].y.size); for every estimator in the tree n = 2
    (the mass marginal) and s = 2."""
    spec: EstimatorSpec
    x: np.ndarray            # (N, n) reported
    P: np.ndarray            # (N, n, n)
    x_unproj: np.ndarray     # (N, n)
    P_unproj: np.ndarray     # (N, n, n)
    status: list[Status]
    stat: np.ndarray         # (N,) consistency stat vs declared constraint (NaN if none)
    threshold: float | None  # None iff no constraint set was declared
    flag: np.ndarray         # (N,) bool: debounced exceedance (computed for every spec)
    res_pre: np.ndarray      # (N, rows)
    res_post: np.ndarray     # (N, rows)  NaN where no projection applied
    corr: np.ndarray         # (N, n)     NaN where no projection applied
    latency_s: np.ndarray    # (N,)
    # Innovation record, indexed by SAMPLING step j (NaN where the sensor was missing,
    # where the estimator has no prediction, or where the observation never arrived).
    innov: np.ndarray        # (N, s)  nu = y - H x_pred
    innov_var: np.ndarray    # (N, s)  S_ii
    innov_z: np.ndarray      # (N, s)  nu / sqrt(S_ii)
    # Evidence record, indexed by SAMPLING step j: True where sensor i's sample j was
    # present (Observation.mask) and the observation was ingested within the run -- what
    # the estimator was given, whatever the estimator did with it.
    observed: np.ndarray     # (N, s) bool
    # CUSUM on z, indexed by REPORT step k: the statistic after the last observation
    # ingested at step k (before the reset an alarm applies; NaN if the channel is off),
    # and whether an alarm was raised while ingesting at step k. Because the alarm is
    # stamped at the report step, its delay includes the observation's arrival delay.
    cusum_stat: np.ndarray   # (N, s)
    cusum_alarm: np.ndarray  # (N, s) bool
    # Provenance, indexed by REPORT step k: the evidence ids (Observation.evidence_ids,
    # concatenated in sampling order) of every observation ingested at step k.
    ingested_evidence: list[tuple[str, ...]]
    # Augmented estimators only (else None): for each name in the estimator's aug_names,
    # "<name>_hat" and "<name>_sd" (N,) on the report clock and, unless the estimator
    # declares that component's nominal None, "flag_<name>" (N,) bool, the debounced
    # |hat - nominal| / sd > PARAM_FLAG_Z. x / P above stay the first n_report components.
    extra: dict[str, np.ndarray] | None = None


def run(
    inputs: PublicInputs,
    obs: list[Observation],
    cs: ConstraintSet | None,
    spec: EstimatorSpec,
    declared_m0: tuple[float, ...],
    declared_m0_std: float = 5.0,
    kf_cfg: KFConfig = KFConfig(),
    aug_cfg: AugConfig = AugConfig(),
    closedq_cfg: ClosedQConfig = ClosedQConfig(),
    oracle_cfg: OracleConfig = OracleConfig(),
    *,
    oracle_inputs: tuple[np.ndarray, np.ndarray] | None = None,
) -> RunResult:
    """Run `spec` over the observation record `obs` (one Observation per step of
    inputs.t, indexed by sampling step). `oracle_inputs` is the hidden (u_actual, leak)
    pair: required for spec.kind == "oracle", refused for every other kind."""
    if not isinstance(inputs, PublicInputs):
        raise TypeError(f"run() takes PublicInputs, not {type(inputs).__name__}")
    if spec.kind == "oracle" and oracle_inputs is None:
        raise ValueError("spec kind 'oracle' needs oracle_inputs=(u_actual, leak), passed explicitly by the caller")
    if spec.kind != "oracle" and oracle_inputs is not None:
        raise ValueError(f"oracle_inputs are hidden arrays for the oracle bound only; spec {spec.name!r} "
                         f"is kind {spec.kind!r}")
    n = len(obs)
    if n < 2:
        raise ValueError("run() needs at least two steps")
    if inputs.t.shape[0] != n:
        raise ValueError(f"{n} observations against a clock of {inputs.t.shape[0]} steps; "
                         "observations are indexed by sampling step on inputs.t")
    n_sensors = int(np.asarray(obs[0].y).size)
    if any(np.asarray(o.y).size != n_sensors for o in obs):
        raise ValueError(f"every observation must carry {n_sensors} sensor value(s), as obs[0] does")
    dt = float(inputs.t[1] - inputs.t[0])
    est_cls = ESTIMATORS[spec.kind]
    cfg = {KFConfig: kf_cfg, AugConfig: aug_cfg, ClosedQConfig: closedq_cfg, OracleConfig: oracle_cfg}[est_cls.config_cls]
    est_kwargs: dict = {}
    if spec.kind == "oracle":
        # THE ONE PLACE the hidden arrays cross to the estimator side, and only because the
        # caller passed them explicitly: the oracle (bound) gets the actual pump parameter
        # rate and the leak as known inputs. No other kind receives them.
        est_kwargs["oracle_inputs"] = oracle_inputs
    est = est_cls(declared_m0, declared_m0_std, dt, inputs.u_commanded, cfg, **est_kwargs)
    n_report = int(est.n_report)
    aug_names = tuple(est.aug_names)
    aug_nominal = tuple(est.aug_nominal)
    if len(aug_nominal) != len(aug_names):
        raise ValueError(f"{est_cls.__name__}: {len(aug_names)} aug_names but {len(aug_nominal)} aug_nominal entries")
    n_state = n_report + len(aug_names)
    if cs is not None and np.atleast_2d(cs.A).shape[1] != n_report:
        raise ValueError(f"constraint {cs.version!r} has {np.atleast_2d(cs.A).shape[1]} columns; "
                         f"{est_cls.__name__} reports {n_report} reconciled components")

    rows = cs.dof if cs is not None else 1
    feasible = cs is not None and is_feasible(cs)
    thr = chi2_quantile(cs.rank, spec.threshold_q) if cs is not None else None

    x = np.zeros((n, n_report)); P = np.zeros((n, n_report, n_report))
    xu = np.zeros((n, n_report)); Pu = np.zeros((n, n_report, n_report))
    status: list[Status] = []
    stat = np.full(n, np.nan)
    flag = np.zeros(n, dtype=bool)
    res_pre = np.full((n, rows), np.nan)
    res_post = np.full((n, rows), np.nan)
    corr = np.full((n, n_report), np.nan)
    lat = np.zeros(n)
    observed = np.zeros((n, n_sensors), dtype=bool)
    cusum = Cusum(spec.cusum, n=n_sensors) if spec.cusum is not None else None
    cusum_stat = np.full((n, n_sensors), np.nan)
    cusum_alarm = np.zeros((n, n_sensors), dtype=bool)
    ingested: list[tuple[str, ...]] = []
    extra: dict[str, np.ndarray] | None = None
    if aug_names:
        extra = {}
        for name, nominal in zip(aug_names, aug_nominal):
            extra[f"{name}_hat"] = np.full(n, np.nan)
            extra[f"{name}_sd"] = np.full(n, np.nan)
            if nominal is not None:
                extra[f"flag_{name}"] = np.zeros(n, dtype=bool)
    aug_streak = [0] * len(aug_names)

    streak = 0
    next_obs = 0   # index of the first observation that has not yet arrived
    fed_j = -1     # last ingested sampling step whose state has already received a projection
    for k in range(n):
        t0 = perf_counter()
        tk = float(inputs.t[k])
        ids_k: tuple[str, ...] = ()
        while next_obs < n and obs[next_obs].arrival_t <= tk:
            o = obs[next_obs]
            est.ingest(o, next_obs)
            observed[next_obs] = o.mask
            ids_k += o.evidence_ids
            next_obs += 1
            if cusum is not None:
                cusum_alarm[k] |= cusum.update(est.innov_z[-1])
        ingested.append(ids_k)
        if cusum is not None:
            cusum_stat[k] = cusum.stat
        xr, Pr = est.report(k)
        if xr.shape != (n_state,):
            raise ValueError(f"{est_cls.__name__} reported a state of shape {xr.shape}; it declares "
                             f"n_report = {n_report} plus {len(aug_names)} aug component(s)")
        if aug_names:
            # the reconciliation stage sees the first n_report components only; the rest is an output
            for i, (name, nominal) in enumerate(zip(aug_names, aug_nominal)):
                c = n_report + i
                hat, sd = float(xr[c]), float(np.sqrt(Pr[c, c]))
                extra[f"{name}_hat"][k] = hat
                extra[f"{name}_sd"][k] = sd
                if nominal is None:
                    continue   # declared "no flag for this component"
                aug_streak[i] = aug_streak[i] + 1 if abs(hat - nominal) > PARAM_FLAG_Z * sd else 0
                extra[f"flag_{name}"][k] = aug_streak[i] >= PARAM_FLAG_DEBOUNCE
            xr, Pr = xr[:n_report], Pr[:n_report, :n_report]

        s = consistency_stat(xr, Pr, cs) if feasible else np.nan
        exceed = feasible and s > thr
        streak = streak + 1 if exceed else 0
        f = streak >= spec.debounce
        hold = spec.guard and f

        se = reconcile(
            xr, Pr, cs, mode=spec.mode, lam=spec.lam, hold=hold, threshold=thr,
            stat=s if feasible else None, t=tk, model_version=est.model_version,
        )
        if spec.feedback and se.status is Status.OK and next_obs > 0 and next_obs - 1 != fed_j:
            # Feed back once per ingested step. If the ingest clock has stalled (a late
            # observation blocks the sampling order), the state at j already carries the
            # projection and is rank-deficient; projecting it again would be refused by
            # the kernel's SPD check, and there is nothing new to feed back anyway.
            fed_j = next_obs - 1
            _feed_back(est, se, cs, spec, k, j=fed_j, n_report=n_report)
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
    innov = np.full((n, n_sensors), np.nan)
    innov_var = np.full((n, n_sensors), np.nan)
    innov_z = np.full((n, n_sensors), np.nan)
    if est.innov:
        j = len(est.innov)
        innov[:j] = np.asarray(est.innov)
        innov_var[:j] = np.asarray(est.innov_var)
        innov_z[:j] = np.asarray(est.innov_z)

    return RunResult(
        spec=spec, x=x, P=P, x_unproj=xu, P_unproj=Pu, status=status, stat=stat, threshold=thr,
        flag=flag, res_pre=res_pre, res_post=res_post, corr=corr, latency_s=lat,
        innov=innov, innov_var=innov_var, innov_z=innov_z, observed=observed,
        cusum_stat=cusum_stat, cusum_alarm=cusum_alarm, ingested_evidence=ingested, extra=extra,
    )


def _feed_back(est, se, cs: ConstraintSet, spec: EstimatorSpec, k: int, j: int, n_report: int) -> None:
    """Push the applied projection back into the estimator's state at its own last
    ingested sampling step j. When j == k the report at k IS the state at j, so the
    reported (x*, P*) goes back verbatim; under arrival delay (j < k) the state at j
    is projected with the same mode. Only the first n_report components (the mass
    marginal) are ever fed back."""
    if j == k:
        xs, Ps = se.x, se.P
    else:
        xj, Pj = est.report(j)
        xj, Pj = xj[:n_report], Pj[:n_report, :n_report]
        if spec.mode == "hard":
            xs, Ps = project_hard(xj, Pj, cs)
        elif spec.mode == "soft":
            xs, Ps = project_soft(xj, Pj, cs, spec.lam)
        else:   # unreachable: __post_init__ requires a mode
            raise ValueError(f"unknown mode {spec.mode!r}")
    est.set_state(xs, Ps)
