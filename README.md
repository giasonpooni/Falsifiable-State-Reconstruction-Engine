# Falsifiable State Reconstruction Engine

State estimates that ship with the statistic that can reject them.

This repository holds the Phase 1 vertical slice of a research program on
evidence-grounded state reconstruction under degraded observability — one state
schema, one observation schema, one small simulator, one estimator family, one
constraint-reconciliation kernel, one evaluation report, so that an observation
can be traced through estimation and reconciliation to a measured error against
hidden simulated truth — and, on top of it, the P2 work that answers what the
Phase 1 review left open: a detector that does not read the constraint, an
estimator that carries the faults the constraint cannot see, and the baselines
that would show the constraint row adds nothing. Nothing more.

As built, it is an *evidence-preserving reconciliation stage over a
state-estimation testbed*. Its lineage is data validation and reconciliation
with a global χ² test (Crowe, 1985) and constrained Kalman filtering. The name
is a program goal — every estimate carries a test able to reject it — not a
delivered property; the results sections say exactly where the current tests are
blind. "State reconstruction" here means estimating a dynamical system's state
from measurements, not quantum-state tomography and not the axiomatic
reconstruction of a physical theory.

```
Truth (hidden)  ──h(·)+degradation──▶  Observation  ──▶  Estimator  ──▶  LCM reconcile  ──▶  StateEstimate
      │                                                                                             │
      └──────────────────────────────── Evaluator (only reader of truth) ◀───────────────────────────┘
```

## Run

```bash
uv run --python 3.13 --dev pytest -q
```

```bash
uv run --python 3.13 python run_experiments.py
```

```bash
uv run --python 3.13 python -m set_lcm.experiments.calibration
```

```bash
uv run --python 3.13 python -m set_lcm.experiments.sweep
```

The grid writes `results/summary.{md,json}`; the other two write
`results/calibration.{md,json}` and `results/sweep.{md,json}` (about a quarter of an
hour together). Full-size runs are also wrapped in tests marked `slow`, which the default
`pytest` skips; `uv run --python 3.13 --dev pytest -m slow` runs them.

`results/` is a verified artifact, not a hand-committed file: the slow test
`tests/test_results_reproduce.py` regenerates the whole grid and asserts equality with
the committed `results/summary.json` value for value (latency and the provenance
stamp excluded). Every results file carries a provenance block — Python and numpy
versions, platform, a line-ending-independent SHA-256 of the source tree, and git
HEAD at generation time (the parent of the commit that contains the results) — and
prints it as the first line of the markdown. Every scenario is
run over 20 seeds (simulation and degradation seeds offset together); tables report
mean ± sd across seeds and the JSON keeps every per-seed metric. A single seed is a
realization, not a result: the first single-seed run of the noisy-valve scenario put
`kf+hard` within 8 % of `kf` during the blackout (0.386 vs 0.419 kg); over 20 seeds it
is 0.55 ± 0.18 vs 0.92 ± 0.44 kg.

## What is in the slice

| Module | Responsibility |
|---|---|
| `src/set_lcm/schema/` | `Observation`; `ConstraintSet` — `A`, `b`, a version and a description, `dof` = rows, `rank` = χ² dof; it carries no uncertainty on `b` (P2b); `StateEstimate` envelope with `Status` (`NOT_CONVERGED` reserved, nothing emits it), unprojected state always retained, correction, residuals pre/post, consistency stat. |
| `src/set_lcm/lcm/` | Hard (KKT closed form) and soft (penalty) linear-equality projection weighted by `P⁻¹`; feasibility check and reduction of dependent rows; SPD / singularity guards; χ² consistency statistic and the detectability `d(f)` of a fault direction; `reconcile()` that never mutates its input. |
| `src/set_lcm/testbed/simulator.py` | Two-reservoir material transfer. Hidden `m`, hidden leak, hidden actual pump *parameter* rate (`u_actual`: the per-step pump fluctuation is process noise, not part of it); public commanded pump rate and declared initial total. |
| `src/set_lcm/testbed/degrade.py` | Noise, random dropout, sensor blackout, undeclared bias, quantization (declared into `R`), arrival delay written into `arrival_t`. Seed-controlled. |
| `src/set_lcm/testbed/estimators.py` | Five estimator kinds behind one `ingest` / `report` interface, each initialised from a *declared* prior and reporting by predicting forward from the last *arrived* observation. `hold_last`. `kf`: linear KF on x = [m1, m2] with a *diagonal* Q, σ_w = 0.05 kg per step (`KFConfig`; closure is the constraint's declared claim, not the filter's). `kf_aug`: x = [m1, m2, α, L] — pump scale and boundary flux as states with their own uncertainty, a time-varying *linear* KF (`AugConfig`: α ~ N(1, 0.1²), L ~ N(0, 0.02²), random walks 1e-3 and 2e-3 kg/s per step, same σ_w). Two baselines built to remove the case for a constraint row: `kf_closedq`, the KF with closure written into its process noise, Q = σ_q² dt² B Bᵀ + ε I with σ_q = 0.01 kg/s (the simulator's declared pump fluctuation) and ε = 1e-8 (`ClosedQConfig`), and `oracle`, a KF given the *hidden* actual pump parameter rate and leak as known inputs with Q = ε I only (`OracleConfig`) — a bound, never a candidate. Every filter records, per ingested step and sensor, the innovation, its variance and the normalised innovation; hold-last records NaN. `kf` and `kf_closedq` implement `set_state(x, P)` for fed-back projection; `kf_aug` (the mass marginal cannot be replaced without its cross-covariances) and hold-last raise `NotImplementedError`. |
| `src/set_lcm/testbed/cusum.py` | Per-sensor two-sided CUSUM on the normalised innovation (`CusumConfig(k=0.5, h=8.0)`): the evidence side's own detector, reading no constraint. |
| `src/set_lcm/testbed/runner.py` | Runs a (scenario, estimator) pair step by step, ingesting observations only once `arrival_t ≤ t_k`. Three detection channels that never talk to each other: the debounced consistency flag (with the optional guard that *holds* projection and reports `MODEL_INCONSISTENT`), the CUSUM on the ingest clock with alarms stamped at the report step of ingestion, and one debounced flag per extra state (\|α̂ − 1\|/σ_α > 3.29, \|L̂\|/σ_L > 3.29, three consecutive reports). Reconciliation always acts on the mass marginal (x[:2], P[:2, :2]); an augmented estimator's extra states go to `RunResult.extra`. `EstimatorSpec.feedback` pushes each applied projection (x*, P*) back into the filter at its own last ingested step, at most once per ingested step. The one place hidden truth crosses to the estimator side is marked here: `truth.u_actual` and `truth.leak` go to the constructor of kind `"oracle"` and to nothing else. |
| `src/set_lcm/testbed/evaluate.py` | RMSE (overall and windowed), 95 % interval coverage, normalised error, error along row(A) and null(A), residuals, correction magnitude, false alarms and censored detection delay for the constraint flag, per sensor for the CUSUM and per parameter for the α / L flags, α error against the hidden pump-rate ratio over pump-on steps, L error against the hidden leak, mean normalised innovation per window, solver failures, latency. Only reader of truth. |
| `src/set_lcm/experiments/phase1.py` | The scenario grid, the eleven estimator specs, multi-seed aggregation and report writer; `run_experiments.py` is a thin CLI over it. |
| `src/set_lcm/experiments/provenance.py` | The provenance block every results file carries: interpreter and numpy versions, platform, source-tree SHA-256, git HEAD and a dirty flag. |
| `src/set_lcm/experiments/calibration.py` | In-loop null of the consistency statistic (mean, tail quantiles, empirical vs nominal exceedance, autocorrelation time), a threshold × debounce sweep of the guard, and the null of the CUSUM channel over the same windows for h ∈ {4, 6, 8, 10}. |
| `src/set_lcm/experiments/sweep.py` | Fault-magnitude sweep on four axes (declared-total error, sensor bias, leak rate, uncertain declared total with soft λ = 1/σ_b²); `kf_aug` runs on the first and third, `kf_closedq` and `kf+hard+fb+guard` on the first. |

## Scenarios

All six declare the same constraint, `m1 + m2 = 100 kg`, version `closed-boundary-v1`.
The estimator is initialised from a *declared* prior (mean and std, stated per
scenario), never from the simulator's state, and it may use an observation only
once `Observation.arrival_t` has passed.

| Scenario | Constraint is | What it tests |
|---|---|---|
| `closed_noise` | true | Does a correct constraint reduce error? Does soft leave a residual where hard does not? |
| `closed_blackout_pumpbias` | true | Sensor 2 dark for 100 steps while the pump delivers 20 % more than commanded (a wrong *parameter*). Does the constraint help when the measured reservoir itself lags? |
| `closed_blackout_noisy_valve` | true | Sensor 2 dark for 300 steps while an unmodeled valve moves mass at random (pure observability loss). Does the constraint carry sensor 1's information into the dark reservoir? |
| `leak_stale_constraint` | **stale** from step 300 | 10 kg leaks out. Does hard projection make the estimate confidently wrong? Does the guard notice and stop enforcing? |
| `bias_quant_delay` | true, evidence is not | Sensor 1 gains an undeclared +3 kg bias. Does the guard flag it? Note it cannot tell bias from leak. |
| `closed_wrong_prior` | true | The declared initial fill is 74/26 kg against a truth of 70/30 (5 kg prior std). Is the wrong prior forgotten from the evidence? (KF: 0.60 kg while settling → 0.23 steady; hold-last stays at ~2.0.) |

The grid runs eleven estimator specs on every scenario:

- Phase 1: `hold_last`, `kf`, `kf+soft(1/λ = 4 kg²)`, `kf+hard`, `kf+hard+guard`.
- P2 augmented state: `kf_aug` — mode None, never projected; the consistency statistic is
  still computed on its mass marginal, and α and L are outputs with their own flags.
- P2 baselines: `kf+hard+fb` and `kf+hard+fb+guard` (the projection fed back into the
  filter), `kf_closedq` and `kf_closedq+hard` (closure in Q instead of, and then as well
  as, a constraint row), and `oracle (bound)` (hidden inputs known; a bound, not a
  candidate).

The `uncertain_total` sweep axis adds `kf+soft(λ = 1/σ_b²)` at each point.

## What a flag means

The consistency statistic `r = A x̂ − b`, `stat = rᵀ(A P Aᵀ)⁻¹ r` tests one joint
hypothesis: *the declared constraint is true, the estimator's model is right, and its
stated uncertainty is calibrated.* A flag rejects that conjunction. It does not say
which conjunct failed — a stale constraint, an undeclared sensor bias, a wrong pump
parameter, and under-modelled process noise all produce flags. Each scenario's
`fault_onset` is the first step at which the conjunction is false; flags before it are
false alarms, flags after it are detections.

## What the Phase 1 results do and do not show

The tables in `results/summary.md` are real, but several of the obvious readings
of them are wrong. An adversarial review of this slice established the following;
the first round of work in this repository addresses them.

Triples below are (RMSE kg, cov95, nz) over the named window, from `results/summary.md`.
nz is the RMS normalised error eᵢ/σᵢ: 1.0 when calibrated, above 1 over-confident.

- **"A true constraint cuts KF RMSE 0.28 → 0.20 kg" is not a finding.** It is the
  analytic 1/√2 of averaging two equal-noise sensors, already reproduced in
  `tests/test_lcm.py`. In `closed_noise`'s steady window: kf 0.23 / 0.99 / 0.73 →
  kf+hard 0.16 / 0.98 / 0.72. The error along row(A) goes to 0.00 exactly; the error
  along null(A) stays 0.23.
- **A stale constraint is the negative control that matters.** In the leak window of
  `leak_stale_constraint`: kf 1.07 / 0.60 / 3.35, kf+hard 3.80 / 0.09 / 16.9 with a
  post-projection residual of 5e-16 — numerically perfect, physically wrong, and
  reported with roughly seventeen times the confidence it deserves. The guard flags
  in 20/20 seeds (median 66 steps after onset), holds projection, and lands at
  1.12 / 0.55 / 3.87.
- **Hard projection is over-confident by construction.** It sets `A P Aᵀ = 0`, i.e. it
  asserts the constraint is exactly true. In-window it reads 1.00 / 0.02 / 3.75
  (pump-bias blackout), 1.41 / 0.05 / 5.88 (post-bias) and 3.80 / 0.09 / 16.9 (leak);
  the whole-run cov95 column (0.67, 0.36, 0.53) hides that. In the bias scenario
  kf+hard's post-bias RMSE is well below the unconstrained filter's (1.41 against
  1.98) with the worst calibration in the table (cov95 0.05, shared only with its
  fed-back variants) — RMSE alone would pick the wrong estimator.
- **The consistency statistic is not χ²(1) in the loop.** `results/calibration.md`
  pools it from the filter's own reported P over the windows where the joint
  hypothesis holds: mean 0.46–0.64 instead of 1.0, q999 between 5.5 and 9.8 instead of
  10.8, and lag-1 autocorrelation 0.76–0.94 — an integrated autocorrelation time of
  7–30 steps, so the 12,000 samples in `closed_noise` are about 400 independent ones.
  The deflation comes from the diagonal Q asserting sum-direction process noise that
  the closed simulator never generates (the same thing shows as nz = 0.73 for the
  unconstrained KF). The pre-fault count of one flagged step in 6,000 is therefore a
  property of a deflated, autocorrelated statistic, not of the guard. Sweeping the
  threshold on the leak scenario (q ∈ {0.95, 0.99, 0.999} × debounce ∈ {1, 3, 10}):
  every setting detects the 10 kg leak in 20/20 seeds; what the setting buys is
  pre-onset false-alarm rate against delay, from 1.6e-2 per step at a median 44-step
  delay (q = 0.95, debounce 1) to 0 at 73 steps (q = 0.999, debounce 10).
- **A single sum constraint is blind to the difference direction.** For
  `A = [1, 1]` and the pump or valve fault along `(−1, 1)`, the detectability
  `d(f) = fᵀAᵀ(APAᵀ)⁻¹Af` is exactly 0 for every estimator in every seed (`d(f)`
  column). Those faults are structurally undetectable by this test — 0/20 seeds flag
  within 100 steps and the censored median delay is "> 550" — while calibration
  collapses: kf in the pump-bias blackout reads 1.31 / 0.21 / 2.90 with error
  0.71 along row(A) versus 1.70 along null(A). For the leak and the sensor bias,
  d(f) is 4.9 and 4.4 and 20/20 seeds flag. The answer to that blindness is not a
  better test but more state: `kf_aug` (P2, "The augmented state") estimates the
  null-direction fault and reads 0.50 / 0.96 / 0.83 in the same window.
- **The projection is oblique.** The `P⁻¹`-weighted correction moves along `P Aᵀ`,
  which has a null(A) component whenever `P` is anisotropic. In the pump-bias
  blackout the null-direction error goes 1.70 → 1.41 under hard projection; in
  `closed_noise`, where `P` is nearly isotropic, it is unchanged (0.23 → 0.23).
- **The guard has a dead band, and it sits at 1–3σ of the constraint's own
  uncertainty.** From `results/sweep.md` (20 seeds, steady window; kf reads
  0.23 / 0.99 / 0.73 throughout): with the declared total wrong by 0.25 kg, hard
  projection still helps (0.21 / 0.97 / 0.92); at 0.5 kg it is already worse than the
  unconstrained filter (0.30 / 0.88 / 1.33) and the guard fires in 0/20 seeds; at 1 kg
  hard reads 0.53 / 0.35 / 2.34 and the guard fires in 3/20; only from 2 kg does the
  guard catch it in 20/20 and hand back the unconstrained answer (0.29 / 0.94 / 1.04).
  The sum's own uncertainty is A P Aᵀ ≈ 0.2 kg², i.e. σ ≈ 0.45 kg, so the dead band is
  the 1–3σ region where a wrong constraint is neither negligible nor rejectable. The
  leak axis has the same shape: a 1 kg leak (0.005 kg/s) already costs hard projection
  0.41 / 0.65 / 1.81 against kf's 0.24 / 0.99 / 0.75, with the guard silent. The 10 kg
  leak and 3 kg bias of the main grid are the easy regime. `kf_aug`, which never
  projects, reads 0.32 / 0.98 / 0.80 at every value of the declared-total error: immune
  to a wrong constraint because it does not use the constraint, and paying for that on
  the nominal system (0.32 against kf+hard's 0.16 at δ = 0).
- **When the constraint's uncertainty is declared, use it.** On the `uncertain_total`
  axis (b = total0 + N(0, σ_b²) drawn per seed) the soft mode with λ = 1/σ_b² — the
  pseudo-measurement whose variance *is* the declared uncertainty — stays calibrated
  at every σ_b (cov95 0.98–0.99, nz 0.72–0.74) at or below kf's RMSE, while hard
  projection and the guard, which treat the constraint as exact, degrade to
  0.75 / 0.37 / 3.34 and 0.41 / 0.67 / 1.75 at σ_b = 2 kg. An arbitrary λ is not a
  mode; a declared σ_b is. That the σ_b has to be smuggled in from the sweep side, because
  `ConstraintSet` cannot carry it, is a P2b item.
- **Two truth leaks, now closed.** The estimator used to be initialised from the
  simulator's exact initial mass and delay was passed out of band. It is now
  initialised from a declared prior (`closed_wrong_prior` exercises a wrong one), and
  the runner ingests an observation only once its `arrival_t` has passed — a test
  corrupts every observation sampled after step *k − d* and checks that reports up to
  *k* are bit-identical.
- **The kernel now refuses what it cannot interpret.** A rank-deficient or indefinite
  `P`, or a singular `A P Aᵀ`, raises instead of yielding `stat = 0.0` with `status =
  ok` (which is what a hard-projected covariance fed back in used to produce).
  Dependent constraint rows are reduced to an independent set and the χ² dof is
  `rank(A)`, not `rows(A)`.
- **Not evidence of anything:** "no solver failures" (nothing can emit
  `not_converged`; the closed-form kernel has no iteration to fail, and the first
  producer is a P2b item); the latency column, tagged "(this machine)" — interpreter
  overhead on 2×2 matrices (4×4 for `kf_aug`) on one laptop, and what it counts has
  grown: the kernel guards' eigenvalue and condition checks every step, the per-sensor
  CUSUM update on every ingest, the 4-state predict and update, and the projection
  itself for the projected variants (`closed_noise` p50: kf 328 µs, kf_aug 353,
  kf+hard 499); determinism beyond same-process (the reproduction test proves
  same-build, same-machine reproduction, nothing more); the shipped soft variant at
  1/λ = 4 kg² as a meaningfully different estimator against S ≈ 0.2 kg² — the
  `uncertain_total` sweep is where soft mode earns its place.

## P2: seeing the null space

The Phase 1 review left three things open. The consistency test is structurally blind to
null(A), and the two faults that live there — a mis-scaled pump and a boundary flux — are
exactly the ones the constraint author did not know about. The test is one test of one
conjunction, with nothing independent of the constraint to corroborate or contradict it.
And every sentence saying the constraint row adds value was measured against a filter
that did not know the boundary was closed, so the value could have been Q's, not the
row's. P2 built one thing for each: an evidence-side channel that reads no constraint,
an estimator that carries α and L as state, and the baselines that would show the row
adds nothing. What follows is what `results/` says about each, and then what it still
does not say.

### The evidence-side channel (CUSUM)

The consistency flag is one test of one joint hypothesis, and it is structurally blind
to anything in null(A). The slice now carries a second detector that is independent
of the constraint: every Kalman ingest records, per sensor, the innovation
ν = y − H x_pred, its variance Sᵢᵢ and the normalised innovation z = ν/√Sᵢᵢ (hold-last
has no prediction and records NaN), and the runner runs a two-sided CUSUM on z per
sensor — g⁺ ← max(0, g⁺ + z − k), g⁻ ← max(0, g⁻ − z − k), alarm and reset when
max(g⁺, g⁻) > h, with k = 0.5 and h = 8 (`CusumConfig` on `EstimatorSpec`, on by
default). It is updated when an observation is ingested and the alarm is stamped at
that report step, so its delay includes arrival delay. `RunResult` gains `innov`,
`innov_var`, `innov_z` on the sampling clock and `cusum_stat`, `cusum_alarm` on the
report clock; the evaluator scores it per sensor with the same false-alarm and
censored-delay rules as the flag; the reconciliation stage never reads it. From
`results/summary.md` (20 seeds; cells are seeds alarmed within 100 steps of onset and
the censored median delay):

- **Sensor bias.** In `bias_quant_delay` the channel on the biased sensor 1 alarms in
  20/20 seeds at a median 14 steps against the constraint test's 20/20 at 36, and it
  names the sensor, which the constraint test cannot (a leak and a biased sensor
  produce the same flag). Sensor 2's channel reads 0/20, "> 400"; `CUSUM FA max` is
  0 / 0.
- **Leak.** In `leak_stale_constraint` sensor 2's channel alarms in 20/20 seeds at a
  median 60 steps (constraint test: 20/20 at 66), from the evidence alone: the filter
  lags the drain and sensor 2's mean normalised innovation over the leak window is
  −0.67 (`z̄ s1/s2` column). Sensor 1 stays at 0/20. Over the 20 seeds there was one
  pre-leak alarm on each sensor, in two different seeds (`CUSUM FA max` 1 / 1; the
  per-seed counts are in `results/summary.json`): one in 6,000 sensor-steps per sensor.
- **What it does not see.** The pump-rate error in `closed_blackout_pumpbias` shifts
  sensor 1's normalised innovation by −0.38 per step during the blackout and −0.40 in
  recovery — below k = 0.5, so the statistic has nothing to accumulate: 2/20 seeds
  within 100 steps, median "> 550", against the constraint test's 0/20 (d(f) = 0).
  Sensor 2's channel is dark for the blackout; per seed (`cusum_detection_delay_steps`
  in `results/summary.json`) it alarms in 18 of 20 seeds only once that sensor returns
  and sees what the dark reservoir accumulated (z̄ = +0.58 in the recovery window) — a
  median 164 steps after onset, i.e. 14 after the blackout ends — never alarms in one
  seed, and in one seed alarmed 34 steps after onset, before the blackout began, which
  is the 1/20 in the "≤ 100" cell. A per-step shift below k is invisible to a CUSUM
  tuned for a half-sigma shift, whatever h is; the same for the zero-mean valve (2/20
  and 1/20, both censored).
- **Its null is measured, not assumed.** `results/calibration.md` runs the channel
  over the same nominal windows as the consistency statistic (46,000 sensor-steps,
  20 seeds) for h ∈ {4, 6, 8, 10}: 203, 17, 2 and 0 alarms. The smallest h with zero
  alarms on this sample is 10; the largest un-reset excursion is 9.12. h = 8 stays the
  shipped default: its 2 alarms are 4.3e-5 per sensor-step, a quarter of the
  constraint guard's own pre-onset rate (1.7e-4 per step at q = 0.999, debounce 3),
  and zero alarms at h = 10 is a statement about this sample, not a bound. Across the
  whole grid the only pre-onset alarms at h = 8 are those two and one in
  `closed_wrong_prior` (`CUSUM FA max` 1 / 0), which is not the wrong prior showing
  through: the settle-window z̄ there is −0.07 / +0.03, the same closed system with the
  correct prior (`closed_noise`) reads 0 / 0, and one alarm in 12,000 sensor-steps is
  of the same order as the channel's measured null rate (4.3e-5 per sensor-step).

### The augmented state (α, L)

The two faults the sum constraint cannot see — a mis-scaled pump, which moves mass along
null(A), and a boundary flux, which makes the constraint stale — are not fixed by a better
test on the same two states. They are fixed by more state. `kf_aug` carries
x = [m1, m2, α, L]: α scales the commanded pump rate and L is a flux out of reservoir 2,

    m1' = m1 − α u dt,   m2' = m2 + α u dt − L dt,   α' = α,   L' = L,

which is linear in the state for the known commanded u, so it is a time-varying linear
Kalman filter, not an EKF. Its priors are declared, not tuned: α ~ N(1, 0.1²),
L ~ N(0, 0.02²), random walks of 1e-3 and 2e-3 kg/s per step, and the same diagonal
0.05 kg mass process noise as `kf` (`AugConfig`). The reconciliation stage sees only the
mass marginal (x[:2], P[:2, :2]) and, at mode None, computes the consistency statistic
but never projects; α and L are *outputs* with their own σ, and the runner raises a flag
on each when it departs from its no-fault value by more than 3.29 σ (two-sided 0.001)
for three consecutive reports. The evaluator scores α against the hidden parameter
ratio over the steps where the pump is commanded on and L against the hidden leak. From
`results/summary.md` (20 seeds; triples are RMSE kg / cov95 / nz over the named window):

- **Pump bias during a blackout.** In `closed_blackout_pumpbias` kf_aug reads
  0.50 / 0.96 / 0.83 over the blackout against kf 1.31 / 0.21 / 2.90 and kf+hard
  1.00 / 0.02 / 3.75, and 0.43 / 0.97 / 0.88 in recovery against 1.01 / 0.25 / 2.67 and
  0.89 / 0.06 / 3.61. The null-direction error is 0.53 kg where kf carries 1.70 and hard
  projection 1.41: the fault the constraint test is structurally blind to (d(f) = 0, and
  still 0/20 flags on kf_aug's own marginal) is estimated instead of tested for. α̂
  converges from sensor 1 alone — RMSE against the hidden ratio of 1.2 is 0.084 over the
  blackout and 0.035 over recovery, σ_α = 0.041 at run end — and the α flag fires in
  20/20 seeds (`detected_any` in `results/summary.json`), but slowly: 0/20 within 100
  steps of onset and a median of 146, about when the blackout ends. With 2 kg sensors,
  one of them dark, a 20 % pump error takes ~150 steps to reach 3.29 σ. No false alarms
  (FA max 0 / 0).
- **Leak under a stale constraint.** In `leak_stale_constraint` kf_aug reads
  0.40 / 0.95 / 0.94 over the leak window against kf 1.07 / 0.60 / 3.35, kf+hard
  3.80 / 0.09 / 16.87 and kf+hard+guard 1.12 / 0.55 / 3.87. It beats kf+hard+guard by a
  factor of 2.8 and is the only candidate that stays calibrated through the leak (the
  oracle bound, which knows the leak, reads 0.11 / 0.88 / 1.08; see "Baselines"); L̂ tracks
  the 0.05 kg/s drain with an RMSE of 0.025 kg/s over the window, onset and shut-off
  transients included. The L flag fires in 20/20 seeds, 9/20 within 100 steps, median 103
  — slower than the constraint test (20/20 at 66) and the CUSUM (20/20 at 60), because a
  0.05 kg/s leak is 3.5 σ_L of the filter's own steady-state L uncertainty (0.0144 kg/s):
  the flag needs L̂ almost fully converged. The tracking itself does two things the flag
  does not: the consistency statistic on kf_aug's marginal still rejects the stale
  constraint in 20/20 seeds at a median 54 steps, and sensor 2's CUSUM goes quiet
  (1/20, z̄ −0.01 against kf's −0.67) because the prediction no longer lags the drain.
  On the `leak_rate` axis of `results/sweep.md` the L flag's dead band is *wider* than
  the guard's — 0/20 seeds at 0.005, 0.01 and 0.02 kg/s (the guard: 0/20, 0/20, 3/20) and
  9/20 at 0.05 (the guard: 20/20) — but the estimate does not wait for the flag: kf_aug
  reads 0.30 / 0.99 / 0.74, 0.30 / 0.99 / 0.75 and 0.31 / 0.99 / 0.77 at the three small
  leaks where the guard reads 0.41 / 0.66 / 1.81, 0.52 / 0.64 / 2.28 and 0.59 / 0.65 / 2.37
  and even the unconstrained filter degrades to 0.47 / 0.77 / 1.47 at 0.02 kg/s. A 1–4 kg
  leak that the L flag never rejects, and the guard rejects in at most 3/20 seeds, is
  absorbed by L̂ without a flag, which is the point: the parameter is estimated, not merely
  tested.
- **What it costs.** Two extra states are two extra ways to be wrong. On the nominal
  system (`closed_noise`) kf_aug reads 0.32 / 0.98 / 0.80 in steady state against kf
  0.23 / 0.99 / 0.73 and kf+hard 0.16 / 0.98 / 0.72: a 40 % RMSE premium for freedom it
  does not use, still calibrated. In `closed_blackout_noisy_valve` — pure observability
  loss, no parameter fault — it is *worse* than the unconstrained filter over the 300-step
  blackout, 1.54 ± 1.39 / 0.89 / 1.12 against kf 0.92 ± 0.44 / 0.73 / 1.67 and kf+hard
  0.55 / 0.67 / 1.96: L is unobservable while sensor 2 is dark, so m2 is dead-reckoned
  from the pre-blackout L̂ (σ_L ≈ 0.014 kg/s over 300 steps is ±4 kg). Its intervals admit
  it (cov95 0.89 against kf's 0.73); the constraint fixes it. In `bias_quant_delay` the α
  flag fires in 15/20 seeds at a median 57 steps after the +3 kg sensor bias: a 3 kg rise
  in m1 while the pump is on can only be explained by the model through α, so the flag
  names a parameter, not a cause — the estimator side cannot tell a biased sensor from a
  mis-scaled pump, just as the constraint test cannot tell a leak from a bias. Before any
  onset, in every scenario, neither parameter flags (FA max 0 / 0 throughout).
- **Plainly.** kf_aug beats kf+hard+guard in both fault scenarios — 0.50 vs 1.02 over the
  pump-bias blackout, 0.40 vs 1.12 over the leak — and is calibrated there where the guard
  is not (cov95 0.96 and 0.95 against 0.02 and 0.55). It loses to hard projection on the
  nominal system (0.32 vs 0.16) and on the noisy valve (1.54 vs 0.55). Neither is "the"
  estimator; each carries a failure the results name. And its flags are not the
  constraint's: with the stated priors a 0.05 kg/s leak sits at 3.5 σ_L of the filter's
  steady-state uncertainty and a 20 % pump error needs ~150 steps to reach 3.29 σ_α, so
  both are seen but neither is seen quickly, and a 0.02 kg/s leak is not flagged at all.

### Baselines: does the row add value?

Before any sentence saying the constraint row adds value, the grid carries the estimators
that would show it does not. `kf_closedq` is the same KF with closure written into its
process noise instead of into a constraint row (Q = σ_q² dt² B Bᵀ + ε I, σ_q = 0.01 kg/s
— the simulator's declared pump fluctuation — ε = 1e-8): it "knows" the boundary is closed
through Q, has no row, and does not know b. `oracle (bound)` is a KF given the *hidden*
actual pump parameter rate and the hidden leak as known inputs with Q = ε I only — a bound
on what a perfect model of the inputs could do, never a candidate; the runner marks the one
place those arrays cross to the estimator side. `kf+hard+fb` feeds each applied projection
(x*, P*) back into the filter instead of keeping it as a post-stage over a retained
unprojected state. From `results/summary.md` (20 seeds; RMSE kg / cov95 / nz):

| estimator | closed_noise, steady | leak_stale_constraint, leak window | leak flag (k/n ≤ 100, median) |
|---|---|---|---|
| kf | 0.23 / 0.99 / 0.73 | 1.07 / 0.60 / 3.35 | 20/20, 66 |
| kf_closedq | 0.11 / 0.98 / 0.75 | 3.44 / 0.13 / 25.11 | 15/20, 92 |
| kf_closedq+hard | 0.08 / 0.99 / 0.69 | 4.17 / 0.10 / 34.62 | 15/20, 92 |
| kf+hard | 0.16 / 0.98 / 0.72 | 3.80 / 0.09 / 16.87 | 20/20, 66 |
| kf+hard+guard | 0.16 / 0.98 / 0.72 | 1.12 / 0.55 / 3.87 | 20/20, 66 (held 235 steps) |
| kf+hard+fb+guard | 0.16 / 0.98 / 0.72 | 3.80 / 0.09 / 16.87 | 0/20, > 300 (held 0) |
| kf_aug | 0.32 / 0.98 / 0.80 | 0.40 / 0.95 / 0.94 | 20/20, 54 |
| oracle (bound) | 0.10 / 0.92 / 1.02 | 0.11 / 0.88 / 1.08 | 20/20, 14 |

What the numbers say. On the nominal system, LCM's estimator value when the constraint is
true is recovered by a filter that encodes closure in Q: `kf_closedq` reads 0.11 against
`kf+hard`'s 0.16 in `closed_noise` (and 0.10 against 0.16 in `closed_wrong_prior`) — the 1/√2
sensor-averaging gain and more, because its declared σ_q also makes its difference-direction
process noise 0.014 kg per step against `kf`'s untuned 0.05 — and the row on top of that Q
buys only the exact total (0.11 → 0.08, row error 0.08 → 0.00), which is the same knowledge
that makes both confidently wrong when b is stale (3.44 and 4.17, cov95 0.13 and 0.10). That
recovery does not extend to the two scenarios where the constraint is true but the filter's
model is not: over the pump-bias blackout `kf_closedq` reads 1.56 / 0.00 / 7.64 against
`kf+hard`'s 1.00 / 0.02 / 3.75, and over the noisy-valve blackout 0.91 / 0.31 / 4.54 against
0.55 / 0.67 / 1.96 — the row corrects the sum direction whatever Q says, and the small
structural Q is what lags there. What a filter with closure in Q does not have is anything to
stop enforcing: its own consistency statistic still rejects the stale constraint (15/20
within 100 steps), but nothing acts on it, whereas `kf+hard+guard` flags 20/20, holds
projection for 235 steps and lands at 1.12 / 0.55. Feeding the projection back destroys
that test: `kf+hard+fb+guard` has the same RMSE as `kf+hard` wherever the constraint is
tested (3.80 / 0.09 in the leak window, and 1.41 / 0.05 post-bias in `bias_quant_delay`;
where the two differ at all it is by a few hundredths, 0.96 against 1.00 over the pump-bias
blackout) and never flags (0/20 in both), because a filter that has been told the constraint
every step no longer disagrees with it — its residual and its A P Aᵀ shrink together — so
the retained unprojected state is what the consistency test runs on, not an implementation
detail. That closes the Phase 1 question of whether projection should feed back by default:
it was built, measured, and it stays a post-stage. The bound row is a bound only where its
model is exact: the oracle is below everything in the leak window (0.11 against `kf_aug`'s
0.40) and in the pump-bias blackout (0.18 against 0.50), but on the nominal system
`kf_closedq+hard` beats it (0.08 against 0.10) because the row carries the exact total, which
is knowledge of the *state* that a perfect model of the *inputs* cannot supply; and on the
noisy valve, whose transfer it does not model, it is not a bound at all (1.03 / 0.25 / 6.26
over the blackout against `kf+hard`'s 0.55 / 0.67 / 1.96). The smaller structural Q also has
a price the nominal window hides: in `closed_blackout_pumpbias` `kf_closedq` lags the
null-direction fault more than even the unconstrained `kf` (1.56 / 0.00 / 7.64 over the
blackout against 1.31 / 0.21 / 2.90) and is still at 0.68 / 0.20 in the steady window where
`kf` has forgotten it (0.24 / 0.99). On the `declared_total_error` axis of
`results/sweep.md` the fed-back guard is dead at every δ > 0 (0/20 detected and, per
`results/sweep.json`, 0 steps held; RMSE 2.01 / 0.00 / 8.91 at δ = 4 kg, the same as
`kf+hard`, where `kf+hard+guard` hands back 0.23 / 0.99 / 0.73), and `kf_closedq` reads
0.11 / 0.98 / 0.75 at every δ because, like `kf_aug`, it never uses b.

### What P2 still does not claim

- **That the constraint row makes the estimator better on the nominal system.** It does
  not, beyond what Q can do: `kf_closedq` 0.11 against `kf+hard` 0.16. What the row does
  that Q cannot is carry the exact declared total when it is true (0.11 → 0.08) and carry a
  test that something acts on when it is not (guard held 235 steps, 1.12 / 0.55, against
  `kf_closedq`'s 3.44 / 0.13 with a flag nothing reads). The claim has narrowed to that.
- **That the evidence-side channel is a general detector.** At k = 0.5 it sees the 3 kg
  bias (20/20 at 14) and the lag a 0.05 kg/s leak induces (20/20 at 60), and it does not
  see the 20 % pump error (z̄ −0.38 per step, 2/20, "> 550"); a sustained shift below k is
  invisible at any h. Its null rate (4.3e-5 per sensor-step at h = 8) is a count on 46,000
  sensor-steps, not a derived bound.
- **That α and L give timely flags.** The α flag needs ~150 steps for a 20 % pump error
  (0/20 within 100, median 146); the L flag catches a 0.05 kg/s leak in 9/20 seeds within
  100 steps and a 0.02 kg/s leak in 0/20. The augmented state's value is in the estimate
  (0.31 / 0.99 / 0.77 at 0.02 kg/s where the guard reads 0.59 / 0.65 / 2.37), not in its
  flags.
- **That more state is free.** A 40 % RMSE premium on the nominal system (0.32 against
  0.23) and, over the noisy-valve blackout, 1.54 ± 1.39 against `kf+hard`'s 0.55, because
  L is unobservable while sensor 2 is dark.
- **Attribution.** Three channels, no arbiter. The constraint flag cannot tell a leak from a
  bias; the CUSUM names a sensor, not a cause; the α flag fires on a sensor bias (15/20 at
  57) because the model can only explain it through the pump. Nothing in the tree
  combines them, and nothing should until each one's null is on the table.
- **A bound.** The oracle bounds only what its model covers: on the noisy valve it reads
  1.03 / 0.25 / 6.26 against `kf+hard`'s 0.55 / 0.67 / 1.96.
- **Sensitivity to the declared priors.** `AugConfig` is one stated set of values, not a
  sweep; there is no result for how `kf_aug` moves with σ_α0, σ_L0 or the random-walk
  rates.
- **That the constraint and the augmented state ever meet.** `kf_aug` is never projected,
  its `set_state` raises, and the constraint never reads α̂ or L̂; the consistency statistic
  is computed on its marginal and nothing acts on it. Both directions are P2b.

## Deliberately out of scope

- **IMM (deferred, not rejected).** An interacting-multiple-model filter over {closed,
  leaking, mis-scaled pump} hypotheses would be a second, discrete answer to the question
  `kf_aug` already answers continuously with α and L and their own σ, at the price of
  declared mode-transition priors, so it earns a comparison only once P2b has put
  uncertainty on the constraint side and the two can be scored on the same statistic.
- **P2b: inequality constraints and `NOT_CONVERGED` producers.** `Status.NOT_CONVERGED`
  is reserved and nothing emits it because the closed-form kernel has no iteration to
  fail; an inequality row (m2 ≥ 0, a bounded flux) needs an active-set or QP solve, which
  is the first producer of that status and the first place "no solver failures" becomes a
  measurement.
- **P2b: constraint uncertainty declared on `ConstraintSet` itself (`b_var`).** Today the
  set is A x = b, exact by declaration, and the `uncertain_total` sweep brings σ_b in from
  the sweep side as the soft λ. With `b_var` on the set the consistency statistic becomes
  rᵀ(A P Aᵀ + B_var)⁻¹ r, the guard's 1–3σ dead band is tested against the declared
  uncertainty instead of against zero, and soft mode's λ stops being a free parameter.
- **P2b: feeding α̂ and L̂ back into the constraint as a prior.** The augmented filter
  estimates the flux the constraint author did not know about; a constraint set could
  carry it (b − ∫L̂ dt with its variance) instead of going stale. Not built, deliberately:
  a constraint that follows the estimate re-opens the fed-back-projection question — the
  test stops disagreeing with what it tests — so it needs `b_var` first and a negative
  control of its own.
- nonlinear constraints, multi-variable factors beyond one linear row
- out-of-sequence measurement handling beyond uniform delay
- the spectral processor, the Rust ingestion boundary, any manifold machinery
- claims about cross-platform bitwise determinism; determinism here means same seed → identical arrays on one build
