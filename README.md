# Falsifiable State Reconstruction Engine

State estimates that ship with the statistic that can reject them.

This repository holds the Phase 1 vertical slice of a research program on
evidence-grounded state reconstruction under degraded observability: one state
schema, one observation schema, one small simulator, one estimator family, one
constraint-reconciliation kernel, one evaluation report — so that an
observation can be traced through estimation and reconciliation to a measured
error against hidden simulated truth. That is the Phase 1 exit condition, and
nothing more.

As built, it is an *evidence-preserving reconciliation stage over a
state-estimation testbed*. Its lineage is data validation and reconciliation
with a global χ² test (Crowe, 1985) and constrained Kalman filtering. The name
is a program goal — every estimate carries a test able to reject it — not a
delivered property; the results section says exactly where the current test is
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
`results/calibration.{md,json}` and `results/sweep.{md,json}` (about ten minutes
together). Full-size runs are also wrapped in tests marked `slow`, which the default
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
| `src/set_lcm/schema/` | `Observation`, `ConstraintSet` (`dof` = rows, `rank` = χ² dof), `StateEstimate` envelope with `status`, unprojected state, correction, residuals pre/post, consistency stat. |
| `src/set_lcm/lcm/` | Hard (KKT closed form) and soft (penalty) linear-equality projection weighted by `P⁻¹`; feasibility check and reduction of dependent rows; SPD / singularity guards; χ² consistency statistic; `reconcile()` that never mutates its input. |
| `src/set_lcm/testbed/simulator.py` | Two-reservoir material transfer. Hidden `m`, hidden leak; public commanded pump rate and declared initial total. |
| `src/set_lcm/testbed/degrade.py` | Noise, random dropout, sensor blackout, undeclared bias, quantization (declared into `R`), arrival delay written into `arrival_t`. Seed-controlled. |
| `src/set_lcm/testbed/estimators.py` | Hold-last baseline; linear Kalman filter with a *diagonal* Q (closure is the constraint's declared claim, not the filter's). Both start from a declared prior and report by predicting forward from the last *arrived* observation. The filter records, per ingested step and sensor, the innovation, its variance and the normalised innovation; hold-last records NaN. |
| `src/set_lcm/testbed/cusum.py` | Per-sensor two-sided CUSUM on the normalised innovation (`CusumConfig(k=0.5, h=8.0)`): the evidence side's own detector, reading no constraint. |
| `src/set_lcm/testbed/runner.py` | Runs a (scenario, estimator) pair step by step, ingesting observations only once `arrival_t ≤ t_k`; debounced consistency flag; optional guard that *holds* projection and reports `MODEL_INCONSISTENT`; runs the CUSUM on the ingest clock and stamps its alarms at the report step of ingestion. |
| `src/set_lcm/testbed/evaluate.py` | RMSE (overall and windowed), 95 % interval coverage, residuals, correction magnitude, false alarms and detection delay for the constraint flag and per sensor for the CUSUM, mean normalised innovation per window, solver failures, latency. Only reader of truth. |
| `src/set_lcm/experiments/phase1.py` | The scenario grid, estimator specs, multi-seed aggregation and report writer; `run_experiments.py` is a thin CLI over it. |
| `src/set_lcm/experiments/calibration.py` | In-loop null of the consistency statistic (mean, tail quantiles, empirical vs nominal exceedance, autocorrelation time), a threshold × debounce sweep of the guard, and the null of the CUSUM channel over the same windows for h ∈ {4, 6, 8, 10}. |
| `src/set_lcm/experiments/sweep.py` | Fault-magnitude sweep on four axes (declared-total error, sensor bias, leak rate, uncertain declared total with soft λ = 1/σ_b²). |

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

Estimator variants: `hold_last`, `kf`, `kf+soft(1/λ = 4 kg²)`, `kf+hard`, `kf+hard+guard`.

## What a flag means

The consistency statistic `r = A x̂ − b`, `stat = rᵀ(A P Aᵀ)⁻¹ r` tests one joint
hypothesis: *the declared constraint is true, the estimator's model is right, and its
stated uncertainty is calibrated.* A flag rejects that conjunction. It does not say
which conjunct failed — a stale constraint, an undeclared sensor bias, a wrong pump
parameter, and under-modelled process noise all produce flags. Each scenario's
`fault_onset` is the first step at which the conjunction is false; flags before it are
false alarms, flags after it are detections.

## An evidence-side channel

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
  Sensor 2's channel is dark for the blackout and, in all but one seed, fires only once
  that sensor returns and sees what the dark reservoir accumulated (z̄ = +0.58 in the
  recovery window): a median 164 steps after onset, i.e. 14 after the blackout ends.
  The 1/20 within 100 steps is a single seed that alarmed 34 steps after onset, before
  the blackout began (per-seed `cusum_detection_delay_steps` in `results/summary.json`).
  A per-step shift
  below k is invisible to a CUSUM tuned for a half-sigma shift, whatever h is; the
  same for the zero-mean valve (2/20 and 1/20, both censored).
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
  kf+hard has the *lowest* post-bias RMSE of any variant and the worst calibration —
  RMSE alone would pick the wrong estimator.
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
  d(f) is 4.9 and 4.4 and 20/20 seeds flag.
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
  leak and 3 kg bias of the main grid are the easy regime.
- **When the constraint's uncertainty is declared, use it.** On the `uncertain_total`
  axis (b = total0 + N(0, σ_b²) drawn per seed) the soft mode with λ = 1/σ_b² — the
  pseudo-measurement whose variance *is* the declared uncertainty — stays calibrated
  at every σ_b (cov95 0.98–0.99, nz 0.72–0.75) at or below kf's RMSE, while hard
  projection and the guard, which treat the constraint as exact, degrade to
  0.75 / 0.37 / 3.34 and 0.41 / 0.67 / 1.75 at σ_b = 2 kg. An arbitrary λ is not a
  mode; a declared σ_b is.
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
  `not_converged` yet); the latency column, tagged "(this machine)" — interpreter
  overhead on 2×2 matrices on one laptop, and roughly doubled since the kernel guards
  run an eigenvalue and a condition check every step; determinism beyond same-process
  (the reproduction test proves same-build, same-machine reproduction, nothing more);
  the shipped soft variant at 1/λ = 4 kg² as a meaningfully different estimator
  against S ≈ 0.2 kg² — the `uncertain_total` sweep is where soft mode earns its place.

## Deliberately out of scope for Phase 1

- inequality constraints, nonlinear constraints, multi-variable factors beyond one linear row
- feeding the projected state back into the filter (projection is a post-stage here; the unprojected state is always retained)
- out-of-sequence measurement handling beyond uniform delay
- the spectral processor, the Rust ingestion boundary, any manifold machinery
- claims about cross-platform bitwise determinism; determinism here means same seed → identical arrays on one build
