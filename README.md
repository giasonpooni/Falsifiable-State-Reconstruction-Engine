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
that would show the constraint row adds nothing — plus the first P2b item,
uncertainty declared on the constraint itself (`b_var`), a runner that cannot receive
hidden truth and an evaluator that needs none, and the first real-data bridge: NOAA
water levels admitted by DAF (the Data Acquisition Fabric) brought in as `Observation`s,
with refusals and provenance but no model of them. Nothing more.

As built, it is an *evidence-preserving reconciliation stage over a
state-estimation testbed*. Its lineage is data validation and reconciliation
with a global χ² test (Crowe, 1985) and constrained Kalman filtering. The name
is a program goal — every estimate carries a test able to reject it — not a
delivered property; the results sections say exactly where the current tests are
blind. "State reconstruction" here means estimating a dynamical system's state
from measurements, not quantum-state tomography and not the axiomatic
reconstruction of a physical theory.

```
Truth (hidden) ──h(·)+degradation──▶ Observation ──▶ Estimator ──▶ LCM reconcile ──▶ RunResult ──▶ truth-free evaluator
  │ │                                                    ▲                               │         (reads the record only)
  │ └─── PublicInputs.from_truth: t, u_commanded ────────┘                               │
  └──────────────────────────────────── Evaluator (scores against truth) ◀───────────────┘

DAF observation dicts ──bridge.daf──▶ Observation + PublicInputs(t grid, u = 0) + provenance ──▶ run() ──▶ truth-free evaluator
(data/daf; no truth)       (refuses what it cannot represent; resamples, converts and averages nothing)
```

The estimator side sees the observations, the public inputs, a declared prior and a declared
constraint — never the truth. The one labelled exception is the oracle bound, whose hidden
inputs experiment code passes to `run()` through the keyword-only `oracle_inputs`; `run()`
refuses them for every other kind.

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

```bash
DAF_ROOT=/path/to/daf-checkout uv run --python 3.13 python tools/export_daf_fixtures.py
DAF_ROOT=/path/to/daf-checkout uv run --python 3.13 --dev pytest -q tests/test_bridge_daf.py
```

The first regenerates `data/daf/` from a DAF checkout (it needs DAF's code, never the
network); the second adds the one bridge test that asks DAF's own code to recompute every
committed evidence id, which skips without `DAF_ROOT`. See "DAF bridge".

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
realization, not a result: seed 0 of the noisy-valve scenario on its own puts `kf+hard`
within 8 % of `kf` during the blackout (0.386 vs 0.419 kg, the per-seed entries in
`results/summary.json`); over 20 seeds it is 0.55 ± 0.18 vs 0.92 ± 0.44 kg.

## What is in the slice

| Module | Responsibility |
|---|---|
| `src/set_lcm/schema/` | `Observation`, with `evidence_ids` (default `()`): the ids of the admitted evidence it was built from, carried for provenance and read by no estimator; `ConstraintSet` — `A`, `b`, a version and a description, `dof` = rows, `rank` = χ² dof, and `b_var`, the declared uncertainty of `b`: `None` (the default) declares the set exact, otherwise a `(rows,)` vector of variances or a `(rows, rows)` symmetric PSD covariance Σ_b, validated at construction (wrong shape, a negative or non-finite variance, asymmetry or indefiniteness raise `ValueError`) and stored as a read-only copy; `StateEstimate` envelope with `Status` (`NOT_CONVERGED` reserved, nothing emits it), unprojected state always retained, correction, residuals pre/post, consistency stat. |
| `src/set_lcm/lcm/` | Hard (KKT closed form) and soft (penalty) linear-equality projection weighted by `P⁻¹`; feasibility check and reduction of dependent rows; SPD / singularity guards; χ² consistency statistic and the detectability `d(f)` of a fault direction; `reconcile()` that never mutates its input. With `b_var` declared, S = A P Aᵀ + Σ_b: the statistic is rᵀS⁻¹r, `d(f)` = fᵀAᵀS⁻¹Af, hard projection is the Kalman update with pseudo-measurement noise Σ_b (Joseph form), soft is hard with Σ_b + (1/λ) I; dependent rows are refused rather than reduced, and only the exact (zero-variance) part of a set can be infeasible. With `b_var = None` the arithmetic is the pre-`b_var` arithmetic. |
| `src/set_lcm/testbed/simulator.py` | Two-reservoir material transfer. Hidden `m`, hidden leak, hidden actual pump *parameter* rate (`u_actual`: the per-step pump fluctuation is process noise, not part of it); public commanded pump rate and declared initial total. |
| `src/set_lcm/testbed/degrade.py` | Noise, random dropout, sensor blackout, undeclared bias, quantization (declared into `R`), arrival delay written into `arrival_t`. Seed-controlled. |
| `src/set_lcm/testbed/inputs.py` | `PublicInputs(t, u_commanded)`: the step clock and the commanded input, the only inputs the runner takes besides the observations, the declared prior and the declared constraint. `PublicInputs.from_truth` copies exactly those two fields of a simulated truth, read-only. |
| `src/set_lcm/testbed/estimators.py` | Five estimator kinds behind one `ingest` / `report` interface, each declaring `n_report` (2 for all five: the masses) and, for extra state, `aug_names` with a nominal per component (`None` = recorded, never flagged), each initialised from a *declared* prior and reporting by predicting forward from the last *arrived* observation. `hold_last`. `kf`: linear KF on x = [m1, m2] with a *diagonal* Q, σ_w = 0.05 kg per step (`KFConfig`; closure is the constraint's declared claim, not the filter's). `kf_aug`: x = [m1, m2, α, L] — pump scale and boundary flux as states with their own uncertainty, a time-varying *linear* KF (`AugConfig`: α ~ N(1, 0.1²), L ~ N(0, 0.02²), random walks of 1e-3 (α, dimensionless) and 2e-3 kg/s (L) per step, same σ_w). Two baselines built to remove the case for a constraint row: `kf_closedq`, the KF with closure written into its process noise, Q = σ_q² dt² B Bᵀ + ε I with σ_q = 0.01 kg/s (the simulator's declared pump fluctuation) and ε = 1e-8 (`ClosedQConfig`), and `oracle`, a KF given the *hidden* actual pump parameter rate and leak as known inputs with Q = ε I only (`OracleConfig`) — a bound, never a candidate. Every filter records, per ingested step and sensor, the innovation, its variance and the normalised innovation; hold-last records NaN. `kf` and `kf_closedq` implement `set_state(x, P)` for fed-back projection; `kf_aug` (the mass marginal cannot be replaced without its cross-covariances) and hold-last raise `NotImplementedError`. |
| `src/set_lcm/testbed/cusum.py` | Per-sensor two-sided CUSUM on the normalised innovation (`CusumConfig(k=0.5, h=8.0)`): the evidence side's own detector, reading no constraint. |
| `src/set_lcm/testbed/runner.py` | `run(inputs: PublicInputs, obs, cs, spec, ...)` runs one estimator spec over an observation record step by step, ingesting observations only once `arrival_t ≤ t_k`; it names no `Truth` and takes no truth argument. The number of sensors comes from the observations (`obs[0].y.size`), the reported-state dimension from the estimator's `n_report`, and every `RunResult` array is sized from those. Three detection channels that never talk to each other: the debounced consistency flag (with the optional guard that *holds* projection and reports `MODEL_INCONSISTENT`), the CUSUM on the ingest clock with alarms stamped at the report step of ingestion, and one debounced flag per extra state with a nominal (\|α̂ − 1\|/σ_α > 3.29, \|L̂\|/σ_L > 3.29, three consecutive reports). Reconciliation acts on the first `n_report` components (the mass marginal x[:2], P[:2, :2] for every estimator here); an augmented estimator's extra states go to `RunResult.extra`. `EstimatorSpec.feedback` pushes each applied projection (x*, P*) back into the filter at its own last ingested step, at most once per ingested step. `RunResult.observed` records which samples the estimator was given and `RunResult.ingested_evidence[k]` the evidence ids of everything ingested at report step k (`()` throughout a simulated run). The oracle bound's hidden actual pump rate and leak arrive only through the keyword-only `oracle_inputs`, which `run()` refuses for any other kind and requires for `"oracle"`. |
| `src/set_lcm/testbed/evaluate.py` | RMSE (overall and windowed), 95 % interval coverage, normalised error, error along row(A) and null(A), residuals, correction magnitude, false alarms and censored detection delay for the constraint flag, per sensor for the CUSUM and per parameter for the α / L flags, α error against the hidden pump-rate ratio over pump-on steps, L error against the hidden leak, mean normalised innovation per window, solver failures, latency. The scoring evaluator: it reads the hidden truth after the run; nothing on the estimator side does. |
| `src/set_lcm/testbed/truth_free.py` | `evaluate_truth_free(run, windows)`: reads nothing but the `RunResult`. Per sensor and window: observed samples, fraction missing, mean / RMS / lag-1 autocorrelation of the normalised innovation z and the fraction with \|z\| > 1.96 (sampling clock), CUSUM alarms and the first alarm step (report clock); where a constraint was declared, flag and status counts, the mean consistency statistic and its per-step exceedance; extra-state flag counts; evidence ids ingested; latency. |
| `src/set_lcm/bridge/daf.py` | `bridge(records, *, series, time_zone, cadence_s, arrival_policy, conflict_policy, daf_commit, latency_s, declared_sigma)`: serialized DAF per-measurement NOAA observations → `BridgedSeries` (`PublicInputs` on a uniform grid, one `Observation` per grid point with DAF evidence ids, provenance), refusing what it cannot represent (see "DAF bridge"); `load_records` with a strict JSON reader; `verify_ids(records, daf_root)`, the one function that imports DAF, optional. Imports nothing from DAF at runtime. |
| `tools/export_daf_fixtures.py`, `data/daf/` | Runs DAF's own per-measurement NOAA binding on DAF's committed fixtures (replayed, no network) and writes what DAF admitted with DAF's `observation_to_dict`; `data/daf/PROVENANCE.md` and `manifest.json` record the pins, fixture hashes, binding parameters and command. |
| `src/set_lcm/experiments/phase1.py` | Where hidden truth is read and routed: `run_spec` hands the runner `PublicInputs.from_truth(truth)` and, through `oracle_inputs_for`, the hidden (u_actual, leak) to the oracle kind and to nothing else; the grid, the sweep and the calibration all run through it. The scenario grid with its per-scenario constraint builder (`ExactTotal` for the first six scenarios, `UncertainTotal` for `closed_uncertain_total`), the eleven estimator specs, multi-seed aggregation and report writer; `run_experiments.py` is a thin CLI over it. |
| `src/set_lcm/experiments/provenance.py` | The provenance block every results file carries: interpreter and numpy versions, platform, source-tree SHA-256, git HEAD and a dirty flag. |
| `src/set_lcm/experiments/calibration.py` | In-loop null of the consistency statistic (mean, tail quantiles, empirical vs nominal exceedance, autocorrelation time), a threshold × debounce sweep of the guard, and the null of the CUSUM channel over the same windows for h ∈ {4, 6, 8, 10}. |
| `src/set_lcm/experiments/sweep.py` | Fault-magnitude sweep on four axes (declared-total error, sensor bias, leak rate, uncertain declared total with soft λ = 1/σ_b²); `kf_aug` runs on the first and third, `kf_closedq` and `kf+hard+fb+guard` on the first. Two axes also hand the same `b` to specs whose `ConstraintSet` declares `b_var`: `kf+hard(b_var)` and its guard on the uncertain total (b_var = σ_b²), `kf+hard(b_var=0.25)` and its guard on the declared-total error. |

## Scenarios

The first six declare the same exact constraint, `m1 + m2 = 100 kg`, version
`closed-boundary-v1`, with no uncertainty on `b`. The seventh, `closed_uncertain_total`,
declares a total that is itself off — `100 kg` plus an offset drawn per seed from
N(0, 1 kg²) — together with that uncertainty (`b_var = [1.0]`, version
`closed-boundary-uncertain-v1`); each scenario's constraint comes from its own builder.
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
| `closed_uncertain_total` | true *within its declared uncertainty* | `b` is off by N(0, 1 kg²) per seed and says so (`b_var` = 1 kg²). Does hard projection that honours the declaration stay calibrated, and does the guard stay quiet? See "Declared constraint uncertainty (b_var)". |

The grid runs eleven estimator specs on every scenario:

- Phase 1: `hold_last`, `kf`, `kf+soft(1/λ = 4 kg²)`, `kf+hard`, `kf+hard+guard`.
- P2 augmented state: `kf_aug` — mode None, never projected; the consistency statistic is
  still computed on its mass marginal, and α and L are outputs with their own flags.
- P2 baselines: `kf+hard+fb` and `kf+hard+fb+guard` (the projection fed back into the
  filter), `kf_closedq` and `kf_closedq+hard` (closure in Q instead of, and then as well
  as, a constraint row), and `oracle (bound)` (hidden inputs known; a bound, not a
  candidate).

The `uncertain_total` sweep axis adds `kf+soft(λ = 1/σ_b²)` at each point, and
`kf+hard(b_var)` and its guard, whose constraint set declares b_var = σ_b²; the
`declared_total_error` axis adds `kf+hard(b_var=0.25)` and its guard.

## What a flag means

The consistency statistic `r = A x̂ − b`, `stat = rᵀ(A P Aᵀ)⁻¹ r` tests one joint
hypothesis: *the declared constraint is true, the estimator's model is right, and its
stated uncertainty is calibrated.* A flag rejects that conjunction. It does not say
which conjunct failed — a stale constraint, an undeclared sensor bias, a wrong pump
parameter, and under-modelled process noise all produce flags. Each scenario's
`fault_onset` is the first step at which the conjunction is false; flags before it are
false alarms, flags after it are detections. When the constraint declares `b_var`, the
statistic is rᵀ(A P Aᵀ + Σ_b)⁻¹ r, still χ²(rank A), and "the declared constraint is
true" reads "b is within its declared uncertainty".

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
  column). Those faults are structurally undetectable by this test — for `kf` and its
  projected variants 0/20 seeds flag within 100 steps and the censored median delay is
  "> 550"; `kf_closedq`, whose sum-direction uncertainty is ε-small, is also 0/20 within
  100 with a median 350 — while calibration
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
  mode; a declared σ_b is. The σ_b no longer has to be smuggled in from the sweep side:
  `ConstraintSet` carries it as `b_var` (P2b, "Declared constraint uncertainty (b_var)").
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
  `rank(A)`, not `rows(A)` (for an exact set; with declared `b_var` dependent rows are
  refused, see P2b).
- **Not evidence of anything:** "no solver failures" (nothing can emit
  `not_converged`; the closed-form kernel has no iteration to fail, and the first
  producer is a P2b item); the latency column, tagged "(this machine)" — interpreter
  overhead on 2×2 matrices (4×4 for `kf_aug`) on one laptop, and what it counts has
  grown: the kernel guards' eigenvalue and condition checks every step, the per-sensor
  CUSUM update on every ingest, the 4-state predict and update, and the projection
  itself for the projected variants (`closed_noise` p50: kf 360 µs, kf_aug 390,
  kf+hard 546); determinism beyond same-process (the reproduction test proves
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
  whole grid, on `kf`'s own channels, the only pre-onset alarms at h = 8 are those two
  and one in `closed_wrong_prior` (`CUSUM FA max` 1 / 0; the closed-Q, oracle and
  fed-back rows carry their own counts in that column, at most 2 per sensor in any
  seed), which is not the wrong prior showing
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
L ~ N(0, 0.02²), random walks of 1e-3 (α, dimensionless) and 2e-3 kg/s (L) per step, and the same diagonal
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
on what a perfect model of the inputs could do, never a candidate; experiment code routes
those arrays to it through `run()`'s keyword-only `oracle_inputs`, which `run()` refuses for
every other kind. `kf+hard+fb` feeds each applied projection
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

## P2b

The first P2b item is built: the constraint's uncertainty is declared on the
`ConstraintSet` itself. The others — inequality rows with the first `NOT_CONVERGED`
producer, and feeding α̂ and L̂ back into the constraint — are still under "Deliberately out
of scope".

### Declared constraint uncertainty (b_var)

`ConstraintSet.b_var` declares b = A x_true + e with e ~ N(0, Σ_b), as a `(rows,)` vector of
variances or a `(rows, rows)` PSD covariance; `None` declares the set exact, which is what
every constraint declared before, and for an exact set the kernel runs the arithmetic it
ran before, so every result of the six exact scenarios and of the pre-existing sweep specs
is unchanged value for value. With Σ_b declared and S = A P Aᵀ + Σ_b (kernel tests in
`tests/test_lcm.py`):

- **consistency** is rᵀS⁻¹r, χ²(rank A) under a joint hypothesis that now includes "b is
  within its declared uncertainty" (20,000 i.i.d. draws with b ~ N(b_true, Σ_b): mean ≈ 1,
  ≈ 0.1 % beyond χ²₁(0.999));
- **detectability** is d(f) = fᵀAᵀS⁻¹Af, which shrinks as Σ_b grows; null(A) stays at
  exactly 0;
- **hard** is the Kalman update with pseudo-measurement noise Σ_b: K = P Aᵀ S⁻¹,
  x* = x − K r, P* = (I − K A) P (I − K A)ᵀ + K Σ_b Kᵀ. Σ_b = 0 gives the exact projection to
  1e-12; Σ_b > 0 leaves a residual and a positive-definite P*;
- **soft** is hard with Σ_b + (1/λ) I, an undeclared slack on top of the declared
  uncertainty. On an exact set soft(λ) equals hard with b_var = (1/λ) I to 1e-12 (λ up to
  100; at λ = 1e4 the penalised form itself loses ~1e-10 to its conditioning): soft mode was
  always a pseudo-measurement with an undeclared variance, and `b_var` is where that
  variance belongs;
- **dependent rows** with `b_var` are refused (`ValueError`), not reduced: the SVD reduction
  keeps U_rᵀb and drops U_⊥ᵀb, which carries information about b's errors whenever Σ_b is
  not isotropic across the dependent rows (an exact row duplicated by an uncertain one would
  come out with a positive variance). Only the exact part of a set — null(Σ_b), the
  zero-variance rows of a vector — can be infeasible. `check_spd(P)` still refuses a
  rank-deficient P; with Σ_b > 0, S is non-singular even where A P Aᵀ is not.

**The honest case: `closed_uncertain_total`.** `b` is off by N(0, 1 kg²) per seed and
declares b_var = 1 kg², so the joint hypothesis holds and every flag is a false alarm. From
`results/summary.md` (20 seeds; RMSE kg / cov95 / nz, steady window):

- `kf+hard`, honouring `b_var`, reads 0.21 / 0.99 / 0.69 (whole run 0.26 / 0.99 / 0.76). It
  leaves a mean residual of 0.74 kg of the 0.91 kg it started from (`|res| post`;
  `mean_abs_res_pre` in `results/summary.json`): about a fifth is removed, because the
  declared 1 kg² is several times the filter's own sum-direction variance (A P Aᵀ ≈ 0.2 kg²,
  see the dead-band bullet above). `kf+hard+guard` is identical and never flags — FA
  0 / 0.0e+00, 0 steps held — and no spec in the scenario flags in any seed.
- What that buys over the unconstrained filter here: nothing measurable. `kf` reads
  0.21 / 0.99 / 0.66. A row whose declared uncertainty dominates the filter's own carries
  almost no information; what declaring it buys is that the row stops doing damage. The
  same σ_b = 1 kg treated as exact costs 0.41 / 0.68 / 1.84 on the `uncertain_total` axis of
  `results/sweep.md` (below), where the declared variant reads 0.22 / 0.99 / 0.72.
- **Feedback is not rescued.** `kf+hard+fb` reads 0.45 ± 0.24 / 0.63 / 1.73, with a
  row-direction error of 0.57 kg against the post-stage's 0.20. The filter is told the same
  offset b at every step as if it were a fresh, independent pseudo-measurement and converges
  onto it: its own mean residual before projection is 0.11 kg (`mean_abs_res_pre`) against
  the post-stage's 0.91. Its constraint flag never fires (FA 0); the evidence-side CUSUM
  does (`CUSUM FA max` 4 / 5) — the sensors disagree with a filter that has absorbed a wrong
  total.
- `kf_closedq+hard` reads what `kf_closedq` reads (0.11 / 0.97 / 0.81): with closure in Q
  the filter's sum-direction variance is far below 1 kg² and the declared row barely moves
  it (`|corr|` 0.04 kg).

**Exact against declared against soft, `uncertain_total`.** b = total0 + N(0, σ_b²) per seed;
`kf+hard` and `kf+hard+guard` treat it as exact, `kf+soft(λ = 1/σ_b²)` carries σ_b² as λ on
an exact set, `kf+hard(b_var)` and its guard get the same b with b_var = σ_b² on the set,
and their flags are scored as false alarms because their hypothesis holds. From
`results/sweep.md` (20 seeds, steady window):

| σ_b [kg] | kf+hard (exact) | kf+hard+guard (exact) | kf+soft(λ = 1/σ_b²) | kf+hard(b_var) | exact guard: det, held | b_var guard: FA, held |
|---|---|---|---|---|---|---|
| 0.1 | 0.17 / 0.98 / 0.75 | 0.17 / 0.98 / 0.75 | 0.17 / 0.98 / 0.73 | 0.17 / 0.98 / 0.73 | 0/20, 0 | 0, 0 |
| 0.3 | 0.21 / 0.96 / 0.92 | 0.21 / 0.96 / 0.92 | 0.19 / 0.99 / 0.74 | 0.19 / 0.99 / 0.74 | 0/20, 2 | 0, 0 |
| 0.5 | 0.26 / 0.88 / 1.15 | 0.26 / 0.89 / 1.13 | 0.20 / 0.99 / 0.73 | 0.20 / 0.99 / 0.73 | 0/20, 11 | 0, 0 |
| 1 | 0.41 / 0.68 / 1.84 | 0.33 / 0.81 / 1.43 | 0.22 / 0.99 / 0.72 | 0.22 / 0.99 / 0.72 | 4/20, 72 | 0, 0 |
| 2 | 0.75 / 0.37 / 3.34 | 0.41 / 0.67 / 1.75 | 0.23 / 0.99 / 0.72 | 0.23 / 0.99 / 0.72 | 10/20, 183 | 0, 0 |

Declaring `b_var` on the set gives, to every printed digit, the estimate the sweep used to
get from λ = 1/σ_b² on an exact set — the same pseudo-measurement in two algebraic forms —
and it gives the guard the right hypothesis: the declared guard never fires at any σ_b,
where the exact guard rejects the (false) exact hypothesis in up to 10/20 seeds within 100
steps, holds for 183 steps on average at σ_b = 2 kg, and still lands at 0.41 / 0.67 / 1.75.

**A wrong total with an honest-scale declaration, `declared_total_error`.** b is off by a
fixed δ; `kf+hard(b_var=0.25)` declares σ_b = 0.5 kg, the scale of the exact guard's dead
band. From `results/sweep.md` (20 seeds, steady window; det = seeds flagged within 100
steps):

| δ [kg] | kf+hard (exact) | kf+hard+guard (exact) | kf+hard(b_var=0.25) | kf+hard(b_var=0.25)+guard | det exact / declared | held exact / declared |
|---|---|---|---|---|---|---|
| 0 | 0.16 / 0.98 / 0.72 | 0.16 / 0.98 / 0.72 | 0.19 / 0.99 / 0.67 | 0.19 / 0.99 / 0.67 | FA 0 / FA 0 | 0 / 0 |
| 0.25 | 0.21 / 0.97 / 0.92 | 0.21 / 0.97 / 0.92 | 0.19 / 0.99 / 0.68 | 0.19 / 0.99 / 0.68 | 0/20 / 0/20 | 0 / 0 |
| 0.5 | 0.30 / 0.88 / 1.33 | 0.30 / 0.88 / 1.33 | 0.21 / 0.99 / 0.74 | 0.21 / 0.99 / 0.74 | 0/20 / 0/20 | 0 / 0 |
| 1 | 0.53 / 0.35 / 2.34 | 0.52 / 0.40 / 2.28 | 0.28 / 0.96 / 0.99 | 0.28 / 0.96 / 0.99 | 3/20 / 0/20 | 38 / 0 |
| 1.5 | 0.77 / 0.03 / 3.41 | 0.54 / 0.56 / 2.32 | 0.37 / 0.89 / 1.31 | 0.37 / 0.89 / 1.32 | 13/20 / 2/20 | 289 / 7 |
| 2 | 1.01 / 0.00 / 4.50 | 0.29 / 0.94 / 1.04 | 0.47 / 0.74 / 1.67 | 0.45 / 0.76 / 1.60 | 20/20 / 7/20 | 524 / 142 |
| 4 | 2.01 / 0.00 / 8.91 | 0.23 / 0.99 / 0.73 | 0.90 / 0.04 / 3.20 | 0.23 / 0.99 / 0.73 | 20/20 / 20/20 | 592 / 590 |

Declaring an honest σ_b narrows the damage where the error is within about two declared
σ_b: at δ = 0.5 and 1 kg the declared variant reads 0.21 / 0.99 / 0.74 and
0.28 / 0.96 / 0.99, where exact hard reads 0.30 / 0.88 / 1.33 and 0.53 / 0.35 / 2.34 and the
exact guard, silent or nearly so (0/20, 3/20), has nothing better to hand back. It costs part
of the exact gain when the total is right (0.19 against 0.16 at δ = 0; `kf` 0.23). And it
moves the dead band instead of removing it: the declared test is weaker, so its guard
catches 2/20 at 1.5 kg and 7/20 at 2 kg where the exact guard catches 13/20 and 20/20, and at
δ = 2 kg — four declared σ_b, i.e. a dishonest declaration — the declared variant is worse
than the exact guard (0.45 / 0.76 / 1.60 against 0.29 / 0.94 / 1.04). At 4 kg both guards
catch it in 20/20 and hand back 0.23 / 0.99 / 0.73, while the unguarded declared projection
is still confidently wrong (0.90 / 0.04 / 3.20).

What `b_var` does **not** show:

- **An efficiency gain from an uncertain constraint.** In `closed_uncertain_total`, `kf+hard`
  honouring b_var = 1 kg² reads what `kf` reads (0.21 against 0.21). `b_var` makes an
  uncertain row harmless and its test honest; it does not make it informative.
- **A measured in-loop null of the b_var statistic.** `results/calibration.md` was not
  extended to it. The in-loop evidence is zero flags from every spec over 20 seeds × 600
  steps of `closed_uncertain_total` and FA 0 for the declared guard at every σ_b of the
  `uncertain_total` axis: counts, not a rate estimate. The unconstrained filter's own
  in-loop statistic is deflated (mean 0.46–0.64 in `results/calibration.md`); a large
  declared Σ_b dilutes that deflation, it does not remove it.
- **That feedback becomes legitimate.** 0.45 / 0.63 / 1.73 fed back against 0.21 / 0.99 /
  0.69 as a post-stage: the declared uncertainty of b is one draw per declaration, not white
  noise per step, and a Kalman update repeated every step assumes the latter.
- **That any declared σ_b protects.** Only an honest one: a fixed 2 kg error declared as
  σ_b = 0.5 kg is over-confident (nz 1.67) and its guard is slower than the exact one.
- **Dependent uncertain rows, a correlated Σ_b, or more than one row in the loop.** The
  kernel refuses the first; the other two are exercised only by `tests/test_lcm.py`. Every
  constraint in the grid and the sweep is the single sum row.

## Before the bridge: no truth on the estimator side

A real record — the NOAA water levels the bridge will bring in from DAF's committed fixtures —
has no hidden truth, so the estimator side must not be able to receive one. This stage makes
that structural, removes the runner's two-sensor, two-state assumptions, carries evidence
provenance through the run, and adds the one evaluator that works without a truth. It changes
no number: a fresh grid equals the committed `results/summary.json` value for value (the
reproduction test; latency and provenance excluded), and fresh calibration and sweep runs
equal `results/calibration.json` and `results/sweep.json` the same way.

- **The runner takes public inputs, not a truth.** `run(inputs: PublicInputs, ...)`, where
  `PublicInputs(t, u_commanded)` is the step clock and the commanded input and
  `PublicInputs.from_truth` copies exactly those two fields, read-only. `runner.py` and
  `estimators.py` contain no reference to `Truth` (a test reads their source), and `run()` has
  no truth parameter. The oracle bound's hidden arrays arrive only through the keyword-only
  `oracle_inputs=(u_actual, leak)`; `run()` raises if they are passed for any kind but
  `"oracle"` or are missing for it. Experiment code is the one place that routes them
  (`phase1.oracle_inputs_for`, called by `phase1.run_spec`), and a second source test pins
  where the source reads a hidden field off a truth (`truth.m`, `.leak`, `.u_actual` on any
  name containing "truth"; a syntactic check, so a field reached another way would escape
  it): that function, the observation operator (`degrade.observe` measures the masses) and
  the scoring evaluator.
- **Shapes come from the record and the estimator.** The number of sensors is
  `obs[0].y.size`, the reported-state dimension the estimator's `n_report`; `RunResult`'s state,
  covariance, correction, innovation and CUSUM arrays are sized from those, and reconciliation
  acts on the first `n_report` components. An estimator may declare an extra component's
  nominal `None`: recorded, never flagged.
- **Evidence ids ride through the run.** `Observation.evidence_ids` (default `()`) names the
  admitted evidence an observation was built from; `RunResult.ingested_evidence[k]` lists the
  ids of everything ingested at report step k, in sampling order, and `RunResult.observed`
  which samples the estimator was given. No estimator reads the ids: a record carrying them is
  bit-identical in every estimate to the same record without (tested, with a stalled ingest
  clock).
- **A truth-free evaluator.** `evaluate_truth_free(run, windows)` reads nothing but the
  `RunResult` (listed in the module table). Two quantities in `results/` never needed the truth
  and it reproduces them: its per-window mean z is, to the bit, the `z̄ s1/s2` column (tested)
  — in `bias_quant_delay` sensor 1's moves from −0.01 before the bias to +0.16 after it while
  sensor 2's reads −0.00 in both windows, and in `closed_noise` both read +0.00 / +0.01 — and
  the in-loop null of the consistency statistic in `results/calibration.md` is computed from
  the estimate, its covariance and the declared constraint alone: the evaluator's per-seed
  mean statistic, averaged over the 20 seeds, gives that table's means (0.599, 0.644, 0.570
  and 0.463 over the four nominal windows; tested at 2 seeds).

What this does **not** show:

- **Anything on a real record.** No NOAA observation had been run at this stage. The next
  section brings DAF's NOAA evidence in; it still imports no DAF type or adapter at runtime,
  and nothing in the tree models a water level.
- **A truth-free number in `results/` beyond those two.** The grid does not run
  `evaluate_truth_free`; what it measures on simulated runs (RMS z, lag-1 autocorrelation,
  tail fractions) is stated in `tests/test_truth_free.py`, not in `results/`.
- **That innovations which look calibrated mean a calibrated filter.** The Q over-statement
  that reads nz = 0.73 against truth in `closed_noise` (steady window) is diluted in each
  sensor's innovation by R = 4 kg², which dominates S: in steady state with the pump off the
  filter states S ≈ 4.10 kg² against an actual ≈ 4.05 kg², an RMS z of about 0.99 by that
  arithmetic, within a few percent of 1. Where R dominates, per-sensor innovations are a weak
  witness of Q; the consistency statistic, which the over-statement enters undiluted, is the
  sharper one.
- **Whether an estimate is right.** A sensor bias and a real change the model does not
  explain write the same record; the evaluator names the sensor whose z moves, as the CUSUM
  does, not the cause.
- **Evidence handling.** The runner carries `evidence_ids` as opaque strings — not checked
  against any evidence store or resolved. The DAF bridge (next section) deduplicates identical
  readings and refuses or reports conflicting ones before anything reaches the runner, and
  `verify_ids` can have DAF recompute the ids; nothing here decides when a revised artifact
  should change the state.
- **General estimators.** The runner no longer assumes two sensors or two reported
  components, and a test-only three-sensor estimator exercises that path; every estimator in
  the tree still is the two-reservoir model with `n_report = 2`.

## DAF bridge

DAF — the Data Acquisition Fabric (https://github.com/atomtrapping/Data-Acquisition-Channel) —
acquires and admits evidence; this repository consumes it and owns none of it.
`set_lcm.bridge.daf` turns serialized DAF observations into what `run()` takes — `PublicInputs`
on a uniform clock and one `Observation` per grid point — and refuses what it cannot represent
faithfully. DAF stays read-only at a pinned commit, `6b37859` (vendored substrate
`vendor/scout-retrieval-agent` at `5e146d5`): nothing here writes to a DAF checkout, and nothing
on the bridging path imports DAF, SCOUT or the vendored evidence code (a test reads the module's
imports). This stage changes no simulated result: the grid, sweep and calibration code is
untouched, and nothing below is a number from `results/` — the counts are properties of the
committed evidence in `data/daf/`, pinned by `tests/test_bridge_daf.py`.

**Where the evidence comes from.** `tools/export_daf_fixtures.py` (it needs `DAF_ROOT`) runs
DAF's own per-measurement NOAA binding — DAF's unmodified `NoaaWaterLevelSourceAdapter` and
`NoaaWaterLevelMeasurementExtractor`, through `execute_plan` and SCOUT admission into a
`DurablePool`, as DAF's own live-observation test does — on DAF's committed fixtures, replaying
their bytes through the adapter's own `fetch_bytes` hook, and writes every admitted Observation
with DAF's `observation_to_dict`. `data/daf/` holds the result: NOAA CO-OPS station 8454000
(Providence, RI) on 2024-01-15 on the MLLW datum and on the STND datum, and a preliminary day
(2026-08-23, MLLW), 240 six-minute readings each; and two **SYNTHETIC** four-reading windows —
DAF's hand-written station 9999999, an original and a revised version — used only to exercise
conflicts. `data/daf/PROVENANCE.md` (generated, with `manifest.json`) records both commits,
each source fixture's path and sha256, the extractor and binding parameters (station, datum,
units and the `time_zone=gmt` of the adapter URL), and the command; NOAA CO-OPS data are U.S.
public domain. The fixture bytes are read from DAF's git object store, so a checkout's
line-ending conversion cannot change an id; re-running the tool reproduces every file byte for
byte, and the MLLW file's DAF document id begins `3bc9041f042eb48f`, the version id DAF's
Phase 17 transcript records for the live fetch.

**What the bridge refuses, and why.** `series`, `time_zone`, `cadence_s`, `arrival_policy`,
`conflict_policy` and `daf_commit` are keyword-only with no default (`latency_s` is required
with `"replay"` and refused with `"as_acquired"`; `declared_sigma` exists only where the caller
declares one). Every refusal raises — `BridgeRefusal`, a `ValueError` carrying a reason code and
the evidence ids involved, for anything about the evidence, the zone or the units; a plain
`ValueError` for a malformed argument — and there is no partial record.

- *series* — the (station, datum, unit) groups, one sensor column each. A record from an
  unlisted group is ignored and counted in provenance; a different datum or unit is never
  pooled: MLLW and STND of the same water surface become two columns that differ by 1.064 m at
  every one of the 240 points. One (station, datum) listed in two units is refused: the same
  readings would enter as two sensors, and the bridge converts no units.
- *time_zone* — DAF's `measurement_time` carries no zone (DAF keeps the source's event time in
  the content, the zone only in the adapter URL). Omitting it is a `TypeError`, `None` is
  refused (no naive parsing), and anything but UTC/GMT is refused; a `measurement_time` with a
  zone of its own is refused rather than reconciled.
- *cadence_s* — the grid runs from the first to the last reading; a reading more than 1 s off
  it is refused, every such reading listed. A wrong cadence is refused, not resampled: 720 s
  declared on the six-minute record refuses 120 readings. A grid point with no reading is
  `mask` False, `y` NaN.
- *uncertainty* — R_ii = `uncertainty`² where `uncertainty_kind` is `"stated"` (NOAA's `s`, as
  DAF's extractor records it). A reading without one is refused unless the caller passes
  `declared_sigma` for that series, recorded as consumer-declared; declaring one for a series
  whose readings state their own is refused too, so source-stated and consumer-declared R never
  share a series. A stated 0.000 passes through as R = 0 and is counted (`n_zero`: two readings
  on each 2024-01-15 datum), never floored.
- *arrival_policy* — `"replay"` with an explicit `latency_s` (arrival_t = t + latency_s), or
  `"as_acquired"`, arrival from each record's `extracted_at` on the same clock, refused where it
  is absent, naive or earlier than the measurement. Which one is recorded.
- *conflict_policy* — two records for the same (series, grid point) that disagree in value or
  in the uncertainty that becomes R: `"refuse"` raises naming both ids; `"report_and_keep_both"`
  leaves the point missing — neither value used, neither id ingested — and lists the conflict.
  On the SYNTHETIC pair, 2026-01-03 00:00 reads 1.200 (s 0.011) in the original and 1.207
  (s 0.006) in the revision: refused, or left missing with the conflict listed, while the three
  unchanged readings are deduplicated with both ids kept (8 records in, 6 used, 3
  deduplicated, 2 in conflict). The revision never silently wins.
- Also refused: a record that is not a per-measurement NOAA water-level observation (another
  extraction method, a missing field, a non-finite value), one evidence id carrying two
  contents, a listed series that matched nothing, and a file with a bare NaN / Infinity or a
  repeated key (the strict reader refuses what DAF's `strict_json_loads` refuses, and more).

**What it records.** `BridgedSeries.provenance`: the caller-supplied DAF commit; records in,
used, ignored (by group), in conflict (n_records_in = n_used + n_ignored + n_in_conflict) and
deduplicated; the conflicts; the grid check (tolerance, largest offset accepted); the time zone
declared and applied; the epoch (first grid point, ISO-8601 UTC) and cadence; the arrival
policy; `R_source` per series (source-stated with the σ range and zero count, or
consumer-declared with the σ); and a sha256 over the sorted evidence ids used. Each
`Observation` carries the series ids (`noaa:8454000:MLLW:m`) as `source_ids` and the DAF
observation ids behind its components as `evidence_ids` (per component in
`component_evidence`); `run()` carries them to `RunResult.ingested_evidence`, and on the MLLW
day with replay latency 0 a test-only random walk ingests exactly the 240 ids used, each once.
`verify_ids(records, daf_root)` is the optional check that imports DAF: DAF's own
`observation_from_dict` recomputes every committed id (728 of 728 at `6b37859`; the test skips
without `DAF_ROOT`) and a one-mm edit is caught.

**The DAF invariants it respects** (DAF's `docs/DAF_STATE_SPACE_BOUNDARY.md`, sections 10–13,
18). `t` is the source event time read out of `Observation.content`; `retrieved_at` /
`extracted_at` are never identity — `extracted_at` is read only as the `"as_acquired"` arrival
clock, and deduplication compares content, never stamps. Contradictory observations coexist
and are never averaged. The state-space side needs no `RawDocument`, adapter or DAF type.
Evidence identity is not model identity: DAF's ids ride as provenance that no estimator reads.
A revised artifact does not imply a state transition: the bridge reports the disagreement and
leaves the decision with the caller.

What the bridge does **not** do:

- **Network.** Nothing is fetched; every byte is a fixture committed to DAF.
- **DAF-side changes.** None; the export tool refuses a DAF checkout with tracked changes or a
  substrate off its pin, and writes no bytecode into it.
- **Model water level.** No estimator in the tree models a tide, and no number in `results/`
  comes from NOAA data. The one run in the tests is a test-only random walk that shows the
  plumbing, not an estimate.
- **Decide what a revision means.** A disagreeing revision is a conflict for the caller, not an
  update, and nothing chooses between preliminary and verified.
- **Claim NOAA's `s` is the error of the six-minute value.** It is the dispersion of the
  one-second samples behind it (waves included), reported to 1 mm; R = s² is the source's
  statement passed through, and a stated 0.000 becomes R = 0.
- **Give `"as_acquired"` a real acquisition clock on these files.** Their `extracted_at` is
  DAF's replay stamp (2026-08-25, 2026-08-26 for the revision), not when NOAA served the bytes,
  which DAF does not record; on the 2024 day nothing arrives within the day. The preliminary
  file's datum is the binding default, MLLW, because DAF does not record that request either.
- **Resample, interpolate, convert units, or read any other DAF extractor's content.** One
  content shape, one grid, one zone family.
- **Verify more than the observation id.** `verify_ids` recomputes each Observation's id from its
  own fields with DAF's code; it does not re-walk the record, document or raw bytes behind it.

## Deliberately out of scope

- **IMM (deferred, not rejected).** An interacting-multiple-model filter over {closed,
  leaking, mis-scaled pump} hypotheses would be a second, discrete answer to the question
  `kf_aug` already answers continuously with α and L and their own σ, at the price of
  declared mode-transition priors. `b_var` now puts uncertainty on the constraint side, so
  the two could be scored on the same statistic; that comparison is not built.
- **P2b: inequality constraints and `NOT_CONVERGED` producers.** `Status.NOT_CONVERGED`
  is reserved and nothing emits it because the closed-form kernel has no iteration to
  fail; an inequality row (m2 ≥ 0, a bounded flux) needs an active-set or QP solve, which
  is the first producer of that status and the first place "no solver failures" becomes a
  measurement.
- **P2b: feeding α̂ and L̂ back into the constraint as a prior.** The augmented filter
  estimates the flux the constraint author did not know about; a constraint set could
  carry it (b − ∫L̂ dt with its variance, now expressible as `b_var`) instead of going
  stale. Not built, deliberately: a constraint that follows the estimate re-opens the
  fed-back-projection question — the test stops disagreeing with what it tests — and it
  still needs a negative control of its own. `closed_uncertain_total`'s fed-back row is the
  warning: an uncertain b told to the filter at every step is absorbed as if it were fresh
  evidence (0.45 / 0.63 / 1.73, constraint flag silent).
- nonlinear constraints, multi-variable factors beyond one linear row
- out-of-sequence measurement handling beyond uniform delay
- the spectral processor, the Rust ingestion boundary, any manifold machinery
- claims about cross-platform bitwise determinism; determinism here means same seed → identical arrays on one build
