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

Results land in `results/summary.md` and `results/summary.json`. Every scenario is
run over 20 seeds (simulation and degradation seeds offset together); tables report
mean ± sd across seeds and the JSON keeps every per-seed metric. A single seed is a
realization, not a result: the first single-seed run of the noisy-valve scenario put
`kf+hard` within 8 % of `kf` during the blackout (0.386 vs 0.419 kg); over 20 seeds it
is 0.55 ± 0.18 vs 0.92 ± 0.44 kg.

## What is in the slice

| Module | Responsibility |
|---|---|
| `set_lcm/schema.py` | `Observation`, `ConstraintSet`, `StateEstimate` envelope with `status`, unprojected state, correction, residuals pre/post, consistency stat. |
| `set_lcm/simulator.py` | Two-reservoir material transfer. Hidden `m`, hidden leak; public commanded pump rate and declared initial total. |
| `set_lcm/degrade.py` | Noise, random dropout, sensor blackout, undeclared bias, quantization (declared into `R`), uniform arrival delay. Seed-controlled. |
| `set_lcm/estimators.py` | Hold-last baseline; linear Kalman filter with a *diagonal* Q (closure is the constraint's declared claim, not the filter's). Delay handled as forward prediction from the last arrived state. |
| `set_lcm/lcm.py` | Hard (KKT closed form) and soft (penalty) linear-equality projection weighted by `P⁻¹`; feasibility check; χ² consistency statistic; `reconcile()` that never mutates its input. |
| `set_lcm/runner.py` | Runs a (scenario, estimator) pair step by step; debounced consistency flag; optional guard that *holds* projection and reports `MODEL_INCONSISTENT`. |
| `set_lcm/evaluate.py` | RMSE (overall and windowed), 95 % interval coverage, residuals, correction magnitude, false alarms, detection delay, solver failures, latency. |

## Scenarios

All five declare the same constraint, `m1 + m2 = 100 kg`, version `closed-boundary-v1`.

| Scenario | Constraint is | What it tests |
|---|---|---|
| `closed_noise` | true | Does a correct constraint reduce error? Does soft leave a residual where hard does not? |
| `closed_blackout_pumpbias` | true | Sensor 2 dark for 100 steps while the pump delivers 20 % more than commanded (a wrong *parameter*). Does the constraint help when the measured reservoir itself lags? |
| `closed_blackout_noisy_valve` | true | Sensor 2 dark for 300 steps while an unmodeled valve moves mass at random (pure observability loss). Does the constraint carry sensor 1's information into the dark reservoir? |
| `leak_stale_constraint` | **stale** from step 300 | 10 kg leaks out. Does hard projection make the estimate confidently wrong? Does the guard notice and stop enforcing? |
| `bias_quant_delay` | true, evidence is not | Sensor 1 gains an undeclared +3 kg bias. Does the guard flag it? Note it cannot tell bias from leak. |

Estimator variants: `hold_last`, `kf`, `kf+soft(1/λ = 4 kg²)`, `kf+hard`, `kf+hard+guard`.

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

- **"A true constraint cuts KF RMSE 0.28 → 0.20 kg" is not a finding.** It is the
  analytic 1/√2 of averaging two equal-noise sensors, already reproduced in
  `tests/test_lcm.py`.
- **A stale constraint is the negative control that matters.** After a 10 kg leak,
  hard projection reports 3.80 kg RMSE against 1.07 unconstrained while its
  post-projection residual is 1e-16: numerically perfect, physically wrong. The guard
  flags at ~65 steps, holds projection, and lands at 1.12.
- **Hard projection is over-confident by construction.** It sets `A P Aᵀ = 0`, i.e. it
  asserts the constraint is exactly true. Its in-window coverage is 0.02–0.09 in
  every fault window; the whole-run cov95 column (0.36–0.67) hides that.
- **The consistency statistic is not χ²(1) in the loop.** Its steady-state mean is
  ≈0.42, deflated because the diagonal Q asserts sum-direction process noise that
  the closed simulator never generates. The pre-fault count of one flagged step in
  6,000 is therefore a property of a deflated statistic, not of the guard.
- **A single sum constraint is blind to the difference direction.** For
  `A = [1, 1]` and a pump or valve fault along `[−1, 1]`, the detectability
  `fᵀAᵀ(APAᵀ)⁻¹Af` is exactly zero. Those faults are structurally undetectable by
  this test — coverage collapses with no flag in 18–19 of 20 seeds — not "missed".
- **The guard has a dead band.** A declared-total error of 0.5–1.5 kg already makes
  hard projection worse than the unconstrained filter and does not trip the guard.
  The leak (10 kg) and bias (3 kg) scenarios only show the easy regime.
- **Two truth leaks.** The estimator is initialised from the simulator's exact initial
  mass, and `Observation.arrival_t` is never consumed (delay is passed out of band).
- **Not evidence of anything:** "no solver failures" (nothing can emit
  `not_converged` yet); the latency column (interpreter overhead on 2×2 matrices on
  one laptop); determinism beyond same-process; the soft variant as a meaningfully
  different estimator at 1/λ = 4 kg² against S ≈ 0.2 kg².

## Deliberately out of scope for Phase 1

- inequality constraints, nonlinear constraints, multi-variable factors beyond one linear row
- feeding the projected state back into the filter (projection is a post-stage here; the unprojected state is always retained)
- out-of-sequence measurement handling beyond uniform delay
- the spectral processor, the Rust ingestion boundary, any manifold machinery
- claims about cross-platform bitwise determinism; determinism here means same seed → identical arrays on one build
