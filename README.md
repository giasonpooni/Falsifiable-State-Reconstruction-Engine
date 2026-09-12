# Falsifiable State Reconstruction Engine

**Tells you when a network of fluid sensors stops adding up — and tells you what it cannot
see.**

Python 3.12+, numpy only. No network at runtime. No GPU. `results/` is a verified artifact.

---

## The problem

A reservoir has a storage gauge, an outflow gauge and two inflow gauges. Conservation says
the four must agree: whatever came in, minus whatever went out, is the change in storage.
They never quite do, and the interesting question is whether the gap is the water or the
instruments.

Conventional monitoring answers "that reading looks odd". This answers:

> Over three years the imbalance accumulates to **+1,526 acre-ft (+0.41% of the 368,576
> acre-ft that flowed in)** — but the daily residual has **sd 64.12 acre-ft/day over 1,095
> days**, so that total carries a standard error of 64.12·√1095 ≈ **2,122 acre-ft** (wider
> still at the measured lag-1 of +0.173) and sits **inside one standard error of zero**. The
> cumulative total is *not* the finding. What rejects is the **daily** statistic: `rᵀS⁻¹r` has
> mean **9.69** against a per-day χ²(1) threshold of **10.83** at q = 0.999, exceeding it on
> **38.6% of days**, where under the declared hypothesis its mean would be **1**. Here is the
> residual before and after correction, here is what the correction was, and here are the
> faults this configuration is **structurally blind to**.

Those are real numbers, from `results/real_water_balance.md`, on 1,096 days of USGS gauge
records for Ridgway Reservoir, Colorado.

## What it computes

Four questions, four distinct mechanisms — and the fourth is the unusual one.

| question | mechanism | measured today |
|---|---|---|
| **Is something wrong?** | consistency statistic `rᵀ S⁻¹ r ~ χ²(rank A)` on the constraint residual `r = A x − b`, with `S = A P Aᵀ + Σ_b` | rejects the Ridgway closure on 38.6% of days |
| **Which instrument?** | per-sensor CUSUM on normalised innovations — **reads no constraint** | on an injected +3 kg bias on sensor 1 (`bias_quant_delay`, `kf`): that sensor's channel alarms **20/20 seeds, median 13.5 steps**, against the constraint test's 20/20 at **36.0** — and sensor 2's channel stays **0/20**, so it names the instrument |
| **How much?** | an augmented state carrying the unexplained term (pump scale α, boundary flux L, ungauged volume U) as a first-class output with its own σ | ungauged net inflow +2,068 acre-ft (+0.56%), against +1,526 from model-free arithmetic |
| **What can't I see?** | `detectability(f) = fᵀAᵀS⁻¹A f`, and `fdi.isolability()` on whether two faults are *distinguishable* | `d(f) = 0` exactly along null(A); **no fault in the tree is isolatable** — see below |

## Quickstart

```bash
uv run --python 3.13 --dev pytest -q                # fast suite
uv run --python 3.13 python run_experiments.py      # regenerates results/summary.*
```

Two tanks, one conservation law, an estimate that does not satisfy it:

```python
import numpy as np
from set_lcm.schema import ConstraintSet
from set_lcm.lcm import reconcile, chi2_quantile
from set_lcm.fdi import isolability

# The total is 100 kg -- but that is itself a measurement, known to +/- 0.5 kg, so the
# constraint declares its own uncertainty rather than claiming to be exact.
cs = ConstraintSet(version="total-v1",
                   A=np.array([[1.0, 1.0]]), b=np.array([100.0]),
                   b_var=np.array([0.5 ** 2]),
                   description="m1 + m2 = 100 kg")

x, P = np.array([52.0, 46.0]), np.diag([1.0, 1.0])   # 2 kg unaccounted for
est = reconcile(x, P, cs, mode="hard", threshold=chi2_quantile(cs.rank, 0.999))

est.status             # Status.OK
est.consistency_stat   # 1.78   -- against a threshold of 10.83, so not rejected
est.residual_pre       # [-2.0]      the disagreement
est.residual_post      # [-0.222]    NOT zero: b carried uncertainty, so it is not forced
est.correction         # [0.889, 0.889]
est.x_unprojected      # [52.0, 46.0]  -- always kept, never overwritten
```

Now ask what this configuration could ever have distinguished:

```python
isolability({"leak_from_tank_1": [1.0,  0.0],
             "leak_from_tank_2": [0.0,  1.0],
             "transfer_1_to_2":  [1.0, -1.0]}, P, cs)
```

```
d         {'leak_from_tank_1': 0.444, 'leak_from_tank_2': 0.444, 'transfer_1_to_2': 0.0}
invisible ['transfer_1_to_2']
isolable  []
  leak_from_tank_1 vs leak_from_tank_2   cos=1.0   distinguishable=False
```

A transfer between the tanks is **exactly invisible**: it does not change the total, so no
covariance and no amount of data will ever reveal it. And the two leaks, though both visible,
produce *identical* residual signatures — the statistic cannot tell them apart.

## Know this before you use it: detection is not isolation

Everything the constraint test knows about a fault arrives through the residual `r`. Whiten
it, and a unit fault along `f` has signature `S^(−1/2) A f`, whose squared norm **is** `d(f)`
and whose *direction* is everything else. Two faults with collinear signatures move the
residual identically — observing it is equally consistent with either, and only their ratio
is ever recoverable.

Every constraint here has **rank(A) = 1**, which makes the residual a *scalar*. Every
signature then lies on one axis, so:

> With rank(A) = 1, no fault is isolatable from any other. Not with a better covariance, not
> with more data, not with a longer record.

This is the classical result that a *global* test on constraint residuals detects a gross
error without locating it (Crowe, 1985) — computed here for the topologies actually declared
rather than quoted. `fdi.isolability()` refuses to report an isolation the rank forbids, and
the converse is tested: rank 2 with non-collinear signatures *does* isolate.

**It follows that the per-sensor channel is not a second opinion — it is the only localiser.**
The constraint says *the system is inconsistent*; CUSUM on each sensor's own innovations says
*which instrument's predictions went wrong*. Neither substitutes for the other.

## Use case: a fluid sensor degradation estimator

Gauge networks drift, foul, take rating shifts after floods, and get re-datumed at
maintenance. The product this is built toward answers, for a real network: *is an instrument
degrading, which one, by how much, and how long until you would have known* — and states what
it could never have caught.

Two things make that credible here rather than asserted:

- **Injected-fault validation.** `testbed/degrade.py` puts *known* faults into a simulator
  with hidden truth, so detection delay and false-alarm rate can be measured against the
  answer. The same code then runs truth-free on real gauges where nobody knows the answer.
- **A stated blind spot.** `d(f)` and `isolability()` say up front what a sensor topology
  cannot reveal — including, at Ridgway, that an equal bias on an inflow gauge and the
  outflow gauge cancels inside `(q_in1 + q_in2 − q_out)` *before it reaches any state*, so it
  is invisible to the balance and outside what `d(f)` can even score.

**Where it is not ready**, stated plainly: the real-data false-alarm rate is **unknown**. No
window of the real record is *known* to be fault-free — the reports are truth-free — so there
is nothing on the gauges to calibrate a null against. On the simulator's nominal windows,
where truth is known, the same statistic computed in the loop has mean **0.463–0.644** where
an exact χ²(1) gives 1.0, and lag-1 autocorrelation **0.761–0.935** where independent draws
give 0 (autocorrelation time 7.4–29.7 steps). There the null is measurably not the
distribution the 10.83 threshold comes from; on the real record it has not been measured at
all. So the 38.6% rejection rate **cannot be converted into a false-alarm probability in
either direction** — nothing here says whether the tool over- or under-alarms on those
gauges. The CUSUM threshold (k = 0.5, h = 8) was likewise calibrated on simulated records.
**Do not wire either channel to an alarm until both are calibrated on real records.** Fixing
that, and reaching rank(A) ≥ 2 so isolation becomes possible, are the next two pieces of work
([`docs/COLLABORATOR_BRIEF.md`](docs/COLLABORATOR_BRIEF.md) §5).

## The mandate

Five rules, each enforced by code and pinned by a test:

1. **Every estimate ships with the statistic that can reject it** — and with `d(f)`, which
   says what that statistic is blind to.
2. **Nothing overwrites an observation or the unprojected estimate.** Contradictory readings
   are refused or reported, never averaged; a revision is not a state transition.
3. **An assumption without a source is refused, not defaulted.** An uncertainty the source did
   not state, or a time zone for a bare date, must be declared *with a citation*.
4. **The estimator side never reads hidden truth.** Pinned by an AST test; the one labelled
   exception is an oracle bound, passed explicitly.
5. **Every reported number traces to `results/`, and every result says what it does not
   show.** Real-data reports are truth-free: nobody knows the true water level, so no number
   in them is an error.

Where a result depends on a value no source states, it is **swept** and reported at every
point — as the Ridgway storage σ is, moving the rejection rate 52.0% → 38.6% → 13.3%.

## Map

```
src/set_lcm/
  schema/        Observation, ConstraintSet (with b_var), StateEstimate, Status
  lcm/           the kernel: consistency statistic, hard/soft projection, detectability
  fdi.py         isolability: whether two faults are distinguishable at all
  testbed/       simulator, degradation injection, estimators, runner, CUSUM,
                 truth-scoring and truth-free evaluators
  bridge/daf.py  consumes evidence admitted by DAF; refuses rather than defaults
  experiments/   the simulated grid, calibration, sweeps, and the real-data reports
data/daf/        committed evidence, and the raw recorded responses it replays from
results/         verified artifacts: every number in every report
docs/            the long form
tools/           evidence acquisition and export (the only code that touches a network)
```

## Deeper reading

- [`docs/RESULTS.md`](docs/RESULTS.md) — every result in the order it was built, with what
  each does and does not show. This is where the working is.
- [`docs/COLLABORATOR_BRIEF.md`](docs/COLLABORATOR_BRIEF.md) — the fastest complete
  orientation, including the open questions and where the author is least confident.
- [`HANDOFF.md`](HANDOFF.md) — session-to-session state.
- `data/daf/PROVENANCE.md` — where every committed reading came from, under which upstream
  commit, and which of those commits are published.

Scope note: "state reconstruction" here means estimating a dynamical system's state from
measurements — not quantum-state tomography, and not the axiomatic reconstruction of a
physical theory. The name is a **program goal**, not a delivered property; the results say
exactly where the tests are blind.

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
- **A 31-day live fetch.** The next step for real data is a month of six-minute readings fetched
  live through DAF's own NOAA adapter — enough to separate S2 and N2 from M2 and K1 from O1 — and
  it needs the user's go-ahead: nothing in this repository or its tests makes a network request.
- **Localising the imbalance.** P4b's constraint is real and it is rejected, but a global χ²
  test on constraint residuals cannot say which of the ungauged catchment, evaporation, the
  stage–capacity table or a gauge rating is responsible. That is the documented limit of the
  method (Crowe, 1985), and separating them needs a second, independent measurement of one of
  the same quantities — another gauge on the same reach, or a lake-evaporation record.
