# Collaborator brief — FSRE

*Written for a second AI assistant (ChatGPT) joining this project alongside Claude, who has
been building it. Self-contained: you can work from this document without repository access,
though you will be more useful with it. Last updated at commit `a2ac1eb`.*

---

## 1. What this is, in one paragraph

**FSRE (Falsifiable State Reconstruction Engine)** reconciles state estimates against
declared conservation relations in fluid systems where a balance must close and the evidence
does not — reservoirs, river reaches, pipe and cooling networks. Its lineage is **data
validation and reconciliation** with a global χ² test (Crowe, 1985) and **constrained Kalman
filtering**. As built it is an *evidence-preserving reconciliation stage over a
state-estimation testbed*. Its tagline: **state estimates that ship with the statistic that
can reject them.**

**The use case has just been tightened to: a fluid sensor degradation estimator.** That is
the frame for everything below.

---

## 2. The mandate — read this before proposing anything

These five rules are enforced by code and pinned by tests. A suggestion that violates one
will be rejected, so check against them first. They are not style preferences; they are the
reason the project exists.

1. **Every estimate ships with the statistic that can reject it.** `reconcile()` returns a
   consistency statistic `rᵀS⁻¹r ~ χ²(rank A)`, its threshold, and residuals before *and*
   after projection. `detectability(f)` reports, per fault direction, what that statistic is
   structurally blind to.
2. **Nothing overwrites an observation or the unprojected estimate.** Disagreement between
   evidence and model is preserved, never reconciled away. A revision is not a state
   transition: contradictory readings are refused or reported, never averaged, and the later
   one never silently wins.
3. **An assumption without a source is refused, not defaulted.** A measurement variance the
   source did not state must be declared **with a citation**; a bare date must be given a
   zone **with a citation**. The bridge raises rather than guesses.
4. **The estimator side never reads hidden truth.** Pinned by an AST test. The single
   labelled exception is an oracle bound, via an explicit argument.
5. **Every reported number traces to `results/`, and every result says what it does not
   show.** Real-data reports are **truth-free** — nobody knows the true water level, so no
   number in them is an error, and each carries its own "cannot validate" list.

**Standing working rules:** `results/` is a verified artifact with a source hash and
reproduction tests; never tune configs or seeds to make numbers look better; identified
parameters are fitted on one window and evaluated on a disjoint one; the DAF repository
(a separate upstream) is **never pushed to**.

---

## 3. What exists today — measured, not estimated

**Scale.** Everything here is small. The largest matrix operation is a 9×9 solve. The
longest real record is 1,096 daily points; the tide-gauge records are 240 six-minute points.
The full simulated grid runs in about a minute of CPU. The project is **numpy-only**. *No GPU
or ML tooling is warranted at this scale — please do not propose any.*

**Machinery:**

| piece | what it does |
|---|---|
| `lcm/` | reconciliation kernel: χ² consistency statistic, hard/soft projection, feasibility, `detectability(f)`, declared constraint uncertainty (`b_var`) |
| `fdi.py` | **new** — whether two faults are *distinguishable* (see §4) |
| `testbed/cusum.py` | per-sensor CUSUM on normalised innovations; **reads no constraint**. Indicates a channel, not a faulty instrument — see §3 |
| `testbed/degrade.py` | injects known sensor faults into a simulator with hidden truth |
| `testbed/estimators*.py` | Kalman family; augmented filters carrying a pump scale α, a boundary flux L, an ungauged volume U as first-class outputs with their own σ |
| `bridge/daf.py` | consumes evidence admitted by DAF (a separate acquisition repo); refuses rather than defaults |
| `experiments/` | the simulated grid, calibration, sweeps, and two real-data reports |

**Real evidence committed:** NOAA tide gauge 8454000 (Providence RI) six-minute water levels;
Ridgway Reservoir CO daily means for storage, outflow and two inflows, WY2023–2025, 1,096
days each, gauged fraction 0.929.

**Key measured numbers** (all from `results/`, all reproducible):

- Ridgway model-free closure residual: mean **+0.703 ft³/s** against 169.5 ft³/s mean gauged
  inflow; three-year cumulative **+1,526 acre-ft = +0.41%** of inflow — **but that total is
  0.72 standard errors from zero** (sd 64.12 acre-ft/day over 1,095 days ⇒ se ≈ 2,122), so it
  is *not* evidence of a net imbalance. The daily statistic is what rejects.
- Declared-closure rejection: consistency statistic mean **9.69** vs χ²(1) threshold 10.83,
  exceeded on **38.6%** of days.
- The augmented filter is never rejected (mean 0.28) — it *cannot* be, which is what makes
  it an estimate rather than a test.
- **The in-loop null is not χ²(1).** Mean **0.463–0.644** against a nominal 1.0, lag-1
  autocorrelation **0.761–0.935**, autocorrelation time **7.4–29.7 steps**.
- On an injected +3 kg bias on sensor 1 (`bias_quant_delay`, `kf`), that sensor's CUSUM
  channel alarms **20/20 seeds at median 13.5 steps**, against the constraint test's 20/20 at
  **36.0**, with sensor 2 at **0/20**.
- **But CUSUM does not name the faulty instrument.** On `leak_stale_constraint` — a 10 kg
  *process* leak with `bias: null`, no sensor fault at all — it names sensor 2 at **20/20,
  median 59.5**, with the same signature it gives when correct. It indicates the channel whose
  own predictions broke. Treating that as instrument identification is the single easiest
  mistake to make with this tree.
- A declared storage σ *no source states* moves the rejection rate **52.0% → 38.6% → 13.3%**
  across its declared sweep (50/200/800 acre-ft).

---

## 4. The result that shapes the current direction

**Detection is not isolation, and rank(A) = 1 is why.**

Everything the constraint test knows about a fault arrives through the residual
`r = A x − b` with covariance `S = A P Aᵀ + Σ_b`. Whiten it: a unit fault along `f` has
signature `S^(−1/2) A f`, whose squared norm **is** `d(f)` and whose *direction* is
everything else the residual carries. Two faults with collinear signatures move the residual
identically — observing it is equally consistent with either, and only their ratio is ever
recoverable.

Every constraint in the repository has **rank(A) = 1** (the two-reservoir sum `[1,1]`, the
reservoir closure `[1,−1]`, the augmented closure `[1,−1,−1]`). A rank-1 `A` makes the
residual a **scalar**, so every signature lies on one axis and every detectable pair is
collinear by construction. Therefore:

> With rank(A) = 1, no fault is isolatable from any other. Not with a better covariance, not
> with more data, not with a longer record.

This is the classical statement that a *global* test on constraint residuals detects a gross
error without locating it. It is implemented and tested in `fdi.py`, including the converse
(rank 2 with non-collinear signatures *does* isolate) so it reads as a finding rather than a
missing feature.

**Two further blind spots, measured on the real topology:**
- The null direction of the balance — storage and cumulative gauged inflow rising together —
  has `d(f) = 0` exactly.
- **An equal bias on an inflow gauge and the outflow gauge cancels inside
  `(q_in1 + q_in2 − q_out)` before it reaches any state.** It is invisible to the balance
  *and* outside what `d(f)` can even score, because `d(f)` measures directions in state
  space and this never becomes one.

**What follows architecturally:** sensor-level localisation cannot come through the residual,
because the residual is one number. It must come from a channel that does not pass through
it — which is exactly what the per-sensor CUSUM channel is. The two are not redundant: the
constraint says *the system is inconsistent*, the per-sensor channel says *which instrument's
own predictions went wrong*.

---

## 5. The development plan under discussion

Ordered by what moves the product, not by the old roadmap:

1. **Multi-gauge river reach with Muskingum routing** — one constraint row per reach, so
   three gauges give rank 2 and isolation becomes possible for the first time. Still linear;
   the existing kernel handles it unchanged. *This is the item that changes the product's
   category.*
2. **Empirical null on real, autocorrelated records** (block bootstrap / surrogate series)
   → per-station thresholds with a *measured* false-alarm rate. Currently the real-data
   false-alarm rate is explicitly **unknown**. This is the deployability gate.
3. **Realistic degradation modes** in `degrade.py`: drift (slow ramp), fouling (progressive
   gain loss), rating shift (step change after a flood), ice effect (seasonal, correlated),
   datum step. Today it only has a *step* bias.
4. **Detection-delay curves** at a fixed false-alarm rate, per fault mode — the product's
   spec sheet.
5. **Join the sources' own QC flags to detector output, in the report only** — NOAA QC flags
   and USGS `ESTIMATED` qualifiers are a partial ground truth currently unused.
6. **Multi-site panel** (2–3 more reservoirs) — every real-data claim is currently one-site.
7. **Physical bounds as inequality rows** (storage ≥ 0, flow ≥ 0) — the cheapest unambiguous
   sensor-fault detector, and the declared first producer of the reserved `NOT_CONVERGED`
   status.

---

## 6. Failure modes to avoid — specific to this project

An assistant unfamiliar with the mandate will reliably make these mistakes. Please don't:

- **Invent a number.** If you don't have a source for an uncertainty, say so and propose a
  *sweep* over declared values, which is what the project does for the storage σ.
- **Plot or compute "estimate vs. truth" on real data.** There is no truth in the real-data
  reports. Two separate evaluators exist for exactly this reason.
- **Claim isolation.** Until rank(A) ≥ 2 with non-collinear signatures, naming which sensor
  is at fault from the constraint test is unsupported.
- **Treat the CUSUM channel as a second opinion on the constraint test.** It is the *only*
  localiser, precisely because it reads no constraint.
- **Propose GPU, NVIDIA, Omniverse, or ML.** The compute is 9×9 matrices. It would be
  cargo-culting.
- **Suggest a dashboard before the false-alarm rate is known.** A visualiser showing alarms
  at an unmeasured rate is worse than none.
- **Say "just use a rolling z-score / threshold on the raw series."** The whole point is that
  a fault must be distinguished from real dynamics, which is what the constraint and the
  innovation-based channel do and a raw threshold cannot.
- **Assume `Observation`/`PublicInputs` are dicts.** They are frozen, validating dataclasses.

---

## 7. Suggested division of labour

Claude has repository access, runs the tests, and makes the commits. The highest-value things
you can do are the ones that are *independent* of that:

**Best fit for you:**
- **Independent derivation and literature work.** Redundancy classification and
  observability in process networks, parity-space FDI, structured residuals, the classical
  results on how many simultaneous gross errors are identifiable. Give theorems with sources.
- **Cross-check the rank-1 result in §4 from scratch.** If it is wrong, the current direction
  is mis-set — this is the single most valuable thing you can check.
- **Design the Muskingum reach topology concretely:** state vector, the `A` matrix it
  produces, its rank, and which single-gauge biases become isolatable. Show the linear
  algebra.
- **Domain knowledge:** how stream gauges and reservoir stage–capacity relations actually
  degrade in the field; USGS/NOAA operational practice; what a real network operator already
  has on their desk and what they'd pay attention to.
- **Adversarial reading of the reports.** `results/real_water_balance.md` and
  `results/real_noaa.md` state conclusions and a "what these numbers do not show" list. Try
  to find a claim that overreaches.

**Leave to Claude:** writing code into the repo, running the suites, regenerating `results/`
(they are hash-verified artifacts), and anything touching the DAF acquisition path.

---

## 8. Concrete first questions

Checkable, and genuinely open:

1. Is the rank-1 isolation bound in §4 correct as stated? Derive it independently. Is there a
   case where a rank-1 constraint *can* distinguish two faults that I have missed — for
   example using the *time profile* of the residual rather than its instantaneous direction?
   (That last one is the most interesting loophole: a drift and a step both move a scalar
   residual, but not with the same shape over time. Does that recover isolation, and under
   what assumptions?)
2. For the Ridgway topology — four gauges, one conservation relation — what is the redundancy
   degree in the classical sense, and what is the maximum number of simultaneous gross errors
   identifiable? Cite the theorem.
3. Does adding per-sensor bias states to the augmented filter make gauge biases identifiable,
   or does it just relocate the unidentifiability into the state vector? Show the argument.
4. For a three-gauge reach with Muskingum routing: how many independent constraint rows, and
   which single-gauge biases become isolatable?
5. What are the real degradation signatures of a USGS stream gauge and of a reservoir
   stage–capacity relation — with sources — and which of them would the current CUSUM-on-
   innovations channel actually catch?

---

## 9. Where Claude is least confident

Please push hardest here:

- **The time-profile loophole in question 1.** The rank-1 bound is about the instantaneous
  residual direction. Sequential detection over a record may recover something the static
  argument forbids. This has not been worked out.
- **Whether "sensor degradation estimator" is the right product framing at all**, versus the
  narrower and more defensible "closure monitor that says when a gauge network stops adding
  up".
- **Whether the declared flow σ of 5% is defensible.** It is read from USGS's "Good" rating
  class (95% of daily values within 10%, read as 2σ), but the site-and-period rating was
  never acquired and the normality assumption is the consumer's.
- **The identifiability claim for the reach.** Asserted from the row count, not yet derived.

---

## 10. Provenance and honesty conventions

If you propose text that will end up in the repository, match these:

- Every number in a report traces to a `results/` file; the report and JSON are generated
  together so they cannot disagree.
- Consumer-declared uncertainties carry their citation verbatim through the bridge into the
  report, including a statement of which assumptions are the consumer's and not the source's.
- Reports state what they *cannot* validate as prominently as what they show.
- Where a result depends on a declared value with no source, it is **swept**, and every
  conclusion is reported at every point of the sweep.
