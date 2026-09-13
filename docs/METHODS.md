# Methods and interpretation

FSRE combines established estimation and reconciliation methods with preserved evidence,
explicit assumptions and reproducible experiments. Numerical results are in the generated
[reports](../results/); future work is in the [roadmap](ROADMAP.md).

## Reconciliation and the reference null

For an unprojected estimate `xhat` with covariance `P`, declare `A x = b`. The kernel uses

```text
r = A xhat - b
S = A P A' + Sigma_b
T = r' S^-1 r
```

`Sigma_b` is the declared reference covariance; omitting `b_var` declares the reference
exact. This covariance addition assumes independent estimate and reference errors.

If `r` is centered Gaussian and `S` is its correct covariance, `T` has a chi-square
distribution with degrees of freedom equal to the number of independent residual
components, `rank(A)` for the supported independent constraint rows. This is a mathematical
result under a calibrated null, not an unconditional property of the filter's reported
covariance. Biased evidence, misspecified dynamics, fitted uncertainty and dependence can
invalidate it. Temporal correlation changes repeated-test, debounce and alarm-episode
behavior even when the marginal reference distribution is correct.

The test challenges a conjunction: the physical relationship, estimation model and
uncertainty description must jointly explain the evidence. Rejection does not identify
which part failed; non-rejection does not prove the state is correct.

Exact hard reconciliation minimizes the correction weighted by `P^-1` subject to `A x=b`.
It can make a residual zero even under a wrong relationship. With `b_var` declared, hard
reconciliation is a Gaussian pseudo-measurement update with noise `Sigma_b`, so its residual
need not vanish. Soft reconciliation adds slack `I/lam`, for positive `lam`. The runner's
guard separately decides whether to withhold correction. An `OK` status describes an
applied update, not certification of the measurements. See [Usage](USAGE.md).

## Static fault geometry uses the residual vector

Assume exactly one candidate fault `i`, a fixed state direction `f_i`, and an unknown,
unrestricted signed amplitude `a`. Its local response is

```text
r = a A f_i + noise
w_i = S^(-1/2) A f_i
d(f_i) = ||w_i||^2
```

Structural invisibility is `A f_i=0`. Positive but small `d(f_i)` indicates a small response
relative to the declared uncertainty at the supplied magnitude, not nullspace membership.
Under Gaussian noise with unchanged covariance, `a^2 d(f_i)` is the alternative statistic's
noncentrality parameter; it is not a guaranteed increase in each observed `T`.

Unrestricted signed amplitude makes the possible noiseless residuals a line `span(w_i)`.
Collinear nonzero signatures cannot distinguish competing candidates instantaneously. With
one effective residual dimension, all visible pairs are collinear. Noncollinear lines
permit noiseless single-fault separation, but do not establish finite-noise classification
accuracy. Known signs, magnitudes or other restrictions can change the conclusion.

`fdi.py` implements this static comparison. It does not implement temporal classification,
simultaneous-fault recovery or an operational diagnostic decision rule. The scalar `T`
discards direction: with `S=I`, `[3,0]` and `[0,3]` both produce `T=9`. Isolation based on
different directions must retain and use the residual vector.

Each `FaultPair` also exposes `orthogonal_fraction`, the sine of the angle between the
whitened signatures, and `isolation_amplification`, its reciprocal. For a fraction of
0.002, only about one five-hundredth of a signature lies outside the alternative fault
line. This measures geometric separation; it does not set an alarm threshold or guarantee
correct attribution. The values depend on covariance and do not change the structural
labels. `as_dict()` archives them beside `cos` for every pair, because archiving
`distinguishable` without the separation behind it records a yes/no that reads stronger than
the geometry supports. Pairs containing an invisible fault have undefined angular
diagnostics, which serialize as non-finite values a caller must map itself.

`diagnostics.diagnose()` reports the same thing for a finite record rather than a static
structure. Each observable candidate carries its nearest rival in the whitened post-nuisance
coordinates the test used — the cosine, the fraction of its own signature that rival cannot
explain, and the reciprocal of that fraction — and `DiagnosticResult.min_separation` is the
tightest such pair in the catalogue. Every verdict states it, `identified` included: an
identification drawn from a catalogue whose closest pair is barely separated is not the claim
it appears to be. An exactly collinear pair reports separation `0.0` and no amplification,
because no amplitude separates it; the committed camera baseline is that case, and its
declared gauge-drift and camera-drift candidates differ only in sign.

For a sensor-space fault dictionary `D` in `r=D e`, uniqueness of every explanation with
at most `s` nonzeros requires no nonzero vector in `null(D)` with support at most `2s`.
Otherwise it can be split into two competing sparse explanations. For an underdetermined
dictionary this is the familiar `spark(D)>2s` condition. Multiple nonzero columns in one
residual dimension cannot locate an arbitrary unknown single error. A known faulty location
can still support amplitude estimation under the other assumptions. This is not a bound
on temporal or sign-constrained models.
[Donoho and Elad, original paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC153464/).

The static module's fault directions live in state space. The new measurement baseline maps
raw sensor profiles through an interval operator before diagnosis. This representation must
retain information about cancellation: equal additive inflow and outflow biases, for
example, cancel in net flow. The balance channel cannot see that cancellation; other
evidence may still reveal a change.

## Temporal information and nuisance processes

Rank one at each instant does not imply rank one over a record. Suppose a known onset
introduces a persistent storage offset `beta_S` and constant flow-rate bias `beta_q`.
In consistent time and volume units, a simplified cumulative balance is

```text
r_t = beta_S - t beta_q + epsilon_t
```

Distinct times give a stacked design matrix with rows `[1,-t]`, of rank two. For
`t=1,...,5`, noiseless residuals `[1,-1,-3,-5,-7]` identify `beta_S=3`, `beta_q=2`.
Whitening with a nonsingular joint temporal covariance preserves rank.

This counterexample does not establish isolation at Ridgway. It assumes known baseline/onset,
specified fault profiles and an adequate physical model. Storage drift can mimic flow bias;
ungauged inflow can mimic both. Distinguishing a step from a ramp does not identify a sensor
if either sensor can produce either profile. Finite-horizon analysis must account for
nuisance responses and conditioning as well as rank.

Adding bias states creates a representation, not new evidence. In a constant system with
`y=x+b`, the augmented measurement matrix `[I,I]` cannot distinguish changes `[delta,-delta]`.
A prior may choose among explanations without identifying them from data. Known different
dynamics or independent references may resolve the ambiguity.

## Finite imbalance uncertainty and feedback

The augmented water-balance relation is `S_storage-G-U=S0`, with cumulative gauged volume
`G` and modeled unexplained volume `U`. Some `U` can satisfy it algebraically. Its finite
prior and process variance nevertheless allow sufficiently large disagreement to reject
the model statistically. Non-rejection on one record is conditional on those scales.

Gauge observations alone do not update `U` in this model. Reconciliation moves its reported
value, and feedback carries the change forward. `U` can absorb ungauged water, evaporation,
precipitation, gauge bias and storage-model error. Its posterior standard deviation is
conditional on that model, not an independently validated error bar on ungauged inflow.

Balance feedback preserves the conditional distribution of rate states given the reconciled
marginal, updating rate means and cross-covariances too. It matches a full-state Gaussian
constraint update; see [feedback tests](../tests/test_balance_feedback.py). This fixes
covariance consistency, but does not make repeated use of the same uncertain reference
independent evidence.

## Measurement baseline and remaining filter gaps

The [implemented fluid baseline](FLUID_BASELINE.md) now provides a direct measurement
calculation `r=H y`, `C=H Cov(y) H.T`, with matched interval support and shared-reference
covariance. It tests conditional fault explanations over a fixed horizon after projecting
out declared nuisance effects. Its analytical and synthetic validation does not calibrate
the real records or retrofit the historical estimator study. The following limitations
continue to apply to that study's filters and reconciliation path.

Ridgway's reference `b=S0` is the first storage reading, which also enters the estimate.
For shared evidence the general residual covariance is

```text
Var(A xhat - b) = A P A' + Var(b)
                 - A Cov(xhat,b) - Cov(b,xhat) A'
```

`b_var` currently represents `Var(b)`, not these cross terms. Their influence can decay
without feedback, but must be modeled before claiming calibrated rejection probabilities.
An explicit shared initial-state reference remains a possible filter design. Flow and storage
uncertainty, process noise, serial correlation and common errors also need sensitivity
analysis; a citation alone does not calibrate them.

Daily-mean storage is not an instantaneous day-boundary reading. With net flow constant
within each day, conservation gives

```text
Sbar[k+1] - Sbar[k] = c * (q[k] + q[k+1]) / 2
```

Here `c` converts daily flow to volume. Pairing the difference with only `q[k]` produces
an apparent residual `c*(q[k+1]-q[k])/2` even with perfect sensors. General within-day flow
needs the appropriate averaging operator or subdaily data. The experiment compares
arithmetic alignments; the full filter and guard still use their declared same-day
approximation. Smaller scatter after averaging does not prove correct timing because
averaging also changes measurement noise. The new measurement baseline applies the correct
averaging operator under the stated within-day model; a general filter treatment remains
a method-development gap.

The uncertainty of a cumulative residual needs its joint temporal covariance:
`Var(sum(r)) = 1' Cov(r) 1`. The shortcut `sd(r) * sqrt(n)` assumes independent,
equal-variance errors. Positive lag-1 correlation alone does not bound the contribution
from all other lags. Residual scatter can also contain real dynamics and model error;
it is not automatically calibrated measurement uncertainty. A cumulative total alone
therefore does not establish a statistically significant physical imbalance.

## Planned three-gauge Muskingum design

**This design is not an implemented reach model or a validated capability.** For a
fixed-parameter reach use

```text
q_down,t = C0 q_up,t + C1 q_up,t-1 + C2 q_down,t-1
C0 + C1 + C2 = 1
```

These are the standard Muskingum coefficients. Parameters must be declared or fitted on
separate data, with uncertainty and applicability assessed.
[USACE formulation](https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/channel-flow/muskingum-model).

For successive reaches and measurement vector
`[q1,t,q2,t,q3,t,q1,t-1,q2,t-1,q3,t-1]`, the two residual rows are

```text
A = [ -C0,   1, 0, -C1, -C2,   0 ]
    [   0, -D0, 1,   0, -D1, -D2 ]
```

Persistent additive biases `e=[e1,e2,e3]`, present at both times, give

```text
D_bias = [ -a,  a, 0 ]       a = 1-C2
         [  0, -b, b ]       b = 1-D2
```

For nonzero `a,b`, this dictionary has rank two and nonzero, pairwise noncollinear columns.
The residual vector therefore distinguishes each nonzero single-gauge bias from the other
single-gauge alternatives in this noiseless model. Two arbitrary simultaneous biases are
not uniformly identifiable; the common offset `[1,1,1]` is invisible to the balance.

Two arbitrary independent lateral inflows can occupy both residual dimensions and mimic
faults. Routing errors can resemble faults too. The shared middle gauge, reused lagged
samples and parameter uncertainty require joint covariance. Adding rows alone neither
supplies that covariance nor implements isolation in the presence of nuisance processes.

### Alternative planned balance: cooling-loop energy

A second relation could instead come from energy conservation over the same fluid pipes.
Mass flow multiplied by enthalpy contributes to energy flow; heat exchange and changing
stored energy must also be represented. A second equation supports diagnosis only when
candidate fault signatures remain distinguishable after these nuisance terms and state
observability are considered.

If temperature-derived enthalpies are used as measured coefficients, their uncertainty
creates an errors-in-variables problem. Shared measurements can correlate coefficient
errors, flow errors and the mass/energy residuals. A joint nonlinear state model is another
possible design, and remains unimplemented.

`ConstraintSet.A_var` now declares that uncertainty, and the consistency statistic uses it.
With `A = A_bar + E` the residual gains `E x`, so `S` gains

    Cov(E x)_ij = x^T Sigma_ij x + tr(Sigma_ij P)

where `Sigma_ij` is the declared covariance of rows `i` and `j` of `A`. Unlike `Sigma_b` this
is a **quadratic form in the state**: the same constraint set has a different residual
covariance at a different operating point, and `detectability` stops being a property of the
set alone — it requires a state and raises without one.

Omitting the term is not conservative, it is wrong in the rejecting direction, because the
term belongs in the denominator. Measured on a null that is true by construction
([`results/errors_in_variables.md`](../results/errors_in_variables.md)): treating an uncertain
relation as exact rejects a healthy system at **26 to 30 times** the nominal rate, with the
statistic's mean at 7.1–8.3 against a rank of 2. A full declaration is calibrated at 1.2x.

Independent coefficient variances alone still do not represent dependence between rows, and
the size of that shortcut is measured rather than asserted: where the rows share evidence, a
per-row declaration rejects at 2.1x nominal against a full `vec(A)` declaration's 1.2x. That
is much smaller than ignoring `A` altogether, which is an argument for the shortcut where
evidence genuinely is not shared, not for using it everywhere.

Two things the correction does not do. It is first order in the coefficient error — the
residual miscalibration is measured shrinking from 1.24x to 0.98x as the declared variance
falls a hundredfold, which is what a dropped second-order term does. And it corrects the
consistency **test** only: reconciliation still solves against `A_bar` as though exact, so a
reported correction stays conditional on that. Nothing declares dependence between `A`'s
error and `b`'s, or between `A`'s error and the state estimate.
The measurement model must represent significant uncertainty sources and their correlations;
see [NIST's law of uncertainty propagation](https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-appendix-law-propagation-uncertainty).

## From channel anomalies to validated diagnostics

CUSUM detects persistent prediction disagreement. Its channel identifies where disagreement
is observed, not necessarily its cause. Mean-shift monitoring is not a universal detector
of variance growth, lost response or timing errors. Physical bounds require justified
quantity definitions; a violation alone does not prove failed hardware.

The committed [simulation report](../results/summary.md) supplies a counterexample:
`leak_stale_constraint` injects a process leak with no sensor bias, yet the ordinary KF's
sensor-2 CUSUM alarms within 100 steps in all 20 seeds, with median delay 59.5 steps.
Identifying the channel that disagrees is therefore insufficient to identify faulty hardware.

Discharge often depends on a stage-rating relationship changed by erosion, deposition,
vegetation, debris or ice while the stage sensor still functions.
[USGS streamgaging practice](https://www.usgs.gov/mission-areas/water-resources/science/streamgaging-basics).
Reservoir surveys likewise update storage-elevation relationships.
[USBR survey example](https://data.usbr.gov/catalog/4574/item/11370).

Source QC metadata is corroboration, not fault truth. NOAA's preliminary `f` begins with a
count of high-frequency outliers, and `s` describes sample dispersion; neither directly
establishes a failed gauge or the error variance of the reported mean.
[NOAA field definitions](https://api.tidesandcurrents.noaa.gov/api/prod/responseHelp.html).

A bootstrap of unlabeled records estimates alert behavior under its resampling assumptions,
not a known healthy null. Operational validation needs reviewed healthy periods and events,
separate threshold-selection and test sets, alarm/reset rules, and uncertainty on event-level
false-alarm rates. Measure detection delay, incorrect attribution and magnitude error
separately, retaining misses and ambiguous cases.
