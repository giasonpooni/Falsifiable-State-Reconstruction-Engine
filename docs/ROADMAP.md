# Development roadmap

The target is validated measurement diagnostics for fluid networks. Current code supports
reconciliation, static fault geometry, simulated fault experiments and real-record
consistency reports. Dependable field diagnosis and degradation magnitude remain development
targets. [Methods](METHODS.md) states the assumptions behind these extensions.

**The foundation and the first fluid-measurement baseline are implemented on `main`.**
The baseline has analytical and synthetic validation, not industrial field validation.
The sequence is ordered by the evidence each stage adds.

## Scope: fluid systems

Keep development focused on fluids. The camera prototype supports fixed-marker vertical
image registration for level measurements. General 3-D camera pose and odometry remain
planned, contingent on a specific measurement experiment and a validated geometric model.
Telemetry is the transport and reporting of measurements and status; its timing and missingness already
have representations through `arrival_t`, `mask` and the bridge's `arrival_policy`.
These fields do not imply support for every transport protocol or arrival pattern.

Thermal measurements are a candidate source of a second conservation relation on the same
cooling-loop pipes: mass and energy balances. This is an alternative planned experiment to
the river reach, not an implemented temperature estimator or a generic-domain expansion.

## 1. Foundation — implemented

Repair balance feedback so the full state and covariance remain consistent; validate
numerical inputs and soft-projection parameters; enforce consistent ingestion units; and
separate structural blindness from low sensitivity in FDI. Correct scientific claims,
add optional constraint row-unit metadata, provide a practical README and usage example,
and establish automated checks. Unit metadata describes the rows; it does not convert
quantities or establish a physical model.

Acceptance requires focused regressions, an independent full-state Joseph reference,
clear invalid-input failures, CI across supported Python versions, and regeneration and
reproduction checks for affected reports. These establish software contracts, not the
scientific claims of subsequent stages.

## 2. Sensor-space and finite-horizon analysis — baseline implemented

The [fluid baseline](FLUID_BASELINE.md) maps storage/flow measurements and fault profiles
through one interval operator. It propagates the joint measurement covariance, including
shared references, and compares known-onset offset, drift and gain hypotheses after removing
declared nuisance effects. It returns ambiguous and insufficient-evidence outcomes.

Boundary measurements support exact interval balances; mean storage requires an explicit
constant-net-flow assumption. The new Ridgway measurement replay uses that assumption and
sweeps both of its consumer choices: the declared storage uncertainty and the window length,
which moves the rejection rate further than the uncertainty does. The historical estimator
study retains its documented approximation.

Analytical tests and the synthetic report cover identifiable cases, collinear signatures,
step/ramp ambiguity, common-mode cancellation, nuisance confounding and conditional amplitude
coverage. Unknown-onset search, simultaneous faults and uncertain routing remain planned.
Adding bias states must not be presented as automatic identification.

## Additive invariant layer — implemented

The [invariant state/error layer](INVARIANT_LAYER.md) supports affine fluid models on
the additive group, including full covariance, masked observations and fixed affine
coordinate charts. Independent Gaussian conditioning and ordinary KF comparisons
establish this special case. The generated experiment checks consistent estimates and
innovation statistics under units, basis, origin and measurement-order changes.

This does not add fault identifiability or a nonlinear IEKF advantage. Extend to a
nonlinear group only with an explicit fluid model, a derivation of its error dynamics
and observation properties, and matched EKF comparisons. Field measurements remain the
next source of evidence; this layer does not replace their validation.

## Camera and gauge measurements — synthetic prototype implemented

The [camera baseline](CAMERA_BASELINE.md) starts with rendered grayscale frames, extracts
a declared horizontal water edge, and uses a fixed marker to compensate vertical image
translation. It admits frames using capture IDs, timestamps, missingness and a declared
duplicate-content policy. Calibration and camera/gauge fusion carry shared uncertainty
across the record; raw measurements and disagreement remain available. A separate held-out
reference and synthetic truth support evaluation. The [report](../results/camera_baseline.md)
records the comparisons without claiming field performance.

Spectral summaries use only complete, contiguous windows with a uniform capture clock.
They are advisory features, not calibrated fault probabilities or a way to recover an
incorrectly declared clock. Metadata timing checks do not establish operational event
detection rates. Static scenes can repeat legitimately, so duplicate-content refusal
must be an explicit policy.

Recording preparation is now available through a blank evidence kit and metadata/file/clock
preflight. The [controlled recording protocol](TANK_RECORDING_PROTOCOL.md) specifies separate
calibration, development and held-out sessions, original evidence and uncertainty declarations.
The checker reports readiness for review; it does not validate equipment or run real-data
inference. No real tank measurements or field-accuracy results have been obtained by this step.

The next gate remains a controlled tank recording with a fixed, front-facing camera, a fixed
fiducial, documented capture timing, and independently calibrated gauge/reference readings.
Synthetic calibration treats anchor pixels as exact; uncertain real anchors need an
errors-in-variables treatment or a validated alternative. Current covariance propagation
is first order, and repeated frames do not create independent calibration trials.
Perspective, rotation and moving-camera 3-D pose need a separate experiment and observation
model before extending the registration or claiming odometry support. No FluidNexus code
or generated-view measurements are imported into this prototype.

## 3. A second independent balance — built

**Delivered: [`results/muskingum_reach.md`](../results/muskingum_reach.md)**, on the design the
study below selected. The rule is the residual vector and its joint covariance through
`diagnose()`, not a rank calculation and not a scalar threshold. Against the stage's own
acceptance criteria, on 64 evaluation seeds disjoint from 16 development seeds:

- **Each nonzero single-gauge persistent bias is recovered** at the strong declared magnitude:
  6 of 6 declared instrument faults identified correctly in 100% of records with routing.
  Continuity alone recovers 0% of the two storage biases and 0% of a common flow drift,
  because it cannot see them at any magnitude.
- **A common offset invisible to the balance** is named rather than hoped for: a coordinated
  offset of both storage sensors by K times a common offset on all three flow gauges produces
  exactly zero residual. That is the topology's blind spot, stated.
- **Ambiguity when a nuisance mimics a fault** is delivered by the catalogue, not the topology.
  An ungauged lateral inflow produces the same residual direction as an inflow-gauge bias, so
  with only instrument candidates declared the rule confidently names an instrument in 100% of
  records. Declaring the physical explanation converts that to 100% ambiguous with the correct
  explanation among them. A catalogue of only instrument faults will blame an instrument for a
  river.
- **Parameter error is not a fault.** With the true reach at K = 1.15, x = 0.28 while the rows
  declare K = 1.0, x = 0.2, a healthy river raises a false alarm in 100% of records with A
  treated as exact, and 0% once the parameter uncertainty is declared (stage 3b).

Remaining for this stage: correlated observations beyond the shared storage reading the design
already carries, interval semantics other than the declared trapezoidal mean, and an onset that
is searched rather than supplied.

## 3a. The design study that chose it

**The design study is committed: [`results/second_balance.md`](../results/second_balance.md).**
It runs `fdi.isolability()` on both candidates before either is built, because a second
balance buys only the directions its rows separate. Its results decide this stage:

- Every constraint in the repository has rank(A) = 1, so **0 of 4** declared faults are
  isolable today. That is why this stage exists.
- A second *conservation* row is not enough: continuity alone reaches **1 of 7**, and the
  cooling loop's mass+energy pair also **1 of 7**, leaving six perfectly confounded pairs
  because one energy equation gives one residual direction.
- The **constitutive** relation is what separates sensors. With Muskingum routing the river
  reaches **5 of 7**; with the heat-exchanger duty relation the loop reaches **2 of 7**.
- The river's one remaining confound is physical — an ungauged lateral inflow against an
  inflow-gauge bias — which no topology fixes and only another gauge would. The loop's is
  instrumental.
- Every constitutive row carries declared parameters, so the isolation it buys is
  conditional on uncertainty the kernel cannot yet represent. See stage 3b.

Implement the two-reach Muskingum design in Methods, with the routing rows: continuity alone
is measured to be nearly worthless for isolation. Begin with declared parameters and
known synthetic truth, then introduce parameter error, lateral inflow, correlated
observations and correctly modeled interval semantics.

Acceptance requires recovery of each nonzero single-gauge persistent bias in the ideal
model, a common offset invisible to the balance, and appropriate ambiguity when nuisance
processes mimic faults. Implement a decision rule using the residual vector and joint
covariance; a rank calculation or scalar threshold is not that rule.

An alternative is a cooling loop with mass and energy balances over the same pipes. The study
above did exactly the check this paragraph asks for and the alternative came second: 2 of 7
against 5 of 7, an instrumental rather than physical residual confound, and a declared prior
in kilograms and joules that `check_spd` refuses at 2 of 3 swept scales, which is an argument
for nondimensionalising that design before building it. Before choosing it anyway, show which
candidate faults remain observable and have noncollinear signatures
after heat exchange, energy storage and other nuisance terms are included. Temperature-
derived enthalpy coefficients are measured quantities with uncertainty; shared flow and
temperature evidence can correlate the rows. `ConstraintSet.A_var` carries that uncertainty
now (stage 3b); declare the full `vec(A)` covariance rather than per-row variances where the
evidence is shared, because the per-row shortcut is measured to reject at 2.1x nominal there
against a full declaration's 1.2x. A joint nonlinear state model remains an alternative and
remains unimplemented. Neither thermal modeling nor this alternative experiment is delivered
in the foundation stage.

## 3b. Uncertainty on the relation itself — implemented for the test, not the projection

Every constitutive row the stage above needs carries measured coefficients, and the kernel
declared uncertainty only on `b`. `ConstraintSet.A_var` now declares it on `A`, as a per-row
stack or a full `vec(A)` covariance, and the consistency statistic adds `Cov(E x)`.

[`results/errors_in_variables.md`](../results/errors_in_variables.md) measures what that is
worth against a null that is true by construction: treating an uncertain relation as exact
rejects a healthy system at 26–30x the nominal rate. It is not a conservative simplification.

What remains: the **projection** still solves against `A_bar` as though it were exact, so a
reported correction is conditional on that; the correction is first order, with its residual
measured; and no dependence between `A`'s error and `b`'s, or between `A`'s error and the
state, is declared or inferred. A full errors-in-variables reconciliation is the next step
here, and it is a harder problem than the test was.

## 4. Degradation benchmark — offset/drift/gain baseline implemented; expansion planned

The committed fixed-horizon benchmark supplies offset, drift and gain at two locked
magnitudes, with disjoint development/evaluation seeds, physical controls, per-seed results
and explicit scoring denominators. It reports misses, ambiguity, wrong attribution and
conditional interval coverage. No operational detection delay or field accuracy is claimed.

Expand the fault matrix to cover unknown-onset steps, slow drift, nonlinear rating shifts, stuck readings,
quantization, variance growth and timing faults. Match these with legitimate physical
changes, ice, changed routing, omitted flux and changed stage-to-storage relationships.
Distinguish instrument faults from measurement-model errors.

Acceptance requires delay versus magnitude or drift rate, missed events, wrong-sensor
attribution, fault-class confusion and magnitude-interval coverage across independent seeds.
Include ambiguous/insufficient-evidence outcomes. Common-mode controls may alarm other
channels but must not acquire unsupported balance-based diagnoses.

## 5. Operational calibration and external validation — planned

The recording protocol and preflight kit prepare evidence collection for a controlled pilot.
Actual acquisition, justified real-image calibration, measurement alignment and held-out
evaluation remain required before a field-validation claim.

Define alarm episodes, resets and units such as alarms per sensor-month or network-month.
Choose thresholds on designated development data and lock them before evaluation. Use a
small site/season panel with reviewed healthy periods, maintenance or calibration events,
and reference measurements. A controlled fluid test loop with known interventions is
another useful external validation route.

Acceptance requires event-level false-alarm rates with uncertainty, delay distributions
retaining misses, attribution performance and validation of identifiable magnitudes. Source
QC flags are weak corroborating labels. Bootstrap alert rates on unadjudicated history must
not be called measured fault-free performance.

## Later integration choices

After demonstrating value, define a versioned input/output contract, supported deployment
settings and an operator report showing evidence, assumptions, candidate causes and
uncertainty. Inequalities, additional models and interfaces should answer demonstrated
needs; they do not replace validation. Physical limits need explicit domain assumptions
and do not make causal diagnosis unambiguous.

Every stage retains evidence and unprojected estimates, documents assumptions and exclusions,
and regenerates reports from code. Current numerical claims belong in
[generated results](../results/), not in this planning document.
