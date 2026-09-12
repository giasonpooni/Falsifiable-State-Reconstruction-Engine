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
an uncertainty sweep. The historical estimator study retains its documented approximation.

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

The next gate is a controlled tank recording with a fixed, front-facing camera, a fixed
fiducial, documented capture timing, and independently calibrated gauge/reference readings.
Synthetic calibration treats anchor pixels as exact; uncertain real anchors need an
errors-in-variables treatment or a validated alternative. Current covariance propagation
is first order, and repeated frames do not create independent calibration trials.
Perspective, rotation and moving-camera 3-D pose need a separate experiment and observation
model before extending the registration or claiming odometry support. No FluidNexus code
or generated-view measurements are imported into this prototype.

## 3. A second independent balance — planned

Implement the two-reach Muskingum design in Methods. Begin with declared parameters and
known synthetic truth, then introduce parameter error, lateral inflow, correlated
observations and correctly modeled interval semantics.

Acceptance requires recovery of each nonzero single-gauge persistent bias in the ideal
model, a common offset invisible to the balance, and appropriate ambiguity when nuisance
processes mimic faults. Implement a decision rule using the residual vector and joint
covariance; a rank calculation or scalar threshold is not that rule.

An alternative is a cooling loop with mass and energy balances over the same pipes. Before
choosing it, show which candidate faults remain observable and have noncollinear signatures
after heat exchange, energy storage and other nuisance terms are included. Temperature-
derived enthalpy coefficients are measured quantities with uncertainty; shared flow and
temperature evidence can correlate the rows. Use an explicit errors-in-variables treatment
or a joint nonlinear state model where needed. An independently declared `A_var` by itself
would not account for those correlations. Neither thermal modeling nor this alternative
experiment is delivered in the foundation stage.

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
