# Development roadmap

The target is validated measurement diagnostics for fluid networks. Current code supports
reconciliation, static fault geometry, simulated fault experiments and real-record
consistency reports. Dependable field diagnosis and degradation magnitude remain development
targets. [Methods](METHODS.md) states the assumptions behind these extensions.

**The foundation stage is implemented on `main`. Later stages are planned work.**
The sequence is ordered by the evidence each stage adds.

## Scope: fluid systems

Keep development focused on fluids. No odometry development is planned. Telemetry is the
transport and reporting of measurements and status; its timing and missingness already
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

## 2. Sensor-space and finite-horizon analysis — planned

Map faults from individual channels before aggregation. Add explicit offset, drift and gain
hypotheses, temporal responses and nuisance inputs such as ungauged flow and routing error.
Assess rank and conditioning after accounting for nuisance effects. Address the shared
reference covariance and interval-averaging gaps in Methods.

Acceptance requires examples separating identifiable cases from collinear signatures,
step/ramp ambiguity and common-mode cancellation. State single-fault and simultaneous-fault
assumptions separately. Adding bias states must not be presented as automatic identification.

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

## 4. Degradation benchmark — planned

Build a fault matrix covering steps, slow drift, gain/rating shifts, stuck readings,
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
