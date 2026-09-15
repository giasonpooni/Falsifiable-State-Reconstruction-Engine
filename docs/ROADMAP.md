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
  cooling loop's mass+energy pair also **1 of 7**, leaving fifteen perfectly confounded pairs
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

## 3a-ii. The cooling design, taken up on its own terms

**Committed: [`results/cooling_circuits.md`](../results/cooling_circuits.md).** The paragraph
above asks for exactly one thing before choosing the cooling design anyway — which candidate
faults stay observable with noncollinear signatures once the real nuisance structure is in —
and this study answers it for the arrangement a moulding shop actually has: a manifold feeding
several metered circuits rather than one loop. It does the nondimensionalising the study above
called for, declares the shared supply thermocouple as the full `vec(A)` covariance stage 3b
asks for, and sweeps the circuit count so the procurement question is answered before hardware
is bought. What it found:

- **Conservation alone isolates nothing, at any circuit count**, and not because the geometry
  is tight: a fouling circuit returns less enthalpy because it absorbed less heat, so it
  satisfies the energy balance exactly and is **invisible**, in `null(A)`.
- **The duty row is what buys isolation** — 18 of 19 declared faults at six circuits — and a
  metered header closes the catalogue by making the header meter itself visible.
- **The count is then the wrong thing to read.** Every declared fault is structurally isolable
  with the header metered, which is the reading `fdi.Isolability` warns against in its own
  docstring. The tightest separated pair is what governs, and without a header meter it is
  *identical* at one circuit and at six: extra circuits buy coverage, not conditioning.
- **It is never one pair.** A manifold is symmetric under relabelling its circuits, so the
  minimum is attained once per circuit — six tied pairs at six circuits. The study reports the
  size and shape of that family instead of naming an arbitrary member.
- **The declared heat load is the lever; the meter count is not.** Going from one metered
  circuit to six moves the hardest pair by a factor of 1.14; declaring the heat load a hundred
  times looser moves it by 88. Buy a credible heat load before a sixth flow meter.
- **Which question binds is conditional on that prior, not on the count alone.** With the
  header metered the binding family asks *which meter* at small counts and *thermocouple or
  fouling, inside this circuit* at large ones. At the declared prior that handover falls at
  four circuits; a hundredfold looser it falls at three; a hundredfold tighter it never falls
  in this sweep. The study tabulates that rather than resolving it.
- **The supply thermocouple is not the binding instrument**, declared uncertain at the full
  `vec(A)` covariance: no isolable set moves and the hardest pair moves by under 1%.

It also closed two defects it depended on. `fdi` computed `S` without `Cov(E x)` whatever
`A_var` declared, so a declared matrix uncertainty never reached the fault geometry;
`residual_covariance`, `whitened_signature` and `isolability` now take the operating point
that quadratic form needs, and refuse a declared `A_var` without one. And `second_balance`
read confounding off `orthogonal_fraction == 0.0` rather than the structural label, which
under its own kg/J conditioning reported a confounded pair as its tightest *separated* one at
2.21x; corrected, its confound counts rise from 6 to 15 and from 1 to 4, and its tightest
genuinely separated pair reads 1.38x. No isolable count in that study changed.

Not established: nothing here is measured on an installed manifold, the form of the duty
relation is not varied, and fouling is *declared* to be a process change that conserves
energy rather than found to be one.

## 3b. Uncertainty on the relation itself — implemented for the test and the projection

Every constitutive row the stage above needs carries measured coefficients, and the kernel
declared uncertainty only on `b`. `ConstraintSet.A_var` now declares it on `A`, as a per-row
stack or a full `vec(A)` covariance, and the consistency statistic adds `Cov(E x)`.

[`results/errors_in_variables.md`](../results/errors_in_variables.md) measures what that is
worth against a null that is true by construction: treating an uncertain relation as exact
rejects a healthy system at 26–30x the nominal rate. It is not a conservative simplification.

The **projection** now follows the same declaration.
[`results/eiv_projection.md`](../results/eiv_projection.md) measures it on the same system,
scored on the estimate instead of the statistic: a nominal 95% region covers 1.5% at the
widest operating point when the relation is treated as exact, and the over-confidence follows
`1 + c·scale²` to within 0.10% across a 16x sweep. That last number is the mechanism rather
than a summary — a constant mis-declaration of `Sigma_b` or `P` would have cost the same at
every scale, so the slope is specifically the relation's own coefficients acting on a larger
state. Treating it as exact is worse than not projecting at all from 1x upward; the declared
version is never worse, at any scale tested.

What remains: the correction is first order, with its residual measured; `Cov(E x)` is taken
at the incoming state rather than iterated to a fixed point, with that cost measured at no
more than 0.03 NEES against a target of 3; and no dependence between `A`'s error and `b`'s, or
between `A`'s error and the state, is declared or inferred. Shared evidence produces exactly
those dependences, and they are the remaining piece of this stage.

## 3c. A site declared as data — implemented for the first site

Every `ConstraintSet` in this repository was built by a function inside one experiment module,
with the site's gauge ids, drainage areas, declared sigmas and citations as module globals
beside it. That made a second reservoir cost a second experiment module, and it put the
kernel's capabilities — `diagnose`, `detectability`, the isolability tables, the design study,
a declared `A_var` — out of reach of anyone not editing this codebase.

The abstraction was not invented for this: `experiments.second_balance.Topology` already
carried states, faults, variants and declared priors across three instances. `declarations/`
promotes that shape out of one experiment and makes it loadable, and
`src/set_lcm/declaration.py` reads it with the refusals the format needs to be worth trusting.

[`declarations/ridgway.toml`](../declarations/ridgway.toml) is the first instance, and
`results/real_water_balance` regenerates from it with no value moved. That is what makes the
format measured rather than merely plausible: the same report, from a document instead of from
constants.

What remains, and it is the substantive half: **a second site.** Every real-data claim in this
repository is n=1 site, and the claim that a declaration is now sufficient is untested until a
second one costs a declaration rather than a module. The design-study topologies are not yet
expressed in the format either, so `faults` — which `Topology` carries and this schema does
not — is the next field it needs. Estimator configuration stays in code deliberately: a
declaration says what the system is and what evidence exists for it, not how to filter it.

## 3d. A second real site — built, and it answered the question it was built to ask

`declarations/` was added on the claim that a second reservoir would cost a declaration rather
than a module. [`results/real_taylor_park.md`](../results/real_taylor_park.md) is the test of
that claim, and the answer is a declaration **plus two one-time costs that only a second site
could have exposed**: `estimators_balance` assumed exactly two gauged inflows (`N_SENSORS = 4`
and three hand-indexed state layouts), and the study's functions read one module's globals
rather than a declaration. No test could have found either — every test had two inflow gauges,
because every site did. Neither is paid again by a third site.

Taylor Park is deliberately not a clone: three gauged inflows, two of them seasonal gauges
reporting on 642 of 1,096 days, against Ridgway's four complete series. So the bridge's
missing-reading path, which no real evidence had ever exercised here, now carries 454 absent
days on each of two columns.

What it found is not what Ridgway found. Ridgway's balance closes to 0.41% of gauged inflow
with an interval containing zero; Taylor Park's does not close, and the shared `claims()`
**refused to print Ridgway's sentences about it** — the guard firing on a site transfer rather
than on a wording slip. Splitting the record by whether every gauge reported separates an
imbalance the evidence shows from one its absence creates: on the 641 fully reported days the
residual is still 9.80% of gauged inflow. The absent readings account for about half of the
apparent imbalance and no more.

That 9.80% sits within a percentage point of this site's ungauged drainage fraction (8.90%),
and the augmented variant carries exactly that term. It is not concluded: the same comparison
at Ridgway is 0.41% against 7.09%, no correspondence at all. A relationship that holds at one
site and fails at the other is a coincidence or a mechanism, and two sites cannot tell which.

What remains: n=2 is not a sample, and no fault is diagnosed at either site. The design-study
topologies are still not expressed in the declaration format, and `faults` — which
`second_balance.Topology` carries and the schema does not — is the next field it needs.

## 3e. The diagnostic surface on real evidence

Until now every record `diagnose()` judged was one this repository generated. `muskingum_reach`,
`fluid_baseline` and `camera_baseline` are synthetic; `real_fluid_baseline` calls it on a real
record but with an **empty** hypothesis set, as a consistency test with no candidates. The
isolation half — candidate fits, amplitude intervals, `min_separation`, the `ambiguous` and
`insufficient_evidence` outcomes — had never met real evidence.

Stage 3d produced what was missing: a real record whose balance does not close by a margin
that is not arguable. [`results/real_diagnosis.md`](../results/real_diagnosis.md) asks what
could explain it, with the candidate catalogue declared in each site's own TOML
(`[[fault]]`, resolved against the record through a named profile, the same split a constraint
row uses when its right-hand side is a reading).

Two results. **Structural:** `ungauged_constant` and `outflow_reads_low` — water no gauge sees,
and an outlet gauge reading low — are exactly collinear on a closure residual at both sites
(cos = -1.0000). A practitioner separates them; one rank-1 row never can, at any covariance,
with any amount of data. **Empirical:** the covariance is built from the declared instrument
sigmas, and the unmodelled daily-mean alignment error is swept rather than chosen. Reading the
sweep downward, Taylor Park goes `unexplained` (nothing in the catalogue accounts for the
record) to `ambiguous` (several candidates at once) to `consistent` (nothing left to explain).
Ridgway never reaches ambiguity. **At no point is a single cause named**, which was predicted
before the table was computed and is the honest outcome for this constraint and catalogue.

Adding a site also stopped costing an hour: `tools/export_daf_fixtures.py --session NAME`
merges one recorded session into the existing manifest, refusing when the DAF or substrate pins
differ because a merge would then describe two states. It reproduces a full export's manifest
byte for byte in 1m39s instead of 57 minutes.

What remains: the catalogue bounds the answer, so a cause absent from it cannot be found; the
alignment term is swept, never measured; and no fault is asserted to exist at either site.

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
