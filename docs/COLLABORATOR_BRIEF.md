# Contributor brief

`main` is the canonical development version. Build, validate and publish changes directly
there; do not maintain separate development branches unless the user requests one.
See [AGENTS.md](../AGENTS.md) for the repository workflow.

The Fluid State Reconstruction Engine (FSRE) is a research toolkit for testing agreement between estimated states and declared physical relationships while retaining the evidence and every correction. Its development target is measurement-system diagnostics in fluid networks, including instrument drift, rating-curve errors and storage-model errors. Current real-data experiments demonstrate consistency checking, not validated causal diagnosis or degradation magnitude.

Start with the [README](../README.md) and [usage guide](USAGE.md). Read [Methods](METHODS.md) before interpreting scores or fault geometry, and [Roadmap](ROADMAP.md) before extending the scope. Current numerical claims belong in the generated [reports](../results/); [Results](RESULTS.md) preserves the research history.

Development remains focused on fluids. The camera prototype delivers fixed-marker vertical image registration; general 3-D camera pose and odometry remain planned pending a specific experiment and geometric model. Telemetry timing and missingness are already represented through `arrival_t`, `mask` and `arrival_policy`. A mass-plus-energy cooling-loop experiment is a planned alternative to the river reach, not delivered thermal support. Its fault signatures need analysis after heat/storage nuisance and correlated measurement uncertainty are included; see the roadmap.

## Architecture

| Location | Responsibility |
|---|---|
| `src/set_lcm/schema/` | Observation, estimate and constraint contracts. |
| `src/set_lcm/lcm/` | Linear reconciliation, feasibility, residuals and consistency statistics. |
| `src/set_lcm/fdi.py` | Static single-fault geometry of the residual vector. |
| `src/set_lcm/measurement.py` | Explicit interval balances and joint measurement/reference covariance. |
| `src/set_lcm/diagnostics.py` | Conditional fixed-horizon single-fault fits, nuisance projection and explicit ambiguity. |
| `src/set_lcm/invariant.py`, `coordinates.py` | Additive invariant error-state filtering and fixed affine coordinate charts. |
| `src/set_lcm/frame_quality.py` | Frame identity, provenance and capture-clock admission; advisory spectra on complete uniform windows. |
| `src/set_lcm/camera.py` | Declared horizontal-edge extraction, vertical marker registration and first-order calibrated level covariance. |
| `src/set_lcm/camera_fusion.py` | Camera/gauge comparison and per-time fusion with full shared covariance and retained raw disagreement. |
| `src/set_lcm/recording.py`, `recording_clock.py` | Blank recording kits and bounded preflight checks of declared files, identities, observation support and capture clocks. |
| `src/set_lcm/testbed/` | Estimators, arrival-aware runner, CUSUM, simulation and evaluators. |
| `src/set_lcm/bridge/daf.py` | Explicit selection and admission of supported DAF evidence. |
| `src/set_lcm/experiments/` | Reproducible scenarios, calibration, sweeps and real-data reports. |
| `data/daf/` | Committed replay evidence, manifest and acquisition provenance. |

## Evidence and contribution rules

- Preserve observations and unprojected estimates. Derived values and corrections must stay distinguishable from evidence. Refuse or report contradictory revisions; do not silently average them or select the latest one as truth.
- Declare units, interval semantics, arrival policy and uncertainties. Carry citations where required by the bridge. If a source does not supply a quantity, identify the consumer's assumption and assess sensitivity; a citation does not validate an assumption it does not support.
- Keep simulated hidden truth outside estimators. The labeled oracle is the explicit exception. Evaluate synthetic errors against truth and real-record consistency with the truth-free evaluator.
- Keep camera inference inputs separate from held-out reference readings and synthetic truth. Preserve capture times and frame hashes; do not silently interpolate, repair frame rate or promote generated views to independent observations. Shared calibration uncertainty must remain correlated across frames and sources.
- Do not select configurations or seeds to obtain a preferred conclusion. State fitting and evaluation windows, keep them disjoint where parameters are fitted, and distinguish model-development data from independent validation data.
- Change report generators and their evidence together. Follow the usage guide to regenerate affected Markdown and JSON reports, retain provenance, and run reproduction checks. Do not hand-edit result numbers to match a narrative. Update claims when behavior changes.
- Add focused tests for corrected failures and meaningful method contracts. Compare numerical changes with independent calculations where practical; document remaining limitations.
- DAF is a separate upstream repository. **Do not push to DAF.** Changes here must not silently alter upstream acquisition, evidence identity or revision policy.

## Interpretation rules

The [invariant layer](INVARIANT_LAYER.md) is the additive-group affine Gaussian case,
equivalent to an ordinary KF. It retains pre-update evidence and full covariance. Its
coordinate consistency does not establish physical conservation, sensor identifiability,
nonlinear robustness or general Lie-group IEKF support. Those require separate models
and evidence. Do not infer noise or hide uncertain shared references inside fixed charts.

The [fluid baseline](FLUID_BASELINE.md) is delivered with analytical tests, an offset/drift/
gain benchmark and a truth-free Ridgway replay. Candidate signatures and onset are supplied;
`identified` means unique adequacy within that declared catalogue, not verified causality.
Amplitude intervals are conditional and not adjusted for model selection. The new measurement
path does not silently replace the historical water-balance filters or their assumptions.

The [camera baseline](CAMERA_BASELINE.md) processes actual rendered grayscale frames and
gauge readings under a fixed synthetic protocol. It has no field validation. Detector
thresholds are quality rules, not confidence estimates; uncertainty is separately declared.
GLS calibration treats pixel anchors as exact, and height covariance uses first-order
propagation with declared independence assumptions. Neither real-anchor errors-in-variables
calibration nor exact Gaussian coverage after image selection is established. A reused
calibration realization is shared evidence, not an independent trial for each frame.

Fixed-marker compensation estimates vertical image translation only. It does not deliver
general 3-D pose, perspective correction or odometry. Preserve raw camera/gauge disagreement:
signed drift explanations can remain confounded, and shared errors can escape a difference
check. Spectral features are advisory and cannot verify an incorrectly declared physical
clock without an external reference. No FluidNexus code or generated views are used.

The [recording kit](TANK_RECORDING_PROTOCOL.md) prepares a real experiment; it does not
perform one. `ready_for_review` means checked artifacts and declarations are present,
not that clocks, calibration or covariance have been scientifically validated. Do not
turn absent equipment/data/uncertainty into synthetic defaults. Preserve native clocks
that the bounded timestamp parser cannot accept; never round them to obtain admission.

A consistency threshold is a reference under a calibrated Gaussian null. An alarm is not a diagnosis. Per-channel CUSUM locates prediction disagreement, not necessarily a faulty instrument. Static rank-one geometry does not establish impossibility over every time record; known dynamics can add information. Finite uncertainty on an augmented imbalance term does not make its model statistically unfalsifiable.

Treat source QC flags as corroborating metadata, not ground-truth fault labels. A constraint violation may reflect physical change, an omitted flux, a unit or timing problem, or a model error. Report ambiguous explanations explicitly. Describe unbuilt methods as planned work, and keep detection, isolation and magnitude estimation as separate claims.
