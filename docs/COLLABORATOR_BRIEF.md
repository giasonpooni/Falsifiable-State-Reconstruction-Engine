# Contributor brief

`main` is the canonical development version. Build, validate and publish changes directly
there; do not maintain separate development branches unless the user requests one.
See [AGENTS.md](../AGENTS.md) for the repository workflow.

The Fluid State Reconstruction Engine (FSRE) is a research toolkit for testing agreement between estimated states and declared physical relationships while retaining the evidence and every correction. Its development target is measurement-system diagnostics in fluid networks, including instrument drift, rating-curve errors and storage-model errors. Current real-data experiments demonstrate consistency checking, not validated causal diagnosis or degradation magnitude.

Start with the [README](../README.md) and [usage guide](USAGE.md). Read [Methods](METHODS.md) before interpreting scores or fault geometry, and [Roadmap](ROADMAP.md) before extending the scope. Current numerical claims belong in the generated [reports](../results/); [Results](RESULTS.md) preserves the research history.

Development remains focused on fluids; no odometry development is planned. Telemetry timing and missingness are already represented through `arrival_t`, `mask` and `arrival_policy`. A mass-plus-energy cooling-loop experiment is a planned alternative to the river reach, not delivered thermal support. Its fault signatures need analysis after heat/storage nuisance and correlated measurement uncertainty are included; see the roadmap.

## Architecture

| Location | Responsibility |
|---|---|
| `src/set_lcm/schema/` | Observation, estimate and constraint contracts. |
| `src/set_lcm/lcm/` | Linear reconciliation, feasibility, residuals and consistency statistics. |
| `src/set_lcm/fdi.py` | Static single-fault geometry of the residual vector. |
| `src/set_lcm/testbed/` | Estimators, arrival-aware runner, CUSUM, simulation and evaluators. |
| `src/set_lcm/bridge/daf.py` | Explicit selection and admission of supported DAF evidence. |
| `src/set_lcm/experiments/` | Reproducible scenarios, calibration, sweeps and real-data reports. |
| `data/daf/` | Committed replay evidence, manifest and acquisition provenance. |

## Evidence and contribution rules

- Preserve observations and unprojected estimates. Derived values and corrections must stay distinguishable from evidence. Refuse or report contradictory revisions; do not silently average them or select the latest one as truth.
- Declare units, interval semantics, arrival policy and uncertainties. Carry citations where required by the bridge. If a source does not supply a quantity, identify the consumer's assumption and assess sensitivity; a citation does not validate an assumption it does not support.
- Keep simulated hidden truth outside estimators. The labeled oracle is the explicit exception. Evaluate synthetic errors against truth and real-record consistency with the truth-free evaluator.
- Do not select configurations or seeds to obtain a preferred conclusion. State fitting and evaluation windows, keep them disjoint where parameters are fitted, and distinguish model-development data from independent validation data.
- Change report generators and their evidence together. Follow the usage guide to regenerate affected Markdown and JSON reports, retain provenance, and run reproduction checks. Do not hand-edit result numbers to match a narrative. Update claims when behavior changes.
- Add focused tests for corrected failures and meaningful method contracts. Compare numerical changes with independent calculations where practical; document remaining limitations.
- DAF is a separate upstream repository. **Do not push to DAF.** Changes here must not silently alter upstream acquisition, evidence identity or revision policy.

## Interpretation rules

A consistency threshold is a reference under a calibrated Gaussian null. An alarm is not a diagnosis. Per-channel CUSUM locates prediction disagreement, not necessarily a faulty instrument. Static rank-one geometry does not establish impossibility over every time record; known dynamics can add information. Finite uncertainty on an augmented imbalance term does not make its model statistically unfalsifiable.

Treat source QC flags as corroborating metadata, not ground-truth fault labels. A constraint violation may reflect physical change, an omitted flux, a unit or timing problem, or a model error. Report ambiguous explanations explicitly. Describe unbuilt methods as planned work, and keep detection, isolation and magnitude estimation as separate claims.
