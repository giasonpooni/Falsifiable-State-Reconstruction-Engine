# Fluid-Sensor Reconstruction Testbed (WIP)

**Check whether measurements in a fluid system agree with its physical relationships.**

Short name **FSRE**, which is what the code, the docs and the module namespace use.
FSRE is a Python research toolkit for engineers and researchers working with water levels,
flow gauges and storage measurements. It estimates the state of a system, checks the estimate
against a declared balance, and keeps a record of the disagreement and any correction.

The aim is to help investigate degrading measurements: **when did the readings stop agreeing,
what could explain the difference, and what can this sensor arrangement actually detect?**

## A practical example

A reservoir has measurements of stored water, incoming flow and outgoing flow. Over the same
time interval, conservation relates them:

```text
change in stored water = water in − water out
```

If the measurements do not support that balance, FSRE can flag the disagreement and show its
size under your stated uncertainties. The cause could be a drifting instrument, an outdated
rating curve, an unmeasured inflow, or an unsuitable model. **An alarm starts an investigation;
it does not, by itself, identify a broken sensor.**

## What you get

| Question | FSRE provides |
|---|---|
| Do the estimates agree with the declared balance? | A consistency score and the residual before correction. |
| What changed during reconciliation? | The original estimate, corrected estimate, correction and resulting uncertainty. |
| Which measurement channels need investigation? | Per-channel checks for persistent disagreement with their predictions. |
| Could this arrangement distinguish the suspected faults? | An analysis of visible, invisible and confusable fault directions under a declared model, with how much larger a fault must be to name than to detect. |
| Is the relation itself uncertain? | Declared uncertainty on the constraint matrix, not only its right-hand side, so a measured coefficient does not read as a fault. |
| Can several fault explanations fit the same record? | A fixed-record comparison that returns explicit ambiguous or insufficient-evidence outcomes. |
| Can I estimate fluid states in consistent units and coordinates? | An additive invariant filter with full covariance propagation and tested coordinate transformations. |
| Can a camera provide a second level measurement? | A synthetic camera-and-gauge baseline with frame admission, calibration, vertical-marker compensation and correlated uncertainty. |
| How well does a method work? | Simulated fault experiments with known answers, and replay reports for real measurements. |

## Try it

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) and a checkout of this
repository. The project supports Python 3.12 and 3.13; its runtime dependency is NumPy.

```bash
git clone https://github.com/giasonpooni/Fluid-Sensor-Reconstruction-Testbed.git
cd Fluid-Sensor-Reconstruction-Testbed
uv run --frozen --python 3.13 python examples/quickstart.py
```

The example uses two tanks and a declared total of 100 kg. It shows two cases:

- A small disagreement: reconcile the estimate while retaining the original values.
- A large disagreement: flag it and hold the correction so the balance cannot hide the problem.

The example uses synthetic values and makes no network requests. The first `uv` run may
download Python and dependencies. See the [usage guide](docs/USAGE.md) for the code, expected
output, and how to use your own estimates or measurement records.

To try the fluid-measurement baseline:

```bash
uv run --frozen --python 3.13 python examples/fluid_baseline.py
```

This example checks matching storage/flow intervals and shows a conserving record, an
identifiable storage step, and a drift that cannot be distinguished from a flow offset.
The [baseline guide](docs/FLUID_BASELINE.md) explains its inputs and assumptions.

To try the invariant state/error layer:

```bash
uv run --frozen --python 3.13 python examples/invariant_fluid.py
```

It estimates two tank masses and verifies the same answer in transformed coordinates.
For the supported affine models, it is equivalent to an ordinary Kalman filter.
The [invariant layer guide](docs/INVARIANT_LAYER.md) explains its scope and inputs.

To try the camera-and-gauge prototype:

```bash
uv run --frozen --python 3.13 python examples/camera_baseline.py --out-dir work/camera-demo
```

It creates and replays a labeled fixture of rendered grayscale frames and gauge readings.
A fixed marker compensates vertical image motion; frame timing and shared calibration
uncertainty remain explicit. The [camera guide](docs/CAMERA_BASELINE.md) explains the
measurement pipeline and how to plan a controlled recording.

To prepare a real tank recording:

```bash
uv run --frozen --python 3.13 python -m set_lcm.recording init work/tank-recording
uv run --frozen --python 3.13 python -m set_lcm.recording check work/tank-recording --out-report work/tank-preflight.json
```

This creates a blank evidence kit. Its first check is expected to report `incomplete`
and exit with code 2. Follow the [recording protocol](docs/TANK_RECORDING_PROTOCOL.md)
to supply equipment declarations and acquired evidence. Preflight checks metadata, files
and declared clocks; it does not run real-data inference or establish field accuracy.

## What is ready today?

**Available:** a tested linear reconciliation kernel, simulation and fault injection,
per-channel innovation monitoring, static fault-signature analysis, and a fluid-measurement
baseline with explicit intervals, shared-reference covariance and conditional offset/drift/
gain comparisons. The baseline has analytical and synthetic validation plus a real-record
consistency replay. Original evidence is retained; derived outputs remain separate.
An additive invariant filtering layer is also available, with correlated/missing readings
and coordinate-consistency tests. General nonlinear Lie-group IEKF models remain planned.
A camera-and-gauge prototype now exercises real pixel extraction from synthetic frames,
vertical image registration, timing checks and fusion with full shared covariance.
An empty recording kit and preflight checker support preparation for a controlled real trial.
It has no field validation; general 3-D camera pose and odometry remain planned work.

A site is declared as data rather than as code. `declarations/ridgway.toml` carries that
reservoir's states and their units, its constraint variants, its four gauges with the
uncertainty this consumer declares for each and the citation behind it, the record the
evidence lives in, and every other declared number — process-noise scales, prior widths,
drainage areas. `results/real_water_balance` regenerates from it with no value moved, which is
what says the format is sufficient for a real site rather than merely plausible. Whether a
SECOND site costs only a declaration is not yet measured; that is the next thing to find out.

**Still being developed:** diagnosis of events with unknown onset, field-validated degradation
magnitudes, and operational alert thresholds. Current fault fits assume specified profiles,
onset and uncertainty; their amplitude intervals are conditional on that model. Real-data reports measure
consistency under declared assumptions; they do not establish the true state or which sensor
is faulty. Some different faults are indistinguishable with the available measurements.

Use FSRE now to evaluate a measurement model, compare methods, replay evidence and investigate
possible faults. Treat it as research software when deciding whether it is suitable for an
operational workflow.

## Explore the examples

| Example | What it demonstrates | Report |
|---|---|---|
| Camera and gauge | Synthetic grayscale measurements, fixed-marker motion compensation, timing problems and disagreement under shared calibration uncertainty. | [Camera baseline results](results/camera_baseline.md) |
| Invariant filtering | Ordinary KF equivalence under unit, coordinate and measurement-order changes, including missing and biased readings. | [Invariant layer results](results/invariant_layer.md) |
| Fluid fault benchmark | Offset, drift and gain across 128 evaluation seeds per case, including confounded faults and physical changes. | [Baseline results](results/fluid_baseline.md) |
| Ridgway measurement baseline | Matching daily intervals and full residual covariance, under declared timing/model and uncertainty assumptions. | [Measurement replay](results/real_fluid_baseline.md) |
| Two-reservoir simulation | Noise, missing readings, biased sensors and stale balances, scored against hidden simulated truth. | [Simulation results](results/summary.md) |
| NOAA tide gauge | Measurement replay, water-level filters and checks that do not require known truth. | [Water-level results](results/real_noaa.md) |
| NOAA tide gauge, one month | What record length changes: which tidal constituents 31 days separate and 15 do not, a q fitted on the first half and scored on the second, and the stated per-reading uncertainty against a model-free bound. | [Month results](results/real_noaa_month.md) |
| Ridgway filter study | Historical estimator comparison, retaining its documented daily-mean/reference approximation. | [Water-balance results](results/real_water_balance.md) |
| Two-reach river | The first topology here that can name an instrument: six declared faults recovered in 100% of records with routing, none of the storage ones without it. | [Muskingum results](results/muskingum_reach.md) |
| Second-balance design study | Which proposed topology could actually isolate a fault, computed before either is built: conservation alone reaches 1 of 7, the constitutive relation reaches 5 of 7. | [Design study](results/second_balance.md) |
| Uncertain relations | What treating a measured coefficient as exact costs, against a null that is true by construction: 26 to 30 times the nominal false-alarm rate. | [Calibration](results/errors_in_variables.md) |
| Uncertain relations, reconciled | The same cost to the estimate rather than the test: a nominal 95% region that actually covers 1.5%, and an over-confidence measured to be quadratic in the operating point to within 0.10%. | [Projection](results/eiv_projection.md) |

To run the default test suite:

```bash
uv run --frozen --python 3.13 --dev pytest -q
```

The [usage guide](docs/USAGE.md#reproduce-the-reports) lists report-generation commands and
the slower reproducibility checks. Reports are generated from code and include provenance.

## Read further

- [Usage guide](docs/USAGE.md): installation, inputs, outputs and integration examples.
- [Fluid baseline](docs/FLUID_BASELINE.md): interval handling, shared uncertainty and conditional fault explanations.
- [Camera baseline](docs/CAMERA_BASELINE.md): frame admission, level calibration, vertical registration and camera/gauge comparison.
- [Controlled recording protocol](docs/TANK_RECORDING_PROTOCOL.md): equipment evidence, calibration, timing and held-out evaluation preparation.
- [Invariant filtering](docs/INVARIANT_LAYER.md): additive state/error geometry, estimation and coordinate consistency.
- [Methods and interpretation](docs/METHODS.md): the mathematics, fault-identifiability limits
  and the meaning of a consistency score.
- [Development roadmap](docs/ROADMAP.md): the steps toward validated measurement diagnostics.
- [Detailed research history](docs/RESULTS.md): experiments and their limitations.
- [Contributor brief](docs/COLLABORATOR_BRIEF.md): architecture and working conventions.
- [Data provenance](data/daf/PROVENANCE.md): where the committed measurements came from.

“State reconstruction” means estimating a physical system from measurements. FSRE retains
the evidence needed to challenge its estimates. Whether a fault can be detected depends on
the measurements, model and declared uncertainty.

## License

FSRE is available under the [MIT License](LICENSE).
