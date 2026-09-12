# Falsifiable State Reconstruction Engine

**Check whether measurements in a fluid system agree with its physical relationships.**

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
| Could this arrangement distinguish the suspected faults? | An analysis of visible, invisible and confusable fault directions under a declared model. |
| How well does a method work? | Simulated fault experiments with known answers, and replay reports for real measurements. |

## Try it

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) and a checkout of this
repository. The project supports Python 3.12 and 3.13; its runtime dependency is NumPy.

```bash
git clone https://github.com/giasonpooni/Falsifiable-State-Reconstruction-Engine.git
cd Falsifiable-State-Reconstruction-Engine
uv run --frozen --python 3.13 python examples/quickstart.py
```

The example uses two tanks and a declared total of 100 kg. It shows two cases:

- A small disagreement: reconcile the estimate while retaining the original values.
- A large disagreement: flag it and hold the correction so the balance cannot hide the problem.

The example uses synthetic values and makes no network requests. The first `uv` run may
download Python and dependencies. See the [usage guide](docs/USAGE.md) for the code, expected
output, and how to use your own estimates or measurement records.

## What is ready today?

**Available:** a tested linear reconciliation kernel, simulation and fault injection,
per-channel innovation monitoring, static fault-signature analysis, and replay examples using
NOAA water levels and USGS reservoir measurements. Original evidence is retained; derived
estimates and corrections are separate outputs.

**Still being developed:** dependable sensor diagnosis, degradation magnitude estimates, and
operational alert thresholds validated on independent field events. Real-data reports measure
consistency under declared assumptions; they do not establish the true state or which sensor
is faulty. Some different faults are indistinguishable with the available measurements.

Use FSRE now to evaluate a measurement model, compare methods, replay evidence and investigate
possible faults. Treat it as research software when deciding whether it is suitable for an
operational workflow.

## Explore the examples

| Example | What it demonstrates | Report |
|---|---|---|
| Two-reservoir simulation | Noise, missing readings, biased sensors and stale balances, scored against hidden simulated truth. | [Simulation results](results/summary.md) |
| NOAA tide gauge | Measurement replay, water-level filters and checks that do not require known truth. | [Water-level results](results/real_noaa.md) |
| Ridgway Reservoir | A balance built from storage and flow measurements, with explicit uncertainty assumptions. | [Water-balance results](results/real_water_balance.md) |

To run the default test suite:

```bash
uv run --frozen --python 3.13 --dev pytest -q
```

The [usage guide](docs/USAGE.md#reproduce-the-reports) lists report-generation commands and
the slower reproducibility checks. Reports are generated from code and include provenance.

## Read further

- [Usage guide](docs/USAGE.md): installation, inputs, outputs and integration examples.
- [Methods and interpretation](docs/METHODS.md): the mathematics, fault-identifiability limits
  and the meaning of a consistency score.
- [Development roadmap](docs/ROADMAP.md): the steps toward validated measurement diagnostics.
- [Detailed research history](docs/RESULTS.md): experiments and their limitations.
- [Contributor brief](docs/COLLABORATOR_BRIEF.md): architecture and working conventions.
- [Data provenance](data/daf/PROVENANCE.md): where the committed measurements came from.

“State reconstruction” means estimating a physical system from measurements. The project name
expresses the goal that estimates carry evidence capable of challenging them; it is not a
guarantee that every possible fault can be detected.
