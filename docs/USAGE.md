# Using FSRE

FSRE is a Python library and a collection of reproducible experiments. Start with the small
example, then choose whether you need to check raw fluid measurements, reconcile an existing
estimate or replay a sequence of measurements through an estimator.

## Install and run the example

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone the repository,
and run the following from its root:

```bash
uv run --frozen --python 3.13 python examples/quickstart.py
```

`uv` creates an isolated environment using the committed lockfile. Python 3.12 is also
supported. Running the example itself does not fetch data.

The example prints:

```text
Small disagreement
  Score: 1.78; reference threshold: 10.83
  Action: ok
  Original estimate: [52.0, 46.0] kg
  Output estimate: [52.889, 46.889] kg
  Residual before: -2.000 kg
  Residual after: -0.222 kg

Large disagreement
  Score: 44.44; reference threshold: 10.83
  Action: model_inconsistent
  Original estimate: [60.0, 50.0] kg
  Output estimate: [60.0, 50.0] kg
  Residual before: 10.000 kg
  Correction held: investigate the measurements and model.
```

The remaining residual in the first case is intentional: the total itself has uncertainty.
`ok` means that the requested reconciliation was applied; it does not certify the model or
the measurements. The second case illustrates a policy that holds reconciliation when the
unprojected estimate exceeds the reference threshold.

## Check raw storage and flow measurements

```bash
uv run --frozen --python 3.13 python examples/fluid_baseline.py
```

`BalanceRecord` declares measurement intervals, units and joint covariance.
`balance_residuals()` applies the same linear operator to readings and covariance;
`diagnose()` tests a fixed record against supplied fault profiles and nuisance effects.
The [baseline guide](FLUID_BASELINE.md) describes the array ordering, interval-mean assumption,
shared-reference treatment and the five outcomes. Start here for the new fluid-measurement
baseline. It is an offline calculation with known candidate profiles/onset, not an online
fault detector.

## Estimate with the invariant layer

```bash
uv run --frozen --python 3.13 python examples/invariant_fluid.py
```

Use `GaussianState`, `predict` and `update` from `set_lcm.invariant` for a declared
affine fluid model. The layer estimates additive invariant error, applies its correction,
and retains the innovation and full covariance. `AffineCoordinates` changes state units,
basis and origin while transforming the model consistently. The [guide](INVARIANT_LAYER.md)
describes masks, correlated uncertainty, numerical refusals and assumptions. This additive
case equals the ordinary KF; general nonlinear IEKF support remains planned.

## Compare camera and gauge levels

```bash
uv run --frozen --python 3.13 python examples/camera_baseline.py --out-dir work/camera-demo
```

This example creates and replays a labeled synthetic recording bundle. It extracts a water
edge from grayscale frames, compensates vertical image motion using a fixed marker, checks
frame IDs and capture times, and compares camera and gauge levels with full shared covariance.
It preserves raw readings and their difference alongside combined estimates.

The bundle separates inference inputs in `observations.npz` from the held-out reference and
synthetic truth in `evaluation.npz`, with a checksum manifest. The [camera guide](CAMERA_BASELINE.md)
describes the measurement contract; the [recording protocol](TANK_RECORDING_PROTOCOL.md)
describes preparation for acquired evidence. This example does not decode compressed video.
Its exact pixel calibration anchors and declared uncertainties belong to the synthetic
experiment; real recordings need their own calibration evidence. Vertical marker registration
does not estimate general 3-D camera pose or odometry. See the [generated report](../results/camera_baseline.md)
for the synthetic comparisons and limitations.

## Prepare a controlled tank recording

Create a blank kit in a new or empty directory:

```bash
uv run --frozen --python 3.13 python -m set_lcm.recording init work/tank-recording
```

The kit contains `recording.json`, separate camera/gauge observation tables and held-out
reference/event tables, plus folders for originals and supporting evidence. Initialization
does not overwrite an existing recording. Equipment-dependent values remain `null`; the
kit contains no acquired measurements or assumed sensor uncertainties.

Follow the [controlled recording protocol](TANK_RECORDING_PROTOCOL.md): declare the equipment,
measurand and datum; link calibration, development, timing and uncertainty evidence; lock the
configuration; then retain originals and populate the tables from the actual acquisition.
Keep reference readings and intervention labels in `evaluation/`. The older
[combined planning CSV](../examples/camera_recording_template.csv) is not imported by this checker.

Inspect the kit and save its complete JSON preflight report outside the recording folder:

```bash
uv run --frozen --python 3.13 python -m set_lcm.recording check work/tank-recording --out-report work/tank-preflight.json
```

The default console output is a short summary. Add `--json` to print the complete report:

```bash
uv run --frozen --python 3.13 python -m set_lcm.recording check work/tank-recording --json
```

`check` exits with code **0** for `ready_for_review`, or **2** for `incomplete` or `invalid`.
An empty kit is expected to return `incomplete` and code 2. Resolve missing declarations
from evidence; do not fill unknown uncertainty or clock values with guessed defaults.

Preflight checks metadata, retained-file checksums and differences between declared capture
times. `ready_for_review` does not validate the physical clock, calibration, covariance or
equipment, and does not mean that real-data inference or quantitative evaluation ran.
Video decoding, uncertain-anchor calibration and any asynchronous measurement alignment
still need a justified real-recording workflow.

## Reconcile your own estimate

The [complete example](../examples/quickstart.py) calls three public functions:

```python
threshold = chi2_quantile(constraint.rank, 0.999)
score = consistency_stat(estimate, covariance, constraint)
result = reconcile(
    estimate, covariance, constraint, mode="hard",
    hold=score > threshold, threshold=threshold, stat=score,
)
```

This snippet assumes the imports and input definitions in the complete example. The required
inputs are:

| Input | Meaning |
|---|---|
| `estimate` | One finite value for each state component. |
| `covariance` | A compatible symmetric positive-definite covariance, including correlations. |
| `ConstraintSet.A`, `.b` | The linear relationship `A @ state = b`, with compatible units. |
| `ConstraintSet.b_var` | Optional variance/covariance of the reference `b`; `None` declares it exact. |
| `ConstraintSet.row_units` | Optional unit for each row of `A @ state` and `b`, for example `("kg",)`. Metadata only: no automatic conversion or dimensional checking. |
| `mode` | `hard` applies the declared constraint update; `soft` adds slack through positive `lam`; `None` checks without projection. |
| `hold` | Your decision to withhold correction; the kernel does not choose a debounce or alarm policy. |

Supplying `threshold` alone does **not** hold a correction. The example explicitly sets
`hold`; the time-series runner has a separate guard/debounce policy.

When declaring row units, covariance entry `b_var[i, j]` has the product of row units
`i` and `j`; a variance vector uses each row's unit squared. The caller must supply the
coefficients and input values in compatible units. Mixed mass/energy rows can be labeled,
but this does not add an energy-balance model or uncertainty on the coefficients of `A`.

The result retains `x_unprojected`, `P_unprojected`, `x`, `P`, `residual_pre`,
`residual_post`, `correction`, `consistency_stat` and `status`. A held correction has no
post-correction residual. Invalid numerical inputs raise `ValueError` rather than becoming
successful estimates. `INFEASIBLE` means the declared exact relationships contradict one
another. `NOT_CONVERGED` is reserved for future iterative solvers and is not emitted here.

The chi-square threshold is a reference under a calibrated Gaussian null, not an established
operational false-alarm rate. See [Methods](METHODS.md) before interpreting it as a probability.

## Replay measurement records

For time-series work, `testbed.runner.run()` takes `PublicInputs`, a sequence of
`Observation` objects, a declared prior, an `EstimatorSpec` and an optional constraint.
Each observation includes measurement values, covariance, a missing-value mask, sample and
arrival times, sensor identities, and optional evidence IDs.

The DAF bridge accepts the supported serialized evidence format. It requires explicit
series selection, units, time semantics, conflict policy and uncertainty declarations.
It is not a generic CSV importer. For your own format, write and test an adapter to those
public inputs; do not treat missing units or unknown uncertainty as zero.

Start from the relevant working integration:

- [`muskingum_reach.py`](../src/set_lcm/experiments/muskingum_reach.py): two river reaches with
  continuity and Muskingum routing, the first topology here whose residual is not a scalar, with
  fault recovery scored against known injected truth.
- [`real_noaa.py`](../src/set_lcm/experiments/real_noaa.py): one gauge, water-level filters,
  no conservation constraint, training and evaluation on separate days.
- [`real_noaa_month.py`](../src/set_lcm/experiments/real_noaa_month.py): the same gauge over a
  month, with the record length as the declared axis: the Rayleigh pair table per window, a
  q fitted on the first half and scored on the disjoint second half both cold and continued,
  and NOAA's stated per-reading sigma described by its median as well as its second moments.
- [`real_water_balance.py`](../src/set_lcm/experiments/real_water_balance.py): storage and
  three flow series, explicit unit conversion and uncertainty assumptions.
- [`phase1.py`](../src/set_lcm/experiments/phase1.py): simulated measurements and known faults,
  with hidden truth used only by evaluation and the explicitly labeled oracle.
- [`real_fluid_baseline.py`](../src/set_lcm/experiments/real_fluid_baseline.py): direct
  interval-aware measurement balances, with shared-reference covariance and no sensor labels.
- [`camera_baseline.py`](../src/set_lcm/experiments/camera_baseline.py): synthetic image
  measurements, frame/timing admission and camera/gauge fusion, with evaluation evidence kept separate.

The current runner gates use by arrival time and consumes the supplied sample order; a
delayed sample can hold up later samples. General out-of-sequence measurement handling is
not implemented. The historical reservoir filter example uses an acknowledged approximation for daily
mean storage/flow alignment. Neither should be silently generalized to a new deployment.

## Declare a site

A site is a TOML document under `declarations/`, not a Python module. It declares the states
and their units, one or more constraint variants over them, the sensors with the uncertainty
this consumer declares and the citation for it, the committed record, and any other declared
number. `set_lcm.declaration.load` reads it and `Declaration.constraint_set(variant, ...)`
builds the `ConstraintSet` the kernel takes.

```python
from set_lcm.declaration import load

site = load("declarations/ridgway.toml")
cs = site.constraint_set("open", readings={"closure": s0}, variances={"closure": s0_var})
```

A row whose right-hand side is a *reading* cannot be resolved when the file is read, so the
declaration names the sensor and the index and the value arrives at build time. Supplying a
reading for a row that declared a constant, or omitting one a row asked for, raises rather
than quietly substituting one for the other.

The format refuses more than it accepts, because a declaration that guesses is worse than no
declaration: an unknown key raises and names itself, a row using a state its variant does not
declare raises and names both, a declared state that no row constrains raises unless the
variant lists it under `unconstrained`, and a declared sigma without a citation raises. The
one implicit rule is that a coefficient omitted from a row is zero — the orphan-state check is
what keeps that from hiding a mistake. Every declared string is stripped of surrounding
whitespace, which is the only value the loader alters; `src/set_lcm/declaration.py` says why.

## Reproduce the reports

These commands regenerate experiments from committed code and, where applicable,
committed measurement inputs, writing the corresponding files under `results/`:

```bash
uv run --frozen --python 3.13 python run_experiments.py
uv run --frozen --python 3.13 python -m set_lcm.experiments.calibration
uv run --frozen --python 3.13 python -m set_lcm.experiments.sweep
uv run --frozen --python 3.13 python -m set_lcm.experiments.second_balance --quiet
uv run --frozen --python 3.13 python -m set_lcm.experiments.muskingum_reach --quiet
uv run --frozen --python 3.13 python -m set_lcm.experiments.errors_in_variables --quiet
uv run --frozen --python 3.13 python -m set_lcm.experiments.eiv_projection --quiet
uv run --frozen --python 3.13 python -m set_lcm.experiments.real_noaa
uv run --frozen --python 3.13 python -m set_lcm.experiments.real_noaa_month --quiet
uv run --frozen --python 3.13 python -m set_lcm.experiments.real_water_balance
uv run --frozen --python 3.13 python -m set_lcm.experiments.real_taylor_park --quiet
uv run --frozen --python 3.13 python -m set_lcm.experiments.real_diagnosis --quiet
uv run --frozen --python 3.13 python -m set_lcm.experiments.fluid_baseline --quiet
uv run --frozen --python 3.13 python -m set_lcm.experiments.real_fluid_baseline --quiet
uv run --frozen --python 3.13 python -m set_lcm.experiments.invariant_layer --quiet
uv run --frozen --python 3.13 python -m set_lcm.experiments.camera_baseline --out-dir results
```

The default test suite includes small experiments and regeneration of the real-data reports.
The slow tests additionally regenerate the full simulation grid, calibration and sweeps:

```bash
uv run --frozen --python 3.13 --dev pytest -q
uv run --frozen --python 3.13 --dev pytest -q -m slow
```

Some small-matrix workloads are slower when the numerical library starts many worker threads.
CI sets `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`; use the same process environment
when reproducing its timings. Runtime depends on the machine and is not a performance claim.

Two DAF validation tests require a separate upstream checkout specified by `DAF_ROOT`.
They are skipped when it is absent. Normal use of the committed replay examples does not
require that checkout. See [data provenance](../data/daf/PROVENANCE.md) for its exact pins.
