# Camera and gauge level baseline

This prototype compares a camera level estimate, a gauge reading, and their
covariance-weighted combination on controlled synthetic records. Camera inputs
are actual rendered grayscale arrays: the detector must find the water edge and
a separate fixed marker in those arrays. A held-out reference and the simulator's
true level are used for evaluation only.

The implemented camera model is deliberately narrow: a fixed, front-parallel
view of a horizontal intensity step in a declared region of interest. A separate
marker supports vertical image translation. There is no field validation, general
water segmentation, perspective correction, camera rotation model, mobile-camera
tracking, or 3-D odometry. No FluidNexus code or generated-view reconstruction is
imported. Synthetic rendered test frames retain `source_kind="synthetic_test"`;
generative views are not independent measurements.

## Run and inspect a labeled fixture

From the repository root:

```sh
uv run --frozen --python 3.13 python examples/camera_baseline.py --out-dir work/camera-demo
```

The example creates and replays a labeled synthetic fixture. Its recording bundle
keeps inference inputs in `observations.npz` and held-out reference observations
and synthetic truth in `evaluation.npz`. A JSON manifest records the declarations
and file checksums. The separate evaluation file is not an estimator input. This
is an array-bundle workflow; it does not decode compressed video or import real
recording tables. The separate preparation kit described below checks recording
metadata and retained evidence without running this synthetic inference wrapper.

`frame-preview.png` is an 8-bit display preview; inference reads the original
floating-point arrays in the archive. The synthetic renderer declares a 1e-9
intensity grid to stabilize exact content hashes across supported builds. This
quantization is part of the test-image model, not a repair applied to acquired video.

The benchmark's [`camera_baseline` experiment](../src/set_lcm/experiments/camera_baseline.py)
provides `build_case`, `infer_measurements`, and `run_experiment`. The
[generated report](../results/camera_baseline.md) records the actual run results
and their provenance. Interpret synthetic performance as evidence about the
declared renderer, interventions, covariance, and quality policy; it does not
establish camera accuracy on a real tank.

The experiment's inference wrapper uses a fixed synthetic protocol, including
its ROI and uncertainty declarations. It is not a configurable real-recording
adapter. Evaluation records reuse one calibration realization from separate
development seeds; pooled coverage is empirical conditional on that calibration.
Frames and paired scenarios are not independent calibration trials.

## What enters the estimate

The stages preserve raw evidence and exclusions:

1. [`frame_quality.audit_frames`](../src/set_lcm/frame_quality.py) retains capture
   counters, original row indices, actual timestamps, exact-content hashes,
   source kinds, missing markers, and per-row admission flags. It never sorts,
   inserts, or retimes frames.
2. [`camera.detect_level`](../src/set_lcm/camera.py) measures a declared horizontal
   edge. The caller supplies ROI bounds, edge polarity, contrast, column-spread,
   and profile tolerances. The detector checks individual column profiles, then
   combines accepted edge rows. A failed check yields a missing level, not a
   substituted estimate. Contrast and quality thresholds do not estimate variance
   or provide a probability that the measurement is correct.
3. `register_vertical_marker` measures a separate stationary marker in its own
   ROI. Its displacement from the declared reference row is subtracted from the
   water row. Missing registration does not silently fall back to uncompensated
   pixels. This compensates one image-translation coordinate only.
4. `fit_linear_calibration` and `LinearLevelCalibration.map_rows` map pixel rows
   to metres over a declared validity range, preserving missing rows and shared
   calibration uncertainty. Out-of-range rows are excluded, not extrapolated.
5. [`camera_fusion.compare_level_sources`](../src/set_lcm/camera_fusion.py) returns
   gauge-only, camera-only, and combined level series, plus the original
   gauge-minus-camera disagreement. Fusion does not discover or automatically
   remove a biased sensor. Reference measurements and fault labels do not enter
   this function.

Use raw detected water rows with `map_rows(..., marker_displacement=...)` to apply
registration and propagate its uncertainty together. `compensate_level` also
provides a single-detection correction; do not subtract the same marker again
when mapping an already corrected row.

Valid camera and gauge readings must have exactly matching timestamps on the
same declared seconds/origin axis. Excluded camera frames may retain their bad
timestamps for inspection. The fusion helper performs no nearest-time matching,
interpolation, or temporal smoothing. It computes a two-source best linear
unbiased estimate at each admitted time under the supplied covariance. Weights
can be negative when sources are correlated.

The benchmark also checks the preserved gauge-camera difference with the
finite-record diagnostic module. Its gauge-drift and camera-drift templates
describe the same line with opposite signs and unrestricted signed amplitudes;
that evidence cannot uniquely attribute the drift to one instrument. The onset
and profiles are supplied design information, not discovered from the video.
A shared bias can leave the difference consistent while both instruments are
wrong relative to the independent reference. Fusion cannot eliminate a shared
calibration error merely by combining the two readings.

## Uncertainty and calibration assumptions

The fitted calibration is `h = a*p + b`, with parameter covariance ordered
`[a, b]`. GLS uses the supplied physical-reference covariance and treats the
anchor pixel coordinates as exact. That is a controlled synthetic assumption,
not an appropriate default for uncertain anchors measured in real images. An
errors-in-variables calibration method is not implemented. The fit also does not
estimate a noise scale from residuals or decide whether the linear calibration
is acceptable for a real optical setup.

For raw water-row covariance `Cw`, marker-displacement covariance `Cm`, and their
declared cross covariance `Cwm`, corrected rows have covariance

```text
Cp = Cw + Cm - Cwm - Cwm.T
Ch = J @ Ccal @ J.T + a**2 * Cp,    J[i] = [corrected_row[i], 1]
```

Marker reference-row uncertainty belongs in `Cm`, including the shared effect
across frames. Omitting `Cwm` explicitly assumes independence between water-row
and marker errors. Calibration-parameter errors are assumed independent of
localization and marker errors. `Ch` is a first-order propagation: the
slope/pixel product term is omitted, and quality selection can change the error
distribution. Exact Gaussian interval coverage is not asserted after these
operations. First-order propagation with covariance terms follows the approach
described in [NIST TN 1297, section 5](https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-5-combined-standard-uncertainty).

The covariance of a `LevelSeries` is indexed by `observed_indices`, while its
height/status vectors retain every original row. Fusion accepts full gauge
covariance and a required gauge-camera cross-covariance declaration. Passing
`None` explicitly declares independence; a matrix states the dependence in the
documented valid-camera ordering. The entire joint covariance must be valid.
Shared calibration and shared instrument references must not be replaced by
independent per-frame noise. The full record covariance is propagated through
the fusion weights, even though those weights do not smooth across time.

An independent evaluation reference still has uncertainty. Preserve that
uncertainty and any shared-reference correlations when assessing errors against
it. Simulator truth is available only in the synthetic evaluation artifact; it
must never be substituted for an inference input or treated as available in a
real recording.

## Frame timing and spectral diagnostics

Frame IDs are capture counters, not newly assigned positions after dropping
frames. `audit_frames` checks the expected elapsed interval as the positive
counter difference times the declared period. A frame following a skipped
capture can remain individually admitted when its actual timestamp matches that
elapsed interval, but the gap remains flagged. A gap in original source indices
does not alter expected physical time. Duplicate/backward IDs or timestamps,
missing rows, generated sources, and clock violations are refused by the
metadata policy.

Hashes cover exact logical pixel bytes plus dtype, byte order, and shape. They
detect repeated content, not whether a camera is truthful. Static water can
legitimately produce identical frames. Repetition is advisory by default;
`reject_repeated_content=True` makes exclusion an explicit policy. Original
timestamps and flags remain available either way. An externally marked missing
placeholder may carry a caller-declared expected slot time, but that declaration
is not a measured capture and the row remains excluded.

`frame_spectrum` analyzes a supplied scalar feature, such as edge motion,
brightness, or a residual, only on a complete, contiguous window of at least
four admitted frames. It checks original indices, capture counters, and actual
capture intervals. The audit's permitted acquisition jitter does not authorize
spectral resampling: a window must separately meet the numerical uniform-clock
tolerance. Gaps and irregular clocks are refused, not silently filled or treated
as a uniform sequence.

The helper detrends, applies a symmetric Hann or boxcar window, and reports a
one-sided power spectral density in `feature_unit**2/Hz`. DC and Nyquist bins are
not doubled. The peak is a frequency-bin feature, not an anomaly score or a
calibrated fault probability. Original values, indices, timestamps, hashes, and
source kinds remain attached. Applying it to successive complete windows can
show changes in frequency content. Sampling frequency, detrending, window
choice, and PSD normalization are explicit choices in the
[SciPy STFT documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html);
FSRT implements its bounded NumPy periodogram directly and does not require
SciPy or provide a general STFT reconstruction pipeline.

A spectrum cannot recover missing physical timestamps or recognize a consistently
wrong frame-rate/time-unit declaration by itself. Aliasing and a mislabeled
clock require an independent timing reference. Periodic motion or illumination
can be legitimate; spectral features alone do not identify a sensor fault.

## Plan a first real recording

Start with the blank recording kit and [controlled tank protocol](TANK_RECORDING_PROTOCOL.md):

```sh
uv run --frozen --python 3.13 python -m set_lcm.recording init work/tank-recording
uv run --frozen --python 3.13 python -m set_lcm.recording check work/tank-recording --out-report work/tank-preflight.json
```

Initialization creates a manifest, separate observation/evaluation tables and
evidence folders, with equipment-dependent declarations left unknown. The empty
kit is expected to return `incomplete` and exit code 2. Supply actual equipment,
calibration, timing and uncertainty evidence; no measurements or uncertainties
are supplied by the kit. The [usage guide](USAGE.md#prepare-a-controlled-tank-recording)
explains the output modes and exit codes.

Preflight checks metadata, file checksums and declared capture-clock differences.
`ready_for_review` does not establish synchronization, calibration validity,
sensor accuracy or real-data inference. Plan the controlled acquisition before
extending the measurement model:

1. Fix the camera in front of a tank with a clearly visible horizontal level
   edge, stable lighting, and a separate stationary fiducial. Record water and
   marker ROIs, polarity, exposure settings, and the physical calibration
   evidence. Keep the edge within the calibration range. Perspective, rotation,
   moving-camera operation, and changing lens geometry need additional models.
2. Retain original recordings and device clocks, available acquisition counters
   with their declared source, decoded-frame indices and evidenced clock mappings.
   Preserve drops, repeats and hashes. Follow the protocol's supported metadata
   format without inventing capture times. Retrieval or file-write time does not
   establish capture time; expected missing slots are not measured captures.
3. Record the gauge and a separate evaluation reference with their own capture
   times, level units in metres, standard uncertainties, and uncertainty evidence.
   Establish synchronization and latency before claiming matching instants;
   the current fusion helper does not align an asynchronous real recording.
4. Preserve localization, marker/reference-row, calibration, gauge, and evaluation
   reference uncertainties. Use linked covariance evidence for shared and
   cross-time terms; scalar uncertainty columns alone cannot express the full
   covariance contract. Real noisy calibration anchors require a justified
   calibration procedure beyond the delivered exact-pixel GLS case.
5. Keep held-out reference columns and any intervention labels in an evaluation
   artifact, separate from the camera/gauge inference bundle. Compare nominal
   records and controlled timing, visibility, and vertical-motion changes before
   making a field-performance claim. Retain failures and source disagreement
   alongside the combined estimate.

The older [`camera_recording_template.csv`](../examples/camera_recording_template.csv)
remains a header-only planning aid and is not imported by the preflight checker.
Its decoded-array hash field uses `frame_content_hash`, including dtype and shape.
The kit's `frames.csv` instead declares a `frame_sha256` file checksum. Neither
hash validates the physical clock, calibration, reference or image model.
