# Controlled tank recording protocol

This is an acquisition plan for the first synchronized tank video, gauge and
independent reference experiment. No real recording or field performance is
claimed here. Choose the equipment, operating range, timing tolerances and
evaluation targets before acquisition; this protocol supplies no assumed sensor
accuracy, uncertainty, sampling rate or required run duration.

The first question is whether the camera can supply a defensible second level
measurement on a controlled tank, including during vertical image motion and
visibility loss. The comparison is gauge-only, camera-only and their combination
on the same physical times, while preserving the original disagreement. A
combined estimate is not evidence that either instrument is healthy.

Read the [camera baseline guide](CAMERA_BASELINE.md) for delivered capabilities
and the [recording template](../examples/camera_recording_template.csv) for a
header-only planning aid. The synthetic example and its bundle loader implement
a fixed synthetic protocol; they must not relabel captured frames or supply
their synthetic covariance to a real recording. General video decoding,
asynchronous alignment and uncertain-pixel calibration are not established by
the synthetic benchmark.

## Prepare the recording kit

From the repository root, create a new kit in the ignored `work/` folder:

```sh
uv run --frozen --python 3.13 python -m set_lcm.recording init work/tank-recording
uv run --frozen --python 3.13 python -m set_lcm.recording check work/tank-recording --out-report work/tank-preflight.json
```

The second command initially reports `incomplete` and exits with code 2. That is
expected: the kit contains a blank `recording.json` and CSV headers, with no
measurements, equipment specifications or invented uncertainties. Use `--json`
to print the full report. The normal display lists the first ten issues; the
saved JSON contains every issue. Reports must be written outside the recording
folder so the checker cannot overwrite source evidence.

1. Fill the equipment identities, measurand, datum and prospective acceptance
   criteria in `recording.json`. Keep unknown values as `null`.
2. Attach calibration, development, clock and uncertainty evidence, then lock
   the configuration before examining the held-out reference. Each evidence
   reference uses a portable relative `path` and lowercase `sha256` file digest.
   One document may satisfy several related references; separate documents are
   not required for every field.
3. Preserve original recordings under `raw/`, listing their IDs, paths and
   hashes in `raw_videos`. `frame_processing_evidence` documents the decoder,
   version, frame extraction, intensity conversion and source-frame mapping.
   Never apply the synthetic intensity grid to acquired evidence.
4. Populate `observations/frames.csv`, `observations/gauge.csv`, and
   `evaluation/reference.csv`. Their headers are the implemented input contract.
   Keep intervention notes in `evaluation/events.csv`; the preflight does not
   interpret event labels. The older combined planning CSV is a different
   planning aid and is not imported by this checker.
5. Run preflight and review every issue. `invalid` means the supplied files or
   declarations violate this import contract, not that an instrument has failed.
   `ready_for_review` exits with code 0 and means the checked artifacts and
   declarations are present. It does not certify the evidence in those documents,
   numerically validate their covariance, run an estimator or measure accuracy.

Frame `present` values are `true` or `false`; a missing payload has no substituted
path or checksum. Blank gauge/reference levels mean missing readings. Their
standard uncertainties must be supplied for present values. Matching `sample_id`
values declare a common acquisition event; they do not make unequal clocks equal.
The report retains missing-source IDs and reports actual declared skew without
retiming or interpolation. `frame_sha256` checks the stored frame file bytes,
not the decoded-array hash used by `frame_content_hash` in the synthetic baseline.

Declare `frame_counter_kind` as `hardware_capture`, `decoder_index`, or
`unavailable`. A missing hardware counter does not invalidate a physical
measurement, but prevents claims that numbering alone establishes acquisition
gaps. When the counter is unavailable, leave `frame_id` blank. Present frames
still identify the retained original recording and source-frame index; source
indices must increase within each recording.
Decoder counters may restart in a different retained recording. A hardware
counter reset requires its own acquisition model; the checker does not infer one.
Register each original recording once: byte-identical originals under multiple
IDs are refused by this pilot contract. They cannot establish additional capture
events without further acquisition-identity evidence.

This checker currently accepts capture timestamps in the explicit
`YYYY-MM-DDTHH:MM:SS[.ffffff]Z` or timezone-offset form, with up to six fractional
digits. Keep raw device clocks and their mappings in clock evidence. A common
monotonic clock can be scientifically valid, but requires an adapter beyond this
bounded CSV contract. Do not invent a UTC mapping or truncate a finer timestamp
to pass the checker. Preserve unsupported source records for that adapter.

`capture_utc` must be the capture instant within the camera exposure or sensor
support interval; instantaneous observations have equal support endpoints.
Delivery/readout timestamps belong in raw clock evidence unless a justified
mapping establishes capture time. Equal normalized capture timestamps still
do not prove equal physical observation support or sufficient synchronization.

## 1. Define the measurement and equipment roles

Write a short session plan identifying the tank, camera, gauge, reference and
operator. Define level in metres relative to a surveyed or otherwise documented
physical datum. Specify whether a reading represents a local surface height,
spatial average or time average, and how the camera edge corresponds to it.
Meniscus, waves and different sensor locations can create real differences
between these quantities. Do not force them into agreement by changing a datum
after viewing validation results.

| Role | Evidence to retain before recording |
|---|---|
| Tank and datum | Geometry, intended level range, measurement locations, datum definition and its uncertainty, and a setup photograph or drawing. |
| Camera | Device/lens identity, mount, image dimensions, focus, exposure and shutter behavior, frame-rate mode, image processing settings and timestamp source. |
| Fixed marker | A marker fixed to the physical reference structure, separate from the water ROI; its reference row, physical placement and uncertainty. |
| Gauge under test | Sensor identity, raw output and conversion to level, calibration/corrections, range, sampling/averaging interval, latency and uncertainty evidence. |
| Held-out reference | A separately acquired physical measurement with its own calibration, timing, location, datum and uncertainty; not a copy of the gauge or an annotation produced by the camera estimator. |
| Clock evidence | Trigger or synchronization mechanism, raw clock logs, offset/drift checks, acquisition latency, exposure timing and uncertainty of any clock mapping. |

A different device or separately recorded reference is not automatically
statistically independent. Identify shared calibration standards, datum surveys,
environmental corrections and clocks. Prefer a reference measurement chain that
can reveal errors shared by the camera and gauge. If only a common standard is
available, preserve its covariance and describe the weaker independence claim.
Do not call the reference exact ground truth.

Preflight refuses the crudest form of this only: three instruments whose declared
evidence is literally the same file. That is a document check, not an independence
finding — two genuinely separate documents can still describe one calibration chain,
one datum survey or one clock. `uncertainty.reference_dependence` is where the real
claim goes, and nothing in the kit can verify it.

Start with a fixed, approximately front-parallel view, stable lighting and a
visible horizontal water edge. Preserve the settings actually used, including
any automatic processing that cannot be disabled. The current detector expects
a declared horizontal intensity-step profile; first establish on development
images whether the real surface, reflections and tank wall satisfy that model.
Failure is useful evidence that a different measurement model is needed.

Place the marker so it tracks the intended image translation without being
carried by the water. Document its depth and relationship to the water plane.
Rotation, perspective, lens distortion, depth-dependent motion and deformation
are not corrected by subtracting one marker row. The vertical-motion trial
below tests this assumption; it does not establish general 3-D odometry.

## 2. Separate calibration, development and validation sessions

Assign immutable session IDs and roles before looking at held-out results. Use
whole recording sessions as the split; adjacent frames from the same recording
are not an independent validation set.

1. **Calibration:** collect physical level anchors over the intended range,
   with repeated independent settings or readings as the setup permits.
   Preserve the anchor images, measured levels, uncertainties, correlations,
   datum evidence and how each anchor was established. Include checks for
   hysteresis or approach direction when relevant. Do not extrapolate beyond
   the supported calibration range.
2. **Development:** choose ROIs, edge polarity, quality thresholds, marker
   policy, timing/alignment rules and covariance model. Use these sessions to
   assess localization and registration errors and the suitability of a linear
   pixel-to-level relationship. Repeated annotations of one image do not by
   themselves measure capture-to-capture error or shared calibration error.
3. **Lock:** record the code revision, calibration version, parameter files,
   frame-admission rules, evaluation windows and success criteria. Choose the
   sampling rate and record length from the actual dynamics, expected timing
   error and precision question, rather than inheriting synthetic constants.
4. **Held-out validation:** acquire separate sessions under the locked setup.
   Keep reference readings and intervention logs in evaluation artifacts.
   Produce camera/gauge estimates before exposing those artifacts to scoring.

The delivered GLS calibration treats anchor pixels as exact. Real extracted
anchor rows usually have uncertainty: quantify it and use a justified
errors-in-variables or other validated calibration method before assigning a
physical camera covariance. Do not set pixel uncertainty to zero to fit the
existing API. A documented approximation requires a sensitivity study showing
why it is adequate for the particular range and uncertainty budget.

Record pre/post-session calibration checks without silently refitting to
held-out discrepancies. A changed mount, focus, zoom or geometry can invalidate
the calibration. If results motivate a model or threshold change, retain that
run as development evidence and obtain a new held-out session for the revised
claim. Reusing one calibration across sessions remains a shared source of
uncertainty; evaluating calibration variability requires separate calibrations.

## 3. Establish the physical timeline

Retain each device's original clock values and time units. Also record UTC when
available and a seconds axis with a named origin. Preserve the mapping between
them, including clock offset, drift, uncertainty and any discontinuity. A
file-write timestamp, nominal playback FPS or container presentation timestamp
alone does not establish the physical exposure time.

Use a shared acquisition trigger or a synchronization event observed and logged
by the relevant acquisition systems where the equipment permits. Check timing
near the beginning and end so one synchronization point is not mistaken for
proof of no drift. A flash visible only in the camera cannot establish the
gauge's capture time unless its relation to the gauge clock is also recorded.
Document whether a timestamp describes exposure start, midpoint, end, readout
or delivery, and record integration/exposure durations and sensor averaging.
Rolling shutter may give the marker and water rows different exposure times.

Preserve original capture counters when the device provides them. Keep decode
order/source indices separately. If hardware counters are unavailable, label
the available indices honestly; newly numbering decoded frames cannot prove
that acquisition had no drops. Retain missingness, repeats and irregular
intervals. Do not repair them by changing FPS, inserting frames or overwriting
capture times. An expected missing slot is a declaration, not a captured image.

For a moving level, verify that timing uncertainty and different averaging
intervals are acceptable for the intended comparison. If not, acquire better
synchronization or build and validate an explicit alignment/measurement-window
model. Keep any converted common-time axis as a derived field alongside raw
times, its mapping and propagated uncertainty. Do not estimate a favorable time
shift from held-out reference agreement. Rounding timestamps until they match
does not establish synchronization.

The current fusion function requires exactly matching declared times for valid
camera/gauge pairs and does no interpolation. Do not feed asynchronous data
into it by overwriting one clock. A manual reference sampled only at settled
plateaus can validate those plateaus; it cannot validate the intervening moving
trajectory without an additional justified model. Unequal sample rates are an
acquisition fact, not permission to invent reference readings.

## 4. Acquire controls and interventions

Record the same basic controls in development and held-out sessions, with
intervention order and repetition planned in advance. Keep the times actually
executed and any deviations, even when an intervention has little visible
effect. Separate physical interventions from changes made later to a copy of
the data. Avoid changing several conditions at once in the first trial.

| Session condition | What to record and compare |
|---|---|
| Settled levels | Several supported levels, with camera, gauge and reference observed together. Verify the chosen averaging interpretation; preserve repeated identical images as potentially legitimate static content. |
| Changing level | Controlled filling and draining within the calibration range. Retain the independent reference at a rate and integration time appropriate to the dynamics. Test physical change without labeling it a sensor fault. |
| Water-edge obstruction | Introduce and remove a documented occlusion while retaining the gauge and reference. Record admission, missing output and recovery; do not judge only the surviving easy frames. |
| Vertical camera motion | Keep the physical level independently observed while applying a controlled vertical displacement with orientation/geometry as stable as possible. Compare registered and unregistered camera estimates on common support, and retain marker displacement and uncertainty. |
| Marker obstruction | Obscure the fixed marker separately. Check that compensation refuses unsupported readings rather than silently using an uncompensated estimate. |
| Timing/data-integrity control | Preserve any actual acquisition drops or stalls. Separately labeled replay copies may inject drops, duplicated payloads or clock offsets to test software admission; these are injected data defects, not observed device faults. |

Declare intervention magnitudes from the equipment's range and the experimental
question. Keep that declaration separate from any magnitude estimated by FSRT.
Do not infer that visibility loss, frame repetition or disagreement proves a
particular broken sensor. In particular, camera and gauge drift can remain
confounded in their difference, while a shared level bias may cancel entirely.

## 5. Build an uncertainty budget from evidence

For every uncertainty term, retain its source, unit, method, applicability and
dependence on other terms. Distinguish standard uncertainty from an expanded
uncertainty or a stated tolerance. A calibration certificate's coverage factor
and stated coverage must travel with its value; do not enter an expanded
uncertainty directly as a one-standard-deviation input. NIST defines expanded
uncertainty as a coverage factor times combined standard uncertainty in
[TN 1297, section 6](https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-6-expanded-uncertainty).
Unknown uncertainty is missing information, not zero. Preflight keeps those apart:
a blank standard uncertainty is `incomplete`, while a declared `0` is accepted and
raised for review, because zero is a claim that the instrument is exact and the
reviewer should see it made rather than inherit it. NIST distinguishes statistical (Type A) and
other (Type B) evaluations and represents dependencies by covariance; these
categories do not simply mean random versus systematic error, as explained in
[NIST TN 1297, section 2](https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-2-classification-components-uncertainty).

The budget should address the following where material to this setup:

- Water-edge localization, illumination/reflections, finite pixel resolution
  and the validity of the declared edge profile.
- Marker localization, its reference row, shared reference-row error and any
  dependence between water and marker extraction.
- Calibration slope/intercept, anchor position/level uncertainty, datum and
  optical geometry, including correlations with later observations.
- Gauge and reference calibration, resolution, drift, physical corrections,
  location, response/integration time and shared standards.
- Capture timing and clock mappings, including shared offsets and the effect
  of timing uncertainty during physical level changes.

Retain applied corrections separately from raw readings and retain uncertainty
in those corrections. Propagate covariance terms as well as individual
variances; a shared datum or calibration error does not shrink just because a
record contains many frames. This follows the combined-standard-uncertainty
approach in [NIST TN 1297, section 5](https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-5-combined-standard-uncertainty).

The current camera propagation is first order and assumes calibration errors
independent of row/marker errors. Validate those assumptions for the recording
or extend the model. Quality selection, temporal dependence and nonlinear
effects can make nominal Gaussian intervals inappropriate. Do not tune the
covariance on held-out discrepancies to obtain a preferred coverage rate.

## 6. Preserve a replayable evidence package

Keep the original device video/files unchanged, including compression and
container metadata when video is the source. Retain acquisition logs, gauge and
reference exports, calibration evidence, setup photographs and intervention
logs. Record file checksums before creating derivatives.

For decoded frames, preserve source file identity, original frame/packet timing,
capture counter if known, decode/source index, dimensions, dtype, color format
and exact-content hash. Record the decoder version and all conversions, crops,
resizing or color-to-grayscale operations. Keep the original video even when
lossless decoded frames are also stored; re-encoding a lossy source does not
recover its lost information. Preview images are display artifacts.

Use `source_kind="measured"` for acquired measurement frames and clearly label
synthetic or generated derivatives. Do not treat generative views as additional
observations. A file checksum identifies file bytes; `frame_content_hash`
identifies decoded array bytes plus shape and dtype. Neither validates a
physical clock, sensor, calibration or measurement model.

Separate the package by purpose, with a manifest that links the pieces:

| Artifact role | Contents |
|---|---|
| Acquisition evidence | Original video/frames and exports, physical clocks, IDs, missingness, setup and device metadata. |
| Inference inputs | Camera and gauge measurements, permitted calibration/development evidence, locked policies and covariance declarations. |
| Evaluation evidence | Held-out reference readings with timing/covariance, independently recorded intervention logs and adjudication notes. No synthetic truth column is populated for real data. |
| Derived results | Extracted rows, registration, exclusions/reasons, original and combined level estimates, raw disagreement, uncertainty and evaluation report. |

The CSV planning template cannot express the full covariance by scalar columns
alone. Link a versioned covariance artifact with explicit ordering, units and
evidence. Retain raw observations even when conversion or admission fails.
Version the adapter, parameters and processing code, so replay reproduces the
derived results without opening evaluation evidence during inference.

## 7. Review the first validation record

Lock the questions and comparison rules before scoring. Start with measurement
validity and availability, then assess uncertainty and comparative performance.
Report all acquired opportunities, admission/exclusion reasons and missing
reference periods. Compare gauge-only, camera-only and combined estimates on
the same valid times, and separately report each method's full availability.
Include the registered/unregistered comparison for the vertical-motion control.

Call a difference from the reference a reference discrepancy, not known true
error. Its covariance is

```text
C_difference = C_estimate + C_reference - C_estimate,reference - C_reference,estimate
```

The simpler sum requires an explicit independence justification. Report signed
discrepancies, RMSE on declared common support, interval results with their
denominators, and variation across sessions. Distinguish per-frame results from
independent session or calibration replicates; do not count autocorrelated frames
as independent evidence for a precise performance claim. If sampling of the
reference is too sparse or its uncertainty dominates, report that limitation.

Retain failures and unexplained disagreements. A failed image model or unknown
timing/covariance is a reason to withhold a quantitative fusion or accuracy
claim, while still preserving and inspecting the acquisition. Revisit the
development design before obtaining fresh held-out evidence. The first recording
can establish feasibility under its documented conditions; it cannot establish
industrial accuracy, operational false-alarm rates or general field diagnosis.

Spectral summaries may be inspected only on complete, contiguous, uniformly
timestamped feature windows. Their peaks describe frequency content, not fault
truth. They do not establish that the physical clock is correct, and the
spectral path must not interpolate across gaps to obtain a result.
