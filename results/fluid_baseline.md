# Synthetic fluid measurement benchmark

Generated with Python 3.13.12, numpy 2.5.3 on Linux-6.18.44-fc-v24-x86_64-with-glibc2.39; source sha256 1baf7ee49925, git 9602053a79 (source dirty). Latency columns are wall-clock on this machine and are not a claim.

**Synthetic research evidence only. No field validation or operational alarm performance is established.**

Each case has 16 development records and 128 disjoint evaluation records. Configurations are fixed; neither set selects thresholds. The table uses evaluation records only.

- **Scope:** Synthetic Gaussian fixed-horizon evidence only; no field validation or operational false-alarm claim.
- **Selection:** All noise scales, templates, amplitudes, horizon and alpha were declared before evaluation. No parameters or thresholds are fitted on either seed set.
- **Independence:** Independent noise per seed within a case; seeds reused across cases for paired comparisons. Do not treat pooled case outcomes as independent.
- **Onset:** Known onset and nominal commanded flow pattern; gain templates use that declared profile, not noisy evaluation observations.
- **Measurement:** Interval-mean storage and flows, constant net flow within each interval, full H C H' covariance including one shared Gaussian calibration reference.
- **Amplitudes:** Arbitrary declared simulation magnitudes in m3, m3/s or fractional gain, with alternating signs; not industry sensitivity targets.
- **Noise:** Independent per-reading Gaussian noise plus a shared rank-one Gaussian calibration perturbation. Faults alter means only; covariance is fixed, including for gain.
- **Event:** One test after a fixed record per seed; detection is rejection of no-fault after declared nuisance. No repeated-window or online delay claim.
- **Baseline:** Simple full-record balance-only Mahalanobis detector at the same alpha, with no nuisance or attribution. Physical omitted flow can rightly violate its gauged balance.
- **Intervals:** Candidate-conditional Gaussian intervals, not selection-adjusted. Report both true-profile coverage and coverage conditional on correct identification with explicit denominators.
- **Causal limits:** A channel/profile explanation is conditional on the candidate and nuisance catalogue. Consistent does not mean healthy; an unmodeled physical input can mimic a fault.

Not tested: unknown onset, multiple faulty instruments, uncertain gain templates, colored or nonlinear model mismatch, real sensor degradation, thermal balances.

| case / magnitude (±) | diagnostic reject | balance-only reject | consistent | identified | ambiguous | unexplained | insufficient | wrong / identified | covered / correctly identified intervals |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_fault/weak: 0 m3 | 0/128 | 0/128 | 128 | 0 | 0 | 0 | 0 | n/a (0) | n/a (0) |
| storage_step/weak: 0.02 m3 | 0/128 | 0/128 | 128 | 0 | 0 | 0 | 0 | n/a (0) | n/a (0) |
| storage_step/strong: 1.6 m3 | 128/128 | 128/128 | 0 | 128 | 0 | 0 | 0 | 0/128 | 123/128 |
| storage_drift/weak: 0.001 m3/s | 0/128 | 0/128 | 128 | 0 | 0 | 0 | 0 | n/a (0) | n/a (0) |
| storage_drift/strong: 0.16 m3/s | 128/128 | 128/128 | 0 | 0 | 128 | 0 | 0 | n/a (0) | n/a (0) |
| inflow_offset/weak: 0.001 m3/s | 0/128 | 0/128 | 128 | 0 | 0 | 0 | 0 | n/a (0) | n/a (0) |
| inflow_offset/strong: 0.16 m3/s | 128/128 | 128/128 | 0 | 0 | 128 | 0 | 0 | n/a (0) | n/a (0) |
| inflow_gain/weak: 0.0005 fraction | 0/128 | 0/128 | 128 | 0 | 0 | 0 | 0 | n/a (0) | n/a (0) |
| inflow_gain/strong: 0.08 fraction | 128/128 | 128/128 | 0 | 9 | 119 | 0 | 0 | 0/9 | 8/9 |
| constant_flow_offset/weak: 0.001 m3/s | 0/128 | 0/128 | 128 | 0 | 0 | 0 | 0 | n/a (0) | n/a (0) |
| constant_flow_offset/strong: 0.16 m3/s | 128/128 | 128/128 | 0 | 0 | 128 | 0 | 0 | n/a (0) | n/a (0) |
| constant_flow_gain/weak: 0.0005 fraction | 0/128 | 0/128 | 128 | 0 | 0 | 0 | 0 | n/a (0) | n/a (0) |
| constant_flow_gain/strong: 0.08 fraction | 128/128 | 128/128 | 0 | 0 | 128 | 0 | 0 | n/a (0) | n/a (0) |
| storage_common_offset/weak: 0.02 m3 | 0/128 | 0/128 | 128 | 0 | 0 | 0 | 0 | n/a (0) | n/a (0) |
| storage_common_offset/strong: 1.6 m3 | 0/128 | 0/128 | 128 | 0 | 0 | 0 | 0 | n/a (0) | n/a (0) |
| legitimate_flow_change/weak: 0.001 m3/s | 0/128 | 0/128 | 128 | 0 | 0 | 0 | 0 | n/a (0) | n/a (0) |
| legitimate_flow_change/strong: 0.16 m3/s | 0/128 | 0/128 | 128 | 0 | 0 | 0 | 0 | n/a (0) | n/a (0) |
| storage_drift_with_nuisance/weak: 0.001 m3/s | 0/128 | 0/128 | 128 | 0 | 0 | 0 | 0 | n/a (0) | n/a (0) |
| storage_drift_with_nuisance/strong: 0.16 m3/s | 0/128 | 128/128 | 128 | 0 | 0 | 0 | 0 | n/a (0) | n/a (0) |
| omitted_flow_with_nuisance/weak: 0.001 m3/s | 0/128 | 0/128 | 128 | 0 | 0 | 0 | 0 | n/a (0) | n/a (0) |
| omitted_flow_with_nuisance/strong: 0.16 m3/s | 0/128 | 128/128 | 128 | 0 | 0 | 0 | 0 | n/a (0) | n/a (0) |
| omitted_flow_unmodeled/weak: 0.001 m3/s | 0/128 | 0/128 | 128 | 0 | 0 | 0 | 0 | n/a (0) | n/a (0) |
| omitted_flow_unmodeled/strong: 0.16 m3/s | 128/128 | 128/128 | 0 | 0 | 128 | 0 | 0 | n/a (0) | n/a (0) |
| storage_step_unrestricted_nuisance/weak: 0.02 m3 | 0/128 | 0/128 | 0 | 0 | 0 | 0 | 128 | n/a (0) | n/a (0) |
| storage_step_unrestricted_nuisance/strong: 1.6 m3 | 0/128 | 128/128 | 0 | 0 | 0 | 0 | 128 | n/a (0) | n/a (0) |

A missed instrument event includes a structurally invisible or nuisance-confounded fault. A rejected gauged balance during omitted real flow is not proof of a failed instrument. The JSON retains every seed, candidate fit, interval, status and scoring denominator.

## Declared cases

- **no_fault:** Conserving mean measurements with exactly the declared Gaussian noise.
- **storage_step:** One instrument, one declared fault profile; unknown unrestricted signed amplitude.
- **storage_drift:** One instrument, one declared fault profile; unknown unrestricted signed amplitude.
- **inflow_offset:** One instrument, one declared fault profile; unknown unrestricted signed amplitude.
- **inflow_gain:** One instrument, one declared fault profile; unknown unrestricted signed amplitude.
- **constant_flow_offset:** Constant inflow makes offset and gain proportional; storage drift also confounds offset.
- **constant_flow_gain:** Constant inflow makes offset and gain proportional; storage drift also confounds offset.
- **storage_common_offset:** A storage offset present throughout the record cancels under differencing.
- **legitimate_flow_change:** A real inflow change and its exactly conserving storage response; no instrument fault.
- **storage_drift_with_nuisance:** Storage drift is indistinguishable from an unrestricted constant ungauged-flow nuisance.
- **omitted_flow_with_nuisance:** Ungauged real inflow changes storage but is absent from gauged flow: physical model discrepancy.
- **omitted_flow_unmodeled:** Ungauged real inflow changes storage but is absent from gauged flow: physical model discrepancy.
- **storage_step_unrestricted_nuisance:** An unrestricted nuisance at every residual consumes all information; diagnosis must abstain.

## Frozen configuration

```json
{
  "intervals": 24,
  "interval_seconds": 1.0,
  "onset_interval": 6,
  "alpha": 0.01,
  "interval_level": 0.95,
  "storage_sd_m3": 0.12,
  "flow_sd_m3_per_s": 0.025,
  "reference_storage_m3": 0.08,
  "reference_inflow_m3_per_s": 0.015,
  "reference_outflow_m3_per_s": 0.01,
  "weak_storage_m3": 0.02,
  "strong_storage_m3": 1.6,
  "weak_rate_m3_per_s": 0.001,
  "strong_rate_m3_per_s": 0.16,
  "weak_gain": 0.0005,
  "strong_gain": 0.08
}
```
