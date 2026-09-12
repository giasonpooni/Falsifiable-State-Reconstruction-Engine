# Real fluid-measurement baseline — Ridgway

Generated with Python 3.13.12, numpy 2.5.3 on Windows-11-10.0.26200-SP0; source sha256 edf14f9bd452, git f7d165b424. Latency columns are wall-clock on this machine and are not a claim.

Real-record conditional consistency; no known fault labels or ground truth.

Storage and flow are interpreted as matching UTC daily means under the inherited consumer
time-zone assumption. Constant net flow within each day is also a consumer assumption.
The calculation propagates the joint measurement covariance
through the balance operator, including the shared initial reading in cumulative residuals.

Fixed windows contain at most 32 days. No sensor cause is assigned.

| Storage sigma [acre-ft] | Tested windows | Rejected windows | Tested adjacent pairs | Untested pairs |
|---|---:|---:|---:|---:|
| 50 | 35 | 8 | 1061 | 34 |
| 200 | 35 | 1 | 1061 | 34 |
| 800 | 35 | 0 | 1061 | 34 |

## Interpretation

The USGS Water Data OGC API defines a daily item's `time` only as the date the observation represents and states no time zone (measured: every value is a bare 'YYYY-MM-DD'). Legacy NWIS practice is the site's local standard-time day, which this API does not say. Read here as a UTC day anchored at its midpoint -- an assumption of this consumer, not a statement of the source. It shifts every series by the same amount, so it cannot change a same-day difference between them; it would matter to an alignment against a sub-daily series, and none is used.

Uncertainty is the existing consumer-declared sensitivity sweep, not a field calibration.
The JSON retains every window's joint statistic, threshold, final cumulative residual,
conditional standard deviation, shared-reference covariance and supporting evidence IDs.
Adjacent and cumulative residuals give the same joint statistic: cumulative summation
does not create independent evidence.

- A rejected window is not a verified sensor fault or a measured false alarm.
- Unmeasured water, varying within-day flow, calibration error and model error remain possible causes.
- The nominal per-window threshold is conditional on Gaussian errors and the declared covariance.
- No network-wide or repeated-window alarm probability is claimed.
- Cross-window adjacent pairs and incomplete windows are explicitly counted, not silently filled.
- Legacy wb_open/wb_aug/wb_closed filters retain their documented approximation; this path works directly on measurements.
