# Real fluid-measurement baseline — Ridgway

Generated with Python 3.13.12, numpy 2.5.3 on Linux-6.18.44-fc-v24-x86_64-with-glibc2.39; source sha256 c2436dd517fa, git 25970ab2d5. Latency columns are wall-clock on this machine and are not a claim.

Real-record conditional consistency; no known fault labels or ground truth.

Storage and flow are interpreted as matching UTC daily means under the inherited consumer
time-zone assumption. Constant net flow within each day is also a consumer assumption.
The calculation propagates the joint measurement covariance
through the balance operator, including the shared initial reading in cumulative residuals.

The detailed sweep immediately below uses windows of at most 32 days; that length is itself swept in the table after it. No sensor cause is assigned.

| Storage sigma [acre-ft] | Tested windows | Rejected windows | Tested adjacent pairs | Untested pairs |
|---|---:|---:|---:|---:|
| 50 | 35 | 8 | 1061 | 34 |
| 200 | 35 | 1 | 1061 | 34 |
| 800 | 35 | 0 | 1061 | 34 |

## The window length decides the rate as much as the declared sigma does

The window length is a consumer choice. A longer window accumulates more of a
persistent misfit while its chi-square threshold grows more slowly, so the same
measurements reject more often at a longer window. Both axes are therefore swept,
and the median window's statistic as a fraction of its own threshold is reported
beside each count: a cell with no rejections whose typical window sits just under
threshold is not the same finding as one whose typical window sits far below it.

| Storage sigma | Window days | Windows | Rejected | Rate | dof/window | median stat/threshold | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 50 | 3 | 365 | 0 | 0.0% | 2 | 0.022 | 0.886 |
| 50 | 8 | 137 | 7 | 5.1% | 7 | 0.102 | 2.245 |
| 50 | 32 *(declared)* | 35 | 8 | 22.9% | 31 | 0.551 | 4.358 |
| 50 | 64 | 18 | 8 | 44.4% | 63 | 0.925 | 3.988 |
| 200 | 3 | 365 | 0 | 0.0% | 2 | 0.001 | 0.097 |
| 200 | 8 | 137 | 0 | 0.0% | 7 | 0.008 | 0.565 |
| 200 | 32 *(declared)* | 35 | 1 | 2.9% | 31 | 0.122 | 1.468 |
| 200 | 64 | 18 | 2 | 11.1% | 63 | 0.359 | 1.149 |
| 800 | 3 | 365 | 0 | 0.0% | 2 | 0.000 | 0.006 |
| 800 | 8 | 137 | 0 | 0.0% | 7 | 0.001 | 0.048 |
| 800 | 32 *(declared)* | 35 | 0 | 0.0% | 31 | 0.009 | 0.295 |
| 800 | 64 | 18 | 0 | 0.0% | 63 | 0.044 | 0.599 |

Neither axis is a measurement of the gauges alone. Reading any one cell as the
record's rejection rate reads a consumer choice as a property of the instruments.

## Interpretation

**Every uncertainty and interval here is consumer-declared and carries its citation,
printed where the numbers are read and not only stored in the JSON.**

- Declared time interval: The USGS Water Data OGC API defines a daily item's `time` only as the date the observation represents and states no time zone (measured: every value is a bare 'YYYY-MM-DD'). Legacy NWIS practice is the site's local standard-time day, which this API does not say. Read here as a UTC day anchored at its midpoint -- an assumption of this consumer, not a statement of the source. It shifts every series by the same amount, so it cannot change a same-day difference between them; it would matter to an alignment against a sub-daily series, and none is used.
- Declared flow sigma: USGS rates a daily discharge record 'Good' when about 95% of daily values are within 10% of their true value (USGS surface-water accuracy classes, as published per site and period in the annual water-data reports). Read as a two-sided 95% normal interval, that is 2 sigma, so sigma = 5% of the reading. Two assumptions here are this consumer's and not USGS's: that this site and period are rated 'Good' (the rating itself was not acquired -- it is not in the daily-values API), and that the error is normal. Floored at 1.0 ft^3/s because a percentage states nothing at zero flow.
- Declared storage sigma: NOT a source statement. Reservoir storage is derived from a measured lake elevation through a stage-capacity table whose uncertainty at this reservoir this repository has no source for. This value is one point of a declared sweep (STORAGE_SIGMA_SWEEP); every conclusion in the report is computed at each point, and the report says which survive the range.
- Within-interval model (`constant_net_flow`): consumer assumption; not established by daily means.
- Window length (32 days, swept over [3, 8, 32, 64]): consumer choice; the declared length is one point of the swept axis in this report and is not privileged by the record.

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
- The rejected-window rate is a joint property of the record and the declared window length; both axes are swept and neither is a measurement of the gauges alone.
- Legacy wb_open/wb_aug/wb_closed filters retain their documented approximation; this path works directly on measurements.
