# P4c: a month of real water levels, and what a month buys over a day

Generated with Python 3.13.12, numpy 2.5.3 on Linux-6.18.44-fc-v24-x86_64-with-glibc2.39; source sha256 c2436dd517fa, git 25970ab2d5. Latency columns are wall-clock on this machine and are not a claim.

One month of real six-minute water levels. Truth-free: no number here is an error.

NOAA CO-OPS station 8454000 on MLLW, 2024-01-01 to 2024-01-31: 7,440 six-minute steps, 0.0% missing, 21,360 evidence ids. Acquired through DAF's adapter and replayed from `data/daf/noaa_8454000_202401_mllw.observations.json`.

## The windows

| window | days | steps | span | missing | evidence ids | role |
|---|---|---:|---|---:|---:|---|
| `fit` | days 1-15 | 3,600 | 2024-01-01T00:00 .. 2024-01-15T23:54 | 0.0% | 10,080 | the q grids are searched here, and nowhere else |
| `held_out` | days 16-31 | 3,840 | 2024-01-16T00:00 .. 2024-01-31T23:54 | 0.0% | 11,280 | held out: a disjoint window, no shared evidence |
| `month` | days 1-31 | 7,440 | 2024-01-01T00:00 .. 2024-01-31T23:54 | 0.0% | 21,360 | the whole record; in-sample for every fitted q |

The fit and held-out windows share no step, no calendar day and no evidence id; the check that says so runs before any filter does.

## What each window's length can separate

Two constituents need a record at least 1 / |n1 - n2| long to be separated.

| window | days | pairs resolved | unresolved | unresolved among the modelled six |
|---|---:|---:|---:|---|
| `fit` | 15.0 | 19 | 2 | M2/N2 |
| `held_out` | 16.0 | 19 | 2 | M2/N2 |
| `month` | 31.0 | 20 | 1 | none |

**The split is not free.** 15.0 days is long enough for S2 and O1 but not for N2: separating N2 from M2 needs 27.6 days, which only the whole month reaches. So `tide_month` declares six constituents on the strength of the month, and the q it is scored with was fitted on a window where one of its declared pairs is unresolved. Both facts are above; neither is hidden in the other.

P1 is not modelled anywhere. Separating it from K1 needs 182.6 days, so a month cannot; it is absorbed into the K1 coefficients, and no K1 number here is clean of it.

## The fitted q, on the fit window only

| estimator | constituents | q_scale | units | log-likelihood | grid interior |
|---|---|---:|---|---:|---|
| `level_trend` | — | 1e-06 | m s^-3/2 | 10,213.7 | yes |
| `tide_kf` | M2, K1, O1, M4 | 0.000316 | m s^-1/2 | 9,544.4 | yes |
| `tide_month` | M2, S2, N2, K1, O1, M4 | 0.000316 | m s^-1/2 | 9,462.2 | yes |

A q at a grid edge is a statement about the grid, not about the record, and is marked as one.

## Scored on the held-out half, two ways

`fresh` starts a filter at the beginning of the held-out window: every state comes from the declared prior, and nothing of the fit window survives except q. `continued` runs one filter over the whole month and scores the held-out steps only, so the state crossing the boundary is the first half's. The ratio is how much of the held-out fit is that burn-in.

| estimator | z RMS fresh | z RMS continued | ratio | z lag-1 fresh | z lag-1 continued | R share fresh | R share continued |
|---|---:|---:|---:|---:|---:|---:|---:|
| `level_trend` | 0.797 | 0.797 | 1.00 | +0.553 | +0.553 | 0.236 | 0.237 |
| `tide_kf` | 0.656 | 0.657 | 1.00 | +0.772 | +0.773 | 0.161 | 0.163 |
| `tide_month` | 0.566 | 0.567 | 1.00 | +0.771 | +0.772 | 0.129 | 0.132 |

**The burn-in buys nothing measurable here, and that is a result rather than an omission.** The largest fresh/continued ratio is 0.9973: over 3,840 steps, a filter started cold from the declared prior reaches the same z RMS as one carrying fifteen days of state. What that measures is the transient's length against the window's, not that state does not matter -- a shorter window would not show the same thing, and the day report's own transient is why it excludes its first 20 steps from the datum check.

CUSUM alarm steps below are in each run's own coordinates: the continued run starts at the beginning of the month, so its step 3,863 and the fresh run's step 263 are the same reading. The JSON carries both forms.

| estimator | window | z mean | z RMS | z lag-1 | \|z\| > 1.96 | CUSUM alarms | first alarm |
|---|---|---:|---:|---:|---:|---:|---:|
| `level_trend` | fit (in-sample) | +0.003 | 1.002 | +0.605 | 5.4% | 20 | 1519 |
| `level_trend` | held out, fresh | +0.005 | 0.797 | +0.553 | 2.4% | 6 | 263 |
| `level_trend` | held out, continued | +0.005 | 0.797 | +0.553 | 2.4% | 6 | 263 |
| `level_trend` | whole month (in-sample) | +0.004 | 0.902 | +0.584 | 3.9% | 26 | 1519 |
| `tide_kf` | fit (in-sample) | -0.002 | 0.920 | +0.785 | 4.1% | 37 | 1470 |
| `tide_kf` | held out, fresh | +0.009 | 0.656 | +0.772 | 1.0% | 4 | 259 |
| `tide_kf` | held out, continued | +0.011 | 0.657 | +0.773 | 1.0% | 4 | 257 |
| `tide_kf` | whole month (in-sample) | +0.004 | 0.795 | +0.780 | 2.5% | 41 | 1470 |
| `tide_month` | fit (in-sample) | -0.004 | 0.806 | +0.782 | 3.1% | 25 | 1471 |
| `tide_month` | held out, fresh | +0.006 | 0.566 | +0.771 | 0.6% | 3 | 266 |
| `tide_month` | held out, continued | +0.011 | 0.567 | +0.772 | 0.6% | 3 | 259 |
| `tide_month` | whole month (in-sample) | +0.004 | 0.693 | +0.779 | 1.8% | 28 | 1471 |

A z RMS below 1 means the filter's stated innovation variance is larger than the innovations it saw, which is a statement about the declared R and q together and not a score. CUSUM here reads no constraint: with one sensor and no declared relation there is nothing for it to localise, and an alarm count is a property of this record against the declared model.

## What NOAA's stated sigma says, and what one reading does to it

R throughout is NOAA's own stated sigma^2 per reading, carried by the bridge and never edited. It is not uniform, and its second moments are not robust:

| window | median | RMS | RMS without the extremes | min | max | distinct values | stated 0.000 | over 10x median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fit` | 0.0070 | 0.0103 | 0.0101 | 0.0000 | 0.0880 | 50 | 20 | 1 |
| `held_out` | 0.0070 | 0.0333 | 0.0080 | 0.0000 | 2.0020 | 30 | 17 | 1 |
| `month` | 0.0070 | 0.0249 | 0.0091 | 0.0000 | 2.0020 | 51 | 37 | 2 |

**One reading states 2.002 m** at 2024-01-19T22:18 (record step 4,543), 286 times the month's median of 0.0070 m. It alone raises the month's RMS stated sigma from 0.0091 m to 0.0249 m, and it lands in the held-out half, which is why that half's RMS is the larger one. Any quantity below that is a second moment of the stated sigma -- the model-free comparison, its expected mean square, the relative scatter of that -- is a statement about this one reading as much as about the month. The median is unchanged by it, and both are given.

37 readings state sigma = 0.000 m. The bridge carries that as R = 0 exactly, which declares the reading exact and gives the filter no room to disagree with it. They are counted, never adjusted.

## The model-free bound, per window

No filter: if each reading's error were white with variance sigma_e^2 and independent of the water level, the expected mean square of the series' second differences would be at least 6 sigma_e^2, so sigma_e <= rms(d2 y) / sqrt(6) in expectation. The month has 31 times a day's second differences, so its own sampling scatter is smaller -- which is the whole point of reporting it per window.

| window | second differences | sigma_e bound (m) | median stated sigma (m) | rms stated sigma (m) | stated var / bound var | null rel. sd of the mean square |
|---|---:|---:|---:|---:|---:|---:|
| `fit` | 3,598 | 0.0043 | 0.0070 | 0.0103 | 5.799 | 8.01% |
| `held_out` | 3,838 | 0.0028 | 0.0070 | 0.0333 | 137.767 | 133.34% |
| `month` | 7,438 | 0.0036 | 0.0070 | 0.0249 | 48.183 | 122.45% |

The bound is an upper bound on a white error, in expectation. A time-correlated error is not bounded by it, and a bound is not a measurement of NOAA's stated sigma.

The last two columns are second moments of the stated sigma, so the single 2.002 m statement drives them: a null relative scatter above 100% says the expected mean square they compare against is itself dominated by one term. The ratio that is not is the median's:

| window | median stated sigma / bound | rms stated sigma / bound |
|---|---:|---:|
| `fit` | 1.64 | 2.41 |
| `held_out` | 2.47 | 11.74 |
| `month` | 1.95 | 6.94 |

**The stated sigma exceeds the white-error bound in every window, on the median reading and on the RMS alike** -- 1.64 to 2.47 times on the median, more on the RMS where the 2.002 m statement lands. So the direction of the finding does not come from that one reading, even though its size in the RMS column does. It is the same fact the filters report as a z RMS below 1, which 11 of the 12 scorings above are: a declared R larger than the innovations it predicts.

It is not a correction to make, and the bound does not say the declared R is wrong. The bound assumes a white error independent of the water level, so a time-correlated component of the stated sigma is not bounded by it -- and NOAA's stated sigma is a published accuracy statement, not a per-reading white-noise variance. What the comparison supports is that this record is smoother than the declared R treats it as being, which is a statement about the pair and not about either alone.

## What these numbers do not show

- Nobody knows the water level, so no number here is an error against it.
- A fitted q_scale is a property of the fit window and the declared grid, not a calibration.
- The 15-day fit window leaves 1 declared pair(s) unresolved (M2/N2 needs 27.6 days), so tide_month's q was fitted where a pair its model declares is not separable.
- P1 is not modelled anywhere. Separating it from K1 needs 182.6 days, so a month cannot; it is absorbed into the K1 coefficients, and no K1 number here is clean of it.
- One station, one month, one datum. A month of another season could order the filters differently.
- The continued scoring is not a cold start: the state entering the held-out window is the fit window's.
- A cold-started filter matches a continued one over this 3,840-step window. That measures the transient's length against the window's, not that carried state does not matter.
- 2 readings state a sigma above 10x the median, the largest 2.002 m; every second moment of the stated sigma here is dominated by that one.
- The stated sigma is 1.95x the white-error bound on the median reading over the month. The bound assumes a white error independent of the level, so a correlated part of a published accuracy statement is not bounded by it and nothing here shows the declared R to be wrong.
- 37 readings state sigma = 0.000 m, carried as R = 0: the filter is asked to treat them as exact. They are counted, not adjusted.
