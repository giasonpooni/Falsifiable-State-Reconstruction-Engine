# A second reservoir, and what a second reservoir cost

Generated with Python 3.13.12, numpy 2.5.3 on Linux-6.18.44-fc-v33-x86_64-with-glibc2.39; source sha256 1d22ad6e1b9a, git 385bafd04b. Latency columns are wall-clock on this machine and are not a claim.

Taylor Park Reservoir, Taylor River, Colorado, from `declarations/taylor_park.toml`. Three water years of USGS daily values on the same grid as Ridgway: reservoir storage, the outlet gauge and **three** inflow gauges.

## What the second site cost

When Ridgway's constants moved into a declaration, the claim was that a second reservoir would cost a declaration rather than a module. That was a hypothesis, and this is its test. The answer is a declaration **plus two one-time costs that only a second site could have exposed**:

- estimators_balance assumed exactly two gauged inflows (N_SENSORS = 4 and three hand-indexed state layouts); the count now comes from BalanceConfig.n_inflows
- the study's functions read this module's globals rather than a declaration; experiments.balance_site.Site is the view that replaced them
- the closure and alignment residuals summed two named inflow columns; they now sum every inflow column the site declares

The declaration itself needed nothing new — same schema, no added fields. What it could not cover was code that had quietly encoded the first site's shape. No test could have found it: every test had two inflow gauges, because every site did. None of it is paid again by a third site.

## What the evidence looks like here

| series | missing days of 1,096 |
| --- | ---: |
| `usgs:USGS-09108500:00054:00003` | 6 |
| `usgs:USGS-09109000:00060:00003` | 0 |
| `usgs:USGS-09107000:00060:00003` | 0 |
| `usgs:USGS-09107500:00060:00003` | 454 |
| `usgs:USGS-09108250:00060:00003` | 454 |

Two of the three inflow gauges are seasonal: they report for part of each year only. Every Ridgway series is complete, every day, so **the bridge's missing-reading path had never been exercised by real evidence** — only by synthetic tests. Here it carries 454 absent days on each of two columns.

## The balance does not close, and the absent readings are not the whole reason

Ridgway's three-year cumulative residual is indistinguishable from zero. This one is not, and the study's shared `claims()` **refused to print Ridgway's sentences about it** — the guard working on a site transfer rather than on a wording slip.

Where gauges are intermittent, an imbalance the evidence shows and an imbalance the evidence's absence creates would otherwise be confounded. This record can separate them:

| days | n | mean (acre-ft per day) | cumulative | residual / gauged inflow |
| --- | ---: | ---: | ---: | ---: |
| every gauge reporting | 641 | 48.22 | 30,910 | 9.80% |
| a seasonal gauge absent | 444 | 73.34 | 32,562 | 94.30% |
| all days | 1,085 | 58.50 | 63,472 | 18.14% |

The last column divides by gauged inflow **as measured on those same days**, so the middle row's 94.30% is not an imbalance of that size: on a day a seasonal creek is absent it is missing from the denominator too, and the remaining gauges are at their winter low. It is reported because the absence is the point, not because the ratio is comparable to the row above it.

So the absent readings account for part of the apparent imbalance and not all of it. On the 641 days when **every gauge reports**, storage still rises 48.22 acre-ft per day more than the gauges account for — 9.80% of gauged inflow. That is present in fully observed data and this study does not say what it is.

## One comparison, and the conclusion not drawn from it

| | Ridgway | Taylor Park |
| --- | ---: | ---: |
| inflow gauges | 2 | 3 |
| series complete | yes | no |
| ungauged drainage | 7.09% | 8.90% |
| residual / gauged inflow | 0.41% | 9.80% (fully reported days) |

At Taylor Park the residual on fully reported days, 9.80%, sits within 0.91 percentage points of its ungauged drainage fraction, 8.90%. It is the obvious explanation and the augmented variant carries exactly that term.

**It is not concluded here.** The same comparison at Ridgway is 0.41% against 7.09% ungauged — no correspondence at all. A relationship that holds at one site and fails at the other is a coincidence or a mechanism, and two sites cannot tell which. Gauge bias, the stage-capacity table, the daily-mean alignment and real ungauged inflow all remain live, and nothing here separates them.

`results/real_diagnosis` takes that question up with a declared catalogue and reaches the same place by a different route: two of the candidates are exactly collinear on this residual, so no amount of data separates them, and the engine reports ambiguity rather than naming one.

## What the augmented model says the ungauged term is

The augmented variant carries a cumulative ungauged state, and one run estimates it rather than reporting its prior: `wb_aug+hard+feedback`. At this site it ends at **29,750 acre-ft** — 6.85 standard deviations from zero, 8.50% of gauged inflow. At Ridgway the same run ends at 2,497 acre-ft, 2.42 sd, 0.68%.

**This is not independent confirmation of anything above**, and it would be easy to present it as though it were. The only run in which U is estimated rather than reported is the one with feedback, and a constraint fed back is absorbed as if it were evidence; U therefore carries the constraint residual rather than measuring it a second time. The number close to the fully-reported cumulative is close to it because it largely IS it.

What it adds is an uncertainty and a contrast. The raw residual is an arithmetic fact with no error bar of its own; this is the declared model's answer with one attached, and it stays more than 3 sd from zero at every point of the declared storage-sigma sweep. Between the two sites it separates by a factor of 2.8 in standard deviations, which is the same direction the closure residual and the drainage fractions point, by the same evidence.

## What is declared here, and on whose authority

USGS states no per-value uncertainty in the daily-values API, so every sigma below is this consumer's declaration and travels with its reason. They are the same declarations the Ridgway study makes, which is part of what makes the two sites comparable — and the reason the second site's imbalance cannot be attributed to a different uncertainty budget.

- NOT a source statement. Reservoir storage is derived from a measured lake elevation through a stage-capacity table whose uncertainty at this reservoir this repository has no source for. This value is one point of a declared sweep (STORAGE_SIGMA_SWEEP); every conclusion in the report is computed at each point, and the report says which survive the range.
  - *declared at: declared.storage_sigma_citation, provenance.bridge.R_source.usgs:USGS-09108500:00054:00003.citation*
- The USGS Water Data OGC API defines a daily item's `time` only as the date the observation represents and states no time zone (measured: every value is a bare 'YYYY-MM-DD'). Legacy NWIS practice is the site's local standard-time day, which this API does not say. Read here as a UTC day anchored at its midpoint -- an assumption of this consumer, not a statement of the source. It shifts every series by the same amount, so it cannot change a same-day difference between them; it would matter to an alignment against a sub-daily series, and none is used.
  - *declared at: declared.day_zone_citation, provenance.bridge.time.day_zone_citation*
- USGS rates a daily discharge record 'Good' when about 95% of daily values are within 10% of their true value (USGS surface-water accuracy classes, as published per site and period in the annual water-data reports). Read as a two-sided 95% normal interval, that is 2 sigma, so sigma = 5% of the reading. Two assumptions here are this consumer's and not USGS's: that this site and period are rated 'Good' (the rating itself was not acquired -- it is not in the daily-values API), and that the error is normal. Floored at 1.0 ft^3/s because a percentage states nothing at zero flow.
  - *declared at: declared.flow_sigma_citation, provenance.bridge.R_source.usgs:USGS-09107000:00060:00003.citation, provenance.bridge.R_source.usgs:USGS-09107500:00060:00003.citation*

## What this does not establish

- Two sites is not a sample. Nothing here is evidence about reservoirs in general.
- No fault is diagnosed. A balance that does not close says the declared model and the declared uncertainties disagree with the record; it does not say which is wrong.
- The seasonal gauges' absence is treated as absence, not as zero flow. A creek that is not reporting is not a creek that is not flowing, and the split above is what keeps those apart rather than a correction that merges them.
- The same daily-mean alignment approximation the Ridgway study documents applies here, and is not re-derived.
- The declared storage sigma is this consumer's, not USGS's, at both sites alike.
