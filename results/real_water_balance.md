# P4b: a real reservoir water balance through the reconciliation kernel

Generated with Python 3.13.12, numpy 2.5.3 on Linux-6.18.44-fc-v24-x86_64-with-glibc2.39; source sha256 18f5a0d2b23b, git 75ca1e857b. Latency columns are wall-clock on this machine and are not a claim.

Ridgway Reservoir, Uncompahgre River, Colorado, 1096 days from 2022-10-01, four USGS daily-mean series admitted by DAF. Gauged drainage 246.2 of 265 sq mi = 0.929.

**Truth-free.** Nobody knows how much water is in this reservoir, so no number below is an error. Every number is computed from the readings, or from a run's own record.

This is the first constraint in the repository that is a physical law over evidence from independent instruments, rather than a declared relation in a simulator. It is **expected to fail**: about 7% of the catchment is ungauged, and evaporation, precipitation on the lake and any gauge bias land in the same unmeasured term. The question is whether it fails for the right reason and by the right amount.

## The record

| series | role | USGS | parameter / statistic | range |
|---|---|---|---|---|
| `usgs:USGS-09147022:00054:00003` | storage | USGS-09147022 | 00054 Acre-ft, statistic 00003 | 51,520 – 83,400 acre-ft |
| `usgs:USGS-09147025:00060:00003` | outflow | USGS-09147025 | 00060 ft^3/s, statistic 00003 | 33.2 – 912.0 ft³/s |
| `usgs:USGS-09146200:00060:00003` | inflow uncompahgre | USGS-09146200 | 00060 ft^3/s, statistic 00003 | see combined inflow below |
| `usgs:USGS-09147000:00060:00003` | inflow dallas | USGS-09147000 | 00060 ft^3/s, statistic 00003 | see combined inflow below |

Combined gauged inflow 45.1 – 1,293.0 ft³/s. Missing readings per series: {'usgs:USGS-09147022:00054:00003': 0, 'usgs:USGS-09147025:00060:00003': 0, 'usgs:USGS-09146200:00060:00003': 0, 'usgs:USGS-09147000:00060:00003': 0}.

## The closure residual, before filtering

Arithmetic on the readings, without a filter or uncertainty weights. The daily-mean time pairing is an approximation, so even perfect gauges need not give zero:

    r_k = (S_{k+1} − S_k) − c (q_in1 + q_in2 − q_out)_k,    c = 1.9834710744 acre-ft per ft³/s-day

over 1095 days, in acre-ft per day:

| mean | sd | median | 5th | 95th | min | max | lag-1 |
|---|---|---|---|---|---|---|---|
| 1.39 | 64.12 | 4.74 | -100.47 | 81.61 | -552.23 | 555.82 | +0.173 |

As a flow: mean **+0.703 ft³/s**, sd 32.33 ft³/s, against a mean gauged inflow of 169.5 ft³/s. Over the whole record the imbalance accumulates to 1,526 acre-ft, +0.41% of the 368,576 acre-ft that flowed in.

**That cumulative is not evidence of a net imbalance, and must not be read as one.** It is a sum of 1,095 daily residuals, so it grows as sqrt(n) even when the gauges close exactly. Against its own standard error it is indistinguishable from zero:

| the cumulative | standard error of the sum | in standard errors | 95% interval |
|---|---|---|---|
| 1,526 acre-ft | 2,122 independent, 2,527 under AR(1) | 0.72 independent, 0.60 under AR(1) | [-3,426, 6,478] |

The interval contains zero, so **the three-year total is consistent with the gauges closing exactly**; the AR(1) column uses the measured lag-1 of +0.173 and is a model, not a measurement. What the record does show is a **daily** disagreement of tens of acre-feet, which is a different statement and the one the sections below test.

### Which day's flow the storage change belongs with

Storage here is a daily **mean**, so the pairing is not obvious. All three are computed:

| pairing | mean (ft³/s) | sd (ft³/s) | lag-1 |
|---|---|---|---|
| same day | +0.703 | 32.33 | +0.173 |
| centred | +0.725 | 20.62 | +0.468 |
| next day | +0.748 | 27.38 | +0.464 |

The **centred** pairing has the smallest scatter, which is what a change between two daily means should pair with. It is not proof: the centred pairing also averages two flow readings, which reduces the flow's own noise contribution by sqrt(2) whether or not the alignment is right; a smaller sd there is therefore not by itself evidence of the alignment.

## What the constraint is blind to

For the open constraint `A = [[1.0, -1.0]]`, d(f) = fᵀAᵀ(A P Aᵀ + Σ_b)⁻¹A f per unit direction:

| direction | d(f) |
|---|---|
| storage_only | 1.11111e-05 |
| cumulative_inflow_only | 1.11111e-05 |
| row_space (S - G) | 2.22222e-05 |
| null_space (S + G) | 0 (structural null) |

- **Structurally invisible:** storage and cumulative gauged inflow rising together by the same volume. d(f) = 0 exactly: no covariance makes this visible to the constraint.
- **Not even a direction:** an equal bias on one inflow gauge and the outflow gauge. It cancels inside (qin1 + qin2 - qout) before it reaches any state, so it never perturbs x at all: it is invisible to the balance AND outside what d(f) can score, because d(f) measures directions in the reported state space. The individual flow channels may still disagree with their predictions; attribution needs additional information beyond this balance.

(a representative reported covariance: the declared storage sigma squared, and (100 acre-ft)^2 on the cumulative gauged volume. d(f) scales with P, so these are for comparing directions with each other, not an absolute sensitivity.)

## Through the kernel

R is **consumer-declared** throughout: USGS states no per-value uncertainty anywhere in the daily-values API, so the bridge required a citation for every σ and carried it here.

- Flows: σ = 5% of the reading, floored at 1.0 ft³/s. USGS rates a daily discharge record 'Good' when about 95% of daily values are within 10% of their true value (USGS surface-water accuracy classes, as published per site and period in the annual water-data reports). Read as a two-sided 95% normal interval, that is 2 sigma, so sigma = 5% of the reading. Two assumptions here are this consumer's and not USGS's: that this site and period are rated 'Good' (the rating itself was not acquired -- it is not in the daily-values API), and that the error is normal. Floored at 1.0 ft^3/s because a percentage states nothing at zero flow.
- Storage: **swept**, not declared once — [50.0, 200.0, 800.0] acre-ft. NOT a source statement. Reservoir storage is derived from a measured lake elevation through a stage-capacity table whose uncertainty at this reservoir this repository has no source for. This value is one point of a declared sweep (STORAGE_SIGMA_SWEEP); every conclusion in the report is computed at each point, and the report says which survive the range.
- The daily interval itself is an interpretation, not a source statement: The USGS Water Data OGC API defines a daily item's `time` only as the date the observation represents and states no time zone (measured: every value is a bare 'YYYY-MM-DD'). Legacy NWIS practice is the site's local standard-time day, which this API does not say. Read here as a UTC day anchored at its midpoint -- an assumption of this consumer, not a statement of the source. It shifts every series by the same amount, so it cannot change a same-day difference between them; it would matter to an alignment against a sub-daily series, and none is used.

Process noise, declared and never fitted: q_storage = 500 acre-ft/√day, q_flow = 50 ft³/s/√day, q_ungauged = 200 acre-ft/√day (wb_aug only).

### Storage σ = 50 acre-ft

| estimator | consistency stat mean | max | over threshold | flagged steps | |correction| mean | |residual| after |
|---|---|---|---|---|---|---|
| `wb_open` | 14.54 | 65.59 | 52.0% | 561 | — | — |
| `wb_open+hard` | 14.54 | 65.59 | 52.0% | 561 | 2,280.8 | 26.7 |
| `wb_open+hard+guard` | 14.54 | 65.59 | 52.0% | 561 | 1,423.3 | 13.2 |
| `wb_aug` | 0.28 | 0.68 | 0.0% | 0 | — | — |
| `wb_aug+hard` | 0.28 | 0.68 | 0.0% | 0 | 2,271.3 | 0.3 |
| `wb_aug+hard+feedback` | 0.07 | 6.55 | 0.0% | 0 | 31.5 | 1.8 |
| `wb_closed` | — | — | — | 0 | — | — |

Threshold χ²(1) at q = 0.999 is 10.828. Under the declared hypothesis the statistic would have mean 1.

Cumulative ungauged net inflow U, acre-ft:

| run | final | sd | in sd | min | max | as % of gauged inflow |
|---|---|---|---|---|---|---|
| `wb_aug` | 0 | 6,618 | 0.00 | 0 | 0 | +0.00% |
| `wb_aug+hard` | 2,078 | 1,043 | 1.99 | -2,705 | 5,029 | +0.56% |
| `wb_aug+hard+feedback` | 2,568 | 1,013 | 2.53 | -2,530 | 5,355 | +0.70% |

**U is observed by nothing except the constraint.** In `wb_aug`, with no projection, it stays at its prior of 0 for the whole record while its sd grows — the augmented state is not identified by the data at all, unlike the simulated `kf_aug`, whose boundary flux L is identified through the dynamics. `+hard` moves the REPORTED U; only `+feedback` puts the projection back into the full filter state and covariance. A constraint fed back is absorbed as if it were fresh evidence; subsequent agreement is therefore not independent validation. That is also why `+feedback` carries the largest `in sd` above: feeding the constraint back shrinks the very sd that column divides by, so its apparent separation from zero is the least trustworthy of the three, not the most. The finite prior and process variance on U still permit rejection of sufficiently large disagreements on other records.

Cross-check: `wb_aug+hard` puts the three-year imbalance at 2,078 acre-ft (+0.56% of gauged inflow), against 1,526 acre-ft (+0.41%) from the arithmetic at the top of this report. These are two calculations from the same measurements: a filter under declared noise, and a sum under a time-pairing assumption. They are not required to agree: the filter's U is a smoothed quantity under a declared random walk, and the arithmetic is not.

**And their agreement is not evidence of an imbalance.** The filter reports its own sd of 1,043 acre-ft on that 2,078 — 1.99 sd from zero — and the arithmetic total is 0.60 standard errors from zero, an interval that contains zero. Two numbers this uncertain landing near each other is agreement between two uncertain numbers, not corroboration of a net imbalance.

### Storage σ = 200 acre-ft

| estimator | consistency stat mean | max | over threshold | flagged steps | |correction| mean | |residual| after |
|---|---|---|---|---|---|---|
| `wb_open` | 9.69 | 26.33 | 38.6% | 417 | — | — |
| `wb_open+hard` | 9.69 | 26.33 | 38.6% | 417 | 2,066.4 | 159.5 |
| `wb_open+hard+guard` | 9.69 | 26.33 | 38.6% | 417 | 1,157.0 | 142.2 |
| `wb_aug` | 0.28 | 0.67 | 0.0% | 0 | — | — |
| `wb_aug+hard` | 0.28 | 0.67 | 0.0% | 0 | 2,264.4 | 4.6 |
| `wb_aug+hard+feedback` | 0.06 | 1.73 | 0.0% | 0 | 32.6 | 16.0 |
| `wb_closed` | — | — | — | 0 | — | — |

Threshold χ²(1) at q = 0.999 is 10.828. Under the declared hypothesis the statistic would have mean 1.

Cumulative ungauged net inflow U, acre-ft:

| run | final | sd | in sd | min | max | as % of gauged inflow |
|---|---|---|---|---|---|---|
| `wb_aug` | 0 | 6,618 | 0.00 | 0 | 0 | +0.00% |
| `wb_aug+hard` | 2,068 | 1,074 | 1.93 | -2,642 | 4,998 | +0.56% |
| `wb_aug+hard+feedback` | 2,497 | 1,031 | 2.42 | -2,473 | 5,250 | +0.68% |

**U is observed by nothing except the constraint.** In `wb_aug`, with no projection, it stays at its prior of 0 for the whole record while its sd grows — the augmented state is not identified by the data at all, unlike the simulated `kf_aug`, whose boundary flux L is identified through the dynamics. `+hard` moves the REPORTED U; only `+feedback` puts the projection back into the full filter state and covariance. A constraint fed back is absorbed as if it were fresh evidence; subsequent agreement is therefore not independent validation. That is also why `+feedback` carries the largest `in sd` above: feeding the constraint back shrinks the very sd that column divides by, so its apparent separation from zero is the least trustworthy of the three, not the most. The finite prior and process variance on U still permit rejection of sufficiently large disagreements on other records.

Cross-check: `wb_aug+hard` puts the three-year imbalance at 2,068 acre-ft (+0.56% of gauged inflow), against 1,526 acre-ft (+0.41%) from the arithmetic at the top of this report. These are two calculations from the same measurements: a filter under declared noise, and a sum under a time-pairing assumption. They are not required to agree: the filter's U is a smoothed quantity under a declared random walk, and the arithmetic is not.

**And their agreement is not evidence of an imbalance.** The filter reports its own sd of 1,074 acre-ft on that 2,068 — 1.93 sd from zero — and the arithmetic total is 0.60 standard errors from zero, an interval that contains zero. Two numbers this uncertain landing near each other is agreement between two uncertain numbers, not corroboration of a net imbalance.

### Storage σ = 800 acre-ft

| estimator | consistency stat mean | max | over threshold | flagged steps | |correction| mean | |residual| after |
|---|---|---|---|---|---|---|
| `wb_open` | 4.50 | 15.18 | 13.3% | 142 | — | — |
| `wb_open+hard` | 4.50 | 15.18 | 13.3% | 142 | 1,112.9 | 929.8 |
| `wb_open+hard+guard` | 4.50 | 15.18 | 13.3% | 142 | 916.3 | 818.2 |
| `wb_aug` | 0.27 | 0.70 | 0.0% | 0 | — | — |
| `wb_aug+hard` | 0.27 | 0.70 | 0.0% | 0 | 2,200.0 | 69.0 |
| `wb_aug+hard+feedback` | 0.12 | 2.39 | 0.0% | 0 | 59.0 | 132.8 |
| `wb_closed` | — | — | — | 0 | — | — |

Threshold χ²(1) at q = 0.999 is 10.828. Under the declared hypothesis the statistic would have mean 1.

Cumulative ungauged net inflow U, acre-ft:

| run | final | sd | in sd | min | max | as % of gauged inflow |
|---|---|---|---|---|---|---|
| `wb_aug` | 0 | 6,618 | 0.00 | 0 | 0 | +0.00% |
| `wb_aug+hard` | 1,965 | 1,398 | 1.41 | -2,450 | 5,048 | +0.53% |
| `wb_aug+hard+feedback` | 2,435 | 1,109 | 2.20 | -2,323 | 5,166 | +0.66% |

**U is observed by nothing except the constraint.** In `wb_aug`, with no projection, it stays at its prior of 0 for the whole record while its sd grows — the augmented state is not identified by the data at all, unlike the simulated `kf_aug`, whose boundary flux L is identified through the dynamics. `+hard` moves the REPORTED U; only `+feedback` puts the projection back into the full filter state and covariance. A constraint fed back is absorbed as if it were fresh evidence; subsequent agreement is therefore not independent validation. That is also why `+feedback` carries the largest `in sd` above: feeding the constraint back shrinks the very sd that column divides by, so its apparent separation from zero is the least trustworthy of the three, not the most. The finite prior and process variance on U still permit rejection of sufficiently large disagreements on other records.

Cross-check: `wb_aug+hard` puts the three-year imbalance at 1,965 acre-ft (+0.53% of gauged inflow), against 1,526 acre-ft (+0.41%) from the arithmetic at the top of this report. These are two calculations from the same measurements: a filter under declared noise, and a sum under a time-pairing assumption. They are not required to agree: the filter's U is a smoothed quantity under a declared random walk, and the arithmetic is not.

**And their agreement is not evidence of an imbalance.** The filter reports its own sd of 1,398 acre-ft on that 1,965 — 1.41 sd from zero — and the arithmetic total is 0.60 standard errors from zero, an interval that contains zero. Two numbers this uncertain landing near each other is agreement between two uncertain numbers, not corroboration of a net imbalance.

## What these numbers do not show

- **That the constraint's rejection measures the ungauged flux.** The statistic rejects the declared closure; it does not say which of the ungauged catchment, evaporation, precipitation on the lake, the stage-capacity table or a gauge rating is responsible. That is the known limit of a global χ² test on constraint residuals (Crowe, 1985), and nothing here escapes it.
- **That R is calibrated.** Every R is a consumer declaration with a citation that says what it assumes. The flow σ assumes a 'Good' rating this repository did not acquire and a normal error; the storage σ has no source at all, which is why it is swept rather than declared.
- **That `wb_aug`'s U is the ungauged inflow.** U is whatever makes the balance close. It absorbs the ungauged catchment, evaporation, precipitation, the stage-capacity table's error and any gauge bias, in one number. Its sd is the filter's own, under declared noise, and is not an error bar on the physical quantity.
- **That `wb_closed` is worse or better.** It declares no constraint, so no statistic here can reject it. It is the baseline that shows what assuming closure looks like, not a candidate that lost.
- **Anything about an equal-and-opposite gauge bias.** It cancels before it reaches the state. The balance cannot see it. Individual channel predictions may still disagree, but that does not by itself establish which gauge is faulty.
- **That the reference is independent.** The first storage reading supplies b and also enters the estimate. The kernel does not include their shared-reference cross-covariance. Repeated feedback of this reference is not independent evidence.
- **That the alignment question is settled.** The centred pairing has the smallest scatter and also averages two readings, which would reduce the scatter regardless. Separating the two needs a sub-daily record, which is a different acquisition.
- **Generality.** One reservoir, three water years, one climate. Winter ice affects the inflow records here — a substantial number of daily values at these sites carry USGS's ESTIMATED qualifier, which DAF keeps out of content by design, so every reading is scored alike above and nothing joins the qualifier to the residual. Such qualifiers would be corroborating metadata, not ground-truth fault labels.
