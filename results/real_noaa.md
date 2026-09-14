# P4: first real observations — NOAA 8454000 water levels through the same runner

Generated with Python 3.13.12, numpy 2.5.3 on Linux-6.18.44-fc-v24-x86_64-with-glibc2.39; source sha256 50a22c67cd05, git cdea185e52. Latency columns are wall-clock on this machine and are not a claim.

Truth-free: nobody knows the water level these readings measured, so no number below is an error. Every number is computed from the run's own record (`testbed.truth_free`), the bridged observations, or the committed evidence.

Evidence: DAF `ff92a737f702` (https://github.com/atomtrapping/Data-Acquisition-Channel), committed fixtures replayed by DAF's own NOAA binding into `data/daf/` — no network. That commit is NOT in DAF's published history: its published ancestor is `6b378590da30`, and the commits above it (carried here as the patch series in `patches/daf/`, never pushed to DAF) add a separate source and leave this binding untouched — the binding version and fixture blob recorded per file in `data/daf/manifest.json` are what fix these bytes, and both are unchanged. Bridge: time_zone UTC, cadence_s 360, arrival_policy replay, latency_s 0.0, conflict_policy refuse.

| day | role | file | first – last grid point (UTC) | readings | stated σ min–max [m], zeros | evidence-id sha256 |
|---|---|---|---|---|---|---|
| 2024-01-15 MLLW | fit window; in-sample when evaluated | `noaa_live_8454000_20240115_mllw.observations.json` | 2024-01-15T00:00:00Z – 2024-01-15T23:54:00Z | 240 of 240 | 0.000–0.027, 2 | `244cc303323772b9` |
| 2026-08-23 MLLW (preliminary) | held out | `noaa_live_8454000_preliminary.observations.json` | 2026-08-23T00:00:00Z – 2026-08-23T23:54:00Z | 240 of 240 | 0.001–0.015, 0 | `0a0c750e92d790d2` |
| 2024-01-15 STND | datum check only | `noaa_live_8454000_20240115_stnd.observations.json` | 2024-01-15T00:00:00Z – 2024-01-15T23:54:00Z | 240 of 240 | 0.000–0.027, 2 | `772c3d2f7e603501` |

## Declared, not fitted

- `level_trend` (water-level-trend-cwna-v1): x = [level, rate], F = [[1, dt], [0, 1]], Q = q² [[dt³/3, dt²/2], [dt²/2, dt]] (continuous white-noise acceleration), H = [1, 0], dt = 360 s from the grid.
- `tide_kf` (water-level-harmonic-rw-m2k1o1m4-v1): x = [mean level, (a, b) for M2 28.9841042 °/h, K1 15.0410686 °/h, O1 13.9430356 °/h, M4 57.9682084 °/h] (Schureman 1958, SP 98), H_k = [1, cos ω t_k, sin ω t_k, …], F = I, Q = q² dt I; it reports the level H_k x_k.
- Priors: level / mean level N(0 m, (10 m)²), rate N(0, (0.001 m/s)²), every harmonic coefficient N(0, (2 m)²). One sensor, n_report = 1, no constraint, CUSUM k = 0.5, h = 8.
- R = NOAA's stated σ² per reading (a stated 0.000 stays R = 0); R × 10 and R × 100 are declared alternatives, not fits.
- q_scale: the grid value that maximises the innovation log-likelihood Σ −½ (log 2πS_j + ν_j²/S_j) over every observed step of **2024-01-15 MLLW only** (declared prior included). Grids: 13 values half a decade apart, `level_trend` 1e-09 … 1e-03 m s^-3/2; `tide_kf` 1e-07 … 1e-01 m s^-1/2.
- Evaluation: the held-out 2026-08-23 MLLW (preliminary) day (a different day, no evidence id in common) and 2024-01-15 MLLW again, labelled in-sample.
- Held out means q_scale only. The prior widths are sized (`testbed/estimators_water.py`) against the largest reading, the fastest six-minute change and the widest half-range in `data/daf/`, all on 2024-01-15 (2.775 m, 0.083 m per step, 0.9575 m) against the held-out day's 1.373 m, 0.041 m and 0.4160 m, so the held-out day sets none of them; but it was in the repository when the priors, the two model structures, the q grids and this report's wording were written.

## Identification on 2024-01-15 MLLW

| i | level_trend q [m s^-3/2] | log-lik | tide_kf q [m s^-1/2] | log-lik |
|---|---|---|---|---|
| 0 | 1.00e-09 | -2669472.1 | 1.00e-07 | -14470.4 |
| 1 | 3.16e-09 | -597767.2 | 3.16e-07 | -14290.6 |
| 2 | 1.00e-08 | -83604.6 | 1.00e-06 | -12890.7 |
| 3 | 3.16e-08 | -10673.2 | 3.16e-06 | -8817.7 |
| 4 | 1.00e-07 | -880.4 | 1.00e-05 | -4784.0 |
| 5 | 3.16e-07 | 495.3 | 3.16e-05 | -1228.9 |
| 6 | **1.00e-06** | **690.7** | 1.00e-04 | 365.8 |
| 7 | 3.16e-06 | 628.8 | **3.16e-04** | **631.2** |
| 8 | 1.00e-05 | 445.0 | 1.00e-03 | 485.0 |
| 9 | 3.16e-05 | 195.4 | 3.16e-03 | 235.4 |
| 10 | 1.00e-04 | -75.4 | 1.00e-02 | -32.0 |
| 11 | 3.16e-04 | -349.9 | 3.16e-02 | -304.1 |
| 12 | 1.00e-03 | -625.0 | 1.00e-01 | -578.6 |

Bold: the fitted value, interior to its grid for both estimators (`level_trend` q = 1.00e-06 m s^-3/2, `tide_kf` q = 3.16e-04 m s^-1/2). A half-decade grid resolves q to that factor and no finer. Per six-minute step the fitted values mean a rate random walk of 1.9e-05 m/s for `level_trend`, and a random walk of 6.0 mm on each of `tide_kf`'s nine states: its harmonic basis is re-fitted as the day goes, not held.

## Per estimator and day (fitted q, R = stated σ²)

| estimator | day | q_scale | log-lik | z mean | z RMS | z lag-1 | \|z\| > 1.96 | CUSUM alarms (first step) | missing | mean R/S |
|---|---|---|---|---|---|---|---|---|---|---|
| level_trend | 2026-08-23 MLLW (preliminary), held out | 1.00e-06 | 709.5 | +0.017 | 0.993 | +0.560 | 0.054 | 0 | 0.000 | 0.217 |
| level_trend | 2024-01-15 MLLW, in-sample (fit day) | 1.00e-06 | 690.7 | +0.020 | 0.793 | +0.703 | 0.021 | 1 (step 10) | 0.000 | 0.280 |
| tide_kf | 2026-08-23 MLLW (preliminary), held out | 3.16e-04 | 673.3 | +0.020 | 0.703 | +0.706 | 0.013 | 0 | 0.000 | 0.120 |
| tide_kf | 2024-01-15 MLLW, in-sample (fit day) | 3.16e-04 | 631.2 | -0.095 | 0.759 | +0.855 | 0.017 | 1 (step 109) | 0.000 | 0.191 |

## R sensitivity at the fitted q (declared alternatives, not fits)

| estimator | day | R scale | log-lik | z RMS | z lag-1 | \|z\| > 1.96 | CUSUM alarms (first step) | mean R/S |
|---|---|---|---|---|---|---|---|---|
| level_trend | 2026-08-23 MLLW (preliminary) | × 1 | 709.5 | 0.993 | +0.560 | 0.054 | 0 | 0.217 |
| level_trend | 2026-08-23 MLLW (preliminary) | × 10 | 584.1 | 0.690 | +0.637 | 0.021 | 0 | 0.414 |
| level_trend | 2026-08-23 MLLW (preliminary) | × 100 | 400.8 | 0.369 | +0.713 | 0.000 | 0 | 0.597 |
| level_trend | 2024-01-15 MLLW | × 1 | 690.7 | 0.793 | +0.703 | 0.021 | 1 (step 10) | 0.280 |
| level_trend | 2024-01-15 MLLW | × 10 | 531.0 | 0.536 | +0.835 | 0.008 | 0 | 0.475 |
| level_trend | 2024-01-15 MLLW | × 100 | 315.7 | 0.383 | +0.919 | 0.000 | 0 | 0.644 |
| tide_kf | 2026-08-23 MLLW (preliminary) | × 1 | 673.3 | 0.703 | +0.706 | 0.013 | 0 | 0.120 |
| tide_kf | 2026-08-23 MLLW (preliminary) | × 10 | 575.1 | 0.591 | +0.722 | 0.013 | 0 | 0.363 |
| tide_kf | 2026-08-23 MLLW (preliminary) | × 100 | 398.4 | 0.379 | +0.746 | 0.000 | 0 | 0.599 |
| tide_kf | 2024-01-15 MLLW | × 1 | 631.2 | 0.759 | +0.855 | 0.017 | 1 (step 109) | 0.191 |
| tide_kf | 2024-01-15 MLLW | × 10 | 504.9 | 0.616 | +0.907 | 0.000 | 1 (step 114) | 0.445 |
| tide_kf | 2024-01-15 MLLW | × 100 | 308.2 | 0.432 | +0.939 | 0.000 | 0 | 0.641 |

## Datum invariance (computed, not assumed)

2024-01-15 on MLLW and on STND, the same q_scale and the same declared prior (0 ± 10 m in both datums). The data differ by a constant; once the prior is forgotten the innovations should not.

| estimator | state | max \|z_MLLW − z_STND\|, steps ≥ 20 (at step) | all steps | STND − MLLW state, last step [m] | range, steps ≥ 20 [m] | STND − MLLW data [m] |
|---|---|---|---|---|---|---|
| level_trend | level | 2.04e-09 (20) | 0.1064 | 1.064000 | 1.0640 – 1.0640 | 1.064000 – 1.064000 |
| tide_kf | mean_level | 6.52e-03 (28) | 0.0988 | 1.063968 | 0.9826 – 1.0640 | 1.064000 – 1.064000 |

By linearity — the two files state the same σ at every step, so both runs use the same gains — the STND − MLLW difference between the two runs is the filter's own response to a constant offset read from a prior mean of 0; both models represent a constant exactly (level or mean level equal to it, every other state 0), so the difference tends to it, and the table measures how fast. `level_trend` absorbs it at the first reading (its 10 m prior std dwarfs √R): its innovations agree to 2.8e-05 from the second reading on and to 2.0e-09 after step 20, and its level states differ by the data's offset. `tide_kf` absorbs it slowly: the first reading splits the offset between the mean level and the harmonic coefficients in proportion to their prior variances, and the coefficients hand it back over hours — its mean-level states differ by 0.9826–1.0640 m after step 20 and by 1.063968 m at the last step, and its innovations by up to 0.0065 in z after step 20 (at step 28). The datum does not change what either filter says about the water once the prior is gone; for `tide_kf` one day is not long enough for it to be entirely gone.

## Model-free series check

If each six-minute value's error were white with variance σ_e² and independent of the water level, the *expected* mean square of the day's second differences would be at least 6 σ_e² (the water's own second differences add to it), so σ_e ≤ rms(Δ²y)/√6 in expectation. One day's mean square scatters about its expectation: under that hypothesis, with independent Gaussian errors of the stated σ_j, the error part alone has the expected mean square and relative sd in the last two columns (the water–error cross term, zero in mean, adds scatter these do not count). A time-correlated error is not bounded by this.

| day | rms Δ²y [m] | white-error bound [m] | rms stated σ [m] | median stated σ [m] | mean σ² / bound² | error-only expected mean square, stated σ [m²] | its relative sd |
|---|---|---|---|---|---|---|---|
| 2024-01-15 MLLW | 0.0072 | 0.0029 | 0.0105 | 0.0090 | 12.9 | 6.64e-04 | 0.17 |
| 2026-08-23 MLLW (preliminary) | 0.0080 | 0.0033 | 0.0068 | 0.0060 | 4.4 | 2.79e-04 | 0.18 |

## What these numbers can and cannot validate

- **NOAA's σ is not the error of the six-minute value.** It is the standard deviation of the one-second samples behind it — waves and seiche included — reported to 1 mm; R = σ² passes the source's statement through. The series check says a white error that large is incompatible with these series: in expectation the second differences allow at most 0.0029 m on 2024-01-15 MLLW and 0.0033 m on the held-out day, where the stated σ has an rms of 0.0105 and 0.0068 m (mean σ² 12.9 and 4.4 times the bound), while one day's mean square would scatter about its expectation by a relative sd of 0.17 and 0.18 under that hypothesis (error part only). A correlated error could be that large; nothing here can tell.
- **z RMS with R = σ² does not say σ² is a calibrated measurement variance for these filters.** R is 0.12–0.28 of the stated innovation variance on average (mean R/S), so the innovations are mostly the filters' own prediction uncertainty; and they are not white (lag-1 +0.56 to +0.86, where 240 white samples would give 0 ± 0.065), so neither model is right about the dynamics. z RMS is 0.993 for `level_trend` on the held-out day and 0.793 in-sample, 0.703 and 0.759 for `tide_kf`. Below 1 the filter states more innovation variance than it sees; a value near 1 next to a lag-1 of +0.56 is not calibration either.
- **R and Q are not separated here, and one gauge cannot separate them without trusting the model.** Only q is fitted; R is held at σ², so the fitted q absorbs whatever σ² does not explain. At the fitted q the log-likelihood falls when R is scaled by 10 and by 100 on both days and for both filters: those alternatives give the readings a lower one-step predictive likelihood *at this q*, which does not make σ² right. A joint fit would split the variance only through the model's own assumptions (a white measurement error, these dynamics); telling sensor error from unmodelled water motion needs an independent measurement of the same water surface, which one gauge does not provide.
- **The CUSUM alarms cannot be classified.** `level_trend`: 1 (first at step 10, 01:00 UTC) on 2024-01-15 MLLW, none on the held-out day; `tide_kf`: 1 (first at step 109, 10:54 UTC) and none. On a simulated record an alarm is scored against a known onset; here there is none, and whether an alarm is a real change in the water, an instrument event or the model's own misfit needs evidence independent of this series. The null behind h = 8 was measured on simulated nominal records (results/calibration.md); these innovations are strongly autocorrelated (the lag-1 column), and a positively autocorrelated z drifts further before it turns, so h = 8's false-alarm rate on this record is unknown.
- **No constraint is applied, so the consistency channel is not exercised on real data yet.** A single series has no conservation relation to declare; every step's status is `skipped`, and the χ² test, the guard and the projection — the reconciliation stage this repository is about — never ran on these records.
- **The harmonic coefficients are not a tidal analysis.** One day of data cannot separate S2 or N2 from M2 (Rayleigh periods 14.8 and 27.6 days), nor K1 from O1 (13.7 days), and on a record of 23.90 h M2 falls short of the criterion even against K1 (25.82 h) and, narrowly, O1 (23.93 h). The coefficients are nuisance states for the next six-minute prediction; nothing reads them as amplitudes.
- **One held-out day is one day.** `level_trend` gives the held-out day's readings a higher one-step predictive log-likelihood than `tide_kf` (709.5 against 673.3), and the fit day's too (690.7 against 631.2); that orders two misspecified filters by how well they predict the next reading on two days — not by how close either is to the water — and a day of another tidal range, season or weather could order them differently. Every reading of the held-out day is preliminary (q = p on 240 of 240), every reading of the fit day verified (q = v on 240 of 240): a revision by NOAA would arrive as a new DAF observation with a new id, and the bridge reports or refuses a disagreement, it does not follow it.
- **Readings NOAA's own QC flagged are scored like every other.** On the held-out day 79 of 240 readings carry a QC flag vector f with a non-zero entry (`0,0,0,0` 161, `1,0,0,0` 79); on 2024-01-15 MLLW 0 do. DAF keeps q and f out of Observation.content by design (revision and acquisition metadata), so the bridge, both filters and every number above treat each reading alike. These counts come from `data/daf/manifest.json`, where the export tool counted them in the raw fixture; they are reported, not used, and what each flag means is NOAA's definition, not interpreted here.
- **Nothing here is an error against the water level.** Every number is a property of the record and the filter together; a filter can be confidently wrong with white, unit-variance innovations if the evidence is wrong in a way its model explains.
