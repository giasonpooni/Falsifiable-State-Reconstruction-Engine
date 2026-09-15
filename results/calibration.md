# In-loop null of the consistency statistic

Generated with Python 3.13.12, numpy 2.5.3 on Linux-6.18.44-fc-v33-x86_64-with-glibc2.39; source sha256 33e4e335962b, git ec69c5a08c. Latency columns are wall-clock on this machine and are not a claim.

Unconstrained KF, 20 seeds, statistic r²/(A P Aᵀ) computed from the filter's own reported P over windows where the joint hypothesis holds. Under an exact χ²(1) null the mean would be 1.0 and the exceedances would equal the nominal tail probabilities. The lag-1 autocorrelation gives an AR(1) integrated autocorrelation time τ; the number of effectively independent samples is n/τ.

| window | n | mean | q95 | q99 | q999 | P(>3.841) [0.05] | P(>6.635) [0.01] | P(>10.828) [0.001] | ρ₁ | τ | n_eff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| closed_noise [0, 600] | 12000 | 0.599 | 2.55 | 4.28 | 6.21 | 0.0160 | 0.0005 | 0.00000 | 0.935 | 29.7 | 403 |
| closed_blackout_pumpbias [0, 50] | 1000 | 0.644 | 2.10 | 3.07 | 5.48 | 0.0050 | 0.0000 | 0.00000 | 0.761 | 7.4 | 136 |
| leak_stale_constraint [0, 300] | 6000 | 0.570 | 2.23 | 4.85 | 9.84 | 0.0162 | 0.0053 | 0.00083 | 0.901 | 19.1 | 314 |
| bias_quant_delay [0, 200] | 4000 | 0.463 | 1.60 | 3.58 | 8.86 | 0.0097 | 0.0032 | 0.00025 | 0.854 | 12.7 | 316 |

# Threshold × debounce sweep (kf+hard+guard, leak_stale_constraint)

Pre-onset false-alarm rate per step, seeds detected within 100 steps of onset, censored median delay, and (RMSE, cov95, nz) over the leak window. The shipped setting is q = 0.999, debounce = 3.

| q | debounce | FA rate | FA max | detected | median delay | RMSE leak | cov95 leak | nz leak | held |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.95 | 1 | 1.6e-02 | 37 | 20/20 | 44 | 1.08 | 0.58 | 3.54 | 260 |
| 0.95 | 3 | 1.1e-02 | 33 | 20/20 | 50 | 1.08 | 0.58 | 3.58 | 256 |
| 0.95 | 10 | 5.7e-03 | 19 | 20/20 | 59 | 1.09 | 0.57 | 3.69 | 247 |
| 0.99 | 1 | 5.3e-03 | 20 | 20/20 | 54 | 1.09 | 0.57 | 3.65 | 249 |
| 0.99 | 3 | 3.7e-03 | 17 | 20/20 | 58 | 1.10 | 0.57 | 3.69 | 245 |
| 0.99 | 10 | 1.7e-03 | 10 | 20/20 | 65 | 1.11 | 0.56 | 3.82 | 238 |
| 0.999 | 1 | 8.3e-04 | 4 | 20/20 | 64 | 1.11 | 0.55 | 3.82 | 237 |
| 0.999 | 3 | 1.7e-04 | 1 | 20/20 | 66 | 1.12 | 0.55 | 3.87 | 235 |
| 0.999 | 10 | 0.0e+00 | 0 | 20/20 | 73 | 1.14 | 0.54 | 4.03 | 228 |

# Null of the evidence-side CUSUM (kf, per sensor)

Two-sided CUSUM on the normalised innovation z = (y − H x_pred)/√Sᵢᵢ with k = 0.5, unconstrained KF, 20 seeds, over the same nominal windows. Cells are alarms s1 / s2 (seeds with at least one alarm s1 / s2) and, in the last column, the largest statistic either sensor reached with no reset (h = ∞). An alarm resets its channel, so the statistic and the alarm count at a given h come from a run at that h, not from thresholding one trace.

| window | sensor-steps | h = 4 | h = 6 | h = 8 | h = 10 | max stat (h = ∞) s1 / s2 |
| --- | --- | --- | --- | --- | --- | --- |
| closed_noise [0, 600] | 12000 | 63 / 34 (19 / 19) | 7 / 2 (6 / 2) | 0 / 0 (0 / 0) | 0 / 0 (0 / 0) | 7.55 / 7.73 |
| closed_blackout_pumpbias [0, 50] | 1000 | 4 / 4 (4 / 4) | 1 / 1 (1 / 1) | 0 / 0 (0 / 0) | 0 / 0 (0 / 0) | 6.04 / 6.05 |
| leak_stale_constraint [0, 300] | 6000 | 23 / 36 (15 / 16) | 2 / 4 (2 / 4) | 1 / 1 (1 / 1) | 0 / 0 (0 / 0) | 9.12 / 8.31 |
| bias_quant_delay [0, 200] | 4000 | 20 / 19 (14 / 11) | 0 / 0 (0 / 0) | 0 / 0 (0 / 0) | 0 / 0 (0 / 0) | 6.00 / 5.91 |

Totals over all nominal windows, both sensors (46000 sensor-steps): h = 4: 203 alarms (4.4e-03 per sensor-step), h = 6: 17 alarms (3.7e-04 per sensor-step), h = 8: 2 alarms (4.3e-05 per sensor-step), h = 10: 0 alarms (0.0e+00 per sensor-step).
The smallest h in {4, 6, 8, 10} with zero alarms over all nominal windows and 20 seeds is h = 10.
The shipped default h = 8 is kept: its 2 nominal alarms are 4.3e-05 per sensor-step, against 1.7e-04 per step for the constraint guard at its shipped q = 0.999, debounce 3 (sweep above), and the grid's 'CUSUM FA max' column reports the per-sensor count in every scenario so they are never hidden. A larger h buys silence on this sample — the largest un-reset excursion was 9.12 — at a delay cost of (h − 8) / (z̄ − k) steps for a sustained shift z̄, i.e. a few steps for the 3 kg bias and more for the slowly building leak lag; zero alarms at a larger h is a statement about this sample, not a bound.
