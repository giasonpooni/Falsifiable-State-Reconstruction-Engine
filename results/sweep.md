# Fault-magnitude sweep

Generated with Python 3.13.12, numpy 2.5.3 on Linux-6.18.44-fc-v24-x86_64-with-glibc2.39; source sha256 83bd31b9b0fe, git 2d859bbc25. Latency columns are wall-clock on this machine and are not a claim.

20 seeds per point. Cells are (RMSE kg / cov95 / nz) over the fault window; det = seeds flagged within 100 steps of onset, or 'FA max / rate' (worst-seed count and mean per-step rate of flags) where the spec's hypothesis holds for the whole run; held = mean steps the guard reported model_inconsistent ('guard held' is kf+hard+guard's). nz = RMS normalised error (1.0 calibrated, >1 over-confident). Specs named (b_var…) are handed a ConstraintSet that declares that variance on b.

## declared_total_error

The declared total is wrong by delta. The joint hypothesis is false from step 0 for delta > 0. kf_aug and kf_closedq read the constraint only through its consistency flag and never project, so their rows are the same at every delta. kf+hard+fb+guard feeds each applied projection back into the filter and stops while the guard holds. kf+hard(b_var=0.25) and its guard get the same wrong total with b_var = 0.25 kg² declared (σ_b = 0.5 kg, the scale of the exact guard's dead band): hard becomes the Kalman update with that pseudo-measurement variance and the guard tests rᵀ(APAᵀ + σ_b²)⁻¹r. A fixed delta is not a draw from N(0, σ_b²), so the onset stays at step 0 for delta > 0 and their flags count as detections.

| declared_total_error [kg] | kf (RMSE/cov/nz) | kf+hard (RMSE/cov/nz) | kf+hard+guard (RMSE/cov/nz) | kf+hard+fb+guard (RMSE/cov/nz) | kf_aug (RMSE/cov/nz) | kf_closedq (RMSE/cov/nz) | kf+hard(b_var=0.25) (RMSE/cov/nz) | kf+hard(b_var=0.25)+guard (RMSE/cov/nz) | kf+hard+guard det | kf+hard+fb+guard det | kf+hard(b_var=0.25)+guard det | guard held | kf+hard+fb+guard held | kf+hard(b_var=0.25)+guard held |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.23 / 0.99 / 0.73 | 0.16 / 0.98 / 0.72 | 0.16 / 0.98 / 0.72 | 0.16 / 0.98 / 0.72 | 0.32 / 0.98 / 0.80 | 0.11 / 0.98 / 0.75 | 0.19 / 0.99 / 0.67 | 0.19 / 0.99 / 0.67 | FA 0 / 0.0e+00 | FA 0 / 0.0e+00 | FA 0 / 0.0e+00 | 0 | 0 | 0 |
| 0.25 | 0.23 / 0.99 / 0.73 | 0.21 / 0.97 / 0.92 | 0.21 / 0.97 / 0.92 | 0.21 / 0.97 / 0.92 | 0.32 / 0.98 / 0.80 | 0.11 / 0.98 / 0.75 | 0.19 / 0.99 / 0.68 | 0.19 / 0.99 / 0.68 | 0/20 | 0/20 | 0/20 | 0 | 0 | 0 |
| 0.5 | 0.23 / 0.99 / 0.73 | 0.30 / 0.88 / 1.33 | 0.30 / 0.88 / 1.33 | 0.30 / 0.88 / 1.33 | 0.32 / 0.98 / 0.80 | 0.11 / 0.98 / 0.75 | 0.21 / 0.99 / 0.74 | 0.21 / 0.99 / 0.74 | 0/20 | 0/20 | 0/20 | 0 | 0 | 0 |
| 1 | 0.23 / 0.99 / 0.73 | 0.53 / 0.35 / 2.34 | 0.52 / 0.40 / 2.28 | 0.53 / 0.35 / 2.34 | 0.32 / 0.98 / 0.80 | 0.11 / 0.98 / 0.75 | 0.28 / 0.96 / 0.99 | 0.28 / 0.96 / 0.99 | 3/20 | 0/20 | 0/20 | 38 | 0 | 0 |
| 1.5 | 0.23 / 0.99 / 0.73 | 0.77 / 0.03 / 3.41 | 0.54 / 0.56 / 2.32 | 0.77 / 0.03 / 3.41 | 0.32 / 0.98 / 0.80 | 0.11 / 0.98 / 0.75 | 0.37 / 0.89 / 1.31 | 0.37 / 0.89 / 1.32 | 13/20 | 0/20 | 2/20 | 289 | 0 | 7 |
| 2 | 0.23 / 0.99 / 0.73 | 1.01 / 0.00 / 4.50 | 0.29 / 0.94 / 1.04 | 1.01 / 0.00 / 4.50 | 0.32 / 0.98 / 0.80 | 0.11 / 0.98 / 0.75 | 0.47 / 0.74 / 1.67 | 0.45 / 0.76 / 1.60 | 20/20 | 0/20 | 7/20 | 524 | 0 | 142 |
| 4 | 0.23 / 0.99 / 0.73 | 2.01 / 0.00 / 8.91 | 0.23 / 0.99 / 0.73 | 2.01 / 0.00 / 8.91 | 0.32 / 0.98 / 0.80 | 0.11 / 0.98 / 0.75 | 0.90 / 0.04 / 3.20 | 0.23 / 0.99 / 0.73 | 20/20 | 0/20 | 20/20 | 592 | 0 | 590 |

## sensor_bias

Undeclared sensor-1 bias from step 200, with 0.5 kg quantization and 5-step delay.

| sensor_bias [kg] | kf (RMSE/cov/nz) | kf+hard (RMSE/cov/nz) | kf+hard+guard (RMSE/cov/nz) | kf+hard+guard det | guard held |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 0.41 / 0.88 / 1.22 | 0.29 / 0.91 / 1.23 | 0.30 / 0.91 / 1.24 | 0/20 | 1 |
| 1 | 0.71 / 0.59 / 2.09 | 0.50 / 0.45 / 2.11 | 0.52 / 0.46 / 2.14 | 1/20 | 9 |
| 2 | 1.34 / 0.52 / 3.95 | 0.95 / 0.08 / 3.98 | 1.29 / 0.44 / 4.02 | 18/20 | 285 |
| 3 | 1.98 / 0.52 / 5.85 | 1.41 / 0.05 / 5.88 | 1.97 / 0.50 / 5.85 | 20/20 | 362 |

## leak_rate

Leak from reservoir 2 over steps 300-500; total lost = 200 x rate. kf_aug L det = seeds whose L flag (|L̂| / σ_L > 3.29, 3 consecutive reports) fired within 100 steps of onset.

| leak_rate [kg/s] | kf (RMSE/cov/nz) | kf+hard (RMSE/cov/nz) | kf+hard+guard (RMSE/cov/nz) | kf_aug (RMSE/cov/nz) | kf+hard+guard det | kf_aug L det | guard held |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.005 | 0.24 / 0.99 / 0.75 | 0.41 / 0.65 / 1.81 | 0.41 / 0.66 / 1.81 | 0.30 / 0.99 / 0.74 | 0/20 | 0/20 | 1 |
| 0.01 | 0.30 / 0.96 / 0.94 | 0.77 / 0.30 / 3.44 | 0.52 / 0.64 / 2.28 | 0.30 / 0.99 / 0.75 | 0/20 | 0/20 | 101 |
| 0.02 | 0.47 / 0.77 / 1.47 | 1.52 / 0.17 / 6.77 | 0.59 / 0.65 / 2.37 | 0.31 / 0.99 / 0.77 | 3/20 | 0/20 | 183 |
| 0.05 | 1.07 / 0.60 / 3.35 | 3.80 / 0.09 / 16.87 | 1.12 / 0.55 / 3.87 | 0.40 / 0.95 / 0.94 | 20/20 | 9/20 | 235 |

## uncertain_total

b = total0 + N(0, sigma_b^2) per seed. The guard tests the constraint as if exact; the soft variant uses the declared uncertainty as the pseudo-measurement variance. kf+hard(b_var) and its guard get the same b with b_var = sigma_b² declared on the ConstraintSet: hard is the Kalman update with that pseudo-measurement variance (the same estimate as the soft variant, which carries it as λ on an exact set) and the guard tests rᵀ(APAᵀ + σ_b²)⁻¹r, whose hypothesis -- b within its declared uncertainty -- holds; their flags are therefore false alarms (onset none; det cells read 'FA max / rate'). kf+hard and kf+hard+guard keep treating b as exact, for contrast.

| uncertain_total [kg (sigma_b)] | kf (RMSE/cov/nz) | kf+hard (RMSE/cov/nz) | kf+hard+guard (RMSE/cov/nz) | kf+soft(lam=1/sigma_b^2) (RMSE/cov/nz) | kf+hard(b_var) (RMSE/cov/nz) | kf+hard(b_var)+guard (RMSE/cov/nz) | kf+hard+guard det | kf+soft(lam=1/sigma_b^2) det | kf+hard(b_var)+guard det | guard held | kf+hard(b_var)+guard held |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 0.23 / 0.99 / 0.73 | 0.17 / 0.98 / 0.75 | 0.17 / 0.98 / 0.75 | 0.17 / 0.98 / 0.73 | 0.17 / 0.98 / 0.73 | 0.17 / 0.98 / 0.73 | 0/20 | 0/20 | FA 0 / 0.0e+00 | 0 | 0 |
| 0.3 | 0.23 / 0.99 / 0.73 | 0.21 / 0.96 / 0.92 | 0.21 / 0.96 / 0.92 | 0.19 / 0.99 / 0.74 | 0.19 / 0.99 / 0.74 | 0.19 / 0.99 / 0.74 | 0/20 | 0/20 | FA 0 / 0.0e+00 | 2 | 0 |
| 0.5 | 0.23 / 0.99 / 0.73 | 0.26 / 0.88 / 1.15 | 0.26 / 0.89 / 1.13 | 0.20 / 0.99 / 0.73 | 0.20 / 0.99 / 0.73 | 0.20 / 0.99 / 0.73 | 0/20 | 0/20 | FA 0 / 0.0e+00 | 11 | 0 |
| 1 | 0.23 / 0.99 / 0.73 | 0.41 / 0.68 / 1.84 | 0.33 / 0.81 / 1.43 | 0.22 / 0.99 / 0.72 | 0.22 / 0.99 / 0.72 | 0.22 / 0.99 / 0.72 | 4/20 | 4/20 | FA 0 / 0.0e+00 | 72 | 0 |
| 2 | 0.23 / 0.99 / 0.73 | 0.75 / 0.37 / 3.34 | 0.41 / 0.67 / 1.75 | 0.23 / 0.99 / 0.72 | 0.23 / 0.99 / 0.72 | 0.23 / 0.99 / 0.72 | 10/20 | 10/20 | FA 0 / 0.0e+00 | 183 | 0 |
