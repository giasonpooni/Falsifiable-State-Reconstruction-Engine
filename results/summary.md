# SET + LCM Phase 1 results

Two-reservoir material transfer, 600 steps, hidden truth, 20 seeds per scenario (mean ± sd across seeds). Declared constraint: m1 + m2 = 100 kg.
RMSE in kg over both reservoirs. cov95 = fraction of steps where truth lies inside the reported ±1.96σ interval. |res| post = mean |A x − b| after projection. A flag rejects the joint hypothesis (constraint ∧ model ∧ calibrated uncertainty); onset = first step at which that hypothesis is false. FA = worst-seed count / mean per-step rate of flags before onset (whole run if no onset). detect = mean steps from onset to first flag / seeds that never flagged. held = mean steps the guard reported model_inconsistent. Flags use χ²₁(0.999) with a 3-step debounce.

### closed_noise

Closed system, 2 kg noise, 5% random dropout. Constraint is TRUE.

| estimator | RMSE all | RMSE steady | cov95 | |res| post | |corr| | FA (max / rate) | detect (mean / missed) | held steps | lat p50 µs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hold_last | 2.01 ± 0.05 | 2.00 ± 0.06 | 0.95 ± 0.01 | — | — | 0 / 0.0e+00 | — | 0 | 62 |
| kf | 0.28 ± 0.05 | 0.23 ± 0.05 | 0.99 ± 0.02 | — | — | 0 / 0.0e+00 | — | 0 | 119 |
| kf+soft(1/lam=4) | 0.26 ± 0.04 | 0.23 ± 0.05 | 0.99 ± 0.01 | 2.7e-01 | 0.01 ± 0.00 | 0 / 0.0e+00 | — | 0 | 219 |
| kf+hard | 0.20 ± 0.04 | 0.16 ± 0.04 | 0.99 ± 0.02 | 3.9e-16 | 0.21 ± 0.04 | 0 / 0.0e+00 | — | 0 | 240 |
| kf+hard+guard | 0.20 ± 0.04 | 0.16 ± 0.04 | 0.99 ± 0.02 | 3.9e-16 | 0.21 ± 0.04 | 0 / 0.0e+00 | — | 0 | 239 |

### closed_blackout_pumpbias

Closed system; sensor 2 dark for steps 100-200 while the pump (on from step 50) actually delivers 0.12 kg/s against a commanded 0.10. Constraint is TRUE; the MODEL is wrong from step 50, so flags after 50 are detections of a parameter error.

| estimator | RMSE all | RMSE blackout | RMSE recovery | RMSE steady | cov95 | |res| post | |corr| | FA (max / rate) | detect (mean / missed) | held steps | lat p50 µs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hold_last | 2.72 ± 0.37 | 4.89 ± 1.20 | 2.03 ± 0.16 | 1.99 ± 0.06 | 0.90 ± 0.01 | — | — | 0 / 0.0e+00 | 128 / 3 | 0 | 65 |
| kf | 0.69 ± 0.07 | 1.31 ± 0.15 | 1.01 ± 0.15 | 0.24 ± 0.04 | 0.76 ± 0.05 | — | — | 0 / 0.0e+00 | 132 / 18 | 0 | 121 |
| kf+soft(1/lam=4) | 0.67 ± 0.07 | 1.27 ± 0.15 | 1.00 ± 0.15 | 0.23 ± 0.04 | 0.76 ± 0.05 | 3.6e-01 | 0.02 ± 0.00 | 0 / 0.0e+00 | 132 / 18 | 0 | 226 |
| kf+hard | 0.55 ± 0.06 | 1.00 ± 0.13 | 0.89 ± 0.16 | 0.18 ± 0.04 | 0.67 ± 0.05 | 3.5e-16 | 0.29 ± 0.05 | 0 / 0.0e+00 | 132 / 18 | 0 | 245 |
| kf+hard+guard | 0.56 ± 0.07 | 1.02 ± 0.15 | 0.90 ± 0.16 | 0.18 ± 0.04 | 0.68 ± 0.05 | 3.5e-16 | 0.29 ± 0.04 | 0 / 0.0e+00 | 132 / 18 | 2 | 244 |

### closed_blackout_noisy_valve

Closed system; sensor 2 dark for steps 100-400 while an unmodeled valve moves mass between reservoirs at random (σ = 0.10 kg/s per step, zero mean; 4x the filter's Q). Pure observability loss, no parameter bias. Constraint is TRUE; the filter's noise model is wrong from step 0, so there is no false-alarm window -- 'detect' shows how many seeds ever flag the miscalibration.

| estimator | RMSE all | RMSE blackout | RMSE recovery | RMSE steady | cov95 | |res| post | |corr| | FA (max / rate) | detect (mean / missed) | held steps | lat p50 µs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hold_last | 6.38 ± 1.01 | 8.79 ± 1.48 | 2.07 ± 0.17 | 2.00 ± 0.09 | 0.74 ± 0.02 | — | — | 0 / 0.0e+00 | 199 / 0 | 0 | 62 |
| kf | 0.76 ± 0.31 | 0.92 ± 0.44 | 0.58 ± 0.27 | 0.51 ± 0.16 | 0.77 ± 0.15 | — | — | 0 / 0.0e+00 | 232 / 19 | 0 | 116 |
| kf+soft(1/lam=4) | 0.71 ± 0.27 | 0.85 ± 0.39 | 0.57 ± 0.27 | 0.51 ± 0.16 | 0.77 ± 0.15 | 5.2e-01 | 0.06 ± 0.03 | 0 / 0.0e+00 | 232 / 19 | 0 | 217 |
| kf+hard | 0.52 ± 0.14 | 0.55 ± 0.18 | 0.49 ± 0.26 | 0.49 ± 0.17 | 0.68 ± 0.13 | 4.0e-16 | 0.48 ± 0.24 | 0 / 0.0e+00 | 232 / 19 | 0 | 237 |
| kf+hard+guard | 0.55 ± 0.25 | 0.59 ± 0.35 | 0.49 ± 0.27 | 0.49 ± 0.17 | 0.68 ± 0.13 | 4.1e-16 | 0.46 ± 0.20 | 0 / 0.0e+00 | 232 / 19 | 6 | 237 |

### leak_stale_constraint

10 kg leaks from reservoir 2 during steps 300-500. The declared closed-boundary constraint becomes STALE at step 300.

| estimator | RMSE all | RMSE pre_leak | RMSE leak_and_after | cov95 | |res| post | |corr| | FA (max / rate) | detect (mean / missed) | held steps | lat p50 µs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hold_last | 1.97 ± 0.03 | 1.96 ± 0.05 | 1.99 ± 0.06 | 0.95 ± 0.00 | — | — | 0 / 0.0e+00 | 167 / 0 | 0 | 62 |
| kf | 0.79 ± 0.04 | 0.33 ± 0.08 | 1.07 ± 0.07 | 0.79 ± 0.01 | — | — | 1 / 1.7e-04 | 65 / 0 | 0 | 118 |
| kf+soft(1/lam=4) | 0.84 ± 0.04 | 0.31 ± 0.07 | 1.15 ± 0.07 | 0.77 ± 0.02 | 2.7e+00 | 0.10 ± 0.00 | 1 / 1.7e-04 | 65 / 0 | 0 | 218 |
| kf+hard | 2.69 ± 0.01 | 0.24 ± 0.07 | 3.80 ± 0.01 | 0.53 ± 0.03 | 5.2e-16 | 1.99 ± 0.06 | 1 / 1.7e-04 | 65 / 0 | 0 | 238 |
| kf+hard+guard | 0.81 ± 0.05 | 0.24 ± 0.07 | 1.12 ± 0.07 | 0.76 ± 0.03 | 6.2e-16 | 0.26 ± 0.06 | 1 / 1.7e-04 | 65 / 0 | 235 | 237 |

### bias_quant_delay

Closed system; sensor 1 acquires an undeclared +3 kg bias at step 200; 0.5 kg quantization; 5-step arrival delay. Constraint is TRUE but the evidence is not.

| estimator | RMSE all | RMSE pre_bias | RMSE post_bias | cov95 | |res| post | |corr| | FA (max / rate) | detect (mean / missed) | held steps | lat p50 µs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hold_last | 2.69 ± 0.05 | 2.03 ± 0.07 | 2.97 ± 0.07 | 0.85 ± 0.02 | — | — | 0 / 0.0e+00 | — / 20 | 0 | 62 |
| kf | 1.63 ± 0.06 | 0.34 ± 0.08 | 1.98 ± 0.07 | 0.67 ± 0.01 | — | — | 0 / 0.0e+00 | 38 / 0 | 0 | 135 |
| kf+soft(1/lam=4) | 1.58 ± 0.06 | 0.31 ± 0.07 | 1.92 ± 0.07 | 0.67 ± 0.01 | 1.8e+00 | 0.08 ± 0.01 | 0 / 0.0e+00 | 38 / 0 | 0 | 236 |
| kf+hard | 1.16 ± 0.06 | 0.24 ± 0.06 | 1.41 ± 0.07 | 0.36 ± 0.01 | 4.3e-16 | 1.34 ± 0.07 | 0 / 0.0e+00 | 38 / 0 | 0 | 256 |
| kf+hard+guard | 1.61 ± 0.06 | 0.24 ± 0.06 | 1.97 ± 0.07 | 0.66 ± 0.01 | 9.2e-16 | 0.29 ± 0.06 | 0 / 0.0e+00 | 38 / 0 | 362 | 201 |
