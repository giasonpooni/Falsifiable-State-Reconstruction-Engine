# SET + LCM Phase 1 results

Generated with Python 3.13.12, numpy 2.5.3 on Linux-6.18.44-fc-v24-x86_64-with-glibc2.39; source sha256 51c948e3468a, git a1761bc717. Latency columns are wall-clock on this machine and are not a claim.

Two-reservoir material transfer, 600 steps, hidden truth, 20 seeds per scenario (mean ± sd across seeds where shown). Declared constraint: m1 + m2 = 100 kg, exact, in every scenario except closed_uncertain_total, which declares m1 + m2 = 100 kg + an offset drawn per seed from N(0, 1 kg²) together with b_var = 1 kg².
RMSE in kg over both reservoirs. cov95 = fraction of steps where truth lies inside the reported ±1.96σ interval. nz = RMS of the normalised error eᵢ/σᵢ (1.0 when calibrated; >1 over-confident). err row/null = RMS error of the reported state along row(A) and null(A) (for A = [1, 1]: the sum direction and the difference direction). z̄ s1/s2 = mean normalised innovation (y − H x_pred)/√Sᵢᵢ per sensor over the window's sampling steps (0 when the filter's prediction is unbiased; — where the sensor is dark or the estimator has no prediction). |res| post = mean |A x − b| after projection. A flag rejects the joint hypothesis (constraint ∧ model ∧ calibrated uncertainty); onset = first step at which that hypothesis is false. FA = worst-seed count / mean per-step rate of flags before onset (whole run if no onset). detected ≤N = seeds flagged within N steps of onset; median delay is the median over all seeds with never-flagged seeds censored at the end of the run ("> T" when the median itself is censored). held = mean steps the guard reported model_inconsistent. d(f) = fᵀAᵀ(APAᵀ)⁻¹Af for the scenario's fault direction on the unprojected P at the end of the first post-onset window; 0 means the consistency test is structurally blind to that fault. Flags use χ²(rank A)(0.999) with a 3-step debounce. Declared constraint uncertainty: where the constraint declares b_var (closed_uncertain_total only), the statistic is rᵀ(APAᵀ + Σ_b)⁻¹r and the joint hypothesis includes 'b is within its declared uncertainty'; hard projection is the Kalman update with pseudo-measurement variance Σ_b (residual post > 0), soft adds its 1/λ on top, and d(f) uses APAᵀ + Σ_b. CUSUM sᵢ = the evidence-side channel: a two-sided CUSUM (k = 0.5, h = 8) on sensor i's normalised innovation z = (y − H x_pred)/√Sᵢᵢ, updated when the observation is ingested and stamped at that report step, so its delay includes arrival delay; cells are (seeds alarmed within N steps of onset, censored median delay) and CUSUM FA max is the worst-seed count of pre-onset alarms per sensor. It reads no constraint; hold-last has no prediction, hence no innovation and no CUSUM. Augmented-state outputs (kf_aug only): the filter's state is [m1, m2, alpha, L] with alpha the pump scale (m1' = m1 − αu dt, m2' = m2 + αu dt − L dt; prior α ~ N(1, 0.1²), L ~ N(0, 0.02²), random walks 1e-3 and 2e-3 kg/s per step); its RMSE / cov95 / nz rows above are the mass marginal, never projected. alpha RMSE (pump on) = RMS of α̂ − u_actual/u_commanded over the steps where the pump is commanded on (α is unobservable when u = 0; u_actual is the hidden parameter rate, so the per-step pump fluctuation counts as process noise, not as α error). L RMSE = RMS of L̂ − leak (kg/s) over the run / window. alpha flag / L flag = |α̂ − 1| / σ_α > 3.29 and |L̂| / σ_L > 3.29 (two-sided 0.001) for 3 consecutive reports; cells and FA max as for the constraint flag, with the same onset. σ_α / σ_L at run end = the filter's own reported sd of each parameter at the last step (mean over seeds). These flags name a parameter, not a cause: a sensor bias that the filter can only explain through the pump will raise the alpha flag. Baselines: kf_closedq is the kf with closure written into its process noise instead of into a constraint row, Q = σ_q² dt² B Bᵀ + ε I with B = (−1, 1), σ_q = 0.01 kg/s (the simulator's declared pump fluctuation) and ε = 1e-8; it has no constraint row and does not know b, and its consistency stat is computed on its own marginal but never projected (kf_closedq+hard adds the row). kf+hard+fb feeds the projected (x*, P*) back into the filter's state after every applied projection (under arrival delay, the projection of the filter's own state at its last ingested step); +guard stops feeding back while the flag holds. oracle (bound) is a KF given the HIDDEN actual pump parameter rate and the hidden leak as known inputs with Q = ε I only: a bound on what a perfect model of the inputs could do, never a candidate; it does not model the per-step pump fluctuation or the valve transfer, so it is over-confident wherever the truth is not deterministic given those inputs.

### closed_noise

Closed system, 2 kg noise, 5% random dropout. Constraint is TRUE.

Accuracy and calibration per window:

| estimator | RMSE steady | cov95 steady | nz steady | err row/null steady | z̄ s1/s2 steady |
| --- | --- | --- | --- | --- | --- |
| hold_last | 2.00 ± 0.06 | 0.95 | 1.00 | 1.98 / 2.01 | — / — |
| kf | 0.23 ± 0.05 | 0.99 | 0.73 | 0.23 / 0.23 | +0.00 / +0.01 |
| kf+soft(1/lam=4) | 0.23 ± 0.05 | 0.99 | 0.72 | 0.22 / 0.23 | +0.00 / +0.01 |
| kf+hard | 0.16 ± 0.04 | 0.98 | 0.72 | 0.00 / 0.23 | +0.00 / +0.01 |
| kf+hard+guard | 0.16 ± 0.04 | 0.98 | 0.72 | 0.00 / 0.23 | +0.00 / +0.01 |
| kf+hard+fb | 0.16 ± 0.04 | 0.98 | 0.72 | 0.00 / 0.23 | -0.01 / -0.01 |
| kf+hard+fb+guard | 0.16 ± 0.04 | 0.98 | 0.72 | 0.00 / 0.23 | -0.01 / -0.01 |
| kf_closedq | 0.11 ± 0.04 | 0.98 | 0.75 | 0.08 / 0.12 | -0.00 / -0.00 |
| kf_closedq+hard | 0.08 ± 0.03 | 0.99 | 0.69 | 0.00 / 0.12 | -0.00 / -0.00 |
| kf_aug | 0.32 ± 0.07 | 0.98 | 0.80 | 0.32 / 0.33 | +0.00 / +0.00 |
| oracle (bound) | 0.10 ± 0.04 | 0.92 | 1.02 | 0.08 / 0.11 | +0.00 / -0.01 |

Reconciliation and detection:

| estimator | RMSE all | cov95 all | nz all | |res| post | |corr| | FA (max / rate) | detected ≤100 (k/n) | median delay | held steps | d(f) | CUSUM s1 (k/n, median) | CUSUM s2 (k/n, median) | CUSUM FA max (s1 / s2) | lat p50 µs (this machine) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hold_last | 2.01 ± 0.05 | 0.95 | 1.00 | — | — | 0 / 0.0e+00 | — | — | 0 | — | — | — | — | 324 |
| kf | 0.28 ± 0.05 | 0.99 | 0.76 | — | — | 0 / 0.0e+00 | — | — | 0 | — | — | — | 0 / 0 | 407 |
| kf+soft(1/lam=4) | 0.26 ± 0.04 | 0.99 | 0.75 | 2.7e-01 | 0.01 ± 0.00 | 0 / 0.0e+00 | — | — | 0 | — | — | — | 0 / 0 | 602 |
| kf+hard | 0.20 ± 0.04 | 0.99 | 0.75 | 3.9e-16 | 0.21 ± 0.04 | 0 / 0.0e+00 | — | — | 0 | — | — | — | 0 / 0 | 642 |
| kf+hard+guard | 0.20 ± 0.04 | 0.99 | 0.75 | 3.9e-16 | 0.21 ± 0.04 | 0 / 0.0e+00 | — | — | 0 | — | — | — | 0 / 0 | 641 |
| kf+hard+fb | 0.20 ± 0.04 | 0.99 | 0.75 | 2.2e-16 | 0.00 ± 0.00 | 0 / 0.0e+00 | — | — | 0 | — | — | — | 0 / 1 | 654 |
| kf+hard+fb+guard | 0.20 ± 0.04 | 0.99 | 0.75 | 2.2e-16 | 0.00 ± 0.00 | 0 / 0.0e+00 | — | — | 0 | — | — | — | 0 / 1 | 655 |
| kf_closedq | 0.21 ± 0.05 | 0.97 | 0.89 | — | — | 0 / 0.0e+00 | — | — | 0 | — | — | — | 1 / 2 | 407 |
| kf_closedq+hard | 0.16 ± 0.05 | 0.98 | 0.86 | 3.6e-16 | 0.13 ± 0.05 | 0 / 0.0e+00 | — | — | 0 | — | — | — | 1 / 2 | 644 |
| kf_aug | 0.36 ± 0.05 | 0.98 | 0.82 | — | — | 0 / 0.0e+00 | — | — | 0 | — | — | — | 0 / 0 | 434 |
| oracle (bound) | 0.21 ± 0.06 | 0.92 | 1.08 | — | — | 0 / 0.0e+00 | — | — | 0 | — | — | — | 1 / 2 | 403 |

Augmented-state outputs (pump scale alpha, boundary flux L in kg/s):

| estimator | alpha RMSE (pump on) | alpha RMSE (pump on) steady | L RMSE | L RMSE steady | alpha flag (k/n, median) | L flag (k/n, median) | FA max (alpha / L) | σ_α / σ_L at run end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kf_aug | 0.025 ± 0.008 | — | 0.0068 ± 0.0011 | 0.0066 | —, — | —, — | 0 / 0 | 0.041 / 0.0144 |

### closed_blackout_pumpbias

Closed system; sensor 2 dark for steps 100-200 while the pump (on from step 50) actually delivers 0.12 kg/s against a commanded 0.10. Constraint is TRUE; the MODEL is wrong from step 50. The fault moves mass along (-1, +1), which lies in null(A).

Accuracy and calibration per window:

| estimator | RMSE blackout | cov95 blackout | nz blackout | err row/null blackout | z̄ s1/s2 blackout | RMSE recovery | cov95 recovery | nz recovery | err row/null recovery | z̄ s1/s2 recovery | RMSE steady | cov95 steady | nz steady | err row/null steady | z̄ s1/s2 steady |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hold_last | 4.89 ± 1.20 | 0.66 | 2.40 | 4.89 / 4.89 | — / — | 2.03 ± 0.16 | 0.95 | 1.01 | 1.94 / 2.10 | — / — | 1.99 ± 0.06 | 0.95 | 1.00 | 1.99 / 2.00 | — / — |
| kf | 1.31 ± 0.15 | 0.21 | 2.90 | 0.71 / 1.70 | -0.38 / — | 1.01 ± 0.15 | 0.25 | 2.67 | 0.41 / 1.36 | -0.40 / +0.58 | 0.24 ± 0.04 | 0.99 | 0.74 | 0.21 / 0.26 | -0.02 / +0.01 |
| kf+soft(1/lam=4) | 1.27 ± 0.15 | 0.19 | 2.93 | 0.64 / 1.67 | -0.38 / — | 1.00 ± 0.15 | 0.24 | 2.70 | 0.38 / 1.35 | -0.40 / +0.58 | 0.23 ± 0.04 | 0.99 | 0.74 | 0.20 / 0.26 | -0.02 / +0.01 |
| kf+hard | 1.00 ± 0.13 | 0.02 | 3.75 | 0.00 / 1.41 | -0.38 / — | 0.89 ± 0.16 | 0.06 | 3.61 | 0.00 / 1.26 | -0.40 / +0.58 | 0.18 ± 0.04 | 0.98 | 0.80 | 0.00 / 0.26 | -0.02 / +0.01 |
| kf+hard+guard | 1.02 ± 0.15 | 0.02 | 3.75 | 0.07 / 1.43 | -0.38 / — | 0.90 ± 0.16 | 0.06 | 3.61 | 0.02 / 1.27 | -0.40 / +0.58 | 0.18 ± 0.04 | 0.98 | 0.80 | 0.00 / 0.26 | -0.02 / +0.01 |
| kf+hard+fb | 0.96 ± 0.13 | 0.02 | 3.71 | 0.00 / 1.36 | -0.47 / — | 0.86 ± 0.16 | 0.05 | 3.58 | 0.00 / 1.22 | -0.42 / +0.48 | 0.18 ± 0.04 | 0.98 | 0.80 | 0.00 / 0.26 | -0.02 / +0.01 |
| kf+hard+fb+guard | 0.96 ± 0.13 | 0.02 | 3.71 | 0.00 / 1.36 | -0.47 / — | 0.86 ± 0.16 | 0.05 | 3.58 | 0.00 / 1.22 | -0.42 / +0.48 | 0.18 ± 0.04 | 0.98 | 0.80 | 0.00 / 0.26 | -0.02 / +0.01 |
| kf_closedq | 1.56 ± 0.13 | 0.00 | 7.64 | 0.34 / 2.18 | -0.65 / — | 2.07 ± 0.14 | 0.00 | 11.42 | 0.29 / 2.90 | -0.94 / +1.17 | 0.68 ± 0.06 | 0.20 | 4.59 | 0.10 / 0.95 | -0.31 / +0.25 |
| kf_closedq+hard | 1.47 ± 0.12 | 0.00 | 9.80 | 0.00 / 2.08 | -0.65 / — | 1.99 ± 0.14 | 0.00 | 14.45 | 0.00 / 2.82 | -0.94 / +1.17 | 0.68 ± 0.06 | 0.13 | 5.50 | 0.00 / 0.96 | -0.31 / +0.25 |
| kf_aug | 0.50 ± 0.24 | 0.96 | 0.83 | 0.44 / 0.53 | -0.15 / — | 0.43 ± 0.10 | 0.97 | 0.88 | 0.41 / 0.42 | -0.06 / -0.00 | 0.31 ± 0.04 | 0.99 | 0.77 | 0.30 / 0.32 | -0.01 / +0.00 |
| oracle (bound) | 0.18 ± 0.07 | 0.96 | 0.93 | 0.16 / 0.17 | -0.02 / — | 0.17 ± 0.07 | 0.93 | 1.06 | 0.14 / 0.18 | -0.01 / +0.05 | 0.11 ± 0.05 | 0.92 | 1.00 | 0.09 / 0.11 | -0.00 / -0.02 |

Reconciliation and detection:

| estimator | RMSE all | cov95 all | nz all | |res| post | |corr| | FA (max / rate) | detected ≤100 (k/n) | median delay | held steps | d(f) | CUSUM s1 (k/n, median) | CUSUM s2 (k/n, median) | CUSUM FA max (s1 / s2) | lat p50 µs (this machine) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hold_last | 2.72 ± 0.37 | 0.90 | 1.35 | — | — | 0 / 0.0e+00 | 2/20 | 130 | 0 | 0 | — | — | — | 324 |
| kf | 0.69 ± 0.07 | 0.76 | 1.66 | — | — | 0 / 0.0e+00 | 0/20 | > 550 | 0 | 0 | 2/20, > 550 | 1/20, 164 | 0 / 0 | 402 |
| kf+soft(1/lam=4) | 0.67 ± 0.07 | 0.76 | 1.68 | 3.6e-01 | 0.02 ± 0.00 | 0 / 0.0e+00 | 0/20 | > 550 | 0 | 0 | 2/20, > 550 | 1/20, 164 | 0 / 0 | 597 |
| kf+hard | 0.55 ± 0.06 | 0.67 | 2.15 | 3.5e-16 | 0.29 ± 0.05 | 0 / 0.0e+00 | 0/20 | > 550 | 0 | 0 | 2/20, > 550 | 1/20, 164 | 0 / 0 | 641 |
| kf+hard+guard | 0.56 ± 0.07 | 0.68 | 2.15 | 3.5e-16 | 0.29 ± 0.04 | 0 / 0.0e+00 | 0/20 | > 550 | 2 | 0 | 2/20, > 550 | 1/20, 164 | 0 / 0 | 643 |
| kf+hard+fb | 0.54 ± 0.06 | 0.67 | 2.13 | 2.3e-16 | 0.00 ± 0.00 | 0 / 0.0e+00 | 0/20 | > 550 | 0 | 0 | 4/20, 134 | 1/20, 196 | 0 / 0 | 655 |
| kf+hard+fb+guard | 0.54 ± 0.06 | 0.67 | 2.13 | 2.3e-16 | 0.00 ± 0.00 | 0 / 0.0e+00 | 0/20 | > 550 | 0 | 0 | 4/20, 134 | 1/20, 196 | 0 / 0 | 648 |
| kf_closedq | 1.13 ± 0.08 | 0.23 | 6.36 | — | — | 0 / 0.0e+00 | 0/20 | 350 | 0 | 0 | 8/20, 104 | 1/20, 158 | 0 / 0 | 399 |
| kf_closedq+hard | 1.09 ± 0.07 | 0.18 | 7.96 | 3.7e-16 | 0.18 ± 0.03 | 0 / 0.0e+00 | 0/20 | 350 | 0 | 0 | 8/20, 104 | 1/20, 158 | 0 / 0 | 641 |
| kf_aug | 0.40 ± 0.07 | 0.98 | 0.83 | — | — | 0 / 0.0e+00 | 0/20 | > 550 | 0 | 0 | 0/20, > 550 | 1/20, > 550 | 0 / 0 | 428 |
| oracle (bound) | 0.21 ± 0.05 | 0.94 | 0.99 | — | — | 0 / 0.0e+00 | 0/20 | > 550 | 0 | 0 | 1/20, > 550 | 0/20, > 550 | 0 / 0 | 405 |

Augmented-state outputs (pump scale alpha, boundary flux L in kg/s):

| estimator | alpha RMSE (pump on) | alpha RMSE (pump on) blackout / recovery / steady | L RMSE | L RMSE blackout / recovery / steady | alpha flag (k/n, median) | L flag (k/n, median) | FA max (alpha / L) | σ_α / σ_L at run end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kf_aug | 0.108 ± 0.013 | 0.084 / 0.035 / — | 0.0073 ± 0.0011 | 0.0069 / 0.0078 / 0.0063 | 0/20, 146 | 0/20, > 550 | 0 / 0 | 0.041 / 0.0144 |

### closed_blackout_noisy_valve

Closed system; sensor 2 dark for steps 100-400 while an unmodeled valve moves mass between reservoirs at random (σ = 0.10 kg/s per step, zero mean; 4x the filter's Q). Pure observability loss, no parameter bias. Constraint is TRUE; the filter's noise model is wrong from step 0, so there is no false-alarm window. The disturbance acts along (-1, +1), which lies in null(A).

Accuracy and calibration per window:

| estimator | RMSE blackout | cov95 blackout | nz blackout | err row/null blackout | z̄ s1/s2 blackout | RMSE recovery | cov95 recovery | nz recovery | err row/null recovery | z̄ s1/s2 recovery | RMSE steady | cov95 steady | nz steady | err row/null steady | z̄ s1/s2 steady |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hold_last | 8.79 ± 1.48 | 0.54 | 4.16 | 8.78 / 8.80 | — / — | 2.07 ± 0.17 | 0.95 | 1.03 | 2.06 / 2.07 | — / — | 2.00 ± 0.09 | 0.95 | 1.00 | 2.01 / 1.99 | — / — |
| kf | 0.92 ± 0.44 | 0.73 | 1.67 | 0.70 / 1.09 | +0.00 / — | 0.58 ± 0.27 | 0.77 | 1.51 | 0.34 / 0.73 | +0.02 / +0.02 | 0.51 ± 0.16 | 0.76 | 1.61 | 0.20 / 0.69 | +0.02 / -0.03 |
| kf+soft(1/lam=4) | 0.85 ± 0.39 | 0.74 | 1.66 | 0.60 / 1.03 | +0.00 / — | 0.57 ± 0.27 | 0.77 | 1.52 | 0.30 / 0.72 | +0.02 / +0.02 | 0.51 ± 0.16 | 0.76 | 1.62 | 0.19 / 0.69 | +0.02 / -0.03 |
| kf+hard | 0.55 ± 0.18 | 0.67 | 1.96 | 0.00 / 0.77 | +0.00 / — | 0.49 ± 0.26 | 0.63 | 1.93 | 0.00 / 0.69 | +0.02 / +0.02 | 0.49 ± 0.17 | 0.62 | 2.17 | 0.00 / 0.69 | +0.02 / -0.03 |
| kf+hard+guard | 0.59 ± 0.35 | 0.67 | 1.96 | 0.07 / 0.81 | +0.00 / — | 0.49 ± 0.27 | 0.63 | 1.93 | 0.02 / 0.69 | +0.02 / +0.02 | 0.49 ± 0.17 | 0.62 | 2.17 | 0.00 / 0.69 | +0.02 / -0.03 |
| kf+hard+fb | 0.54 ± 0.17 | 0.65 | 2.04 | 0.00 / 0.76 | +0.00 / — | 0.50 ± 0.25 | 0.59 | 2.08 | 0.00 / 0.70 | +0.03 / +0.05 | 0.49 ± 0.17 | 0.62 | 2.19 | 0.00 / 0.70 | +0.04 / -0.01 |
| kf+hard+fb+guard | 0.54 ± 0.17 | 0.65 | 2.04 | 0.00 / 0.76 | +0.00 / — | 0.50 ± 0.25 | 0.59 | 2.08 | 0.00 / 0.70 | +0.03 / +0.05 | 0.49 ± 0.17 | 0.62 | 2.19 | 0.00 / 0.70 | +0.04 / -0.01 |
| kf_closedq | 0.91 ± 0.41 | 0.31 | 4.54 | 0.33 / 1.24 | +0.00 / — | 1.02 ± 0.70 | 0.26 | 5.80 | 0.26 / 1.40 | +0.01 / +0.07 | 0.88 ± 0.54 | 0.29 | 5.94 | 0.18 / 1.22 | +0.05 / -0.03 |
| kf_closedq+hard | 0.79 ± 0.32 | 0.24 | 5.40 | 0.00 / 1.12 | +0.00 / — | 0.93 ± 0.67 | 0.23 | 6.88 | 0.00 / 1.32 | +0.01 / +0.07 | 0.88 ± 0.55 | 0.23 | 7.08 | 0.00 / 1.25 | +0.05 / -0.03 |
| kf_aug | 1.54 ± 1.39 | 0.89 | 1.12 | 1.42 / 1.64 | +0.01 / — | 0.57 ± 0.18 | 0.83 | 1.34 | 0.54 / 0.59 | +0.02 / +0.01 | 0.49 ± 0.12 | 0.86 | 1.35 | 0.36 / 0.59 | +0.02 / -0.01 |
| oracle (bound) | 1.03 ± 0.50 | 0.25 | 6.26 | 0.38 / 1.40 | +0.00 / — | 1.38 ± 1.02 | 0.17 | 10.59 | 0.34 / 1.90 | +0.01 / +0.08 | 1.30 ± 1.00 | 0.15 | 12.71 | 0.23 / 1.82 | +0.05 / -0.03 |

Reconciliation and detection:

| estimator | RMSE all | cov95 all | nz all | |res| post | |corr| | FA (max / rate) | detected ≤100 (k/n) | median delay | held steps | d(f) | CUSUM s1 (k/n, median) | CUSUM s2 (k/n, median) | CUSUM FA max (s1 / s2) | lat p50 µs (this machine) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hold_last | 6.38 ± 1.01 | 0.74 | 3.03 | — | — | 0 / 0.0e+00 | 0/20 | 192 | 0 | 0 | — | — | — | 318 |
| kf | 0.76 ± 0.31 | 0.77 | 1.60 | — | — | 0 / 0.0e+00 | 0/20 | > 600 | 0 | 0 | 2/20, > 600 | 1/20, > 600 | 0 / 0 | 392 |
| kf+soft(1/lam=4) | 0.71 ± 0.27 | 0.77 | 1.60 | 5.2e-01 | 0.06 ± 0.03 | 0 / 0.0e+00 | 0/20 | > 600 | 0 | 0 | 2/20, > 600 | 1/20, > 600 | 0 / 0 | 591 |
| kf+hard | 0.52 ± 0.14 | 0.68 | 1.99 | 4.0e-16 | 0.48 ± 0.24 | 0 / 0.0e+00 | 0/20 | > 600 | 0 | 0 | 2/20, > 600 | 1/20, > 600 | 0 / 0 | 636 |
| kf+hard+guard | 0.55 ± 0.25 | 0.68 | 1.99 | 4.1e-16 | 0.46 ± 0.20 | 0 / 0.0e+00 | 0/20 | > 600 | 6 | 0 | 2/20, > 600 | 1/20, > 600 | 0 / 0 | 631 |
| kf+hard+fb | 0.52 ± 0.12 | 0.66 | 2.06 | 2.0e-16 | 0.00 ± 0.00 | 0 / 0.0e+00 | 0/20 | > 600 | 0 | 0 | 2/20, 262 | 1/20, > 600 | 0 / 0 | 643 |
| kf+hard+fb+guard | 0.52 ± 0.12 | 0.66 | 2.06 | 2.0e-16 | 0.00 ± 0.00 | 0 / 0.0e+00 | 0/20 | > 600 | 0 | 0 | 2/20, 262 | 1/20, > 600 | 0 / 0 | 642 |
| kf_closedq | 0.89 ± 0.39 | 0.38 | 4.89 | — | — | 0 / 0.0e+00 | 0/20 | 589 | 0 | 0 | 2/20, 206 | 2/20, 432 | 0 / 0 | 395 |
| kf_closedq+hard | 0.82 ± 0.35 | 0.32 | 5.85 | 3.5e-16 | 0.29 ± 0.13 | 0 / 0.0e+00 | 0/20 | 589 | 0 | 0 | 2/20, 206 | 2/20, 432 | 0 / 0 | 635 |
| kf_aug | 1.18 ± 0.95 | 0.88 | 1.22 | — | — | 0 / 0.0e+00 | 0/20 | > 600 | 0 | 0 | 1/20, > 600 | 0/20, > 600 | 0 / 0 | 427 |
| oracle (bound) | 1.13 ± 0.60 | 0.31 | 8.63 | — | — | 0 / 0.0e+00 | 0/20 | 511 | 0 | 0 | 2/20, 206 | 2/20, 414 | 0 / 0 | 401 |

Augmented-state outputs (pump scale alpha, boundary flux L in kg/s):

| estimator | alpha RMSE (pump on) | alpha RMSE (pump on) blackout / recovery / steady | L RMSE | L RMSE blackout / recovery / steady | alpha flag (k/n, median) | L flag (k/n, median) | FA max (alpha / L) | σ_α / σ_L at run end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kf_aug | 0.061 ± 0.028 | 0.065 / — / — | 0.0103 ± 0.0060 | 0.0085 / 0.0107 / 0.0108 | 0/20, > 600 | 0/20, > 600 | 0 / 0 | 0.042 / 0.0144 |

### leak_stale_constraint

10 kg leaks from reservoir 2 during steps 300-500. The declared closed-boundary constraint becomes STALE at step 300. The fault direction (0, -1) has a component along row(A).

Accuracy and calibration per window:

| estimator | RMSE pre_leak | cov95 pre_leak | nz pre_leak | err row/null pre_leak | z̄ s1/s2 pre_leak | RMSE leak_and_after | cov95 leak_and_after | nz leak_and_after | err row/null leak_and_after | z̄ s1/s2 leak_and_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hold_last | 1.96 ± 0.05 | 0.95 | 0.98 | 1.96 / 1.96 | — / — | 1.99 ± 0.06 | 0.95 | 0.99 | 1.97 / 2.00 | — / — |
| kf | 0.33 ± 0.08 | 0.98 | 0.79 | 0.32 / 0.34 | -0.00 / +0.01 | 1.07 ± 0.07 | 0.60 | 3.35 | 1.07 / 1.05 | +0.00 / -0.67 |
| kf+soft(1/lam=4) | 0.31 ± 0.07 | 0.98 | 0.77 | 0.26 / 0.34 | -0.00 / +0.01 | 1.15 ± 0.07 | 0.56 | 3.65 | 1.23 / 1.05 | +0.00 / -0.67 |
| kf+hard | 0.24 ± 0.07 | 0.97 | 0.81 | 0.00 / 0.34 | -0.00 / +0.01 | 3.80 ± 0.01 | 0.09 | 16.87 | 5.26 / 1.06 | +0.00 / -0.67 |
| kf+hard+guard | 0.24 ± 0.07 | 0.97 | 0.81 | 0.01 / 0.34 | -0.00 / +0.01 | 1.12 ± 0.07 | 0.55 | 3.87 | 1.17 / 1.05 | +0.00 / -0.67 |
| kf+hard+fb | 0.24 ± 0.07 | 0.97 | 0.81 | 0.00 / 0.34 | -0.00 / +0.01 | 3.80 ± 0.01 | 0.09 | 16.87 | 5.26 / 1.06 | -1.31 / -1.98 |
| kf+hard+fb+guard | 0.24 ± 0.07 | 0.97 | 0.81 | 0.00 / 0.34 | -0.00 / +0.01 | 3.80 ± 0.01 | 0.09 | 16.87 | 5.26 / 1.06 | -1.31 / -1.98 |
| kf_closedq | 0.29 ± 0.10 | 0.96 | 0.93 | 0.28 / 0.30 | +0.00 / +0.02 | 3.44 ± 0.06 | 0.13 | 25.11 | 4.08 / 2.65 | -0.43 / -2.18 |
| kf_closedq+hard | 0.21 ± 0.08 | 0.95 | 0.96 | 0.00 / 0.30 | +0.00 / +0.02 | 4.17 ± 0.03 | 0.10 | 34.62 | 5.26 / 2.65 | -0.43 / -2.18 |
| kf_aug | 0.42 ± 0.07 | 0.98 | 0.87 | 0.40 / 0.42 | -0.00 / +0.00 | 0.40 ± 0.07 | 0.95 | 0.94 | 0.40 / 0.38 | +0.00 / -0.01 |
| oracle (bound) | 0.29 ± 0.10 | 0.94 | 1.01 | 0.28 / 0.30 | +0.00 / +0.02 | 0.11 ± 0.06 | 0.88 | 1.08 | 0.09 / 0.11 | +0.00 / +0.00 |

Reconciliation and detection:

| estimator | RMSE all | cov95 all | nz all | |res| post | |corr| | FA (max / rate) | detected ≤100 (k/n) | median delay | held steps | d(f) | CUSUM s1 (k/n, median) | CUSUM s2 (k/n, median) | CUSUM FA max (s1 / s2) | lat p50 µs (this machine) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hold_last | 1.97 ± 0.03 | 0.95 | 0.99 | — | — | 0 / 0.0e+00 | 0/20 | 166 | 0 | 0.125 | — | — | — | 318 |
| kf | 0.79 ± 0.04 | 0.79 | 2.43 | — | — | 1 / 1.7e-04 | 20/20 | 66 | 0 | 4.93 | 0/20, > 300 | 20/20, 60 | 1 / 1 | 398 |
| kf+soft(1/lam=4) | 0.84 ± 0.04 | 0.77 | 2.64 | 2.7e+00 | 0.10 ± 0.00 | 1 / 1.7e-04 | 20/20 | 66 | 0 | 4.93 | 0/20, > 300 | 20/20, 60 | 1 / 1 | 601 |
| kf+hard | 2.69 ± 0.01 | 0.53 | 11.94 | 5.2e-16 | 1.99 ± 0.06 | 1 / 1.7e-04 | 20/20 | 66 | 0 | 4.93 | 0/20, > 300 | 20/20, 60 | 1 / 1 | 637 |
| kf+hard+guard | 0.81 ± 0.05 | 0.76 | 2.80 | 6.2e-16 | 0.26 ± 0.06 | 1 / 1.7e-04 | 20/20 | 66 | 235 | 4.93 | 0/20, > 300 | 20/20, 60 | 1 / 1 | 600 |
| kf+hard+fb | 2.69 ± 0.01 | 0.53 | 11.95 | 2.3e-16 | 0.00 ± 0.00 | 0 / 0.0e+00 | 0/20 | > 300 | 0 | 200 | 9/20, 104 | 20/20, 46 | 1 / 0 | 651 |
| kf+hard+fb+guard | 2.69 ± 0.01 | 0.53 | 11.95 | 2.3e-16 | 0.00 ± 0.00 | 0 / 0.0e+00 | 0/20 | > 300 | 0 | 200 | 9/20, 104 | 20/20, 46 | 1 / 0 | 649 |
| kf_closedq | 2.44 ± 0.04 | 0.55 | 17.77 | — | — | 1 / 1.7e-04 | 15/20 | 92 | 0 | 71.2 | 1/20, 205 | 20/20, 42 | 1 / 1 | 396 |
| kf_closedq+hard | 2.95 ± 0.02 | 0.52 | 24.49 | 4.1e-16 | 0.58 ± 0.07 | 1 / 1.7e-04 | 15/20 | 92 | 0 | 71.2 | 1/20, 205 | 20/20, 42 | 1 / 1 | 637 |
| kf_aug | 0.41 ± 0.05 | 0.96 | 0.91 | — | — | 0 / 0.0e+00 | 20/20 | 54 | 0 | 3.23 | 0/20, > 300 | 1/20, > 300 | 1 / 0 | 427 |
| oracle (bound) | 0.22 ± 0.07 | 0.91 | 1.07 | — | — | 1 / 1.7e-04 | 20/20 | 14 | 0 | 71.3 | 0/20, > 300 | 0/20, > 300 | 1 / 1 | 402 |

Augmented-state outputs (pump scale alpha, boundary flux L in kg/s):

| estimator | alpha RMSE (pump on) | alpha RMSE (pump on) pre_leak / leak_and_after | L RMSE | L RMSE pre_leak / leak_and_after | alpha flag (k/n, median) | L flag (k/n, median) | FA max (alpha / L) | σ_α / σ_L at run end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kf_aug | 0.027 ± 0.008 | 0.027 / — | 0.0183 ± 0.0011 | 0.0079 / 0.0246 | 0/20, > 300 | 9/20, 103 | 0 / 0 | 0.041 / 0.0144 |

### bias_quant_delay

Closed system; sensor 1 acquires an undeclared +3 kg bias at step 200; 0.5 kg quantization; 5-step arrival delay. Constraint is TRUE but the evidence is not. The fault direction (1, 0) has a component along row(A).

Accuracy and calibration per window:

| estimator | RMSE pre_bias | cov95 pre_bias | nz pre_bias | err row/null pre_bias | z̄ s1/s2 pre_bias | RMSE post_bias | cov95 post_bias | nz post_bias | err row/null post_bias | z̄ s1/s2 post_bias |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hold_last | 2.03 ± 0.07 | 0.95 | 1.01 | 2.01 / 2.06 | — / — | 2.97 ± 0.07 | 0.81 | 1.48 | 2.91 / 3.02 | — / — |
| kf | 0.34 ± 0.08 | 0.99 | 0.71 | 0.33 / 0.33 | -0.01 / -0.00 | 1.98 ± 0.07 | 0.52 | 5.85 | 1.96 / 1.99 | +0.16 / -0.00 |
| kf+soft(1/lam=4) | 0.31 ± 0.07 | 0.99 | 0.69 | 0.26 / 0.34 | -0.01 / -0.00 | 1.92 ± 0.07 | 0.52 | 5.77 | 1.86 / 1.99 | +0.16 / -0.00 |
| kf+hard | 0.24 ± 0.06 | 0.99 | 0.73 | 0.00 / 0.34 | -0.01 / -0.00 | 1.41 ± 0.07 | 0.05 | 5.88 | 0.00 / 1.99 | +0.16 / -0.00 |
| kf+hard+guard | 0.24 ± 0.06 | 0.99 | 0.73 | 0.00 / 0.34 | -0.01 / -0.00 | 1.97 ± 0.07 | 0.50 | 5.85 | 1.95 / 1.99 | +0.16 / -0.00 |
| kf+hard+fb | 0.24 ± 0.06 | 0.99 | 0.74 | 0.00 / 0.34 | -0.00 / +0.01 | 1.41 ± 0.07 | 0.05 | 5.88 | 0.00 / 1.99 | +0.83 / +0.67 |
| kf+hard+fb+guard | 0.24 ± 0.06 | 0.99 | 0.74 | 0.00 / 0.34 | -0.00 / +0.01 | 1.41 ± 0.07 | 0.05 | 5.88 | 0.00 / 1.99 | +0.83 / +0.67 |
| kf_closedq | 0.32 ± 0.09 | 0.97 | 0.89 | 0.31 / 0.31 | -0.01 / -0.00 | 1.31 ± 0.07 | 0.19 | 9.29 | 1.03 / 1.54 | +0.67 / +0.16 |
| kf_closedq+hard | 0.22 ± 0.06 | 0.95 | 0.92 | 0.00 / 0.31 | -0.01 / -0.00 | 1.09 ± 0.07 | 0.07 | 8.84 | 0.00 / 1.54 | +0.67 / +0.16 |
| kf_aug | 0.42 ± 0.08 | 0.99 | 0.76 | 0.40 / 0.43 | -0.01 / +0.00 | 2.05 ± 0.07 | 0.51 | 5.97 | 2.06 / 2.04 | +0.11 / -0.02 |
| oracle (bound) | 0.32 ± 0.09 | 0.96 | 0.94 | 0.31 / 0.31 | -0.01 / -0.00 | 1.03 ± 0.08 | 0.50 | 10.75 | 1.03 / 1.02 | +0.84 / -0.02 |

Reconciliation and detection:

| estimator | RMSE all | cov95 all | nz all | |res| post | |corr| | FA (max / rate) | detected ≤100 (k/n) | median delay | held steps | d(f) | CUSUM s1 (k/n, median) | CUSUM s2 (k/n, median) | CUSUM FA max (s1 / s2) | lat p50 µs (this machine) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hold_last | 2.69 ± 0.05 | 0.85 | 1.34 | — | — | 0 / 0.0e+00 | 0/20 | > 400 | 0 | 0.124 | — | — | — | 319 |
| kf | 1.63 ± 0.06 | 0.67 | 4.79 | — | — | 0 / 0.0e+00 | 20/20 | 36 | 0 | 4.37 | 20/20, 14 | 0/20, > 400 | 0 / 0 | 424 |
| kf+soft(1/lam=4) | 1.58 ± 0.06 | 0.67 | 4.73 | 1.8e+00 | 0.08 ± 0.01 | 0 / 0.0e+00 | 20/20 | 36 | 0 | 4.37 | 20/20, 14 | 0/20, > 400 | 0 / 0 | 614 |
| kf+hard | 1.16 ± 0.06 | 0.36 | 4.82 | 4.3e-16 | 1.34 ± 0.07 | 0 / 0.0e+00 | 20/20 | 36 | 0 | 4.37 | 20/20, 14 | 0/20, > 400 | 0 / 0 | 655 |
| kf+hard+guard | 1.61 ± 0.06 | 0.66 | 4.80 | 9.2e-16 | 0.29 ± 0.06 | 0 / 0.0e+00 | 20/20 | 36 | 362 | 4.37 | 20/20, 14 | 0/20, > 400 | 0 / 0 | 468 |
| kf+hard+fb | 1.16 ± 0.06 | 0.36 | 4.82 | 2.0e-16 | 0.00 ± 0.00 | 0 / 0.0e+00 | 0/20 | > 400 | 0 | 33.3 | 20/20, 13 | 12/20, 98 | 0 / 0 | 878 |
| kf+hard+fb+guard | 1.16 ± 0.06 | 0.36 | 4.82 | 2.0e-16 | 0.00 ± 0.00 | 0 / 0.0e+00 | 0/20 | > 400 | 0 | 33.3 | 20/20, 13 | 12/20, 98 | 0 / 0 | 881 |
| kf_closedq | 1.08 ± 0.06 | 0.45 | 7.60 | — | — | 0 / 0.0e+00 | 20/20 | 58 | 0 | 70.2 | 20/20, 13 | 0/20, > 400 | 0 / 0 | 424 |
| kf_closedq+hard | 0.90 ± 0.06 | 0.37 | 7.24 | 4.2e-16 | 0.70 ± 0.06 | 0 / 0.0e+00 | 20/20 | 58 | 0 | 70.2 | 20/20, 13 | 0/20, > 400 | 0 / 0 | 662 |
| kf_aug | 1.69 ± 0.05 | 0.67 | 4.89 | — | — | 0 / 0.0e+00 | 20/20 | 38 | 0 | 2.63 | 19/20, 14 | 0/20, > 400 | 0 / 0 | 476 |
| oracle (bound) | 0.86 ± 0.06 | 0.65 | 8.80 | — | — | 0 / 0.0e+00 | 20/20 | 58 | 0 | 70.2 | 20/20, 13 | 0/20, > 400 | 0 / 0 | 438 |

Augmented-state outputs (pump scale alpha, boundary flux L in kg/s):

| estimator | alpha RMSE (pump on) | alpha RMSE (pump on) pre_bias / post_bias | L RMSE | L RMSE pre_bias / post_bias | alpha flag (k/n, median) | L flag (k/n, median) | FA max (alpha / L) | σ_α / σ_L at run end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kf_aug | 0.050 ± 0.010 | 0.031 / 0.084 | 0.0075 ± 0.0008 | 0.0067 / 0.0078 | 15/20, 57 | 0/20, > 400 | 0 / 0 | 0.041 / 0.0151 |

### closed_wrong_prior

Closed system as closed_noise, but the estimator is initialised from a DECLARED initial fill of 74/26 kg (truth 70/30) with 5 kg prior std. Constraint is TRUE; the prior is wrong and has to be forgotten from the evidence.

Accuracy and calibration per window:

| estimator | RMSE settle | cov95 settle | nz settle | err row/null settle | z̄ s1/s2 settle | RMSE steady | cov95 steady | nz steady | err row/null steady | z̄ s1/s2 steady |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hold_last | 1.97 ± 0.13 | 0.96 | 0.98 | 1.98 / 1.95 | — / — | 2.01 ± 0.05 | 0.95 | 1.00 | 2.02 / 1.99 | — / — |
| kf | 0.60 ± 0.21 | 0.97 | 0.82 | 0.59 / 0.56 | -0.07 / +0.03 | 0.23 ± 0.04 | 1.00 | 0.71 | 0.22 / 0.22 | +0.01 / -0.00 |
| kf+soft(1/lam=4) | 0.50 ± 0.18 | 0.98 | 0.79 | 0.39 / 0.56 | -0.07 / +0.03 | 0.22 ± 0.04 | 1.00 | 0.70 | 0.21 / 0.22 | +0.01 / -0.00 |
| kf+hard | 0.40 ± 0.19 | 0.97 | 0.79 | 0.00 / 0.56 | -0.07 / +0.03 | 0.16 ± 0.05 | 0.99 | 0.69 | 0.00 / 0.22 | +0.01 / -0.00 |
| kf+hard+guard | 0.40 ± 0.19 | 0.97 | 0.79 | 0.00 / 0.56 | -0.07 / +0.03 | 0.16 ± 0.05 | 0.99 | 0.69 | 0.00 / 0.22 | +0.01 / -0.00 |
| kf+hard+fb | 0.40 ± 0.19 | 0.97 | 0.79 | 0.00 / 0.56 | -0.07 / +0.03 | 0.16 ± 0.05 | 0.99 | 0.69 | 0.00 / 0.22 | +0.03 / +0.02 |
| kf+hard+fb+guard | 0.40 ± 0.19 | 0.97 | 0.79 | 0.00 / 0.56 | -0.07 / +0.03 | 0.16 ± 0.05 | 0.99 | 0.69 | 0.00 / 0.22 | +0.03 / +0.02 |
| kf_closedq | 0.59 ± 0.21 | 0.97 | 0.86 | 0.59 / 0.56 | -0.07 / +0.03 | 0.10 ± 0.04 | 0.98 | 0.74 | 0.07 / 0.12 | +0.03 / +0.01 |
| kf_closedq+hard | 0.39 ± 0.19 | 0.97 | 0.82 | 0.00 / 0.56 | -0.07 / +0.03 | 0.09 ± 0.05 | 0.98 | 0.71 | 0.00 / 0.12 | +0.03 / +0.01 |
| kf_aug | 0.60 ± 0.20 | 0.97 | 0.80 | 0.60 / 0.56 | -0.07 / +0.02 | 0.31 ± 0.05 | 0.99 | 0.77 | 0.30 / 0.31 | +0.01 / -0.00 |
| oracle (bound) | 0.59 ± 0.21 | 0.97 | 0.87 | 0.59 / 0.56 | -0.07 / +0.03 | 0.10 ± 0.05 | 0.91 | 0.98 | 0.07 / 0.11 | +0.03 / +0.01 |

Reconciliation and detection:

| estimator | RMSE all | cov95 all | nz all | |res| post | |corr| | FA (max / rate) | detected ≤100 (k/n) | median delay | held steps | d(f) | CUSUM s1 (k/n, median) | CUSUM s2 (k/n, median) | CUSUM FA max (s1 / s2) | lat p50 µs (this machine) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hold_last | 2.00 ± 0.04 | 0.95 | 1.00 | — | — | 0 / 0.0e+00 | — | — | 0 | — | — | — | — | 316 |
| kf | 0.28 ± 0.04 | 0.99 | 0.73 | — | — | 0 / 0.0e+00 | — | — | 0 | — | — | — | 1 / 0 | 389 |
| kf+soft(1/lam=4) | 0.26 ± 0.04 | 0.99 | 0.72 | 2.6e-01 | 0.02 ± 0.00 | 0 / 0.0e+00 | — | — | 0 | — | — | — | 1 / 0 | 594 |
| kf+hard | 0.20 ± 0.05 | 0.99 | 0.72 | 3.6e-16 | 0.20 ± 0.03 | 0 / 0.0e+00 | — | — | 0 | — | — | — | 1 / 0 | 636 |
| kf+hard+guard | 0.20 ± 0.05 | 0.99 | 0.72 | 3.6e-16 | 0.20 ± 0.03 | 0 / 0.0e+00 | — | — | 0 | — | — | — | 1 / 0 | 637 |
| kf+hard+fb | 0.20 ± 0.05 | 0.99 | 0.72 | 2.3e-16 | 0.00 ± 0.00 | 0 / 0.0e+00 | — | — | 0 | — | — | — | 1 / 1 | 646 |
| kf+hard+fb+guard | 0.20 ± 0.05 | 0.99 | 0.72 | 2.3e-16 | 0.00 ± 0.00 | 0 / 0.0e+00 | — | — | 0 | — | — | — | 1 / 1 | 646 |
| kf_closedq | 0.22 ± 0.05 | 0.97 | 0.85 | — | — | 0 / 0.0e+00 | — | — | 0 | — | — | — | 1 / 1 | 397 |
| kf_closedq+hard | 0.16 ± 0.05 | 0.97 | 0.83 | 4.0e-16 | 0.12 ± 0.05 | 0 / 0.0e+00 | — | — | 0 | — | — | — | 1 / 1 | 639 |
| kf_aug | 0.36 ± 0.04 | 0.99 | 0.79 | — | — | 0 / 0.0e+00 | — | — | 0 | — | — | — | 1 / 0 | 422 |
| oracle (bound) | 0.22 ± 0.05 | 0.92 | 1.01 | — | — | 0 / 0.0e+00 | — | — | 0 | — | — | — | 1 / 1 | 398 |

Augmented-state outputs (pump scale alpha, boundary flux L in kg/s):

| estimator | alpha RMSE (pump on) | alpha RMSE (pump on) settle / steady | L RMSE | L RMSE settle / steady | alpha flag (k/n, median) | L flag (k/n, median) | FA max (alpha / L) | σ_α / σ_L at run end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kf_aug | 0.025 ± 0.011 | — / — | 0.0065 ± 0.0007 | 0.0072 / 0.0060 | —, — | —, — | 0 / 0 | 0.041 / 0.0144 |

### closed_uncertain_total

Closed system as closed_noise (own degradation seed). The declared total is 100 kg + an offset drawn per seed from N(0, 1 kg²), and the constraint DECLARES that uncertainty (b_var = 1 kg²): b is within its declared uncertainty, so the joint hypothesis holds and every flag is a false alarm. Every projecting spec honours b_var: hard is the Kalman update with pseudo-measurement variance 1 kg², soft adds its 1/λ on top.

Accuracy and calibration per window:

| estimator | RMSE steady | cov95 steady | nz steady | err row/null steady | z̄ s1/s2 steady |
| --- | --- | --- | --- | --- | --- |
| hold_last | 2.01 ± 0.06 | 0.95 | 1.00 | 1.99 / 2.03 | — / — |
| kf | 0.21 ± 0.04 | 0.99 | 0.66 | 0.20 / 0.22 | +0.00 / -0.00 |
| kf+soft(1/lam=4) | 0.21 ± 0.04 | 0.99 | 0.65 | 0.19 / 0.22 | +0.00 / -0.00 |
| kf+hard | 0.21 ± 0.04 | 0.99 | 0.69 | 0.20 / 0.22 | +0.00 / -0.00 |
| kf+hard+guard | 0.21 ± 0.04 | 0.99 | 0.69 | 0.20 / 0.22 | +0.00 / -0.00 |
| kf+hard+fb | 0.45 ± 0.24 | 0.63 | 1.73 | 0.57 / 0.22 | +0.05 / +0.04 |
| kf+hard+fb+guard | 0.45 ± 0.24 | 0.63 | 1.73 | 0.57 / 0.22 | +0.05 / +0.04 |
| kf_closedq | 0.11 ± 0.05 | 0.97 | 0.81 | 0.08 / 0.13 | +0.00 / +0.00 |
| kf_closedq+hard | 0.11 ± 0.05 | 0.97 | 0.81 | 0.08 / 0.13 | +0.00 / +0.00 |
| kf_aug | 0.29 ± 0.05 | 0.99 | 0.72 | 0.28 / 0.29 | +0.00 / +0.00 |
| oracle (bound) | 0.11 ± 0.05 | 0.87 | 1.14 | 0.08 / 0.13 | +0.00 / +0.00 |

Reconciliation and detection:

| estimator | RMSE all | cov95 all | nz all | |res| post | |corr| | FA (max / rate) | detected ≤100 (k/n) | median delay | held steps | d(f) | CUSUM s1 (k/n, median) | CUSUM s2 (k/n, median) | CUSUM FA max (s1 / s2) | lat p50 µs (this machine) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hold_last | 2.01 ± 0.04 | 0.95 | 1.00 | — | — | 0 / 0.0e+00 | — | — | 0 | — | — | — | — | 262 |
| kf | 0.27 ± 0.04 | 0.99 | 0.72 | — | — | 0 / 0.0e+00 | — | — | 0 | — | — | — | 0 / 0 | 329 |
| kf+soft(1/lam=4) | 0.26 ± 0.04 | 0.99 | 0.71 | 8.7e-01 | 0.03 ± 0.02 | 0 / 0.0e+00 | — | — | 0 | — | — | — | 0 / 0 | 586 |
| kf+hard | 0.26 ± 0.04 | 0.99 | 0.76 | 7.4e-01 | 0.12 ± 0.07 | 0 / 0.0e+00 | — | — | 0 | — | — | — | 0 / 0 | 566 |
| kf+hard+guard | 0.26 ± 0.04 | 0.99 | 0.76 | 7.4e-01 | 0.12 ± 0.07 | 0 / 0.0e+00 | — | — | 0 | — | — | — | 0 / 0 | 568 |
| kf+hard+fb | 0.46 ± 0.23 | 0.65 | 1.72 | 1.0e-01 | 0.01 ± 0.00 | 0 / 0.0e+00 | — | — | 0 | — | — | — | 4 / 5 | 578 |
| kf+hard+fb+guard | 0.46 ± 0.23 | 0.65 | 1.72 | 1.0e-01 | 0.01 ± 0.00 | 0 / 0.0e+00 | — | — | 0 | — | — | — | 4 / 5 | 575 |
| kf_closedq | 0.21 ± 0.04 | 0.96 | 0.89 | — | — | 0 / 0.0e+00 | — | — | 0 | — | — | — | 1 / 0 | 327 |
| kf_closedq+hard | 0.20 ± 0.05 | 0.96 | 0.91 | 8.0e-01 | 0.04 ± 0.02 | 0 / 0.0e+00 | — | — | 0 | — | — | — | 1 / 0 | 565 |
| kf_aug | 0.34 ± 0.04 | 0.99 | 0.78 | — | — | 0 / 0.0e+00 | — | — | 0 | — | — | — | 0 / 0 | 354 |
| oracle (bound) | 0.21 ± 0.04 | 0.90 | 1.09 | — | — | 0 / 0.0e+00 | — | — | 0 | — | — | — | 1 / 0 | 332 |

Augmented-state outputs (pump scale alpha, boundary flux L in kg/s):

| estimator | alpha RMSE (pump on) | alpha RMSE (pump on) steady | L RMSE | L RMSE steady | alpha flag (k/n, median) | L flag (k/n, median) | FA max (alpha / L) | σ_α / σ_L at run end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kf_aug | 0.026 ± 0.010 | — | 0.0066 ± 0.0010 | 0.0058 | —, — | —, — | 0 / 0 | 0.041 / 0.0144 |
