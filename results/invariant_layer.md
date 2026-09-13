# Additive invariant-layer equivalence

Generated with Python 3.13.12, numpy 2.5.3 on Linux-6.18.44-fc-v24-x86_64-with-glibc2.39; source sha256 c2436dd517fa, git 25970ab2d5. Latency columns are wall-clock on this machine and are not a claim.

**This tests numerical equivalence to ordinary Kalman filtering, not a performance improvement.**

Five fixed state charts and four measurement order/unit charts use the same two-tank records. All means and covariances are mapped back before comparison. Processing statuses must match exactly.

Fixed absolute tolerance: 1e-08 in original kg, kg² and dimensionless NIS units.

| scenario | seed | largest mean difference [kg] | largest covariance difference [kg²] | largest NIS difference | equivalent chart runs |
|---|---:|---:|---:|---:|---:|
| clean | 0 | 3.553e-14 | 5.551e-17 | 3.482e-13 | 20/20 |
| clean | 1 | 7.105e-14 | 5.551e-17 | 3.755e-13 | 20/20 |
| clean | 2 | 6.395e-14 | 5.551e-17 | 2.776e-13 | 20/20 |
| clean | 3 | 4.974e-14 | 5.551e-17 | 4.130e-13 | 20/20 |
| clean | 4 | 4.974e-14 | 5.551e-17 | 3.428e-13 | 20/20 |
| clean | 5 | 3.553e-14 | 5.551e-17 | 4.166e-13 | 20/20 |
| clean | 6 | 4.974e-14 | 5.551e-17 | 3.713e-13 | 20/20 |
| clean | 7 | 6.395e-14 | 5.551e-17 | 3.531e-13 | 20/20 |
| biased_gauge | 0 | 6.395e-14 | 5.551e-17 | 8.313e-13 | 20/20 |
| biased_gauge | 1 | 4.974e-14 | 5.551e-17 | 6.679e-13 | 20/20 |
| biased_gauge | 2 | 4.263e-14 | 5.551e-17 | 7.319e-13 | 20/20 |
| biased_gauge | 3 | 4.263e-14 | 5.551e-17 | 7.319e-13 | 20/20 |
| biased_gauge | 4 | 3.553e-14 | 5.551e-17 | 5.045e-13 | 20/20 |
| biased_gauge | 5 | 5.684e-14 | 5.551e-17 | 7.674e-13 | 20/20 |
| biased_gauge | 6 | 5.684e-14 | 5.551e-17 | 7.994e-13 | 20/20 |
| biased_gauge | 7 | 7.105e-14 | 5.551e-17 | 8.740e-13 | 20/20 |

Equivalent comparisons: 320/320.

The JSON retains the model, charts, masks and per-seed/per-chart discrepancies, degrees of freedom and statuses.

## Interpretation

- **Noise:** Independent process and observation noise across times and from the initial error; full correlated R retained. Q is conserving random exchange.
- **Reference:** Independent direct-solve ordinary affine KF with Joseph covariance; receives observations and masks, never physical truth or fault labels.
- **Pairing:** The same eight noise seeds are reused across scenarios and charts. These are paired numerical comparisons, not independent performance trials.
- **Order:** Predict, then assimilate each sample jointly using the observed principal R submatrix. Sensor permutation/scaling preserves its mask and correlations.
- **Chart interpretation:** Fixed affine coordinate changes x'=T x+c, not physical interventions. Translation changes the coordinate origin, not an automorphism of the zero-identity additive group; centered errors transform as T e.
- **Statistic:** NIS is the pre-update innovation Mahalanobis statistic. Update/no_observations statuses are processing outcomes, not fault verdicts. No alarm threshold is used.
- **Limits:** The additive layer equals ordinary KF for this affine Gaussian model. No nonlinear IEKF accuracy, robustness, calibration or sensor-identification gain is claimed. Coordinate changes cannot resolve existing fault ambiguity.

The biased-gauge scenario deliberately violates the no-bias observation model. Equivalence there means both implementations apply the same model to the same faulty record; it is not evidence that either recovers the true state or identifies the faulty gauge.

Measurement scaling would change Gaussian log density by its Jacobian term; this report compares the invariant NIS instead of claiming raw log-likelihood equality.
