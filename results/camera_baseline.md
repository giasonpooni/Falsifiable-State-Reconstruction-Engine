# Synthetic camera/gauge level baseline

Generated with Python 3.13.12, numpy 2.5.3 on Linux-6.18.44-fc-v24-x86_64-with-glibc2.39; source sha256 50a22c67cd05, git cdea185e52. Latency columns are wall-clock on this machine and are not a claim.

**Rendered synthetic frames; no field validation or automatic source isolation.**

A separately calibrated camera detects a water edge and fixed fiducial. The marker registers vertical translation. Gauge-only, camera-only and correlated BLUE estimates are scored on identical valid times. An independent reference enters scoring only.

| scenario | method | available / possible | common samples | truth RMSE [m] | reference discrepancy RMSE [m] | truth interval coverage |
|---|---|---:|---:|---:|---:|---:|
| healthy | gauge_only | 512/512 | 512 | 0.005796 | 0.006050 | 503/512 |
| healthy | camera_only | 512/512 | 512 | 0.000847 | 0.002132 | 512/512 |
| healthy | combined | 512/512 | 512 | 0.000833 | 0.002120 | 512/512 |
| gauge_drift | gauge_only | 512/512 | 512 | 0.014749 | 0.014814 | 344/512 |
| gauge_drift | camera_only | 512/512 | 512 | 0.000847 | 0.002132 | 512/512 |
| gauge_drift | combined | 512/512 | 512 | 0.000966 | 0.002168 | 512/512 |
| camera_obstruction | gauge_only | 512/512 | 416 | 0.005721 | 0.006012 | 409/416 |
| camera_obstruction | camera_only | 416/512 | 416 | 0.000862 | 0.002184 | 416/416 |
| camera_obstruction | combined | 512/512 | 416 | 0.000845 | 0.002172 | 416/416 |
| camera_vertical_drift | gauge_only | 512/512 | 512 | 0.005796 | 0.006050 | 503/512 |
| camera_vertical_drift | camera_only | 512/512 | 512 | 0.000847 | 0.002132 | 512/512 |
| camera_vertical_drift | combined | 512/512 | 512 | 0.000833 | 0.002120 | 512/512 |
| duplicate_frame | gauge_only | 512/512 | 496 | 0.005819 | 0.006067 | 487/496 |
| duplicate_frame | camera_only | 496/512 | 496 | 0.000847 | 0.002153 | 496/496 |
| duplicate_frame | combined | 512/512 | 496 | 0.000832 | 0.002140 | 496/496 |
| dropped_frame | gauge_only | 512/512 | 480 | 0.005843 | 0.006125 | 471/480 |
| dropped_frame | camera_only | 480/512 | 480 | 0.000851 | 0.002143 | 480/480 |
| dropped_frame | combined | 512/512 | 480 | 0.000836 | 0.002133 | 480/480 |
| time_jitter | gauge_only | 512/512 | 512 | 0.005796 | 0.006050 | 503/512 |
| time_jitter | camera_only | 512/512 | 512 | 0.000847 | 0.002132 | 512/512 |
| time_jitter | combined | 512/512 | 512 | 0.000833 | 0.002120 | 512/512 |
| shared_calibration_offset | gauge_only | 512/512 | 512 | 0.031180 | 0.031113 | 1/512 |
| shared_calibration_offset | camera_only | 512/512 | 512 | 0.030078 | 0.030022 | 0/512 |
| shared_calibration_offset | combined | 512/512 | 512 | 0.030097 | 0.030040 | 0/512 |
| physical_level_change | gauge_only | 512/512 | 512 | 0.005796 | 0.006050 | 503/512 |
| physical_level_change | camera_only | 512/512 | 512 | 0.000846 | 0.002133 | 512/512 |
| physical_level_change | combined | 512/512 | 512 | 0.000831 | 0.002120 | 512/512 |

Coverage uses pointwise 95% intervals under the declared first-order covariance and is empirical conditional on one reused calibration. Time samples and paired scenarios are not independent calibration trials.

| scenario | consistent | ambiguous | identified | unexplained | insufficient | null rejections / records | computed / attempted spectral windows |
|---|---:|---:|---:|---:|---:|---:|---:|
| healthy | 16 | 0 | 0 | 0 | 0 | 0/16 | 64/64 |
| gauge_drift | 0 | 16 | 0 | 0 | 0 | 16/16 | 64/64 |
| camera_obstruction | 16 | 0 | 0 | 0 | 0 | 0/16 | 32/64 |
| camera_vertical_drift | 16 | 0 | 0 | 0 | 0 | 0/16 | 64/64 |
| duplicate_frame | 16 | 0 | 0 | 0 | 0 | 0/16 | 48/64 |
| dropped_frame | 16 | 0 | 0 | 0 | 0 | 0/16 | 32/64 |
| time_jitter | 16 | 0 | 0 | 0 | 0 | 0/16 | 0/64 |
| shared_calibration_offset | 16 | 0 | 0 | 0 | 0 | 0/16 | 64/64 |
| physical_level_change | 16 | 0 | 0 | 0 | 0 | 0/16 | 64/64 |

In the vertical-motion case, camera RMSE on the same 512 samples is 0.036888 m without registration and 0.000847 m with it.

## What these results establish

Gauge and camera drift profiles have opposite signs but the same span with unknown signed amplitude. They remain ambiguous when supported. A shared calibration offset can leave their difference consistent while both disagree with the independent reference; combination cannot remove that shared bias.

The JSON retains every seed, frame hashes/times/admission flags, extracted rows, estimates, raw differences, calibration covariance, quality and diagnostic outcomes, spectral windows and scoring denominators. Quality refusal is not a fault label; no faulty-source truth is supplied to inference.

- **Calibration:** Eight separate noiseless anchor images at exact fixed pixel designs; noisy reference heights fit one shared linear calibration. Thresholds and covariance are declared before evaluation, not tuned on evaluation outcomes.
- **Pairing:** Evaluation seeds supply independent per-frame errors, paired across scenarios. All records reuse one calibration realization from disjoint development seeds. Pooled coverage is empirical conditional on that realization, not independent calibration trials or a certified nominal coverage rate.
- **Covariance:** Independent time noise plus one shared reference offset affecting both the gauge and camera calibration. Cg=sigma_g^2*I+sigma_b^2*11'; Cov(g_i,c_j)=sigma_b^2. Camera C=J*Ctheta*J'+a_hat^2*(Cwater+Cmarker), J_i=[corrected_pixel_i,1]. Full cross-time matrices are used throughout; no covariance is estimated from held-out errors.
- **Covariance limits:** Camera propagation is first-order: the slope-error/pixel-error product term is omitted. Intervals use the declared marginal shared-calibration covariance; conditional coverage across this fixed calibration, image gates and fault cases is empirical, not exact Gaussian calibration.
- **Motion:** One-dimensional vertical translation from a fixed image fiducial is subtracted. Marker localization uncertainty is propagated; its reference row is exact in this simulation. This is not rotation, perspective correction or 3-D visual odometry.
- **Estimation:** Camera-only and gauge-only readings and per-time correlated BLUE combined estimates; full covariance propagated, no temporal smoothing, no automatic fault exclusion. All accuracy comparisons use the same physical times and common valid support; availability also counts source fallbacks.
- **Scoring:** Truth and independent reference are read only by simulator/scorer. RMSE versus truth differs from reference discrepancy RMSE. Reference discrepancy covariance is Cestimate+Creference under the declared independence. Pointwise coverage counts use common support, with all denominators retained; the joint reference squared discrepancy uses its full covariance and is reported without a calibrated decision threshold.
- **Spectra:** Advisory linear-detrended Hann periodograms of registered camera level in fixed disjoint eight-frame windows. Only admitted, contiguous, uniform windows are computed; no spectral threshold or clock recovery is claimed.
