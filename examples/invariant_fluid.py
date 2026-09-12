"""Estimate two fluid masses in two coordinate charts using the invariant layer.

All uncertainties are declared synthetic values. This illustrates the additive
group special case, which is equivalent to an ordinary affine Kalman filter.
"""
import numpy as np

from set_lcm.coordinates import AffineCoordinates
from set_lcm.invariant import GaussianState, predict, update


def main():
    # Kilograms in two tanks, with known exchange dynamics per one-second step.
    state = GaussianState([40, 60], [[1, .2], [.2, 1.5]])
    F = np.array([[.96, .03], [.04, .97]])
    drift = np.array([.2, -.2])
    Q = .01 * np.array([[1, -1], [-1, 1]])  # uncertain internal transfer only
    H, offset = np.eye(2), np.array([.7, -.4])
    R = np.array([[.25, .08], [.08, .36]])  # correlated gauge errors, kg^2

    # Both masses in grams, with a deterministic second-coordinate origin.
    # Transforming just the values while leaving covariances/models unchanged is wrong.
    chart = AffineCoordinates([[1000, 0], [0, 1000]], [0, 100])
    chart_state = chart.transform_state(state)
    chart_F, chart_drift, chart_Q = chart.transform_dynamics(F, drift, Q)
    chart_H, chart_offset = chart.transform_observation(H, offset)
    records = (([40.9, 59.6], [True, True]), ([41.4, np.nan], [True, False]))

    for step, (measurement, mask) in enumerate(records, start=1):
        prior = predict(state, F, drift, Q)
        result = update(prior, measurement, H, R, offset, mask)
        chart_prior = predict(chart_state, chart_F, chart_drift, chart_Q)
        chart_result = update(chart_prior, measurement, chart_H, R, chart_offset, mask)
        state, chart_state = result.posterior, chart_result.posterior
        restored = chart.restore_state(chart_state)
        np.testing.assert_allclose(restored.mean, state.mean, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(restored.covariance, state.covariance, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(chart_result.statistic, result.statistic, rtol=1e-11, atol=1e-11)
        print(f"Step {step}: masses {np.round(state.mean, 4).tolist()} kg")
        print(f"  Pre-update innovation: {np.round(result.innovation, 4).tolist()} kg")
        print(f"  Innovation statistic: {result.statistic:.4f}; observed dimension: {result.dof}")
        print("  Equivalent result in the gram/datum chart: verified")
    print("The statistic is model-conditional evidence, not a sensor-health verdict.")


if __name__ == "__main__":
    main()
