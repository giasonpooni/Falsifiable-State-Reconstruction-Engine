"""Two declared tank estimates: one reconciled, one held for investigation.

Run from the checkout: uv run --frozen --python 3.13 python examples/quickstart.py
All values are synthetic teaching inputs, not sensor specifications or field calibration.
"""
from __future__ import annotations

import numpy as np

from set_lcm.lcm import chi2_quantile, consistency_stat, reconcile
from set_lcm.schema import ConstraintSet, StateEstimate


def check_estimate(values: list[float]) -> StateEstimate:
    # The declared total is uncertain: standard deviation 0.5 kg.
    constraint = ConstraintSet(
        version="two-tanks-example-v1",
        A=np.array([[1.0, 1.0]]),
        b=np.array([100.0]),
        b_var=np.array([0.5**2]),
        row_units=("kg",),
        description="The two tank masses total 100 kg, with declared uncertainty.",
    )
    estimate = np.array(values, dtype=float)
    covariance = np.eye(2)  # Independent 1 kg standard deviations in this example.
    threshold = chi2_quantile(constraint.rank, 0.999)
    score = consistency_stat(estimate, covariance, constraint)

    # The caller chooses the policy; a threshold alone does not hold correction.
    return reconcile(
        estimate, covariance, constraint, mode="hard",
        hold=score > threshold, threshold=threshold, stat=score,
    )


def main() -> None:
    for label, values in (("Small disagreement", [52.0, 46.0]),
                          ("Large disagreement", [60.0, 50.0])):
        result = check_estimate(values)
        print(label)
        print(f"  Score: {result.consistency_stat:.2f}; reference threshold: "
              f"{result.consistency_threshold:.2f}")
        print(f"  Action: {result.status.value}")
        print(f"  Original estimate: {result.x_unprojected.tolist()} kg")
        print(f"  Output estimate: {np.round(result.x, 3).tolist()} kg")
        print(f"  Residual before: {result.residual_pre[0]:.3f} kg")
        if result.residual_post is not None:
            print(f"  Residual after: {result.residual_post[0]:.3f} kg")
        else:
            print("  Correction held: investigate the measurements and model.")
        print()


if __name__ == "__main__":
    main()
