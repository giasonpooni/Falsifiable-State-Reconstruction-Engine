"""Create and replay a labeled synthetic tank-camera recording, then score separately."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from set_lcm.experiments.camera_baseline import SCENARIOS, infer_measurements
from set_lcm.experiments.camera_bundle import load_evaluation, load_observations, write_synthetic_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("work/camera-demo"))
    parser.add_argument("--scenario", choices=SCENARIOS, default="camera_vertical_drift")
    parser.add_argument("--seed", type=int, default=10000)
    args = parser.parse_args()
    manifest = write_synthetic_bundle(args.out_dir, seed=args.seed, scenario=args.scenario)
    analysis = infer_measurements(**load_observations(args.out_dir))
    print(f"SYNTHETIC tank-level replay: {args.scenario}; seed={args.seed}")
    print(f"Recording manifest: {manifest.resolve()}")
    print(f"Camera admitted {len(analysis.camera.observed_indices)}/{len(analysis.audit.frame_ids)} level readings")
    print(f"Raw gauge/camera diagnostic: {analysis.diagnostic['status']}")
    # Evaluation is opened only after the complete inference result exists.
    evaluation = load_evaluation(args.out_dir)
    common = sorted(set.intersection(*(set(value.observed_indices) for value in analysis.comparison.series.values())))
    for name, estimate in analysis.comparison.series.items():
        if common:
            rmse = np.sqrt(np.mean((estimate.heights[common] - evaluation["truth"][common])**2))
            print(f"{name}: truth RMSE {rmse:.6f} m on {len(common)} common samples; available {len(estimate.observed_indices)}")
        else:
            print(f"{name}: no common samples for comparison; available {len(estimate.observed_indices)}")
    print("Fixed vertical marker registration only; no general odometry or field validation.")


if __name__ == "__main__":
    main()
