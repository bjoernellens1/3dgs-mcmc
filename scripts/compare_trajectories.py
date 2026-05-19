#!/usr/bin/env python3
"""Compare trajectories from two or more training runs.

Usage:
    python scripts/compare_trajectories.py \
        --runs output/k1_35k_rgbd_odom output/k1_35k_icp \
        --out  output/k1_35k_rgbd_odom/trajectory_eval/vs_icp/

Produces:
  comparison.png  — overlaid trajectories (top-down + side view)
  disagreement.json — ATE-style RMSE between the two methods
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.trajectory_eval import (
    load_tum_trajectory,
    plot_trajectory_pair,
    trajectory_metrics,
)


def _find_tum(run_dir: str) -> str:
    candidate = os.path.join(run_dir, "trajectory_eval", "trajectory_tum.txt")
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(
        f"No trajectory_tum.txt under {run_dir}/trajectory_eval/. "
        "Make sure the run completed with --streaming_trajectory_eval (default on)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare trajectories from two training runs.")
    parser.add_argument("--runs", nargs=2, required=True, metavar="DIR",
                        help="Two run output directories")
    parser.add_argument("--out", required=True, metavar="DIR",
                        help="Output directory for comparison artefacts")
    parser.add_argument("--labels", nargs=2, default=None, metavar="LABEL",
                        help="Labels for the two runs (default: directory basenames)")
    args = parser.parse_args()

    run_a, run_b = args.runs
    labels = args.labels or [os.path.basename(run_a.rstrip("/")),
                              os.path.basename(run_b.rstrip("/"))]

    tum_a = _find_tum(run_a)
    tum_b = _find_tum(run_b)

    print(f"Loading {tum_a}")
    ts_a, c2w_a = load_tum_trajectory(tum_a)
    print(f"Loading {tum_b}")
    ts_b, c2w_b = load_tum_trajectory(tum_b)

    os.makedirs(args.out, exist_ok=True)

    png_path = os.path.join(args.out, "comparison.png")
    plot_trajectory_pair(
        png_path, c2w_a, c2w_b,
        labels=tuple(labels),
        title=f"Trajectory comparison: {labels[0]} vs {labels[1]}",
    )
    print(f"Saved {png_path}")

    # Use the shorter trajectory as the reference for disagreement metrics
    metrics = trajectory_metrics(c2w_a, c2w_b)
    metrics["run_a"] = run_a
    metrics["run_b"] = run_b
    metrics["label_a"] = labels[0]
    metrics["label_b"] = labels[1]
    metrics["n_frames_a"] = len(c2w_a)
    metrics["n_frames_b"] = len(c2w_b)

    json_path = os.path.join(args.out, "disagreement.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {json_path}")
    print(f"Disagreement ATE RMSE: {metrics.get('ate_rmse'):.4f} m  "
          f"RPE trans: {metrics.get('rpe_trans_rmse'):.4f} m  "
          f"RPE rot: {metrics.get('rpe_rot_deg_rmse'):.4f} deg")


if __name__ == "__main__":
    main()
