#!/usr/bin/env python3
"""Evaluate a streaming frame source trajectory against optional TUM GT."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arguments import ModelParams, OptimizationParams, PipelineParams, StreamingParams
from utils.streaming_frames import make_frame_source
from utils.trajectory_eval import (
    associate_by_timestamp,
    load_tum_trajectory,
    plot_trajectory,
    save_tum_trajectory,
    trajectory_metrics,
)


def _frames_to_arrays(frames):
    ts, poses, invalid = [], [], 0
    for frame in frames:
        if getattr(frame, "c2w", None) is None:
            invalid += 1
            continue
        ts.append(float(frame.timestamp))
        poses.append(np.asarray(frame.c2w, dtype=np.float64))
        if getattr(frame, "_odom_valid", True) is False:
            invalid += 1
    if not poses:
        raise RuntimeError("No frames with poses were produced by the frame source.")
    return np.asarray(ts, dtype=np.float64), np.stack(poses), invalid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    ModelParams(parser)
    PipelineParams(parser)
    OptimizationParams(parser)
    StreamingParams(parser)
    parser.add_argument("--label", default="", help="Method label for exported metrics.")
    args = parser.parse_args()

    if not args.model_path:
        args.model_path = os.path.join("output", "odometry_eval")
    args.source_path = os.path.abspath(args.source_path)
    os.makedirs(args.model_path, exist_ok=True)

    source = make_frame_source(args.source_path, args)
    frames = source.get_all() if hasattr(source, "get_all") else list(source)
    ts, c2w, invalid = _frames_to_arrays(frames)

    out_dir = Path(args.model_path)
    save_tum_trajectory(str(out_dir / "trajectory_tum.txt"), ts, c2w)

    gt_path = getattr(args, "tum_gt_path", "") or ""
    metrics = {
        "method": args.label or getattr(args, "orbbec_open3d_odom_method", "estimated"),
        "source_path": args.source_path,
        "n_frames": int(len(c2w)),
        "n_invalid_or_missing": int(invalid),
        "distance_m": None,
        "gt_distance_m": None,
    }
    plot_c2w = c2w
    gt_c2w = None
    if gt_path:
        gt_ts, gt_all = load_tum_trajectory(gt_path)
        matched_ts, matched_c2w, matched_gt, assoc = associate_by_timestamp(
            ts, c2w, gt_ts, gt_all, max_dt=float(getattr(args, "tum_association_max_dt", 0.03))
        )
        metrics["association"] = assoc
        metrics["gt_tum_path"] = gt_path
        if len(matched_ts) > 0:
            save_tum_trajectory(str(out_dir / "trajectory_tum_associated.txt"), matched_ts, matched_c2w)
            save_tum_trajectory(str(out_dir / "groundtruth_tum_associated.txt"), matched_ts, matched_gt)
            metrics.update(trajectory_metrics(matched_c2w, matched_gt))
            plot_c2w = matched_c2w
            gt_c2w = matched_gt
    else:
        metrics.update(trajectory_metrics(c2w, c2w))
        metrics["ate_rmse"] = None
        metrics["ate_mean"] = None
        metrics["rpe_trans_rmse"] = None
        metrics["rpe_rot_deg_rmse"] = None

    plot_trajectory(str(out_dir / "trajectory.png"), plot_c2w, title=metrics["method"], gt=gt_c2w, metrics=metrics)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
