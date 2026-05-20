#!/usr/bin/env bash
set -euo pipefail
docker compose run --rm -T train python scripts/evaluate_streaming_odometry.py -s /mnt/cps_persistent1_shared/datasets/public/TUM/tum_rgbd/freiburg1_desk/rgbd_dataset_freiburg1_desk -m output/tum_ablation_sweeps/tum_odom_20260520_b/runs/odom_freiburg1_desk_gt_raw --tum_gt_path /mnt/cps_persistent1_shared/datasets/public/TUM/tum_rgbd/freiburg1_desk/rgbd_dataset_freiburg1_desk/groundtruth.txt --tum_association_max_dt 0.03 --streaming_max_frames 120 --label gt_raw --tum_sequence freiburg1 --tum_frame_stride 1
