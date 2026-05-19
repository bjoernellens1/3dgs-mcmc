#!/usr/bin/env bash
set -euo pipefail

SRC="${1:-/mnt/cps_persistent1_shared/datasets/christian/embedding_map/bags/slam/kitchen1_slam}"
OUT="${2:-output/orbbec_rosbag_live_o3d_kitchen1_slam}"

python train.py \
  -s "$SRC" \
  -m "$OUT" \
  --eval \
  --iterations 35000 \
  --cap_max 150000 \
  --streaming_replay \
  --streaming_camera_profile orbbec_femto_bolt \
  --orbbec_pose_source open3d_odometry_live \
  --orbbec_open3d_odom_stride 5 \
  --orbbec_open3d_odom_downscale 4 \
  --streaming_frame_admission hybrid_keyframe \
  --streaming_frame_stride 1 \
  --streaming_steps_per_frame 50 \
  --streaming_keyframe_window 60 \
  --streaming_replay_buffer 100 \
  --streaming_global_replay_ratio 0.4 \
  --streaming_max_new_gaussians_per_frame 2000 \
  --streaming_insert_isotropic_scale \
  --streaming_depth_loss_weight 0.1 \
  --streaming_depth_loss_type huber \
  --streaming_free_space_loss_weight 0.01 \
  --streaming_global_reservoir_stride 4 \
  --streaming_eval_hold 8 \
  --mcmc_stop_growth_iter 35000 \
  --densify_until_iter 35000 \
  --streaming_global_maintenance_interval 500 \
  --sh_degree_schedule 500 15000 25000 \
  --test_iterations 10000 20000 35000 \
  --save_iterations 20000 35000
