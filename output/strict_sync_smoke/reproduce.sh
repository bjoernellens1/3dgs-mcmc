#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /mnt/cps_persistent1_shared/datasets/christian/embedding_map/bags/slam/kitchen1_slam -m output/strict_sync_smoke --streaming_replay --orbbec_pose_source open3d_odometry_live --orbbec_sync_threshold_ms 5 --orbbec_open3d_odom_stride 1 --orbbec_open3d_odom_downscale 1 --no-orbbec_open3d_odom_cache --iterations 1000
