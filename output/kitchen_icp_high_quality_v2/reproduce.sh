#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /mnt/cps_persistent1_shared/datasets/christian/embedding_map/bags/slam/kitchen1_slam --config configs/kitchen.json --streaming_replay --orbbec_pose_source open3d_odometry --orbbec_open3d_odom_method icp --orbbec_open3d_odom_downscale 1 --orbbec_open3d_odom_stride 1 --streaming_frame_stride 1 --resolution 1 --model_path output/kitchen_icp_high_quality_v2 --streaming_eval_hold 8 --cap_max 500000 --streaming_max_frames 100
