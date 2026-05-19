#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /mnt/cps_persistent1_shared/datasets/christian/embedding_map/bags/slam/kitchen1_slam --config configs/kitchen.json --streaming_replay --orbbec_pose_source open3d_odometry --orbbec_open3d_odom_method hybrid --orbbec_open3d_odom_downscale 1 --orbbec_open3d_odom_stride 1 --streaming_frame_stride 1 --resolution 1 --streaming_depth_stride 2 --streaming_max_new_gaussians_per_frame 5000 --streaming_depth_loss_weight 0.1 --model_path output/kitchen_hybrid_hq_dense --streaming_eval_hold 8 --cap_max 1000000 --streaming_max_frames 100
