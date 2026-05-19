#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /mnt/cps_persistent1_shared/datasets/bjoern/cps-seggy/bags/rosbag2_2025_06_20-08_40_30 -m output/cps_seggy_smoke --eval --iterations 500 --cap_max 100000 --streaming_replay --streaming_camera_profile orbbec_femto_bolt --orbbec_pose_source open3d_odometry_live --orbbec_open3d_odom_stride 1 --orbbec_open3d_odom_downscale 1 --streaming_frame_admission hybrid_keyframe --streaming_steps_per_frame 25 --streaming_keyframe_window 60 --streaming_export_depth_comparison --test_iterations 500 --save_iterations 500
