#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /mnt/cps_persistent1_shared/datasets/christian/embedding_map/bags/slam/kitchen1_slam -m output/orbbec_live_o3d_smoke_500 --eval --iterations 500 --cap_max 150000 --streaming_replay --streaming_camera_profile orbbec_femto_bolt --orbbec_pose_source open3d_odometry_live --orbbec_open3d_odom_stride 5 --orbbec_open3d_odom_downscale 4 --orbbec_open3d_odom_cache_dir output/open3d_odometry_eval/live_train_cache --streaming_frame_admission hybrid_keyframe --streaming_max_frames 120 --streaming_frame_stride 1 --streaming_steps_per_frame 10 --streaming_keyframe_window 20 --streaming_replay_buffer 40 --streaming_global_replay_ratio 0.3 --streaming_max_new_gaussians_per_frame 1000 --streaming_insert_isotropic_scale --streaming_eval_hold 8 --test_iterations 500 --save_iterations 500 --progress_video_interval 250 --quiet
