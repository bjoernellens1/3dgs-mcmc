#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /mnt/cps_persistent1_shared/datasets/bjoern/realsense_handheld/20260519_realsense435_office1.bag -m output/realsense_office1_fixed --streaming_replay --rosbag_profile realsense --streaming_frame_stride 3 --streaming_steps_per_frame 150 --streaming_keyframe_window 120 --streaming_enforce_full_coverage --cap_max 50000 --iterations 30000 --streaming_max_new_gaussians_per_frame 5000 --streaming_insert_voxel_size 0.01 --streaming_depth_filter_enabled --streaming_depth_loss_weight 0.05 --streaming_global_replay_ratio 0.3 --eval
