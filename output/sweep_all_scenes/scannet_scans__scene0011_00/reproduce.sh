#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /datasets/public/ScanNet/scans/scene0011_00 -m output/sweep_all_scenes/scannet_scans__scene0011_00 --eval --streaming_replay --iterations 6000 --cap_max 100000 --streaming_steps_per_frame 50 --streaming_keyframe_window 60 --streaming_replay_buffer 100 --streaming_global_replay_ratio 0.4 --streaming_eval_hold 8 --scannet_init rgbd --scannet_frame_stride 20 --scannet_max_frames 120 --scannet_init_frames 80 --scannet_depth_stride 8 --scannet_max_init_points 50000 --test_iterations 6000 --save_iterations 6000
