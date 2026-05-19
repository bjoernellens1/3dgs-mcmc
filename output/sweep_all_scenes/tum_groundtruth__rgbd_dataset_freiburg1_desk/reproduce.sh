#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /datasets/public/TUM/groundtruth/rgbd_dataset_freiburg1_desk -m output/sweep_all_scenes/tum_groundtruth__rgbd_dataset_freiburg1_desk --eval --streaming_replay --iterations 6000 --cap_max 200000 --streaming_camera_profile tum_kinect_v1 --streaming_steps_per_frame 50 --streaming_keyframe_window 60 --streaming_replay_buffer 100 --streaming_global_replay_ratio 0.4 --streaming_eval_hold 8 --tum_frame_stride 1 --tum_init rgbd --tum_init_frames 0 --tum_depth_stride 4 --tum_max_init_points 100000 --test_iterations 6000 --save_iterations 6000
