#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /mnt/cps_persistent1_shared/datasets/public/TUM/tum_rgbd/freiburg1_desk/rosbag/rgbd_dataset_freiburg1_desk.bag -m output/test_rosbag_load --streaming_replay --streaming_initial_frames 5 --streaming_keyframe_window 8 --streaming_steps_per_frame 20 --cap_max 50000 --iterations 500 --tum_sequence freiburg1
