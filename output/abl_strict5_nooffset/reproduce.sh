#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /mnt/cps_persistent1_shared/datasets/public/TUM/tum_rgbd/freiburg1_desk/rosbag/rgbd_dataset_freiburg1_desk.bag -m output/abl_strict5_nooffset --orbbec_sync_threshold_ms 5 --no-orbbec_sync_estimate_offset --streaming_replay --streaming_initial_frames 5 --streaming_keyframe_window 40 --streaming_steps_per_frame 50 --streaming_insert_from_depth --cap_max 150000 --iterations 5000 --tum_sequence freiburg1 --eval
