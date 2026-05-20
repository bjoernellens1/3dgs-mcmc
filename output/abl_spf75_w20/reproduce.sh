#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /mnt/cps_persistent1_shared/datasets/public/TUM/tum_rgbd/freiburg1_desk/rgbd_dataset_freiburg1_desk -m output/abl_spf75_w20 --streaming_steps_per_frame 75 --streaming_keyframe_window 20 --iterations 5000 --streaming_replay --streaming_initial_frames 5 --cap_max 150000 --tum_sequence freiburg1 --eval
