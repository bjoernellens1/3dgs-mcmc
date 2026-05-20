#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /mnt/cps_persistent1_shared/datasets/public/TUM/tum_rgbd/freiburg1_desk/rgbd_dataset_freiburg1_desk -m output/test_single_ablation --streaming_replay --streaming_initial_frames 5 --streaming_max_frames 60 --streaming_steps_per_frame 50 --streaming_keyframe_window 40 --iterations 3000 --cap_max 200000 --tum_sequence freiburg1 --quiet
