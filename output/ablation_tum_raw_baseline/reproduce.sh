#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /mnt/cps_persistent1_shared/datasets/public/TUM/tum_rgbd/freiburg1_desk/rgbd_dataset_freiburg1_desk -m output/ablation_tum_raw_baseline --streaming_replay --streaming_initial_frames 5 --streaming_keyframe_window 40 --streaming_steps_per_frame 100 --streaming_insert_from_depth --cap_max 100000 --iterations 2000 --tum_sequence freiburg1 --eval
