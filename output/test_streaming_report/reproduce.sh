#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /mnt/TUM/rgbd_dataset_freiburg1_desk -m output/test_streaming_report --streaming_replay --tum_sequence freiburg1 --cap_max 30000 --iterations 1000 --streaming_initial_frames 5 --streaming_steps_per_frame 50
