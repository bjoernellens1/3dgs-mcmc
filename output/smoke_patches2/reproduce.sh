#!/bin/bash
# Reproduces the training run stored in this output directory.
# Generated automatically at training start.
python train.py -s /data/TUM/rgbd_dataset_freiburg1_desk --tum_sequence freiburg1 --streaming_replay --streaming_free_space_loss_weight 0.01 --cap_max 100000 --iterations 300 --streaming_steps_per_frame 5 --streaming_max_frames 30 --streaming_initial_frames 5 --streaming_eval_hold 8 --quiet -m output/smoke_patches2
