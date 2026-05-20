#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /mnt/TUM/rgbd_dataset_freiburg1_desk -m output/smoke_aniso_dense --streaming_replay --tum_sequence freiburg1 --cap_max 50000 --iterations 15000 --streaming_initial_frames 5 --streaming_steps_per_frame 150 --streaming_global_replay_ratio 0.3 --streaming_keyframe_coverage 0.6 --streaming_max_new_gaussians_per_frame 5000 --streaming_insert_voxel_size 0.01
