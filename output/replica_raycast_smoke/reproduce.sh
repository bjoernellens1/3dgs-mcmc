#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /data/replica/office_0 -m output/replica_raycast_smoke -r 4 --eval --iterations 500 --cap_max 50000 --streaming_replay --streaming_steps_per_frame 100 --streaming_keyframe_window 60 --streaming_insert_isotropic_scale --streaming_depth_loss_weight 0.05 --streaming_depth_loss_type huber --test_iterations 250 500
