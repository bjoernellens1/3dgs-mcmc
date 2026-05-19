#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /data/orbbec/export/10hz/kitchen_icl -m output/orbbec_profile_smoke --eval --iterations 500 --cap_max 20000 --streaming_replay --streaming_camera_profile orbbec_femto_bolt --streaming_steps_per_frame 10 --streaming_keyframe_window 8 --streaming_replay_buffer 16 --streaming_max_frames 20 --streaming_eval_hold 8 --test_iterations 500 --save_iterations 500
