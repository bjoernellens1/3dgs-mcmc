#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /data/orbbec/export/10hz/kitchen_icl -m output/orbbec_profile_uncapped_smoke --eval --iterations 700 --cap_max 150000 --streaming_replay --streaming_camera_profile orbbec_femto_bolt --streaming_steps_per_frame 50 --streaming_keyframe_window 8 --streaming_replay_buffer 16 --streaming_global_replay_ratio 0.1 --streaming_max_frames 20 --streaming_insert_isotropic_scale --streaming_depth_loss_type huber --streaming_global_reservoir_stride 4 --streaming_eval_hold 8 --test_iterations 700 --save_iterations 700
