#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /data/scannet/scans/scene0011_00 -m output/taming_smoke_500b -r 4 --eval --iterations 600 --cap_max 200000 --scannet_frame_stride 8 --scannet_eval_hold 20 --streaming_replay --streaming_steps_per_frame 50 --streaming_keyframe_window 400 --streaming_replay_buffer 50 --streaming_global_replay_ratio 0.4 --streaming_max_new_gaussians_per_frame 500 --streaming_insert_isotropic_scale --streaming_depth_loss_weight 0.05 --streaming_depth_loss_type huber --densification_strategy taming --taming_score_interval 100 --taming_cams 2 --densify_from_iter 100 --densify_until_iter 600 --test_iterations 600 --save_iterations 600
