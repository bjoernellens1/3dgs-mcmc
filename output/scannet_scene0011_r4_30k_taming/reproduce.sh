#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /data/scannet/scans/scene0011_00 -m output/scannet_scene0011_r4_30k_taming -r 4 --eval --iterations 30000 --cap_max 200000 --scannet_frame_stride 8 --scannet_eval_hold 20 --streaming_replay --streaming_steps_per_frame 50 --streaming_keyframe_window 400 --streaming_replay_buffer 200 --streaming_global_replay_ratio 0.4 --streaming_max_new_gaussians_per_frame 2000 --streaming_insert_isotropic_scale --streaming_depth_loss_weight 0.05 --streaming_depth_loss_type huber --streaming_free_space_loss_weight 0.01 --streaming_global_reservoir_stride 4 --densification_strategy taming --taming_score_interval 200 --taming_cams 3 --densify_from_iter 500 --densify_until_iter 30000 --streaming_global_maintenance_interval 1000 --streaming_max_sh_degree 3 --sh_degree_schedule 1000 15000 20000 --test_iterations 5000 10000 20000 30000 --save_iterations 15000 30000
