#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /data/scannet/scans/scene0011_00 -m output/scannet_scene0011_r4_100k_refine_sh3 -r 4 --eval --iterations 100000 --cap_max 200000 --scannet_frame_stride 8 --scannet_eval_hold 20 --streaming_replay --streaming_steps_per_frame 50 --streaming_keyframe_window 400 --streaming_replay_buffer 200 --streaming_global_replay_ratio 0.4 --streaming_max_new_gaussians_per_frame 2000 --streaming_insert_isotropic_scale --streaming_depth_loss_weight 0.05 --streaming_depth_loss_type huber --streaming_free_space_loss_weight 0.01 --streaming_global_reservoir_stride 4 --mcmc_stop_growth_iter 40000 --densify_until_iter 90000 --streaming_global_maintenance_interval 1000 --streaming_max_sh_degree 3 --sh_degree_schedule 1000 50000 70000 --test_iterations 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 --save_iterations 50000 100000
