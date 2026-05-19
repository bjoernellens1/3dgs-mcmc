#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /data/hypersim_ai001001 -m output/hypersim_ai001001_r4_streaming -r 4 --eval --iterations 15000 --cap_max 150000 --streaming_replay --streaming_steps_per_frame 100 --streaming_keyframe_window 50 --streaming_replay_buffer 80 --streaming_global_replay_ratio 0.4 --streaming_max_new_gaussians_per_frame 2000 --streaming_insert_isotropic_scale --streaming_depth_loss_weight 0.05 --streaming_depth_loss_type huber --streaming_free_space_loss_weight 0.01 --streaming_global_reservoir_stride 4 --mcmc_stop_growth_iter 15000 --densify_until_iter 15000 --streaming_global_maintenance_interval 500 --sh_degree_schedule 500 8000 12000 --test_iterations 5000 10000 15000 --save_iterations 10000 15000
