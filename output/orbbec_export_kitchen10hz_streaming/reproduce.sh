#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /data/orbbec/export/10hz/kitchen_icl -m output/orbbec_export_kitchen10hz_streaming --eval --iterations 35000 --cap_max 150000 --streaming_replay --streaming_steps_per_frame 50 --streaming_keyframe_window 60 --streaming_replay_buffer 100 --streaming_global_replay_ratio 0.4 --streaming_max_new_gaussians_per_frame 2000 --streaming_insert_isotropic_scale --streaming_depth_loss_weight 0.05 --streaming_depth_loss_type huber --streaming_free_space_loss_weight 0.01 --streaming_global_reservoir_stride 4 --streaming_eval_hold 8 --mcmc_stop_growth_iter 35000 --densify_until_iter 35000 --streaming_global_maintenance_interval 500 --sh_degree_schedule 500 15000 25000 --test_iterations 10000 20000 35000 --save_iterations 20000 35000
