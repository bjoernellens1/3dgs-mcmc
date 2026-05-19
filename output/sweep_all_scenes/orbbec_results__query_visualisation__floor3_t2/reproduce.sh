#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /datasets/christian/embedding_map/results/query_visualisation/floor3_t2 -m output/sweep_all_scenes/orbbec_results__query_visualisation__floor3_t2 --eval --streaming_replay --iterations 35000 --cap_max 150000 --streaming_camera_profile orbbec_femto_bolt --streaming_steps_per_frame 50 --streaming_keyframe_window 60 --streaming_replay_buffer 100 --streaming_global_replay_ratio 0.4 --streaming_insert_isotropic_scale --streaming_depth_loss_type huber --streaming_global_reservoir_stride 4 --streaming_eval_hold 8 --mcmc_stop_growth_iter 35000 --densify_until_iter 35000 --streaming_global_maintenance_interval 500 --sh_degree_schedule 500 15000 25000 --test_iterations 10000 20000 35000 --save_iterations 20000 35000
