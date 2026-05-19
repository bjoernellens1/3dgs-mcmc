#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /data/TUM/rgbd_dataset_freiburg1_desk --tum_sequence freiburg1 --streaming_replay --densification_strategy gsplat_energy_mcmc --streaming_steps_per_frame 50 --iterations 31000 --mcmc_stop_growth_iter 12000 --streaming_insert_isotropic_scale --streaming_global_replay_ratio 0.3 --streaming_keyframe_window 620 --streaming_free_space_loss_weight 0.01 --streaming_insert_scale_mult 0.5 --streaming_insert_scale_max 0.05 --streaming_insert_normal_scale_ratio 0.15 --streaming_provisional_max_age 20 --streaming_eval_hold 8 --streaming_depth_loss_weight 0.05 --cap_max 200000 --streaming_insertion_debug -m /workspace/3dgs-mcmc/output/lc_baseline
