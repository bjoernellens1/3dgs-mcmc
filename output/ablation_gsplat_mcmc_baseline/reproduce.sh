#!/bin/bash
# Reproduces the training run stored in this output directory.
# Generated automatically at training start.
python train.py -s /data/TUM/rgbd_dataset_freiburg1_desk --tum_sequence freiburg1 --streaming_replay --streaming_insert_isotropic_scale --streaming_steps_per_frame 150 --streaming_global_replay_ratio 0.3 --streaming_keyframe_window 88 --streaming_free_space_loss_weight 0.01 --streaming_insert_scale_mult 0.5 --streaming_insert_scale_max 0.05 --streaming_insert_normal_scale_ratio 0.15 --streaming_provisional_max_age 20 --streaming_eval_hold 8 --streaming_depth_loss_weight 0.05 --cap_max 200000 --iterations 17000 --streaming_max_frames 100 --densification_strategy gsplat_mcmc -m /workspace/3dgs-mcmc/output/ablation_gsplat_mcmc_baseline
