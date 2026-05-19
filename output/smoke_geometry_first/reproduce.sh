#!/bin/bash
# Reproduces the training run stored in this output directory.
# Generated automatically at training start.
python train.py -s /data/TUM/rgbd_dataset_freiburg1_desk --tum_sequence freiburg1 --streaming_replay --streaming_insert_isotropic_scale --streaming_steps_per_frame 5 --streaming_max_frames 20 --streaming_initial_frames 3 --streaming_eval_hold 8 --streaming_depth_loss_weight 0.05 --streaming_free_space_loss_weight 0.01 --streaming_freeze_old_geometry --streaming_young_age_frames 3 --streaming_freeze_new_frame_steps 10 --streaming_new_frame_warmup_steps 5 --streaming_anchor_loss_weight 0.01 --streaming_anchor_decay_steps 200 --streaming_global_reservoir_stride 3 --cap_max 50000 --iterations 300 --quiet -m /workspace/3dgs-mcmc/output/smoke_geometry_first
