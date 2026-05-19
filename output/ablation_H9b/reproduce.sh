#!/bin/bash
# Reproduces the training run stored in this output directory.
# Generated automatically at training start.
python train.py -s /data/tum_raw/rgbd_dataset_freiburg1_desk -m output/ablation_H9b --streaming_replay --streaming_steps_per_frame 150 --streaming_global_replay_ratio 0.3 --streaming_keyframe_window 88 --streaming_insert_isotropic_scale --streaming_free_space_loss_weight 0.01 --streaming_freeze_old_geometry --streaming_young_age_frames 5 --streaming_freeze_new_frame_steps 50 --streaming_initial_frames 5 --streaming_max_frames 100 --cap_max 200000 --iterations 17000 --tum_sequence freiburg1 --save_iterations 17000 --test_iterations 17000 --streaming_new_frame_warmup_steps 10 --streaming_anchor_loss_weight 0.05 --streaming_anchor_decay_steps 500
