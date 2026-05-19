#!/bin/bash
# Reproduces the training run stored in this output directory.
# Generated automatically at training start.
python train.py -s /data/tum/rgbd_dataset_freiburg1_desk -m output/smoke_h7h8h9h10 --streaming_replay --streaming_freeze_old_geometry --streaming_young_age_frames 3 --streaming_freeze_new_frame_steps 20 --streaming_new_frame_warmup_steps 5 --streaming_anchor_loss_weight 0.01 --streaming_global_reservoir_stride 2 --streaming_insert_isotropic_scale --streaming_free_space_loss_weight 0.01 --streaming_initial_frames 3 --streaming_max_frames 20 --streaming_steps_per_frame 10 --streaming_keyframe_window 20 --streaming_global_replay_ratio 0.3 --cap_max 20000 --iterations 300 --tum_sequence freiburg1 --save_iterations 300 --test_iterations 300
