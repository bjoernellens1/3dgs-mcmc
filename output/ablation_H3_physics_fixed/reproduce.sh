#!/bin/bash
# Reproduces the training run stored in this output directory.
# Generated automatically at training start.
python train.py -s /data/TUM/rgbd_dataset_freiburg1_desk --tum_sequence freiburg1 --streaming_replay --streaming_insert_isotropic_scale --streaming_steps_per_frame 150 --streaming_global_replay_ratio 0.3 --streaming_keyframe_window 20 --streaming_free_space_loss_weight 0.01 --cap_max 200000 --iterations 17000 --streaming_max_frames 100 --streaming_initial_frames 5 --streaming_eval_hold 8 --test_iterations 1000 2000 5000 10000 15000 17000 --save_iterations 17000 --streaming_save_frame_interval 50 -m output/ablation_H3_physics_fixed
