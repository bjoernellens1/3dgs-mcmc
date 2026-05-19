#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /datasets/public/Replica-Dataset/apartment_0 -m output/sweep_all_scenes/replica_apartment_0 --eval --streaming_replay --iterations 6000 --cap_max 200000 --streaming_steps_per_frame 50 --streaming_keyframe_window 60 --streaming_replay_buffer 100 --streaming_global_replay_ratio 0.4 --streaming_eval_hold 8 --replica_init mesh --replica_num_views 120 --replica_width 640 --replica_height 480 --replica_max_init_points 100000 --replica_render_points 250000 --test_iterations 6000 --save_iterations 6000
