#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /datasets/public/Replica-Dataset/apartment_0 -m output/replica_view_sampler_smoke --eval --streaming_replay --iterations 300 --cap_max 200000 --streaming_steps_per_frame 20 --streaming_keyframe_window 30 --streaming_replay_buffer 50 --streaming_global_replay_ratio 0.4 --streaming_eval_hold 8 --replica_init mesh --replica_num_views 40 --replica_width 320 --replica_height 240 --replica_max_init_points 50000 --replica_render_points 250000 --test_iterations 300 --save_iterations 300 --progress_video_interval 100 --quiet
