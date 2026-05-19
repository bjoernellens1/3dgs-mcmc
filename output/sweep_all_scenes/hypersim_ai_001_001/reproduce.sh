#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /datasets/public/ml-hypersim/evermotion_dataset/scenes/ai_001_001 -m output/sweep_all_scenes/hypersim_ai_001_001 --eval --streaming_replay --iterations 6000 --cap_max 200000 --streaming_steps_per_frame 50 --streaming_keyframe_window 60 --streaming_replay_buffer 100 --streaming_global_replay_ratio 0.4 --streaming_eval_hold 8 --hypersim_cam_id cam_00 --hypersim_frame_stride 5 --streaming_max_frames 120 --test_iterations 6000 --save_iterations 6000
