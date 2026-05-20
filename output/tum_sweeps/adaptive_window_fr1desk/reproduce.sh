#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /mnt/TUM/rgbd_dataset_freiburg1_desk -m output/tum_sweeps/adaptive_window_fr1desk --streaming_replay --tum_sequence freiburg1 --iterations 89400 --streaming_initial_frames 5 --streaming_eval_hold 8 --streaming_steps_per_frame 150 --cap_max 200000 --streaming_free_space_loss_weight 0.01 --test_iterations 89400 --save_iterations 89400 --tum_gt_path /mnt/TUM/rgbd_dataset_freiburg1_desk/groundtruth.txt --tum_association_max_dt 0.03 --streaming_report_trajectory_max_frames 0 --streaming_max_frames 0
