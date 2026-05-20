#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /mnt/cps_persistent1_shared/datasets/public/TUM/tum_rgbd/freiburg1_desk/rgbd_dataset_freiburg1_desk -m output/tum_sweeps/a2_fix/freiburg1_desk_default_fix --streaming_replay --iterations 89400 --streaming_initial_frames 5 --streaming_eval_hold 8 --streaming_max_frames 0 --streaming_steps_per_frame 150 --cap_max 200000 --streaming_free_space_loss_weight 0.01 --test_iterations 89400 --save_iterations 89400 --tum_gt_path /mnt/cps_persistent1_shared/datasets/public/TUM/tum_rgbd/freiburg1_desk/rgbd_dataset_freiburg1_desk/groundtruth.txt --tum_association_max_dt 0.03 --tum_sequence freiburg1 --streaming_report_trajectory_max_frames 0
