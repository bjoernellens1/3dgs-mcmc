#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /mnt/cps_persistent1_shared/datasets/public/TUM/tum_rgbd/freiburg1_desk/rgbd_dataset_freiburg1_desk -m output/test_ablation_manual --streaming_replay --streaming_initial_frames 5 --streaming_max_frames 80 --streaming_steps_per_frame 50 --streaming_keyframe_window 40 --iterations 4000 --test_iterations 4000 --tum_sequence freiburg1 --tum_gt_path /mnt/cps_persistent1_shared/datasets/public/TUM/tum_rgbd/freiburg1_desk/rgbd_dataset_freiburg1_desk/groundtruth.txt --tum_association_max_dt 0.03 --cap_max 200000 --quiet
