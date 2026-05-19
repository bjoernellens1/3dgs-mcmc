#!/bin/bash
# Reproduces the training run stored in this output directory.
# Generated automatically at training start.
python train.py -s /data/tum_raw/rgbd_dataset_freiburg1_desk -m output/ablation_H11 --streaming_replay --streaming_steps_per_frame 150 --streaming_insert_isotropic_scale --streaming_initial_frames 5 --streaming_max_frames 100 --cap_max 200000 --iterations 17000 --tum_sequence freiburg1 --save_iterations 17000 --test_iterations 17000 --streaming_training_mode submap_stitch --streaming_submap_frames 20 --streaming_submap_iters 3000 --streaming_global_refine_iters 5000
