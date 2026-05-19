#!/bin/bash
# Reproduces the training run stored in this output directory.
# Generated automatically at training start.
python train.py -s /data/tum/rgbd_dataset_freiburg1_desk -m output/smoke_h11_submap --streaming_replay --streaming_training_mode submap_stitch --streaming_submap_frames 5 --streaming_submap_iters 50 --streaming_global_refine_iters 100 --streaming_initial_frames 3 --streaming_max_frames 15 --streaming_steps_per_frame 10 --cap_max 20000 --iterations 300 --tum_sequence freiburg1 --save_iterations 300 --test_iterations 300
