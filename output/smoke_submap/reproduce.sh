#!/bin/bash
# Reproduces the training run stored in this output directory.
# Generated automatically at training start.
python train.py -s /data/TUM/rgbd_dataset_freiburg1_desk --tum_sequence freiburg1 --streaming_replay --streaming_insert_isotropic_scale --streaming_max_frames 20 --streaming_initial_frames 3 --streaming_eval_hold 8 --streaming_training_mode submap_stitch --streaming_submap_frames 5 --streaming_submap_iters 200 --streaming_global_refine_iters 200 --cap_max 50000 --iterations 1000 --quiet -m /workspace/3dgs-mcmc/output/smoke_submap
