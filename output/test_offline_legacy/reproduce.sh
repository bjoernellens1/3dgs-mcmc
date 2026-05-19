#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /data/mipnerf360_v2_dataset/bicycle -m output/test_offline_legacy --cap_max 4000 --iterations 100 --model_layout legacy --densification_strategy mcmc
