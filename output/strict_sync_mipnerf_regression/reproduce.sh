#!/bin/bash
# Reproduces the training run stored in this output directory.
# Auto-generated at training start.

python train.py -s /data/mipnerf360_v2_dataset/bicycle -m output/strict_sync_mipnerf_regression --cap_max 4000 --iterations 100
