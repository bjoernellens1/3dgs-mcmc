#!/bin/bash
# Reproduces the training run stored in this output directory.
# Generated automatically at training start.
python train.py -s /data/scans/scene0011_00 -m output/scannet_scene0011_00_lpips_heavy --cap_max 6000 --lambda_lpips 1.0 --lpips_interval 10 --scannet_frame_stride 1 --scannet_init rgbd --scannet_max_init_points 6000 --model_layout gsplat --iterations 30000
