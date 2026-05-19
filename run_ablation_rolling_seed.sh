#!/bin/bash
# Ablation runner for rolling_seed mode
set -e

BASE_FLAGS="--streaming_replay \
  --streaming_steps_per_frame 150 \
  --streaming_insert_isotropic_scale \
  --streaming_initial_frames 5 \
  --streaming_max_frames 100 \
  --cap_max 200000 \
  --iterations 17000 \
  --tum_sequence freiburg1 \
  --save_iterations 17000 \
  --test_iterations 17000"

SRC="-s /data/tum_raw/rgbd_dataset_freiburg1_desk"

echo "=== Rolling Seed Ablation ==="
python train.py $SRC -m output/ablation_rolling_seed \
  $BASE_FLAGS \
  --streaming_training_mode rolling_seed \
  --streaming_submap_frames 20 \
  --streaming_depth_stride 4 \
  --streaming_global_refine_iters 0 \
  2>&1 | tee output/ablation_rolling_seed.log
echo "Rolling seed done."
