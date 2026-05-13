#!/bin/bash
# Sequential ablation runner for H8–H11 (geometry-first streaming)
# Base config: H4d + H7 flags (freeze_old_geometry, window=88, spf=150, replay=0.3)
# Run with: docker compose run --rm train bash run_ablation_H8_H11.sh

set -e

BASE_FLAGS="--streaming_replay \
  --streaming_steps_per_frame 150 \
  --streaming_global_replay_ratio 0.3 \
  --streaming_keyframe_window 88 \
  --streaming_insert_isotropic_scale \
  --streaming_free_space_loss_weight 0.01 \
  --streaming_freeze_old_geometry \
  --streaming_young_age_frames 5 \
  --streaming_freeze_new_frame_steps 50 \
  --streaming_initial_frames 5 \
  --streaming_max_frames 100 \
  --cap_max 200000 \
  --iterations 17000 \
  --tum_sequence freiburg1 \
  --save_iterations 17000 \
  --test_iterations 17000"

SRC="-s /data/tum_raw/rgbd_dataset_freiburg1_desk"

# ---- H8: H7 + new-frame warmup (10 steps) --------------------------------
echo "=== H8: warmup_steps=10 ==="
python train.py $SRC -m output/ablation_H8 \
  $BASE_FLAGS \
  --streaming_new_frame_warmup_steps 10 \
  2>&1 | tee output/ablation_H8.log
echo "H8 done."

# ---- H9a: H7+H8 + anchor loss weight=0.01 --------------------------------
echo "=== H9a: anchor_weight=0.01 ==="
python train.py $SRC -m output/ablation_H9a \
  $BASE_FLAGS \
  --streaming_new_frame_warmup_steps 10 \
  --streaming_anchor_loss_weight 0.01 \
  --streaming_anchor_decay_steps 500 \
  2>&1 | tee output/ablation_H9a.log
echo "H9a done."

# ---- H9b: H7+H8 + anchor loss weight=0.05 --------------------------------
echo "=== H9b: anchor_weight=0.05 ==="
python train.py $SRC -m output/ablation_H9b \
  $BASE_FLAGS \
  --streaming_new_frame_warmup_steps 10 \
  --streaming_anchor_loss_weight 0.05 \
  --streaming_anchor_decay_steps 500 \
  2>&1 | tee output/ablation_H9b.log
echo "H9b done."

# ---- H10: global reservoir (stride=4, window=20) -------------------------
# Tests whether scene coverage matters more than raw window size.
echo "=== H10: reservoir stride=4, window=20 ==="
python train.py $SRC -m output/ablation_H10 \
  --streaming_replay \
  --streaming_steps_per_frame 150 \
  --streaming_global_replay_ratio 0.10 \
  --streaming_keyframe_window 20 \
  --streaming_global_reservoir_stride 4 \
  --streaming_insert_isotropic_scale \
  --streaming_free_space_loss_weight 0.01 \
  --streaming_freeze_old_geometry \
  --streaming_young_age_frames 5 \
  --streaming_freeze_new_frame_steps 50 \
  --streaming_new_frame_warmup_steps 10 \
  --streaming_initial_frames 5 \
  --streaming_max_frames 100 \
  --cap_max 200000 \
  --iterations 17000 \
  --tum_sequence freiburg1 \
  --save_iterations 17000 \
  --test_iterations 17000 \
  2>&1 | tee output/ablation_H10.log
echo "H10 done."

# ---- H11: submap-stitch mode ---------------------------------------------
echo "=== H11: submap_stitch (20 frames/submap, 3000 iters/submap, 5000 global refine) ==="
python train.py $SRC -m output/ablation_H11 \
  --streaming_replay \
  --streaming_steps_per_frame 150 \
  --streaming_insert_isotropic_scale \
  --streaming_initial_frames 5 \
  --streaming_max_frames 100 \
  --cap_max 200000 \
  --iterations 17000 \
  --tum_sequence freiburg1 \
  --save_iterations 17000 \
  --test_iterations 17000 \
  --streaming_training_mode submap_stitch \
  --streaming_submap_frames 20 \
  --streaming_submap_iters 3000 \
  --streaming_global_refine_iters 5000 \
  2>&1 | tee output/ablation_H11.log
echo "H11 done."

echo ""
echo "=== All H8–H11 ablations complete. Collecting PSNR results... ==="
for d in ablation_H7 ablation_H8 ablation_H9a ablation_H9b ablation_H10 ablation_H11; do
  log="output/$d/train_stdout.log"
  if [ -f "$log" ]; then
    psnr=$(grep -o 'PSNR mean=[0-9.]*dB' "$log" | tail -1)
    echo "  $d: $psnr"
  else
    echo "  $d: log not found"
  fi
done
