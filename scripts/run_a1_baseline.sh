#!/usr/bin/env bash
# Phase A1: Multi-scene baseline (spf50, 8k, 120 frames)
# Usage: bash scripts/run_a1_baseline.sh <scene>
set -euo pipefail

SCENE="$1"
TUM_ROOT="/mnt/cps_persistent1_shared/datasets/public/TUM/tum_rgbd"

case "$SCENE" in
  freiburg1_desk)
    FAMILY=freiburg1
    ;;
  freiburg1_xyz)
    FAMILY=freiburg1
    ;;
  freiburg2_desk)
    FAMILY=freiburg2
    ;;
  freiburg3_long_office_household)
    FAMILY=freiburg3
    ;;
  *)
    echo "Unknown scene: $SCENE"
    exit 1
    ;;
esac

RAW_DIR="$TUM_ROOT/$SCENE/rgbd_dataset_$SCENE"
GT_PATH="$RAW_DIR/groundtruth.txt"
if [ ! -f "$GT_PATH" ]; then
    # Fallback to external groundtruth
    GT_PATH=$(ls "$TUM_ROOT/$SCENE/groundtruth/"*-groundtruth.txt 2>/dev/null | head -1)
fi
OUTPUT="output/tum_sweeps/a1/${SCENE}_baseline_spf50"

echo "=============================================="
echo "Phase A1: $SCENE"
echo "  Family:    $FAMILY"
echo "  Raw:       $RAW_DIR"
echo "  GT:        $GT_PATH"
echo "  Output:    $OUTPUT"
echo "=============================================="

docker compose run --rm -T train \
  python train.py \
    -s "$RAW_DIR" \
    -m "$OUTPUT" \
    --streaming_replay \
    --iterations 8000 \
    --streaming_initial_frames 5 \
    --streaming_eval_hold 8 \
    --streaming_max_frames 120 \
    --streaming_steps_per_frame 50 \
    --streaming_keyframe_window 60 \
    --cap_max 200000 \
    --streaming_free_space_loss_weight 0.01 \
    --test_iterations 8000 \
    --save_iterations 8000 \
    --tum_gt_path "$GT_PATH" \
    --tum_association_max_dt 0.03 \
    --tum_sequence "$FAMILY" \
    --quiet \
    --disable_progress_bar

echo "DONE: $SCENE"
