#!/usr/bin/env bash
# Generic runner for TUM ablation sweep jobs.
#
# Usage: bash scripts/run_sweep_job.sh <scene> <spf> <iterations> <out_dir> [extra_args...]
#
# Scenes: freiburg1_desk, freiburg1_xyz, freiburg2_desk, freiburg3_long_office_household
#
# Example:
#   bash scripts/run_sweep_job.sh freiburg1_desk 50 29800 output/tum_sweeps/a1/freiburg1_desk_full_spf50
#
set -euo pipefail

SCENE="$1"
SPF="$2"
ITERS="$3"
OUTPUT="$4"
shift 4

TUM_ROOT="/mnt/cps_persistent1_shared/datasets/public/TUM/tum_rgbd"

case "$SCENE" in
  freiburg1_desk|freiburg1_xyz)    FAMILY=freiburg1 ;;
  freiburg2_desk)                   FAMILY=freiburg2 ;;
  freiburg3_long_office_household)  FAMILY=freiburg3 ;;
  *)
    echo "Unknown scene: $SCENE"
    exit 1
    ;;
esac

RAW_DIR="$TUM_ROOT/$SCENE/rgbd_dataset_$SCENE"
GT_PATH="$RAW_DIR/groundtruth.txt"
if [ ! -f "$GT_PATH" ]; then
    GT_PATH=$(ls "$TUM_ROOT/$SCENE/groundtruth/"*-groundtruth.txt 2>/dev/null | head -1)
fi

# Compute test_iterations/save_iterations to include final checkpoint
TEST_SAVE_ITERS="$ITERS"

echo "=============================================="
echo "Running: $SCENE  spf=$SPF  iters=$ITERS"
echo "  Output:    $OUTPUT"
echo "  Raw:       $RAW_DIR"
echo "  GT:        $GT_PATH"
echo "  Family:    $FAMILY"
echo "  Extra:     $*"
echo "=============================================="

docker compose run --rm -T train \
  python train.py \
    -s "$RAW_DIR" \
    -m "$OUTPUT" \
    --streaming_replay \
    --iterations "$ITERS" \
    --streaming_max_frames 0 \
    --streaming_initial_frames 5 \
    --streaming_eval_hold 8 \
    --streaming_steps_per_frame "$SPF" \
    --streaming_keyframe_window 60 \
    --cap_max 200000 \
    --streaming_free_space_loss_weight 0.01 \
    --test_iterations "$TEST_SAVE_ITERS" \
    --save_iterations "$TEST_SAVE_ITERS" \
    --streaming_report_trajectory_max_frames 0 \
    --tum_gt_path "$GT_PATH" \
    --tum_association_max_dt 0.03 \
    --tum_sequence "$FAMILY" \
    "$@"

echo "DONE: $SCENE -> $OUTPUT"
