#!/bin/bash
# Growth-fix validation on fr1_desk (full 615 frames).
# Tests whether the occupancy hash rebuild + cover_voxel_multiplier=1.0
# break the 24k Gaussian ceiling observed in all previous runs.
# Also tests lifecycle without opacity anchor (xyz+scale only).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTBASE=/workspace/3dgs-mcmc/output

BASE_ARGS="
  -s /data/TUM/rgbd_dataset_freiburg1_desk
  --tum_sequence freiburg1
  --streaming_replay
  --densification_strategy gsplat_energy_mcmc
  --streaming_steps_per_frame 50
  --iterations 31000
  --mcmc_stop_growth_iter 12000
  --streaming_insert_isotropic_scale
  --streaming_global_replay_ratio 0.3
  --streaming_keyframe_window 620
  --streaming_free_space_loss_weight 0.01
  --streaming_insert_scale_mult 0.5
  --streaming_insert_scale_max 0.05
  --streaming_insert_normal_scale_ratio 0.15
  --streaming_provisional_max_age 20
  --streaming_eval_hold 8
  --streaming_depth_loss_weight 0.05
  --cap_max 200000
  --streaming_insertion_debug
"

run_experiment() {
  local name=$1
  shift
  local extra_args="$*"
  echo ""
  echo "============================================================"
  echo "Starting: $name"
  echo "Extra args: $extra_args"
  echo "============================================================"
  docker compose run --rm \
    -v /home/bjoern/git/3dgs-mcmc:/workspace/3dgs-mcmc \
    -v /mnt/cps_persistent1_shared/peyman/TUM:/data/TUM \
    train python train.py \
      $BASE_ARGS \
      $extra_args \
      -m "${OUTBASE}/${name}"
  echo "Done: $name"
}

# -----------------------------------------------------------------------
# Run 1: Baseline with fix (cover_mult=1.0 + rebuild=200 are now defaults)
# Expect: N grows well past 24k
# -----------------------------------------------------------------------
run_experiment "gfx_baseline" \
  ""

# -----------------------------------------------------------------------
# Run 2: Lifecycle + xyz+scale anchor only (no opacity anchor)
# Rationale: opacity anchor blocked MCMC death→relocation; xyz+scale
# prevent smearing and blowup without locking the birth/death cycle
# Expect: N grows past 24k + anti-fade benefit without MCMC blockage
# -----------------------------------------------------------------------
run_experiment "gfx_lifecycle_xyz_scale" \
  "--streaming_lifecycle_enabled \
   --streaming_mature_age_frames 15 \
   --streaming_mature_min_utility 0.1 \
   --streaming_utility_ema_beta 0.95 \
   --streaming_mature_anchor_xyz_weight 0.05 \
   --streaming_mature_anchor_scale_weight 0.05 \
   --streaming_mature_anchor_opacity_weight 0.0"

# -----------------------------------------------------------------------
# Run 3: Lifecycle + xyz anchor only (minimal anti-fade)
# Rationale: check if scale anchor also causes issues, or only opacity did
# -----------------------------------------------------------------------
run_experiment "gfx_lifecycle_xyz_only" \
  "--streaming_lifecycle_enabled \
   --streaming_mature_age_frames 15 \
   --streaming_mature_min_utility 0.1 \
   --streaming_utility_ema_beta 0.95 \
   --streaming_mature_anchor_xyz_weight 0.05 \
   --streaming_mature_anchor_scale_weight 0.0 \
   --streaming_mature_anchor_opacity_weight 0.0"

echo ""
echo "All 3 growth-fix runs complete."
echo "Results in: ${OUTBASE}/gfx_*"
