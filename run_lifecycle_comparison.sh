#!/bin/bash
# Lifecycle-anti-fade comparison on fr1_desk (full 615 frames).
# Six sequential runs: baseline through full lifecycle+anti-fade config.
# Uses gsplat_energy_mcmc throughout (best SLAM trade-off from full-615 study).
# Run from the project root on the host; source is live-mounted into container.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTBASE=/workspace/3dgs-mcmc/output

# Base config: H4d-physics-fixed (window=88 → all 88 train frames after 8× holdout)
# spf=50 → 31000 iters (same budget as previous full-615 study)
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
# Run 1: Baseline (H4d config, no lifecycle, no anti-fade)
# -----------------------------------------------------------------------
run_experiment "lc_baseline" \
  ""

# -----------------------------------------------------------------------
# Run 2: Lifecycle only (PROVISIONAL→YOUNG→MATURE at age=15, no anchor)
# -----------------------------------------------------------------------
run_experiment "lc_lifecycle_only" \
  "--streaming_lifecycle_enabled \
   --streaming_mature_age_frames 15 \
   --streaming_mature_min_utility 0.1 \
   --streaming_utility_ema_beta 0.95"

# -----------------------------------------------------------------------
# Run 3: Lifecycle + anti-fade anchor losses on MATURE
# -----------------------------------------------------------------------
run_experiment "lc_lifecycle_antifade" \
  "--streaming_lifecycle_enabled \
   --streaming_mature_age_frames 15 \
   --streaming_mature_min_utility 0.1 \
   --streaming_utility_ema_beta 0.95 \
   --streaming_mature_anchor_xyz_weight 0.05 \
   --streaming_mature_anchor_scale_weight 0.05 \
   --streaming_mature_anchor_opacity_weight 0.05"

# -----------------------------------------------------------------------
# Run 4: Lifecycle + anti-fade + stratified sampling
# -----------------------------------------------------------------------
run_experiment "lc_stratified" \
  "--streaming_lifecycle_enabled \
   --streaming_mature_age_frames 15 \
   --streaming_mature_min_utility 0.1 \
   --streaming_utility_ema_beta 0.95 \
   --streaming_mature_anchor_xyz_weight 0.05 \
   --streaming_mature_anchor_scale_weight 0.05 \
   --streaming_mature_anchor_opacity_weight 0.05 \
   --streaming_sampling_mode stratified \
   --streaming_global_reservoir_stride 5"

# -----------------------------------------------------------------------
# Run 5: Full (lifecycle + anti-fade + stratified + KNN dedup)
# -----------------------------------------------------------------------
run_experiment "lc_full_iter_based" \
  "--streaming_lifecycle_enabled \
   --streaming_mature_age_frames 15 \
   --streaming_mature_min_utility 0.1 \
   --streaming_utility_ema_beta 0.95 \
   --streaming_mature_anchor_xyz_weight 0.05 \
   --streaming_mature_anchor_scale_weight 0.05 \
   --streaming_mature_anchor_opacity_weight 0.05 \
   --streaming_sampling_mode stratified \
   --streaming_global_reservoir_stride 5 \
   --streaming_insert_knn_dedup \
   --streaming_insert_knn_radius_factor 0.005"

# -----------------------------------------------------------------------
# Run 6: Full + dataset_fps ingestion (simulated real-time pacing at 30fps)
# -----------------------------------------------------------------------
run_experiment "lc_full_dataset_fps" \
  "--streaming_lifecycle_enabled \
   --streaming_mature_age_frames 15 \
   --streaming_mature_min_utility 0.1 \
   --streaming_utility_ema_beta 0.95 \
   --streaming_mature_anchor_xyz_weight 0.05 \
   --streaming_mature_anchor_scale_weight 0.05 \
   --streaming_mature_anchor_opacity_weight 0.05 \
   --streaming_sampling_mode stratified \
   --streaming_global_reservoir_stride 5 \
   --streaming_insert_knn_dedup \
   --streaming_insert_knn_radius_factor 0.005 \
   --streaming_ingestion_mode dataset_fps \
   --streaming_input_fps_cap 30.0"

echo ""
echo "All 6 lifecycle-anti-fade runs complete."
echo "Results in: ${OUTBASE}/lc_*"
