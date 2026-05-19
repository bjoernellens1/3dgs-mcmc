#!/bin/bash
# Full fr1_desk comparison: 3 strategies × 2 step budgets = 6 sequential runs.
# All 615 frames, window=620 (all frames in window, no forgetting).
# Run inside Docker/Podman via docker compose run.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTBASE=/workspace/3dgs-mcmc/output

BASE_ARGS="
  -s /data/TUM/rgbd_dataset_freiburg1_desk
  --tum_sequence freiburg1
  --streaming_replay
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
"

run_experiment() {
  local strategy=$1
  local spf=$2
  local iters=$3
  local stop_growth=$4
  local name="full615_${strategy}_spf${spf}"

  echo ""
  echo "============================================================"
  echo "Starting: $name  (strategy=$strategy spf=$spf iters=$iters stop_growth=$stop_growth)"
  echo "============================================================"

  docker compose run --rm \
    -v /home/bjoern/git/3dgs-mcmc:/workspace/3dgs-mcmc \
    -v /mnt/cps_persistent1_shared/peyman/TUM:/data/TUM \
    train python train.py \
      $BASE_ARGS \
      --densification_strategy "$strategy" \
      --streaming_steps_per_frame "$spf" \
      --iterations "$iters" \
      --mcmc_stop_growth_iter "$stop_growth" \
      -m "${OUTBASE}/${name}"
}

# spf=50: ~31000 iters, growth stops at 12000 (39%)
run_experiment gsplat_energy_mcmc  50  31000  12000
run_experiment gsplat_mcmc         50  31000  12000
run_experiment gsplat_default      50  31000  12000

# spf=100: ~62000 iters, growth stops at 25000 (40%)
run_experiment gsplat_energy_mcmc  100 62000  25000
run_experiment gsplat_mcmc         100 62000  25000
run_experiment gsplat_default      100 62000  25000

echo ""
echo "All 6 runs complete."
