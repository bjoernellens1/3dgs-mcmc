#!/bin/bash
set -e

# Taming ablation experiments on bicycle scene
# All runs: 6000 iterations, taming strategy

SCENE="bicycle"
BASE_DIR="/workspace/3dgs-mcmc"
DATA_DIR="/data/mipnerf360_v2_dataset/${SCENE}"

COMMON_ARGS=(
  -s "${DATA_DIR}"
  --config "configs/${SCENE}.json"
  --iterations 6000
  --test_iterations 6000
  --save_iterations 6000
  --checkpoint_interval 3000
  --save_interval 1000
  --disable_progress_bar
  --densification_strategy taming
  --parallelism_profile safe
  --quiet
)

# ============================================================
# Run 1: Baseline — just set a budget
# ============================================================
RUN_NAME="${SCENE}_taming_baseline_budget300k_$(date +%Y%m%d_%H%M%S)"
echo ""
echo "=========================================="
echo "RUN 1/5: Baseline (budget=300k)"
echo "  → ${RUN_NAME}"
echo "=========================================="
python train.py \
  "${COMMON_ARGS[@]}" \
  -m "output/${RUN_NAME}" \
  --taming_budget 300000

# ============================================================
# Run 2: Longer pruning — prune through whole training
# ============================================================
RUN_NAME="${SCENE}_taming_prunestop6k_$(date +%Y%m%d_%H%M%S)"
echo ""
echo "=========================================="
echo "RUN 2/5: Longer pruning (prune_stop_iter=6000)"
echo "  → ${RUN_NAME}"
echo "=========================================="
python train.py \
  "${COMMON_ARGS[@]}" \
  -m "output/${RUN_NAME}" \
  --taming_budget 300000 \
  --taming_prune_stop_iter 6000

# ============================================================
# Run 3: Aggressive opacity pruning
# ============================================================
RUN_NAME="${SCENE}_taming_aggr_prune_op02_$(date +%Y%m%d_%H%M%S)"
echo ""
echo "=========================================="
echo "RUN 3/5: Aggressive prune (min_opacity=0.02, prune_stop=6000)"
echo "  → ${RUN_NAME}"
echo "=========================================="
python train.py \
  "${COMMON_ARGS[@]}" \
  -m "output/${RUN_NAME}" \
  --taming_budget 300000 \
  --taming_prune_stop_iter 6000 \
  --taming_min_opacity 0.02

# ============================================================
# Run 4: Count importance in scoring
# ============================================================
RUN_NAME="${SCENE}_taming_countimp1_$(date +%Y%m%d_%H%M%S)"
echo ""
echo "=========================================="
echo "RUN 4/5: Count importance (count_importance=1.0)"
echo "  → ${RUN_NAME}"
echo "=========================================="
python train.py \
  "${COMMON_ARGS[@]}" \
  -m "output/${RUN_NAME}" \
  --taming_budget 300000 \
  --taming_count_importance 1.0

# ============================================================
# Run 5: Full aggressive — all knobs + tight budget
# ============================================================
RUN_NAME="${SCENE}_taming_full_aggro_150k_$(date +%Y%m%d_%H%M%S)"
echo ""
echo "=========================================="
echo "RUN 5/5: Full aggressive (budget=150k, prune_stop=6000, min_op=0.02, count_imp=1.0)"
echo "  → ${RUN_NAME}"
echo "=========================================="
python train.py \
  "${COMMON_ARGS[@]}" \
  -m "output/${RUN_NAME}" \
  --taming_budget 150000 \
  --taming_prune_stop_iter 6000 \
  --taming_min_opacity 0.02 \
  --taming_count_importance 1.0

echo ""
echo "=========================================="
echo "ALL DONE — 5 ablation runs completed"
echo "=========================================="
