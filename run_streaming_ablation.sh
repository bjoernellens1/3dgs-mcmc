#!/bin/bash
# Streaming degradation ablation — fr1_desk, gsplat_energy_mcmc
#
# Baseline (already done): output/bootstrap_vs_iter4000
#   iterations=4000, spf=50, noise_lr=5e5, global_replay_ratio=0.1
#   Curve: iter1000=15.25dB  iter2000=17.58dB  iter3000=15.05dB  iter4000=13.64dB
#
# Note: opacity_reset_interval is NOT used in train_streaming.py (only in taming
# branch of train.py), so the original hypothesis A is reformulated to test noise
# injection directly via --mcmc_noise_stop_iter.
#
# Run inside the container:
#   docker compose run --rm \
#     -v /mnt/cps_persistent1_shared/peyman/TUM:/data/TUM \
#     train bash run_streaming_ablation.sh
#
set -euo pipefail

SCENE_PATH="/data/TUM/rgbd_dataset_freiburg1_desk"
OUTDIR="output"

# Shared args for A/B/C (identical to baseline except the ablated knob)
COMMON_ABC=(
  -s "$SCENE_PATH"
  --tum_sequence freiburg1
  --streaming_replay
  --cap_max 200000
  --iterations 4000
  --streaming_steps_per_frame 50
  --streaming_initial_frames 5
  --streaming_eval_hold 8
  --test_iterations 1000 2000 3000 4000
  --save_iterations 4000
  --save_interval 0
  --quiet
)

# ============================================================
# Experiment A — stop noise at peak (iter 2000)
#   Hypothesis: MCMC noise injection on converged Gaussians
#   drives the iter-2000→4000 regression.
# ============================================================
echo ""
echo "======================================================"
echo "Experiment A: mcmc_noise_stop_iter=2000"
echo "======================================================"
python train.py "${COMMON_ABC[@]}" \
  --mcmc_noise_stop_iter 2000 \
  -m "$OUTDIR/ablation_A_noise_stop_2000" \
  2>&1 | tee "$OUTDIR/ablation_A_noise_stop_2000.log"

# ============================================================
# Experiment B — lower noise_lr (10× reduction: 5e5 → 5e4)
#   Hypothesis: default noise magnitude is too high for
#   streaming; gentler noise should preserve convergence.
# ============================================================
echo ""
echo "======================================================"
echo "Experiment B: noise_lr=50000 (10x reduction)"
echo "======================================================"
python train.py "${COMMON_ABC[@]}" \
  --noise_lr 50000 \
  -m "$OUTDIR/ablation_B_low_noise_lr" \
  2>&1 | tee "$OUTDIR/ablation_B_low_noise_lr.log"

# ============================================================
# Experiment C — higher global replay ratio (0.1 → 0.3)
#   Hypothesis: keyframe-window bias (90% of grads target
#   most recent 8 frames) starves older Gaussians; more replay
#   keeps early regions well-trained.
# ============================================================
echo ""
echo "======================================================"
echo "Experiment C: streaming_global_replay_ratio=0.3"
echo "======================================================"
python train.py "${COMMON_ABC[@]}" \
  --streaming_global_replay_ratio 0.3 \
  -m "$OUTDIR/ablation_C_replay_03" \
  2>&1 | tee "$OUTDIR/ablation_C_replay_03.log"

# ============================================================
# Experiment D1 — steps_per_frame=50, iters float from frames
#   Methodologically correct: iterations = N_frames × spf + tail
#   100 frames × 50 spf + 2000 tail = 7000 iters
#   (Baseline used --iterations 4000 which only let through
#    ~80 frames; here we budget iterations correctly.)
# ============================================================
echo ""
echo "======================================================"
echo "Experiment D1: spf=50, 100 frames, 7000 iters"
echo "======================================================"
python train.py \
  -s "$SCENE_PATH" \
  --tum_sequence freiburg1 \
  --streaming_replay \
  --cap_max 200000 \
  --iterations 7000 \
  --streaming_steps_per_frame 50 \
  --streaming_max_frames 100 \
  --streaming_initial_frames 5 \
  --streaming_eval_hold 8 \
  --test_iterations 1000 2000 3000 4000 5000 6000 7000 \
  --save_iterations 7000 \
  --save_interval 0 \
  --quiet \
  -m "$OUTDIR/ablation_D1_spf50_100frames" \
  2>&1 | tee "$OUTDIR/ablation_D1_spf50_100frames.log"

# ============================================================
# Experiment D2 — steps_per_frame=150, iters float from frames
#   100 frames × 150 spf + 2000 tail = 17000 iters
#   Same frame count as D1; more optimisation time per frame.
#   Prior result (memory notes): 50→150 spf gave +4.85 dB.
# ============================================================
echo ""
echo "======================================================"
echo "Experiment D2: spf=150, 100 frames, 17000 iters"
echo "======================================================"
python train.py \
  -s "$SCENE_PATH" \
  --tum_sequence freiburg1 \
  --streaming_replay \
  --cap_max 200000 \
  --iterations 17000 \
  --streaming_steps_per_frame 150 \
  --streaming_max_frames 100 \
  --streaming_initial_frames 5 \
  --streaming_eval_hold 8 \
  --test_iterations 1000 2000 5000 10000 15000 17000 \
  --save_iterations 17000 \
  --save_interval 0 \
  --quiet \
  -m "$OUTDIR/ablation_D2_spf150_100frames" \
  2>&1 | tee "$OUTDIR/ablation_D2_spf150_100frames.log"

# ============================================================
# Summary
# ============================================================
echo ""
echo "======================================================"
echo "ALL DONE — 5 ablation runs completed"
echo ""
echo "Results:"
for dir in \
  "$OUTDIR/ablation_A_noise_stop_2000" \
  "$OUTDIR/ablation_B_low_noise_lr" \
  "$OUTDIR/ablation_C_replay_03" \
  "$OUTDIR/ablation_D1_spf50_100frames" \
  "$OUTDIR/ablation_D2_spf150_100frames"; do
  report="$dir/comparison/iter_*/report.json"
  for f in $report; do
    [ -f "$f" ] && echo "  $dir:" && python3 -c "import json,sys; d=json.load(open('$f')); print(f'    mean_test_psnr={d[\"mean_test_psnr\"]:.2f} dB  min={d[\"min_test_psnr\"]:.2f}  max={d[\"max_test_psnr\"]:.2f}')" || true
  done
done
echo ""
echo "Baseline (bootstrap_vs_iter4000): 13.64 dB mean test PSNR"
echo "======================================================"
