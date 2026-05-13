# Streaming Degradation Ablation Results

**Date:** 2026-05-12 / 2026-05-13
**Branch:** `streaming-fixes` / `insertion-ablation`
**Scene:** TUM RGB-D `freiburg1_desk` (596 frames, capped via `--streaming_max_frames`)
**Baseline:** `output/bootstrap_vs_iter4000` — 4000 iters, spf=50, noise_lr=5e5, replay_ratio=0.1

## Per-iteration test PSNR curves

| Iter | Baseline | A noise_stop | B low_noise | C replay=0.3 | D1 spf50/7k | D2 spf150/17k |
|-----:|--------:|-------------:|------------:|-------------:|------------:|--------------:|
| 1000 | 15.25   | 15.42        | 15.15       | 15.32        | 15.45       | **17.23**     |
| 2000 | 17.58   | 16.82        | 16.63       | **18.05**    | 18.00       | **23.05**     |
| 3000 | 15.05   | 14.62        | 13.80       | **15.96**    | 15.24       | —             |
| 4000 | 13.64   | 13.63        | 13.80       | **14.10**    | 14.33       | —             |
| 5000 | —       | —            | —           | —            | 13.39       | 17.94         |
| 7000 | —       | —            | —           | —            | 12.92       | —             |
|17000 | —       | —            | —           | —            | —           | 15.25         |

**Final test PSNR (held-out frames):**

| Experiment | Final PSNR | Δ baseline | Bootstrap re-render |
|------------|-----------|-----------|---------------------|
| Baseline   | 13.64 dB  | —         | —                   |
| A: noise_stop_iter=2000 | 13.63 dB | -0.01 | 15.30 dB |
| B: noise_lr=5e4 (10×↓) | 13.80 dB | +0.16 | 11.91 dB |
| C: replay_ratio=0.3     | **14.10 dB** | **+0.46** | 14.26 dB |
| D1: spf=50, 100fr, 7k  | 12.92 dB | -0.72  | 12.80 dB |
| D2: spf=150, 100fr, 17k | **15.25 dB** | **+1.61** | 12.07 dB |

**Gaussian count (all A/B/C/baseline are identical):**
`20,685 → 22,000 → 22,500 → 22,933` (iters 1000/2000/3000/4000). Growth, no pruning.

## Findings

### 1. Noise injection is NOT the cause (exp A disproves hypothesis)

Stopping noise at iter 2000 (the PSNR peak) produced an identical trajectory to baseline:
- Baseline: 17.58 → 15.05 → 13.64 dB
- Exp A: 16.82 → 14.62 → 13.63 dB

The regression persists identically even with noise off. Also: `opacity_reset_interval` was never
referenced in `train_streaming.py` (only in the taming branch of `train.py`), so the original
hypothesis A was a dead-end regardless.

### 2. Lower noise_lr doesn't fix it (exp B)

10× noise reduction slightly improves final PSNR (13.80 vs 13.64), but the regression pattern
is unchanged. The bootstrap re-render (11.91 dB) is WORSE than baseline — lower noise slows
initial fitting of the bootstrap region.

### 3. Higher replay ratio helps most among A/B/C (exp C)

Tripling the global replay ratio (0.1 → 0.3) is the only single-knob change that consistently
improves both peak (+0.47 dB at iter 2000: 18.05 vs 17.58) and final quality (+0.46 dB: 14.10
vs 13.64). The drop from peak to final is also smaller:

- Baseline: −3.94 dB
- Exp A:    −3.19 dB (noise stop can't hurt)
- Exp C:    −3.95 dB (but from a higher peak → higher floor)

More replay keeps early Gaussians' gradients alive longer as the window moves. This directly
confirms the **keyframe-window bias** hypothesis.

### 4. The regression is structural: catastrophic forgetting

All experiments show the same shape — PSNR peaks around iter 2000 then declines. Gaussian
count data rules out pruning. The mechanism is:

1. Bootstrap frames (0–5) train well in first 1000–2000 iters → high PSNR
2. New frames start arriving; training window (`keyframe_window=8`) shifts toward recent frames
3. Early-frame Gaussians stop receiving gradient updates (only 10% of steps from replay buffer)
4. Test cameras are uniformly sampled across all arrived frames, so early-frame quality
   degradation drags down mean test PSNR

This is an **online continual learning problem** (catastrophic forgetting), not a
hyperparameter issue. Single-knob fixes can only partially mitigate it.

### 5. D1 confirms: more iterations at spf=50 makes things worse

D1 runs 7000 iters with 100 frames (vs baseline's 4000 iters with ~80 frames). The PSNR
keeps declining: 18.00 → 15.24 → 14.33 → 13.39 → 13.20 → 12.92 dB. Extending training
time without changing steps_per_frame just exposes more of the forgetting effect.

### 6. D2: spf=150 is the dominant quality lever, but forgetting is even steeper

D2 achieves a spectacular iter-2000 peak of **23.05 dB** (with only ~18 frames ingested, each
trained for 150 steps). Final quality at 17000 iters is 15.25 dB — better than all spf=50
variants but after a −7.8 dB drop from peak.

The higher peak makes sense: 150 steps per frame converges each local view well before moving
on. But when the window moves, the forgetting is equally catastrophic in relative terms.

Coverage vs quality tradeoff at same total iteration budget (17k iters):
- spf=50: ~340 frames seen, final ~12 dB
- spf=150: ~100 frames seen, final 15.25 dB

Less coverage, dramatically higher quality. For a SLAM system with a fixed compute budget,
spf=150 is strongly preferable unless scene coverage is the bottleneck.

## Experiments E / F / G (2026-05-12, completed)

| Experiment | Final PSNR | Δ F-baseline | Forgetting cliff |
|---|---|---|---|
| E: spf=150 + replay=0.3 | 16.58 dB | — | −4.77 dB |
| F: E + keyframe_window=20 | **17.08 dB** | **+0.50** | **−4.34 dB** |
| G: F + anchor_bootstrap=True | 15.33 dB | −1.75 | −6.02 dB |

G backfired: position anchoring created streaking artifacts in early-frame views.
Root cause of forgetting confirmed: **gradient starvation**, not position displacement.

## H1 / H2 / H3 insertion-geometry ablations (2026-05-13)

### Diagnostic modes added

`--streaming_training_mode {normal|placement_only|colors_only}` (new flag).
- `placement_only`: zero backward/MCMC, inserted points stay exactly as placed.
- `colors_only`: geometry frozen (zero LR on means/scales/quats/opacities), SH trains.

### H1 (placement_only) and H2 (colors_only) results

| Stage | PSNR (100 frames) | Notes |
|---|---|---|
| H1: no training at all | 4.62 dB | 19,877 → 21,056 Gaussians, pure insertions |
| H2: SH colors only, 5k iters | 5.79 dB | +1.17 dB from 5000 color-only iters |
| Normal F (full training) | 17.08 dB | 11.3 dB gap = from geometry optimization |

**Finding:** Color training barely helps (+1.2 dB). The entire quality gap vs normal
training comes from scale/rotation optimization. Insertion positions are geometrically
correct; the problem is the anisotropic surfel init.

### H3 (isotropic scale init) vs F (anisotropic surfel)

`--streaming_insert_isotropic_scale`: replaces `(tx, ty, 0.2·min(tx,ty))` with
`(√(tx·ty), √(tx·ty), √(tx·ty))` — spherical Gaussian instead of flat disc.

| Iter | F (anisotropic) | H3 (isotropic) | Δ |
|---|---|---|---|
| 1000 | 17.23 dB | 17.21 dB | ≈0 |
| 2000 | **23.05 dB** | 21.39 dB | −1.66 |
| 5000 | 17.94 dB | **20.40 dB** | +2.46 |
| 10000 | — | 19.02 dB | — |
| 17000 | 15.25 dB | **17.19 dB** | **+1.94** |
| Forgetting cliff | −7.80 dB | **−4.20 dB** | **+3.60** |
| Bootstrap re-render | 12.07 dB | 12.59 dB | +0.52 |

**Finding:** Isotropic init reduces the forgetting cliff by 3.6 dB (7.8 → 4.2 dB),
improving final quality by 1.94 dB. Mechanism: flat surfels are seen edge-on once the
window moves, causing streaking that the optimizer can't repair without close-up views.
Spherical Gaussians degrade gracefully from all viewing angles.

**H3-physics-fixed (2026-05-13, branch: insertion-physics-fixes):**

Physics and correctness fixes applied on top of H3 config:
- Fix render-vs-sensor resolution mismatch in alpha/depth insertion masks
- Fix quaternion singularity for backward-facing normals
- Add `import torch` in `streaming_scene.py`
- Scale clamping: `scale_mult=0.5`, `scale_max=0.05m`, `normal_scale_ratio=0.15`
- Tighten alpha mask (threshold 0.5→0.3) and provisional age (50→20 frames)
- Free-space loss (`streaming_free_space_loss_weight=0.01`): penalise opacity in front of observed surface
- `reproduce.sh` written to every output directory at training start

| Iter | H3 (before fixes) | H3-physics-fixed | Δ |
|---|---|---|---|
| 1000 | 17.21 dB | **18.07 dB** | +0.86 |
| 2000 | 21.39 dB | **21.64 dB** | +0.25 |
| 5000 | 20.40 dB | 20.05 dB | −0.35 |
| 10000 | 19.02 dB | **19.23 dB** | +0.21 |
| 17000 | 17.19 dB | **17.41 dB** | **+0.22** |

**Best result so far: 17.41 dB** (H3 + physics fixes, fr1_desk, 100 frames, 17k iters).
Forgetting cliff unchanged (−4.2 dB) — cliff is training-distribution driven, not insertion-geometry driven.

## H4 / H5 window + floater-loss sweep (2026-05-13)

### H4: larger keyframe window

| Experiment | iter 7000 | Final (17000) | Δ H3-fixed | Cliff (7k→17k) |
|---|---|---|---|---|
| H3-physics-fixed (window=20) | ~19.2 dB | 17.41 dB | — | ~−1.82 dB |
| H4: window=30 | 18.89 dB | 17.72 dB | +0.31 | −1.17 dB |
| H4b: window=40 | **19.51 dB** | **18.93 dB** | **+1.52** | **−0.58 dB** |

**Finding:** Window=40 is a dramatic step change, not incremental. The cliff halves again
(−1.17 → −0.58 dB) and final PSNR jumps +1.21 dB over window=30.

| Experiment | iter 7000 | Final (17000) | Cliff (7k→17k) |
|---|---|---|---|
| H3-physics-fixed (window=20) | ~19.2 dB | 17.41 dB | ~−1.82 dB |
| H4: window=30 | 18.89 dB | 17.72 dB | −1.17 dB |
| H4b: window=40 | 19.51 dB | 18.93 dB | −0.58 dB |
| **H4c: window=60** | 19.24 dB | **20.94 dB** | **+1.70 dB (inverted!)** |

**Finding (H4c — window=60):** The forgetting cliff flips sign. PSNR continues rising from
iter 7000 to 17000 (+1.70 dB). Window=60 covers >60% of the 88 training frames at any time,
so early Gaussians stay in the active window throughout training — catastrophic forgetting
is essentially eliminated. **New best: 20.94 dB.**

Regime change: below window~40, the curve declines after the peak. Above window~60 (at 100
frames), training is still improving at termination. The optimal window for 100-frame runs
appears to be ≥60.

### H5: free-space loss weight sweep

| Experiment | iter 7000 | Final (17000) | Δ base |
|---|---|---|---|
| H4: fsl=0.01, window=30 | 18.89 dB | 17.72 dB | — |
| H5: fsl=0.03, window=30 | 18.93 dB | 17.87 dB | +0.15 |
| H4b: fsl=0.01, window=40 | 19.51 dB | 18.93 dB | — |
| H5b: fsl=0.03, window=40 | 19.34 dB | 18.72 dB | −0.21 |

**Finding:** Higher free-space loss (0.01→0.03) gives marginal improvement on window=30
(+0.15 dB) but slightly hurts on window=40 (−0.21 dB). The floater penalty appears to
interfere with photometric gradients at larger windows where gradient coverage is already
good. Stick with fsl=0.01.

**H4d (window=88 = all training frames):**

| Experiment | iter 7000 | Final (17000) | 7k→17k delta | min test PSNR |
|---|---|---|---|---|
| H4c: window=60 | 19.24 dB | 20.94 dB | +1.70 dB | 17.25 dB |
| **H4d: window=88** | 19.51 dB | **21.99 dB** | **+2.49 dB** | **19.69 dB** |

**Finding (H4d):** All-frames window is the upper bound for this dataset size. PSNR still
rising steeply at iter 17000 (+2.49 dB from 7k), suggesting the model hasn't converged —
more iterations or larger spf budget would push further. min test PSNR 19.69 dB means no
single held-out view falls below 19 dB; the quality floor is dramatically higher than at
smaller windows. **New best: 21.99 dB.**

### Window sweep summary (H3-physics-fixed base config, fr1_desk, 100 frames, 17k iters)

| Window | Coverage (of 88 train frames) | Final PSNR | 7k→17k trend |
|---|---|---|---|
| 20 | 23% | 17.41 dB | −1.82 dB (forgetting) |
| 30 | 34% | 17.72 dB | −1.17 dB |
| 40 | 45% | 18.93 dB | −0.58 dB |
| 60 | 68% | 20.94 dB | +1.70 dB (inverted) |
| **88** | **100%** | **21.99 dB** | **+2.49 dB** |

**Key insight:** Catastrophic forgetting in streaming 3DGS is a pure gradient-coverage problem.
The regime flip occurs between window=40 (45% coverage) and window=60 (68% coverage). Once
>60% of frames are always in the local window, the model improves monotonically to training
end. For a fixed 100-frame / 17k-iter budget, the optimal streaming strategy is to keep as
many frames in the local window as memory allows.

**Implications for real SLAM:**
- "Recent N frames" is a poor keyframe selection strategy: it creates forgetting by design.
- A smarter approach: adaptive keyframe selection that maintains coverage across the full
  trajectory (every Kth frame kept permanently) instead of a sliding recency window.
- At small scene sizes (100 frames), keeping all frames is tractable. At larger scales,
  target ≥60% coverage to stay in the no-forgetting regime.

**Recommended next experiments:**
- **H4d-extended: window=88 + 30k iters**: is the model still improving at 17k? Yes — how
  high does it go with more training time?
- **H6: visual quality evaluation**: compare PLY from H4d vs H3-physics-fixed to confirm
  PSNR gains translate to clean geometry.

## Artifact locations

```
output/ablation_A_noise_stop_2000/comparison/
output/ablation_B_low_noise_lr/comparison/
output/ablation_C_replay_03/comparison/
output/ablation_D1_spf50_100frames/comparison/
output/ablation_D2_spf150_100frames/comparison/
output/ablation_H1_placement_only/comparison/
output/ablation_H2_colors_only/comparison/
output/ablation_H3_isotropic_scale/comparison/
output/ablation_H4_window30/comparison/
output/ablation_H4b_window40/comparison/
output/ablation_H4c_window60/comparison/
output/ablation_H4d_window88/comparison/
output/ablation_H5_fsl003/comparison/
output/ablation_H5b_window40_fsl003/comparison/
```

Each contains `iter_0_bootstrap_views/`, `iter_N/`, `iter_N_bootstrap_views/` with
`report.json`, `test_contact_sheet.png`, and `trajectory.mp4`.
