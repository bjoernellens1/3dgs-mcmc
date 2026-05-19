# Streaming Replay — Bootstrap vs Trained State

**Date:** 2026-05-12
**Branch:** `streaming-fixes`
**Scene:** TUM RGB-D `freiburg1_desk` (596 frames)
**Model:** `gsplat` + `gsplat_energy_mcmc` + `selective_adam`
**Config:** `--streaming_replay --streaming_steps_per_frame 50 --cap_max 200000 --iterations 4000`

## Motivation

Earlier "successful run" report claimed full-sequence streaming worked
because the run completed without error. That run hard-coded
`steps_per_frame` to make total iterations divide the sequence length —
which is methodologically wrong (you don't know `N_frames` in a real
SLAM stream) and reported no test PSNR. Hypothesis raised by user:
in that long run, splats are deleted/moved during streaming and the
end-of-training scene contains *less* information than the bootstrap.

## What was changed in this round (infrastructure)

So we can answer "is the model getting better or worse?" without
hand-waving:

1. **`--eval` defaults to `True`** for the offline path
   (`--no-eval` opts out — `BooleanOptionalAction`).
2. **`--streaming_eval_hold` defaults to `8`** — every 8th arriving
   frame is held out from training/replay, ~12.5% test split spread
   along the trajectory.
3. **`utils/comparison_report.py`** — shared post-training reporter
   used by both `train.py` and `train_streaming.py`. Writes:
   - `comparison/iter_{N}/test/{view}.png` — side-by-side
     `render | gt | |diff|×5`
   - `comparison/iter_{N}/test_contact_sheet.png` — render-top /
     gt-bottom grid
   - `comparison/iter_{N}/trajectory.mp4` — train-camera trajectory
     render
   - `comparison/iter_{N}/report.json` — mean/min/max test PSNR
4. **Streaming pre-training snapshot** at `iter_0_bootstrap_views/`
   plus a re-render of the same bootstrap views at end-of-training
   into `iter_{N}_bootstrap_views/` so the bootstrap-vs-trained diff
   is over identical viewpoints.

Test PSNR and comparison renders are now **mandatory** for both paths
(opt out only via explicit flag).

## Experiment

Single 4000-iteration streaming run, 50 steps/frame:

| Snapshot | Cameras | Views | Mean PSNR | Min | Max |
|---|---|---|---|---|---|
| `iter_0_bootstrap_views` (post-init, before training) | bootstrap (5 frames, 19,877 Gaussians) | 5 | **4.39 dB** | 4.34 | 4.43 |
| `iter_4000_bootstrap_views` (same 5 cams, trained) | bootstrap views re-rendered | 5 | **14.19 dB** | 14.10 | 14.29 |
| `iter_4000/` (held-out streaming test) | every 8th arrival | 10 | **13.64 dB** | 11.43 | 15.79 |

**In-loop test PSNR over the run:**

```
iter 1000:  15.25 dB
iter 2000:  17.58 dB   ← peak
iter 3000:  15.05 dB
iter 4000:  13.64 dB   ← end-of-training
```

**Gaussian count over the run:** 19,936 → 22,933 (net growth ~3k).

## Findings

### 1. Bootstrap is geometrically correct, color-blank
The `iter_0` renders look catastrophically bad (4 dB) because SH
coefficients start near zero — the rendered RGB is near-black. But
the visible dot pattern in the renders shows Gaussians are placed
at correct 3D positions. So "bootstrap shows more detail" is **true
geometrically** (the point cloud has the right outlines), **false in
RGB terms** (no trained colors).

### 2. There IS a training regression — but not from pruning
Quality peaked at iter 2000 (17.58 dB) and dropped **-3.94 dB**
by iter 4000. N *grew* during this period (19.9k → 22.9k), so the
loss is not from MCMC pruning. Likely culprits, in priority order:

1. **`opacity_reset_interval=3000`** lands between the iter-2000 peak
   and the iter-4000 trough. An opacity reset right after a phase of
   convergence is consistent with the observed sharp drop.
2. **MCMC noise injection on already-converged Gaussians.**
   `streaming_mcmc_local_only=True` means new noise hits the
   visible/recent set, which overlaps the converged bootstrap region —
   colors get re-perturbed without enough optimisation budget per
   frame to recover.
3. **Keyframe-window bias.** With `streaming_keyframe_window=8` and
   `streaming_global_replay_ratio=0.1`, ~90% of gradient updates
   target the most recent 8 frames. Older Gaussians drift.

### 3. The visual character of the trained renders is "blob"
Even at peak quality, end-of-training renders are blurred,
over-saturated, and have lost all high-frequency content (keyboard,
papers, monitor edges → uniform smears). This is consistent with
scale-regulariser keeping Gaussians larger than the resolution
demands, but at 50 steps/frame the model also doesn't get enough
optimisation time per arriving frame to fit sharp content before the
window moves on.

### 4. User's original methodological concern is confirmed
Hard-coding `steps_per_frame` to fit `--iterations` makes the
streaming claim meaningless. With the reporter now mandatory, every
streaming run produces test PSNR and comparison artefacts, so this
class of false-success report can't recur.

## Open questions (next experiments)

- **A. Disable opacity reset** (`--opacity_reset_interval 0`) and check
  whether the iter-2000-to-4000 regression disappears. If yes,
  opacity reset is incompatible with streaming and should be
  conditionally disabled in `train_streaming.py`.
- **B. Lower `--noise_lr`** (default `5e5`) and check whether
  bootstrap-view PSNR continues to improve past iter 2000.
- **C. Higher `streaming_global_replay_ratio`** (e.g. `0.3`) to
  counteract keyframe-window bias on older Gaussians.
- **D. Re-do the proper full-sequence sweep** at fixed
  `steps_per_frame` ∈ {50, 150} so total iterations float
  (`N_frames × steps_per_frame + tail`), and compare end-of-stream
  test PSNR. This is the actual test of the user's earlier
  steps-per-frame theory; deferred from this session because the
  iter-2000-regression finding is more actionable first.

## Artifacts on disk

- `output/bootstrap_vs_iter4000/comparison/iter_0_bootstrap_views/`
- `output/bootstrap_vs_iter4000/comparison/iter_4000/`
- `output/bootstrap_vs_iter4000/comparison/iter_4000_bootstrap_views/`

Each directory contains `test_contact_sheet.png`, `trajectory.mp4`,
per-view `test/*.png`, and `report.json`.
