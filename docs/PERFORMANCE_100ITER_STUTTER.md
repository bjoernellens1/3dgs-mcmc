# 100-Iteration Training Stutter

## Observed Behavior

The training loop stutters (brief pause) every ~100 iterations. This is visible in the live web viewer as a frame drop and in `it/s` metrics as a periodic dip.

## Root Cause: Triple Collision at `iteration % 100 == 0`

### Baseline (every iteration)
- `training_report`: `Ll1.item()` + `loss.item()` — 2 CUDA syncs
- `iter_start.elapsed_time(iter_end)` — 1 CUDA event sync
- `log_stage_times`: line-buffered file write (~1–10ms I/O)

### At iteration % 100 == 0, three operations fire simultaneously:

| Operation | File:Line | Cost | CUDA Syncs |
|---|---|---|---|
| `compute_geometry_dashboard()` | `train.py:559` | **High** — `torch.cdist` pairwise distances + multiple tensor reductions | **5–6** `.item()` calls |
| MCMC growth (`grow_interval` starts at 100) | `train.py:674` | **High** — memory reallocation for new Gaussians, parameter cloning | 0–1 |
| Taming scoring (`taming_score_interval` = 100) | `train.py:462` | **Very High** — 10× extra render passes (if taming/strategy enabled) | 0 (no_grad) |

### Why it fades
- `grow_interval` **anneals** from 100 → 2000 over iterations, so growth stutters become less frequent
- Geometry dashboard runs at a **fixed** 100-iter interval, so it persists

### Mitigation
Move `compute_geometry_dashboard` from `iteration % 100` to `iteration % 500` to line up with the progress log interval. This eliminates the 5–6 CUDA sync collision every 100 iters.

## Per-Iteration Interval Summary

| Interval | Operations | Severity |
|---|---|---|
| Every iter | `training_report` (2× .item()), `elapsed_time` (event sync), `log_stage_times` (I/O) | Mild |
| Every 10 | tqdm update | Negligible |
| Every 50 | MCMC relocation (moderate GPU compute) | Minor |
| **Every 100** | **Geometry dashboard + MCMC growth + Taming scoring** | **Severe** |
| Every 500 | PSNR compute + progress print | Moderate |
| Every 2000 | PLY save + checkpoint save (file I/O) | Heavy but rare |

## Key Code Locations

- `compute_geometry_dashboard`: `utils/geometry_metrics.py:124-146`
- MCMC growth: `train.py:674` (inside `if densification_strategy in {"mcmc", "hybrid"}:`)
- Taming scoring: `train.py:462` (inside `if taming_enabled and run_taming_growth:`)
- grow_interval annealing: `utils/mcmc_schedule.py:113`
- geometry dashboard trigger: `train.py:559` (`if iteration % 100 == 0:`)
