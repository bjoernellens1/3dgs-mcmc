# 100-Iteration Training Stutter

## Observed Behavior

The training loop stutters (brief pause) every ~100 iterations. This is visible in the live web viewer as a frame drop and in `it/s` metrics as a periodic dip.

## Root Cause: Original Triple Collision at `iteration % 100 == 0`

### Baseline (every iteration)
- `training_report`: `Ll1.item()` + `loss.item()` — 2 CUDA syncs
- `iter_start.elapsed_time(iter_end)` — 1 CUDA event sync
- `log_stage_times`: line-buffered file write (~1–10ms I/O)

### At iteration % 100 == 0, three operations fire simultaneously:

| Operation | File:Line | Cost | CUDA Syncs |
|---|---|---|---|
| `compute_geometry_dashboard()` | `train.py` | **High** — `torch.cdist` pairwise distances + multiple tensor reductions | Batched scalar extraction |
| MCMC growth (`grow_interval` starts at 100) | `train.py:674` | **High** — memory reallocation for new Gaussians, parameter cloning | 0–1 |
| Taming scoring (`taming_score_interval` was 100) | `train.py` | **Very High** — multiple extra render passes (if taming/strategy enabled) | 0 (no_grad) |

### Why it fades
- `grow_interval` **anneals** from 100 → 2000 over iterations, so growth stutters become less frequent
- Geometry dashboard now runs at `--geometry_log_interval` (default 500), so it no longer adds work every 100 iterations.

### Current Mitigation
The non-quality-affecting expensive intervals are now configurable and the defaults avoid the old viewer/geometry overhead:

- `--geometry_log_interval 500`
- `--sfm_anchor_interval 2000`
- `--web_viewer_image_interval 100`
- `--taming_cams 3`

The live web viewer also reuses the current training render by default. A fixed-camera viewer render is available with `--web_viewer_fixed_camera`, but it intentionally adds an extra rasterization pass at each viewer image interval.

Taming keeps the 100-iteration scoring cadence for quality, but now samples 3 scoring cameras by default. In an isolated 2.5k bicycle benchmark with `--taming_score_interval 100`, changing only `--taming_cams` from 10 to 3 reduced mean scoring-iteration wall time from about 1852 ms to 850 ms, while non-scoring iterations stayed around 34 ms. Raising `--taming_score_interval` to 500 is still a profiling option, but it changed the 6k bicycle metric in evaluation and is intentionally not the default.

## Taming Camera Benchmark

Command shape for both latency runs:

```bash
docker compose run --rm \
  -v "$PWD":/workspace/3dgs-mcmc \
  -v /home/bjoern/Downloads/mipnerf360_v2_dataset:/data/mipnerf360_v2_dataset \
  train python train.py \
    -s /data/mipnerf360_v2_dataset/bicycle \
    -m output/bench_taming_cams{N}_i100 \
    --config configs/bicycle.json \
    --iterations 2500 \
    --test_iterations 2500 \
    --save_iterations 2500 \
    --early_output_iterations 0 \
    --disable_progress_bar \
    --densification_strategy taming \
    --parallelism_profile safe \
    --taming_budget 300000 \
    --taming_cams {N} \
    --taming_score_interval 100 \
    --checkpoint_interval 0 \
    --save_interval 0 \
    --benchmark_dir model
```

Results from `output/bench_taming_cams10_i100/timings.jsonl` and `output/bench_taming_cams3_i100/timings.jsonl`:

| Setting | Scoring wall mean | Scoring wall p95 | Taming scoring mean | Taming scoring p95 | Non-scoring wall mean | 2.5k train PSNR |
|---|---:|---:|---:|---:|---:|---:|
| `--taming_cams 10` | 1852 ms | 1981 ms | 1636 ms | 1911 ms | 34.1 ms | 20.491 |
| `--taming_cams 3` | 850 ms | 896 ms | 635 ms | 775 ms | 34.1 ms | 20.598 |

At 6k with interval 100, `--taming_cams 3` still lowers latency but showed a quality tradeoff on the current bicycle run:

| Run | L1 | Train PSNR |
|---|---:|---:|
| Current 10-camera baseline, `output/bicycle_wv_final_6k` | 0.05013 | 21.862 |
| Patched 10-camera comparison, `output/bicycle_reviewfix_eval_6k` | 0.04990 | 21.923 |
| Patched 3-camera default, `output/bicycle_reviewfix_cams3_i100_6k` | 0.05112 | 21.741 |

## Per-Iteration Interval Summary

| Interval | Operations | Severity |
|---|---|---|
| Every iter | `training_report` (2× .item()), `elapsed_time` (event sync), `log_stage_times` (I/O) | Mild |
| Every 10 | tqdm update | Negligible |
| Every 50 | MCMC relocation (moderate GPU compute) | Minor |
| Every 100 | Taming scoring if enabled; MCMC growth starts here and then anneals | High when Taming is enabled |
| Every 500 | Geometry dashboard, PSNR/progress print | Moderate |
| Every 2000 | SfM-anchor geometry metric, PLY save + checkpoint save | Heavy but rare |

## Key Code Locations

- `compute_geometry_dashboard`: `utils/geometry_metrics.py`
- MCMC growth: `train.py` (inside `if densification_strategy in {"mcmc", "hybrid"}:`)
- Taming scoring: `train.py` (inside `if taming_enabled and run_taming_growth:`)
- grow_interval annealing: `utils/mcmc_schedule.py`
- geometry dashboard trigger: `train.py` (`iteration % args.geometry_log_interval == 0`)
