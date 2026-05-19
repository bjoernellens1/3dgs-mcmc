# Densification vs Pruning Balance — Experiment Notes

**Date:** 2026-05-12
**Scene:** TUM RGB-D freiburg1_desk (596 cameras)
**Model:** gsplat + gsplat_energy_mcmc + selective_adam

## Experiment: High Growth vs Moderate Growth

### Config A (moderate — better quality, faster)
```
--mcmc_stop_growth_iter 8000
--mcmc_growth_factor_start 1.02
--mcmc_grow_interval_min 50
--mcmc_grow_interval_max 1500
--mcmc_dead_opacity_start 0.001
--mcmc_dead_opacity_end 0.005
--mcmc_dead_opacity_power 2.0
--mcmc_relocate_interval_min 25
--mcmc_relocate_interval_max 300
--mcmc_relocate_tau 0.4
--energy_w_dead 2.0
--energy_alpha_dead 0.003
```

### Config B (aggressive — more splats, more floaters)
```
--mcmc_growth_factor_start 1.10
--mcmc_grow_interval_min 50
(same pruning/relocation as above)
```

## Results at iter 4000

| Metric | Config A | Config B |
|---|---|---|
| Gaussian count | 61.7K | 87.6K |
| Growth factor | 1.004 | 1.018 |
| Grow interval | 1681 | 1328 |
| it/s | ~70 | ~60 |
| LSOM | 0.018 | 0.022 |

## Key Finding

Higher densification produces more misplaced splats (floaters) that relocation
cannot keep up with. The metric `LSOM` (Low-Support Opacity Mass) rises,
indicating more unsupported low-quality gaussians. This also reduces
performance (~60 vs 70 it/s) due to the larger count.

**Better strategy:** Moderate growth (factor 1.02–1.05, stop by 8–12K),
aggressive dead-opacity pruning (threshold 0.001→0.005), and frequent
relocation (interval 25–300). This keeps N in the 60–90K range with higher
per-splat quality.

## Recommended Defaults (perf/acceleration branch)

```bash
# From arguments/__init__.py
init_scale_mode = "fixed"      # Avoid O(N²) KNN fallback
cap_max = 500000               # Saner ceiling for debugging
pcd_voxel_size = 0.02          # Open3D downsampling on init
web_viewer_enabled = False     # No periodic CPU/GPU sync
web_viewer_scene_cache_interval = 0
scalar_log_interval = 100

# For benchmarking, add these CLI flags:
--mcmc_stop_growth_iter 8000
--mcmc_growth_factor_start 1.02
--mcmc_grow_interval_min 50
--mcmc_grow_interval_max 1500
--mcmc_dead_opacity_start 0.001
--mcmc_dead_opacity_end 0.005
--mcmc_dead_opacity_power 2.0
--mcmc_relocate_interval_min 25
--mcmc_relocate_interval_max 300
--mcmc_relocate_tau 0.4
--energy_w_dead 2.0
--energy_alpha_dead 0.003
```
