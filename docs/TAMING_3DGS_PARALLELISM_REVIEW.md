# Taming-3DGS Parallelism Review

Date: 2026-05-06

This note tracks Taming-3DGS performance optimizations separately from the densification strategy integration.

## Source Reviewed

- `humansensinglab/taming-3dgs` `origin/rasterizer`
- `changes` patch file in that branch
- Taming README notes about drop-in rasterizer optimizations and sparse Adam

## Optimization Classes

### Python-Level / Training Loop

- `--sh_lower`: update SH coefficients less frequently, once every 16 iterations in the released training loop.
- Separate SH optimizer: Taming separates `_features_rest` into a distinct optimizer group/optimizer.
- Benchmark timing hooks for forward/backward/optimizer step.

Portability: likely portable, but should be opt-in. Less-frequent SH updates can change convergence, especially with this repo's custom SH schedule and ROCm Python-side SH evaluation.

### Optimizer-Level

- Sparse/visible-only Adam via `SparseGaussianAdam`.
- Taming README says sparse Adam can change training behavior, unlike the numerically equivalent rasterizer optimizations.

Portability: not directly portable unless `gsplat` or the ROCm runtime provides an equivalent optimizer. A Python imitation would likely lose the performance benefit and may change optimizer state semantics.

### Renderer / Kernel-Level

- Separate SH/DC paths in the rasterizer API.
- CUDA rasterizer changes for backward/attribute update parallelism.
- Additional per-Gaussian accumulators used by Taming scoring:
  `accum_weights`, `accum_dist`, `accum_blend`, `accum_count`,
  `gaussian_depths`, and `gaussian_radii`.

Portability: partly implemented. The current branch exposes exact Taming accumulator keys from the `gsplat` adapter by replaying the actual rasterized Gaussian/pixel intersections exposed in `gsplat` metadata. This keeps Taming scoring fail-closed: training raises if exact renderer stats are unavailable instead of silently falling back to approximations.

## Recommendation

Do not enable Taming parallelism changes by default in this branch.

## Container Metadata Check

Inside the project container, ROCm `gsplat.rendering.rasterization` returns metadata including:

- `gaussian_ids`
- `radii`
- `means2d`
- `depths`
- `opacities`
- `tiles_per_gauss`
- `isect_ids`
- `flatten_ids`
- `isect_offsets`

This is enough to reconstruct Taming's per-Gaussian accumulation inputs from actual accepted intersections:

- `accum_weights`: sum of Taming pixel weights over contributing Gaussian/pixel pairs.
- `accum_dist`: sum of projected Gaussian-to-pixel distances over contributing pairs.
- `accum_blend`: reconstructed blend contribution using segmented per-pixel transmittance in raster order.
- `accum_count`: count of accepted Gaussian/pixel contributions.

The resulting stats are exact with respect to the `gsplat` intersections used for the rendered image. They should still be performance-profiled against Taming's custom CUDA rasterizer because this implementation reconstructs the accumulators outside the fused raster kernel.

Next investigation steps inside the container:

1. Check whether `gsplat` supports sparse gradients or visible-only optimizer paths that map to Taming's `SparseGaussianAdam`.
2. Profile the exact `accum_*` extraction path on bicycle and compare it with the previous approximate scoring behavior.
3. If sparse/visible-only optimizer support exists, add it behind a separate flag and verify MCMC default behavior in containerized short runs.
