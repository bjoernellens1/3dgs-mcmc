# Taming-3DGS Parallelism Plan

Date: 2026-05-07

## Goal

Bring Taming-3DGS performance ideas into this ROCm/gsplat pipeline without
changing default MCMC behavior. The first implementation stage is opt-in and
uses features already available in the container's `gsplat` runtime. Deeper
kernel work remains benchmark-gated.

## Staged Implementation

### Stage 1: Safe Opt-In Acceleration

- Keep `--densification_strategy mcmc` and plain Adam as defaults.
- Add a `--parallelism_profile safe` preset that enables:
  - `--optimizer_type selective_adam`
  - `--gsplat_sparse_grad`
  - `--sh_update_interval 16`
- Use `gsplat.optimizers.SelectiveAdam` instead of porting Taming's CUDA
  `SparseGaussianAdam`.
- Pass `sparse_grad=True` into `gsplat.rasterization` only when requested.
- Throttle `_features_rest` gradient updates by detaching SH-rest coefficients
  on skipped iterations, while still rendering with current SH values.
- Add structured timing for forward, loss, backward, optimizer, Taming scoring,
  mutation, geometry logging, and total iteration time.
- Keep exact Taming metrics fail-closed. If exact renderer stats are missing,
  Taming scoring must raise instead of falling back to approximations.

### Stage 2: Benchmark-Gated Kernel Work

Stage 2 should only start after Stage 1 timing identifies the actual bottleneck.
The likely candidates are exact Taming accumulator reconstruction and
backward/optimizer update overhead at high Gaussian counts.

Potential work:

- Move exact Taming accumulator computation into a fused ROCm/gsplat rasterizer
  path. This would avoid reconstructing `accum_weights`, `accum_count`,
  `accum_blend`, and `accum_dist` from intersection metadata in Python/Torch.
- Investigate a ROCm equivalent of Taming's per-Gaussian backward traversal.
  Taming's CUDA rasterizer reworks backward accumulation so each warp handles a
  Gaussian bucket and accumulates per-Gaussian gradients before atomics. A direct
  port must be reconciled with ROCm wave32 behavior and this repo's existing
  `gsplat` patches.
- Evaluate a native separated DC/rest SH path in `gsplat`. This repo currently
  evaluates SH in Python to avoid ROCm SH backward instability, so any fused SH
  path must be verified against the documented gfx1151 issues.
- Extend `gsplat` or add a local wrapper only if upstream runtime support is
  insufficient. Prefer upstream-compatible APIs over a private fork unless
  profiling proves the private kernel path is necessary.

## Implications

- `SelectiveAdam` is not behavior-neutral. Invisible Gaussians do not receive
  Adam state updates on skipped iterations, which can affect opacity,
  relocation, pruning, and MCMC utility behavior.
- `sparse_grad=True` can change gradient storage and optimizer expectations.
  Every clone, split, prune, relocation, checkpoint, and restore path must keep
  optimizer state tensors aligned with Gaussian tensors.
- In the current safe v1 implementation, sparse gradients are converted to
  dense contiguous gradients at the `SelectiveAdam` step boundary because the
  fused optimizer kernel requires contiguous dense inputs. This validates the
  runtime path but does not yet capture the full memory/performance upside of a
  fully sparse optimizer stack.
- SH throttling improves speed by reducing updates to view-dependent color
  coefficients, but may slow convergence of specular/view-dependent detail.
- Exact Taming stats are currently more expensive than the old approximation
  path. If Taming performance remains poor, fused exact-stat accumulation is the
  most direct next improvement.
- Default MCMC must remain untouched. Acceleration flags are opt-in until
  repeated containerized bicycle runs show acceptable speed/quality tradeoffs.

## Benchmark Protocol

Run all verification inside the project container. For each benchmark, store
the full run log, TensorBoard metrics, branch, git hash, and run args in the
output directory.

Minimum comparison set:

1. MCMC default, no acceleration, 4000 iterations.
2. MCMC with `--parallelism_profile safe`, 4000 iterations.
3. Taming exact stats, no acceleration, 4000 iterations.
4. Taming exact stats with `--parallelism_profile safe`, 4000 iterations.
5. Hybrid with safe profile only after MCMC and Taming pass.

Report:

- PSNR and L1 at the cap iteration.
- `iter_time` and printed `it/s`.
- Stage timing breakdown.
- Gaussian count and geometry dashboard metrics.
- Any visual coverage or support regressions.

## Acceptance Criteria

- Running without new flags produces the same default behavior.
- Accelerated runs fail clearly if the container lacks `gsplat` `SelectiveAdam`.
- Accelerated runs complete without NaNs, missing exact Taming stats, or
  optimizer state shape errors.
- TensorBoard contains timing scalars and run metadata sufficient to compare
  exact code states across runs.
