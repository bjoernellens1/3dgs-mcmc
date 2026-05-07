# Taming-3DGS Integration Plan

Branch: `include-taming-3dgs`

Date: 2026-05-06

## Goal

Add Taming-3DGS-inspired densification to this ROCm/gsplat 3DGS-MCMC pipeline while keeping the current MCMC behavior intact as the default.

The integration should support three explicit strategies:

- `mcmc`: current behavior, default, regression-sensitive.
- `taming`: constructive score-guided densification toward a deterministic Gaussian budget.
- `hybrid`: Taming-style target budget and scoring for growth, plus MCMC relocation/death handling.

Also investigate Taming-3DGS' training-time parallelism and low-level rasterizer optimizations as a separate track. These optimizations may be valuable independently of the densification strategy, but they must be evaluated against the current ROCm/gsplat backend before porting.

## Constraints

- Current functionality on the `mcmc` path has highest priority.
- The current renderer is a ROCm `gsplat` adapter, not Taming's CUDA `diff_gaussian_rasterization` fork.
- Exact Taming scoring depends on per-Gaussian rasterizer accumulators:
  `accum_weights`, `accum_dist`, `accum_blend`, `accum_count`,
  `gaussian_depths`, and `gaussian_radii`.
- Verification must be run inside the project containerized setup. Host Python checks are not sufficient for this repo because the ROCm/gsplat runtime is container-provided.

## Existing System Summary

The current training loop:

- Requires `--cap_max`.
- Optimizes photometric loss plus opacity/scale regularizers.
- Defaults to `--energy_mcmc`.
- Computes utility from opacity, current-frame visibility, visibility EMA, gradients, and scale penalty.
- Mutates after optimizer step:
  - relocate dead/low-utility Gaussians;
  - grow by sampling parents and applying relocation math;
  - apply stochastic covariance-scaled MCMC noise.
- Logs geometry dashboard metrics such as low-support opacity mass.

This must remain the behavior of `--densification_strategy mcmc`.

## Taming-3DGS Summary

Taming-3DGS differs from this pipeline in two core ways:

- It uses a deterministic count schedule toward an exact budget.
- At densification intervals, it computes multi-camera scores and selects clone/split parents by score, instead of growing via MCMC parent sampling.

The released code scores Gaussians using:

- gradient accumulator;
- opacity;
- depth;
- projected radius;
- scale/volume;
- weighted photometric loss accumulation;
- distance accumulation;
- blend accumulation;
- reverse visibility/count accumulation;
- per-view photometric loss and image edge weighting.

## Integration Decision

Do not replace MCMC. Add Taming as an optional strategy and Hybrid as the likely useful research path.

Reasoning:

- MCMC is the core identity of this repo and has extra ROCm-specific work already wired in.
- Taming's highest-fidelity scoring requires renderer accumulators that are not currently available from the `gsplat` backend.
- Several Taming ideas overlap with the existing energy-guided MCMC utility; the cleanest first step is modular strategy selection and shared scoring.

## Implementation Plan

1. Add strategy plumbing.
   - Add `densification_strategy` with choices `mcmc`, `taming`, `hybrid`.
   - Default to `mcmc`.
   - Add Taming-specific flags:
     - `taming_budget`
     - `taming_budget_mode`
     - `taming_cams`
     - `taming_score_interval`
     - score weights
   - Keep old MCMC flags unchanged.

2. Modularize strategy execution.
   - Move strategy-specific growth/relocation orchestration out of the main training loop.
   - Preserve call order for `mcmc`: optimizer step, MCMC noise, schedule, relocation, growth.
   - Provide a shared strategy context object or simple helper functions to avoid broad refactors.

3. Add Taming utilities.
   - Implement corrected deterministic budget curve.
   - Implement edge-map generation.
   - Implement multi-camera score computation.
   - Require exact renderer stats for Taming score computation.
   - Fail clearly if a renderer cannot provide exact per-Gaussian scoring stats.

4. Extend renderer API conservatively.
   - Add optional `pixel_weights=None` and `return_taming_stats=False` parameters to `gaussian_renderer.render`.
   - Do not change existing return keys.
   - When possible, expose:
     - `gaussian_depths`
     - `gaussian_radii`
     - `accum_weights`
     - `accum_count`
     - `accum_blend`
     - `accum_dist`
   - For `gsplat`, derive exact accumulators from rasterized Gaussian/pixel intersections returned by `rasterize_to_indices_in_range`.

5. Investigate Taming parallelism and rasterizer optimizations.
   - Review the `origin/rasterizer` branch, especially CUDA kernels and optimizer changes.
   - Classify optimizations into:
     - Python-level changes that can be ported directly;
     - renderer API changes that can be mimicked through `gsplat`;
     - CUDA-specific kernel changes that need ROCm/gsplat equivalents or should not be ported.
   - Pay special attention to:
     - sparse/visible-only Adam updates;
     - separate SH optimizer and less-frequent SH updates;
     - parallelized backward/attribute update changes;
     - weighted per-Gaussian accumulation needed for exact Taming scores.
   - Do not enable these by default until short containerized MCMC regression runs show no behavior breakage.

6. Add Gaussian model Taming operations.
   - Add score-guided clone/split methods based on Taming's logic.
   - Guard zero-budget, zero-score, and no-candidate cases.
   - Preserve optimizer state handling and `visibility_ema` resizing.
   - Avoid touching existing `add_new_gs*` and `relocate_gs*` behavior.

7. Implement strategies.
   - `mcmc`: existing code path.
   - `taming`: Taming score-guided constructive clone/split, optional pruning, no MCMC relocation/growth.
   - `hybrid`: MCMC relocation/death handling plus Taming score-guided growth toward the deterministic budget.

8. Verification.
   - Use the containerized setup for runnable checks, for example `docker compose run --rm train ...`.
   - Static compile all changed Python files inside the container.
   - Run parser/help smoke tests inside the container.
   - In the container/runtime with `gsplat`, run:
     - short `mcmc` run and compare logs to current behavior;
     - short `taming` run and confirm Gaussian count follows budget;
     - short `hybrid` run and confirm both relocation and score-guided growth logs appear.

## Acceptance Criteria

- `mcmc` remains the default and its code path remains behaviorally equivalent.
- `--densification_strategy taming` runs without invoking MCMC growth/relocation.
- `--densification_strategy hybrid` keeps MCMC relocation while using score-guided growth.
- Renderer API remains backward-compatible for existing callers.
- Taming scoring requires exact renderer stats and raises if they are unavailable.
- No unrelated refactors or dependency churn.
