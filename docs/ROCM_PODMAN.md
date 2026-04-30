# ROCm Podman Run Instructions

## Quick Start — Bicycle Training

```bash
cd /home/bjoern/git/3dgs-mcmc

podman run --rm --privileged --security-opt label=disable \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video \
  -e HSA_XNACK=1 \
  -e HSA_ENABLE_SDMA=0 \
  -e PYTORCH_ROCM_ARCH=gfx1151 \
  -v /home/bjoern/Downloads/mipnerf360_v2_dataset:/data/mipnerf360_v2_dataset:Z \
  -v $(pwd):/workspace/3dgs-mcmc:Z \
  localhost/3dgs-mcmc-rocm:7.2-tb \
  python train.py -s /data/mipnerf360_v2_dataset/bicycle \
    --config configs/bicycle.json -m output/bicycle --eval
```

The `7.2-tb` tag includes TensorBoard. Logs are written to the output folder and are immediately visible on the host.

## Critical ROCm Environment Variables

| Variable | Value | Why |
|----------|-------|-----|
| `HSA_XNACK` | `1` | **Now safe.** After rebuilding `gsplat` from the `numeric_fixes` branch, `HSA_XNACK=1` works on gfx1151. If you still see page faults, fall back to `0`. |
| `HSA_ENABLE_SDMA` | `0` | Recommended for Strix Halo / gfx1151 to avoid DMA-related hangs. |
| `PYTORCH_ROCM_ARCH` | `gfx1151` | Ensures PyTorch extensions compile for the correct AMD GPU architecture. |

## GPU Access in Podman

Podman needs:
- `--privileged` — required for AMDGPU device access inside the container.
- `--device=/dev/kfd --device=/dev/dri` — passes the ROCm kernel fusion driver and DRI devices.
- `--group-add=video` — adds the container process to the video group for `/dev/dri/card*` access.
- `--security-opt label=disable` — disables SELinux labeling so mounted volumes are accessible.

Without `--privileged`, even simple `torch.cuda` matmuls will fail with AMDGPU VM errors.

## Known Issues

### tile_size must be >= 16 on ROCm gfx1151

**Symptom:** Training starts normally, but loss becomes `nan` within the first few hundred iterations.

**Root cause:** `gsplat`'s HIP backward kernel with `tile_size=8` (the default in some gsplat versions) produces NaN gradients on ROCm/gfx1151 when wave32 mode is enabled.

**Fix:** The codebase now defaults `tile_size=16` (configured via `--tile_size` in `PipelineParams`). Do not override to 8 on ROCm.

### gsplat Backward Kernel Crash

**Symptom:** Training starts, loads cameras, then immediately crashes with:

```text
Memory access fault by GPU node-1 ... Reason: Page not present or supervisor privilege.
```

**Root cause:** The `release/1.5.3b2` branch of ROCm/gsplat has a bug in its HIP backward kernels for gfx1151 (Strix Halo). The forward pass works; the backward pass fails.

**Fix:** Rebuild `gsplat` from the `numeric_fixes` branch inside the container:

```bash
podman run --rm -it --privileged --security-opt label=disable \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video \
  -e HSA_XNACK=1 \
  -e HSA_ENABLE_SDMA=0 \
  -e PYTORCH_ROCM_ARCH=gfx1151 \
  -v $(pwd):/workspace/3dgs-mcmc:Z \
  localhost/3dgs-mcmc-rocm:7.2-tb \
  bash

# Inside container:
export PYTORCH_ROCM_ARCH=gfx1151
pip uninstall -y amd-gsplat gsplat

cd /tmp
rm -rf gsplat
git clone --branch numeric_fixes --depth 1 https://github.com/ROCm/gsplat.git
cd gsplat
git submodule update --init --recursive

# Apply the repo's patches (GLM symlink + wave32 for gfx1151)
python3 /workspace/3dgs-mcmc/docker/patch_gsplat_setup.py
python3 /workspace/3dgs-mcmc/docker/patch_gsplat_warp32.py

python setup.py build_ext --inplace
pip install --no-deps --no-build-isolation .
```

After rebuilding, `gsplat` backward works and training proceeds normally.

## Debugging GPU Faults

If you hit similar faults, run with serialization to pinpoint the failing kernel:

```bash
-e AMD_SERIALIZE_KERNEL=3 \
-e HIP_LAUNCH_BLOCKING=1 \
```

These make kernel launches synchronous and crash at the exact kernel that faults, rather than deferring the error.

## Dataset Structure Expected

The codebase expects COLMAP-format datasets:

```
dataset/bicycle/
  images/              # original images
  images_2/            # 1/2 res
  images_4/            # 1/4 res
  images_8/            # 1/8 res
  sparse/
    0/
      cameras.bin
      images.bin
      points3D.bin
```

The `configs/bicycle.json` sets `"resolution": 4`, so the loader downscales by 4x in memory (or you can point `--images` to `images_4`).

## Resuming Training / Early Stop

Checkpoints (`chkpnt*.pth`) and PLY point clouds (`point_cloud/iteration_*/point_cloud.ply`) are saved every `--checkpoint_interval` / `--save_interval` iterations (default: 2000) into the output folder, which is mounted back to the host via `-v $(pwd):/workspace/3dgs-mcmc:Z`. To resume:

```bash
podman run --rm --privileged --security-opt label=disable \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video \
  -e HSA_XNACK=1 \
  -e HSA_ENABLE_SDMA=0 \
  -e PYTORCH_ROCM_ARCH=gfx1151 \
  -v /home/bjoern/Downloads/mipnerf360_v2_dataset:/data/mipnerf360_v2_dataset:Z \
  -v $(pwd):/workspace/3dgs-mcmc:Z \
  localhost/3dgs-mcmc-rocm:7.2-tb \
  python train.py -s /data/mipnerf360_v2_dataset/bicycle \
    --config configs/bicycle.json -m output/bicycle --eval \
    --start_checkpoint output/bicycle/chkpnt2000.pth
```

## Optimization Launch Configs

These flags trade training speed against final quality. The defaults are safe for full 30k MCMC training. For faster iteration during development, use one of the configs below.

### SH degree schedule (`--sh_degree_schedule`)

Default: `[1000, 2000, 3000]` (degree 0 → 1 at 1000, 1 → 2 at 2000, 2 → 3 at 3000).

Delaying higher-order SH until geometry has stabilized avoids expensive color optimization while Gaussians are still chaotic:

```bash
--sh_degree_schedule 3000 6000 9000
```

This preserves final quality (`sh_degree=3` is still reached) but keeps the early iterations cheaper.

### Densification schedule

| Flag | Default | Fast config | Effect |
|------|---------|-------------|--------|
| `--densify_from_iter` | 500 | 500 | When relocation/growth starts |
| `--densify_until_iter` | 25000 | 15000 | When growth stops (biggest speed win) |
| `--densification_interval` | 100 | 200 | How often relocate/add runs |

Ending growth earlier prevents the Gaussian count from exploding in late training:

```bash
--densify_until_iter 15000 --densification_interval 200
```

### Initialization type

For COLMAP scenes, SfM initialization converges faster than random:

```bash
--init_type sfm
```

For MCMC-paper-faithful experiments, keep `--init_type random`.

### Resolution

For fast debugging or parameter sweeps, lower resolution dramatically speeds up rasterization:

```bash
-r 4   # 1/4 resolution (default in configs/bicycle.json)
-r 2   # 1/2 resolution (higher quality, still fast)
-r 8   # 1/8 resolution (very fast, lower quality)
```

### Recommended fast-quality-safe config

```bash
podman run --rm --privileged --security-opt label=disable \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video \
  -e HSA_XNACK=1 \
  -e HSA_ENABLE_SDMA=0 \
  -e PYTORCH_ROCM_ARCH=gfx1151 \
  -v /home/bjoern/Downloads/mipnerf360_v2_dataset:/data/mipnerf360_v2_dataset:Z \
  -v $(pwd):/workspace/3dgs-mcmc:Z \
  localhost/3dgs-mcmc-rocm:7.2-tb \
  python train.py -s /data/mipnerf360_v2_dataset/bicycle \
    --config configs/bicycle.json -m output/bicycle_fast --eval \
    --init_type sfm \
    --sh_degree_schedule 3000 6000 9000 \
    --densify_until_iter 15000 \
    --densification_interval 200 \
    --test_iterations 30000 \
    --save_iterations 30000
```

For random-init MCMC with the same speedups:

```bash
podman run --rm --privileged --security-opt label=disable \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video \
  -e HSA_XNACK=1 \
  -e HSA_ENABLE_SDMA=0 \
  -e PYTORCH_ROCM_ARCH=gfx1151 \
  -v /home/bjoern/Downloads/mipnerf360_v2_dataset:/data/mipnerf360_v2_dataset:Z \
  -v $(pwd):/workspace/3dgs-mcmc:Z \
  localhost/3dgs-mcmc-rocm:7.2-tb \
  python train.py -s /data/mipnerf360_v2_dataset/bicycle \
    --config configs/bicycle.json -m output/bicycle_fast_mcmc --eval \
    --init_type random \
    --sh_degree_schedule 3000 6000 9000 \
    --densify_until_iter 15000 \
    --densification_interval 200 \
    --test_iterations 30000 \
    --save_iterations 30000
```

## Observed Training Improvements

With the optimizations applied (`--init_type sfm`, `--sh_degree_schedule 3000 6000 9000`, `--densify_until_iter 15000`, `--densification_interval 200`), the full 30k bicycle training on ROCm/gfx1151 shows:

| Metric | Before (random init, default schedule) | After (optimized) |
|--------|----------------------------------------|-------------------|
| Initial Gaussians | 100,000 (random) | 54,275 (SfM) |
| Early iteration speed | ~5–8 it/s (post-densification) | ~28–34 it/s |
| SH degree 0 duration | 0–1000 it | 0–3000 it |
| Densification interval | every 100 it | every 200 it |
| Growth cutoff | 25,000 it | 15,000 it |

**Checkpoint sizes** (indicating controlled Gaussian growth):
- iter 2000: 57 MB
- iter 4000: 93 MB
- iter 6000: 151 MB
- iter 8000: 246 MB

The SfM initialization alone cuts the initial Gaussian count nearly in half, and delaying SH degree growth keeps the early training much faster while still reaching full `sh_degree=3` by iteration 9000.

## Energy-Guided MCMC Observations

### Splat count stalls at ~76k (SfM init)

This occurs on **both 10k and 30k runs** with SfM initialization:

| Iteration | Gaussians | Growth factor | Grow interval | rho | Notes |
|-----------|-----------|---------------|---------------|-----|-------|
| 600 | 54,275 → 56,873 | 1.048 | 150 | 0.009 | Early aggressive growth |
| 975 | 59,590 → 62,184 | 1.044 | 325 | 0.010 | Growth slowing |
| 6,768 | 74,318 → 75,081 | 1.010 | 1,692 | 0.013 | Minimal late growth |
| 9,510 | 75,851 → 76,245 | 1.005 | 1,902 | 0.013 | Nearly stalled |

**Root cause:** The exponential growth-factor decay (`growth_factor_tau=0.35`) is too aggressive for the current cap_max. By `u_growth ≈ 0.78` (reached around iter 9.5k regardless of total run length), `time_decay = exp(-0.78/0.35) ≈ 0.11`, yielding `growth_factor ≈ 1.005`. This is intrinsic to the schedule — not a 10k-run artifact.

**Why this happens with SfM init:**
- SfM starts with only 54k Gaussians (vs 100k random)
- The schedule's `cap_decay = (1-rho)^2 ≈ 0.97` is near 1 because `rho ≈ 0.013`, so cap pressure is negligible
- The **time decay** dominates, shrinking growth factor regardless of how far below cap_max we are

**Side effects:**
- **Far-away objects filtered more aggressively** than with schedule-only MCMC. The utility score penalizes low-visibility and low-gradient Gaussians, which disproportionately affects distant or occluded regions.
- **Dead threshold rises from 0.003 → 0.006**, making late-stage relocation more willing to recycle weak Gaussians.
- **Strong reconstruction with very low splat count**: 76k Gaussians at iter 10k is extremely lean compared to typical 3DGS-MCMC runs (often 200k–500k).

### Random init vs SfM init comparison (energy MCMC, 10k iterations)

| Init | Final N | Test PSNR | Train PSNR | Quality |
|------|---------|-----------|------------|---------|
| SfM | ~76k | ~14.4 | ~11.8 | Clean geometry, well-placed splats |
| Random | ~131k | ~18.9 | ~17.6 | Floaters, splats in wrong locations |

**Key finding: higher splat count is counterproductive without geometric priors.**

Random init starts with 100k Gaussians (vs 54k SfM) and grows to 131k. Despite higher PSNR, the visual quality is worse because:

1. **No geometric prior**: Random points have no scene structure. The energy MCMC utility score tries to guide placement, but without initialization near actual surfaces, many splats converge to wrong depths or become "floaters" in free space.
2. **PSNR is misleading**: PSNR rewards overall pixel similarity but does not penalize localized artifacts strongly. A few hundred misplaced bright splats can inflate PSNR while degrading perceptual quality.
3. **SfM provides surface anchors**: The 54k SfM points are already near actual scene geometry. Energy MCMC then refines and selectively grows from these anchors, keeping the model lean and accurate.
4. **Utility score has limits**: The gradient-based utility term rewards Gaussians that reduce photometric loss. With random init, early gradients are noisy and can reinforce bad placements before the model has learned coarse structure.

**Recommendation:**
- **For COLMAP scenes, always use `--init_type sfm`** unless you specifically need to test MCMC from-scratch reconstruction.
- If you must use random init, consider a much longer stabilization phase before growth (e.g. `--densify_from_iter 2000`) or reduce `--mcmc_growth_factor_start` to limit early chaotic expansion.

**How to increase splat count (SfM init only):**
```bash
# Slower growth-factor decay (default tau=0.35)
--mcmc_growth_factor_tau 0.6

# Higher initial growth factor (default 1.05)
--mcmc_growth_factor_start 1.10

# Extend growth window (default 12_000)
--mcmc_stop_growth_iter 20000

# Or disable energy guidance entirely
--no-energy_mcmc
```

## TensorBoard

### Viewing TensorBoard on localhost

The `localhost/3dgs-mcmc-rocm:7.2-tb` image now includes TensorBoard. Event files are written to the output folder (`output/<name>/`), which is volume-mounted back to the host.

**Option A — Run TensorBoard inside the container (recommended):**

Add port forwarding to your run command:

```bash
podman run --rm --privileged --security-opt label=disable \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video \
  -p 127.0.0.1:6006:6006 \
  -e HSA_XNACK=1 \
  -e HSA_ENABLE_SDMA=0 \
  -e PYTORCH_ROCM_ARCH=gfx1151 \
  -v /home/bjoern/Downloads/mipnerf360_v2_dataset:/data/mipnerf360_v2_dataset:Z \
  -v $(pwd):/workspace/3dgs-mcmc:Z \
  localhost/3dgs-mcmc-rocm:7.2-tb \
  bash -c "tensorboard --logdir /workspace/3dgs-mcmc/output/bicycle --host 0.0.0.0 --port 6006 & \
    python train.py -s /data/mipnerf360_v2_dataset/bicycle \
      --config configs/bicycle.json -m output/bicycle --eval"
```

Then open `http://localhost:6006` on your host.

**Option B — Run TensorBoard on the host (if tensorboard is installed):**

```bash
cd /home/bjoern/git/3dgs-mcmc
tensorboard --logdir output/bicycle --bind_all
```

If tensorboard is not installed on the host, use a disposable container:

```bash
podman run --rm -p 127.0.0.1:6006:6006 \
  -v $(pwd)/output/bicycle:/logs:Z \
  docker.io/tensorflow/tensorflow:latest \
  tensorboard --logdir /logs --host 0.0.0.0
```
