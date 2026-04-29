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

## TensorBoard

TensorBoard summaries are written to the model output folder (e.g. `output/bicycle`), which is volume-mounted back to the host. View them from the host without entering the container:

```bash
cd /home/bjoern/git/3dgs-mcmc
tensorboard --logdir output/bicycle --bind_all
```

Then open `http://<host-ip>:6006` in a browser. If `tensorboard` is not installed on the host, use a temporary venv or a second container:

```bash
podman run --rm -p 6006:6006 \
  -v $(pwd)/output/bicycle:/logs:Z \
  docker.io/tensorflow/tensorflow:latest \
  tensorboard --logdir /logs --host 0.0.0.0
```
