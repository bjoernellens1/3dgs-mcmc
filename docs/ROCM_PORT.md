# ROCm Port for 3DGS-MCMC

This branch replaces the NVIDIA-specific CUDA extension stack with a ROCm-compatible backend based on [`ROCm/gsplat`](https://github.com/ROCm/gsplat).

## What changed

| Component | Before (CUDA) | After (ROCm)
|---|---|---|
| Rasterizer | `diff-gaussian-rasterization` (CUDA extension) | `gsplat` (ROCm/gsplat fork) |
| KNN init | `simple-knn` (CUDA extension) | PyTorch fallback (`utils/rocm_knn_fallback.py`) |
| Relocation | `compute_relocation` CUDA kernel | PyTorch fallback (`utils/rocm_reloc_fallback.py`) |
| Container | None | `Dockerfile` based on `rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0` |
| Package mgr | Conda (`environment.yml`) | `uv` + `pyproject.toml` |

## Quick start with Docker / Podman

Build:
```bash
docker build -t 3dgs-mcmc-rocm:7.2 .
# or
podman build -t 3dgs-mcmc-rocm:7.2 .
```

Run (interactive, with GPU passthrough):
```bash
docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --group-add=render \
  --security-opt seccomp=unconfined \
  --ipc=host \
  -v "$PWD":/workspace/3dgs-mcmc \
  3dgs-mcmc-rocm:7.2
```

Inside the container, verify ROCm:
```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.hip)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

Train a tiny scene:
```bash
python train.py -s /path/to/colmap_scene --cap_max 100000 --iterations 1000
```

## Convention differences: Inria rasterizer vs. gsplat

The adapter in `gaussian_renderer/gsplat_backend.py` hides most differences, but the following are worth knowing if you modify the renderer or camera handling.

### 1. Quaternion order
- **Inria**: `wxyz` (identity = `[1, 0, 0, 0]`)
- **gsplat**: `wxyz` (same!)
- **Action**: No conversion needed. The current repo already stores quaternions as `wxyz`.

### 2. Camera world-to-view matrix
- **Inria**: `world_view_transform` is the **transpose** of the actual world-to-camera matrix. It is passed directly to `GaussianRasterizationSettings(viewmatrix=...)`.
- **gsplat**: `viewmats` expects the **actual** world-to-camera matrix (not transposed).
- **Action**: The adapter transposes `world_view_transform` before passing it to `gsplat.rasterization(...)`.

### 3. Camera intrinsics
- **Inria**: Uses `FoVx`, `FoVy`, `tanfovx`, `tanfovy`, and a 4x4 projection matrix.
- **gsplat**: Uses a 3x3 pinhole intrinsic matrix `K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`.
- **Action**: The adapter computes `fx = W / (2 * tan(FoVx/2))` and builds `K` on the fly.

### 4. SH coefficients layout
- **Inria**: `_features_dc` is `[N, 1, 3]`, `_features_rest` is `[N, K-1, 3]`. `get_features()` concatenates to `[N, K, 3]`.
- **gsplat**: Expects `[N, K, 3]` when `sh_degree` is provided.
- **Action**: `pc.get_features` is passed directly. No transpose needed.

### 5. Color post-processing
- **Inria**: The CUDA rasterizer evaluates SH internally and clamps with `colors = max(SH + 0.5, 0)`.
- **gsplat**: `rasterization(...)` does the same clamping automatically when `sh_degree` is set.
- **Action**: No extra code needed.

### 6. Returned image shape
- **Inria**: `[3, H, W]`
- **gsplat**: `[C, H, W, 3]` (batch of images)
- **Action**: The adapter permutes with `.permute(2, 0, 1)` to match the old shape.

### 7. Visibility / radii metadata
- **Inria**: `radii` has shape `[N]` (one entry per Gaussian, zero if culled).
- **gsplat** (`packed=True`): `radii` and `means2d` are **packed** to only visible Gaussians (`[nnz]`). `meta["gaussian_ids"]` maps packed indices back to the original `[N]` array.
- **Action**: The adapter scatters packed metadata back into full `[N]` arrays so that downstream code expecting `radii > 0` still works.

### 8. Tile size
- **Inria**: Hard-coded to 16 in the CUDA kernel.
- **gsplat** (ROCm fork): Default changed to **8** because it performs better on AMD GPUs.
- **Action**: The adapter explicitly passes `tile_size=8`.

### 9. `screenspace_points` gradient tensor
- **Inria**: Creates a zero `[N, 3]` tensor with `requires_grad=True` so that the rasterizer writes 2D mean gradients into `.grad`.
- **gsplat**: Gradients flow back through `means` directly. `meta["means2d"]` also carries gradients when available.
- **Action**: The adapter still returns a dummy `screenspace_points` for API compatibility. **The MCMC training loop in this repo does not use it**, so this is safe.

## Environment variables (Strix Halo / gfx115x)

If running on Strix Halo or similar RDNA3.5 APU, set these before training:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HSA_XNACK=1
export HSA_ENABLE_SDMA=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
```

For debugging HIP kernel crashes:
```bash
export HIP_LAUNCH_BLOCKING=1
```

## Fallback accuracy notes

- `distCUDA2` fallback uses chunked `torch.cdist` + top-k. It is slower than the CUDA KNN but only runs once during point-cloud initialization.
- `compute_relocation` fallback is an exact vectorised reimplementation of the CUDA kernel (Equation 9 in the 3DGS-MCMC paper). It uses the hockey-stick identity to collapse the double loop into a single sum, giving **bit-identical results** to the original.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `gsplat` import error | Not installed or ROCm PyTorch mismatch | Rebuild container; verify `torch.version.hip` |
| Black renders | Camera matrix convention wrong | Check `world_view_transform.transpose(0,1)` in adapter |
| Inverted / mirrored scene | `K` or `viewmat` transposed | Verify `fx/fy/cx/cy` signs and matrix layout |
| MIOpen compile errors | Hitting BatchNorm or conv from extra deps | Keep deps minimal; avoid importing SAM/feature nets |
| OOM during training | Fragmentation | Ensure `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` |
