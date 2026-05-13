# 3DGS-MCMC Project Instructions

This project is a specialized fork of 3D Gaussian Splatting (3DGS) that implements Markov Chain Monte Carlo (MCMC) for Gaussian densification and relocation. It also includes a comprehensive ROCm port for AMD GPUs using the `gsplat` rasterizer.

## Project Overview
- **Core Methodology:** Treats 3D Gaussian Splatting as an MCMC process to handle densification, relocation, and pruning of Gaussians based on an energy function.
- **ROCm Support:** Replaces NVIDIA-specific CUDA extensions with a ROCm-compatible stack.
- **Key Technologies:** PyTorch, gsplat (ROCm fork), FastAPI (web viewer), uv (package management).

## Architecture & Backends
### Rasterizer
The project supports two main rasterization backends:
- **`gsplat` (Default for ROCm):** Integrated via an adapter in `gaussian_renderer/gsplat_backend.py`. It maps Inria-style camera and SH conventions to `gsplat` requirements.
- **`diff-gaussian-rasterization` (Original):** Used in the CUDA/NVIDIA path.

### MCMC Scheduling
Controlled by `utils/mcmc_schedule.py`. It manages:
- **Relocation:** Moving Gaussians with low utility.
- **Growth:** Adding Gaussians in high-error regions.
- **Death:** Removing Gaussians with low opacity or utility.

### Streaming Replay
A unique mode (`StreamingParams`) for incremental training from video or RGB-D sequences. It supports real-time insertion of Gaussians from depth maps.

## Key Commands

### Training
Standard training for a scene:
```bash
python train.py -s <path_to_data> --config configs/<config>.json --cap_max <max_gaussians>
```
*Note: `--cap_max` is required for MCMC to bound memory usage.*

### ROCm / Docker
Build and run with ROCm support:
```bash
docker build -t 3dgs-mcmc-rocm .
docker run --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host -v "$PWD":/workspace/3dgs-mcmc 3dgs-mcmc-rocm
```

### Evaluation
Compute metrics (PSNR, SSIM, LPIPS):
```bash
python metrics.py -m <model_path>
```

### Rendering
Generate video or images from a trained model:
```bash
python render.py -m <model_path>
```

## Development Conventions

### Camera Conventions (Inria vs. gsplat)
- **World-to-View:** Inria uses transposed W2C matrices. `gsplat_backend.py` handles the transposition automatically.
- **Intrinsics:** `gsplat` expects a 3x3 `K` matrix; the project derives this from `FoVx/FoVy`.
- **Quaternion Order:** Both use `wxyz`.

### SH Evaluation
For performance on ROCm, SH evaluation is often performed in Python via `torch.compile` (`--sh_backend compiled_python`) to avoid heavy HIP kernels for backward passes.

### Adding New Features
- **Arguments:** Update `arguments/__init__.py` to add new parameters to `ModelParams`, `OptimizationParams`, or `StreamingParams`.
- **MCMC Logic:** Modify `utils/energy_mcmc.py` for energy function changes or `utils/mcmc_schedule.py` for scheduling changes.
- **Rasterization:** If modifying the renderer, ensure compatibility in `gaussian_renderer/gsplat_backend.py`.

## ROCm Specifics
- **Environment Variables:** For RDNA3/Strix Halo, set `HSA_OVERRIDE_GFX_VERSION=11.0.0` and `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`.
- **Fallbacks:** `simple-knn` and `distCUDA2` have PyTorch-based fallbacks in `utils/rocm_knn_fallback.py`.

## Memory & Performance
- **Taming-3DGS:** Use `--densification_strategy taming` to enable budget-aware Gaussian management.
- **Sparse Gradients:** Enabled by default (`--gsplat_sparse_grad`) to improve performance and reduce memory during training.
