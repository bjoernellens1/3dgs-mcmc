# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research implementation of **3D Gaussian Splatting as Markov Chain Monte Carlo** (NeurIPS 2024 Spotlight). Built on top of the original 3DGS codebase, this version replaces the NVIDIA-specific CUDA extension stack with a ROCm-compatible backend using [`ROCm/gsplat`](https://github.com/ROCm/gsplat). All training runs inside a Docker/Podman container.

## Running Commands

All runnable verification and training must be done **inside the container**, not host Python.

```bash
# Build the image
docker build -t 3dgs-mcmc-rocm:7.2 .

# Interactive shell with GPU passthrough
docker compose run --rm train bash

# Run training
docker compose run --rm \
  -v /home/bjoern/Downloads/mipnerf360_v2_dataset:/data/mipnerf360_v2_dataset \
  train python train.py \
    -s /data/mipnerf360_v2_dataset/bicycle \
    --config configs/bicycle.json \
    --eval

# Smoke test (fast, CPU-only feasible)
docker compose run --rm train python train.py \
  -s /data/mipnerf360_v2_dataset/bicycle \
  --cap_max 4000 --iterations 100

# Render saved model
docker compose run --rm train python render.py -m output/bicycle

# Compute metrics (PSNR/SSIM/LPIPS)
docker compose run --rm train python metrics.py -m output/bicycle

# TensorBoard
docker compose run --rm -p 6006:6006 \
  -v "$PWD":/workspace/3dgs-mcmc train \
  tensorboard --logdir /workspace/3dgs-mcmc/output --host 0.0.0.0 --port 6006
```

The Mip-NeRF 360 v2 dataset is at `/home/bjoern/Downloads/mipnerf360_v2_dataset`. Mount it read-write if using `--init_type random` (scene loading writes `random.ply` into the scene directory).

Per-scene config files are in `configs/<scene>.json` (e.g., `configs/bicycle.json` sets `resolution: 4` and `cap_max: 5900000`).

## Key Architecture

### Densification Strategies (`--densification_strategy`)
- **`gsplat_energy_mcmc`** (default): Energy-guided MCMC using gsplat model layout. Gaussian utility scores (opacity, visibility, gradient, scale) drive birth/death decisions. Implemented in `utils/energy_mcmc.py` and `utils/strategies/mcmc_strategy.py`.
- **`gsplat_mcmc`**: Plain MCMC without energy guidance, gsplat layout.
- **`mcmc`** / **`hybrid`**: Legacy layout variants.
- **`taming`**: Score-based densification from Taming-3DGS (multi-camera edge/view/loss scores). Implemented in `utils/taming_3dgs.py`.

### Model Layouts (`--model_layout`)
- **`gsplat`** (default): `GsplatGaussianModel` (`scene/gsplat_model.py`) — params stored in `nn.ParameterDict` with keys `means`, `sh0`, `shN`, `opacities`, `scales`, `quats`.
- **`legacy`**: `GaussianModel` (`scene/gaussian_model.py`) — params stored as `_xyz`, `_features_dc`, `_features_rest`, `_opacity`, `_scaling`, `_rotation`.

The renderer (`gaussian_renderer/gsplat_backend.py`) wraps `gsplat.rendering.rasterization` and normalizes both model layouts into a common interface. This is the only rasterizer; the original `diff-gaussian-rasterization` CUDA extension is gone.

### Scene Loading (`scene/__init__.py`)
Scene type is auto-detected from directory layout:
- `sparse/` → COLMAP
- `rgb.txt`+`depth.txt` → TUM RGB-D
- `frames.jsonl`+`intrinsics.json` → Generic RGB-D sequence
- `color/`+`pose/`+`intrinsic/` or `.sens` file → ScanNet
- `mesh.ply` + Replica markers → Replica
- `transforms_train.json` → Blender

### Training Loop (`train.py`)
Key flow: random camera selection → `render()` → L1+SSIM loss + opacity/scale regularizers + energy losses → backward → optimizer step → MCMC noise injection → strategy mutation (relocation/growth).

Parallelism profile `safe` (default) enables `selective_adam` optimizer, sparse gradients (`gsplat_sparse_grad=True`), and `sh_update_interval=16` to reduce SH updates. Controlled by `--parallelism_profile off|safe`.

Checkpoints and PLY files are saved asynchronously via `AsyncSaveWorker` (background thread) to avoid blocking training.

### MCMC Schedule (`utils/mcmc_schedule.py`)
Controls growth/relocation intervals and dead-opacity thresholds over training via `MCMCScheduleConfig`. Growth stops at `--mcmc_stop_growth_iter` (default 12000). LSOM (low-support opacity mass) feedback can suppress growth when floaters accumulate.

### ROCm Fallbacks
- **KNN init**: `utils/rocm_knn_fallback.py` (replaces `simple-knn`)
- **Relocation kernel**: `utils/rocm_reloc_fallback.py` (replaces CUDA `compute_relocation`)
- Both are pure-PyTorch implementations for ROCm compatibility.

### Live Web Viewer
Optional FastAPI/WebSocket viewer at port 6010. Enable with `--web-viewer`. Serves rendered frames and training metrics in real time from a separate process. Requires `fastapi`, `uvicorn`, `websockets`, `cv2`.

## Important Parameters

| Parameter | Default | Notes |
|---|---|---|
| `--cap_max` | 500000 | **Required** (unless taming with budget). Max Gaussian count. |
| `--scale_reg` | 0.01 | Scale regularizer weight. |
| `--opacity_reg` | 0.01 | Opacity regularizer weight (use 0.001 for Deep Blending). |
| `--noise_lr` | 5e5 | MCMC noise learning rate. |
| `--init_type` | `sfm` | `sfm` or `random`. |
| `--densification_strategy` | `gsplat_energy_mcmc` | See strategies above. |
| `--parallelism_profile` | `safe` | `safe` enables sparse training; `off` disables it. |
| `--config` | None | JSON file; CLI args override config values. |

## Streaming Replay Mode (`--streaming_replay`)

Simulates real-time RGB-D input by consuming an existing RGB-D dataset as an ordered frame stream instead of loading all cameras at once. Activated with `--streaming_replay` (off by default).

### Architecture

| Component | File | Role |
|---|---|---|
| `StreamingRGBDFrame` / sources | `utils/streaming_frames.py` | Lightweight metadata records + dataset-specific ordered sources |
| `FrameScheduler` | `utils/stream_scheduler.py` | Controls when frames are released (deterministic or wall-clock) |
| `StreamingScene` | `scene/streaming_scene.py` | Holds arrived cameras, keyframe window, replay buffer; builds init point cloud |
| Training loop | `train_streaming.py` | Full Phase 1+2 loop; dispatched from `training()` in `train.py` |
| `add_points_as_gaussians` | `scene/gsplat_model.py`, `scene/gaussian_model.py` | Appends new Gaussians and extends optimizer state in-place |

**Supported datasets:** generic `RGBDSequence` (frames.jsonl + intrinsics.json), TUM RGB-D, ScanNet. Auto-detected from `--source_path` layout.

### Phase 1 — Ordered windowed training
Gaussian model is bootstrapped from only `--streaming_initial_frames` frames. The training loop samples cameras from the recent `--streaming_keyframe_window` cameras plus an occasional (`--streaming_global_replay_ratio`) older replay frame. New frames arrive every `--streaming_steps_per_frame` iterations.

### Phase 2 — Incremental depth insertion
Each arriving RGB-D frame backprojects its depth to world-space points, voxel-downsamples, removes already-covered regions, and appends up to `--streaming_max_new_gaussians_per_frame` new Gaussians. Controlled by `--streaming_insert_from_depth` (default on).

### Key streaming parameters

| Parameter | Default | Notes |
|---|---|---|
| `--streaming_replay` | `False` | Enable streaming mode |
| `--streaming_steps_per_frame` | `50` | Training iterations between frame releases |
| `--streaming_initial_frames` | `5` | Frames used for bootstrap init |
| `--streaming_keyframe_window` | `8` | Recent cameras for local training |
| `--streaming_replay_buffer` | `32` | Ring buffer size for older frames |
| `--streaming_global_replay_ratio` | `0.1` | Fraction of steps from replay buffer |
| `--streaming_insert_from_depth` | `True` | Phase 2 incremental insertion |
| `--streaming_insert_voxel_size` | `0.02` | Voxel grid for new-point dedup |
| `--streaming_max_new_gaussians_per_frame` | `2000` | Cap per-frame insertion |
| `--streaming_mcmc_local_only` | `True` | Restrict MCMC noise/reloc to visible set |
| `--streaming_global_maintenance_interval` | `500` | Iterations between full MCMC sweeps |
| `--streaming_wallclock` | `False` | Real-time frame pacing (default: deterministic) |

### Example run

```bash
docker compose run --rm \
  -v /path/to/tum_dataset:/data/tum \
  train python train.py \
    -s /data/tum/rgbd_desk \
    -m output/tum_streaming \
    --streaming_replay \
    --streaming_steps_per_frame 50 \
    --streaming_initial_frames 5 \
    --cap_max 200000 \
    --iterations 30000
```

## Output Structure

Each run writes to `output/<model_path>/`:
- `train.log` — stdout/stderr tee
- `cfg_args`, `run_args` — argument snapshots
- `cameras.json` — camera list
- `point_cloud/iteration_N/point_cloud.ply` — saved Gaussian PLY
- `chkpntN.pth` — optimizer checkpoints
- `web_viewer_cache/` — media cache for live viewer
