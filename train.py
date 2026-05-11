#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import json
import math
import time
import warnings
import threading
import queue
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
try:
    from gaussian_renderer import network_gui
except Exception:
    network_gui = None

# Live web viewer (optional; requires fastapi+uvicorn+websockets)
try:
    from gaussian_renderer import web_viewer as _web_viewer_mod
except Exception:
    _web_viewer_mod = None

import sys
from scene import Scene, GaussianModel, GsplatGaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.mcmc_schedule import MCMCScheduleConfig, get_mcmc_schedule
from utils.energy_mcmc import (
    compute_effective_count_loss,
    compute_opacity_entropy_loss,
    compute_gaussian_utility,
)
from utils.compiled_kernels import configure_torch_compile, set_compile_iteration
from utils.geometry_metrics import (
    update_visibility_ema,
    compute_geometry_dashboard,
)
from utils.taming_3dgs import (
    compute_edge_map,
    compute_taming_scores,
    get_taming_budget,
    get_taming_count_array,
    get_taming_score_weights,
    sample_taming_cameras,
)
from utils.web_viewer_media import (
    MultiCameraVideoRecorder,
    ScenePlyCacheWriter,
    parse_video_camera_indices,
)
from utils.strategies import make_mcmc_strategy
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

warnings.filterwarnings("once", message=".*HIPBLAS_STATUS_NOT_SUPPORTED.*")

class TeeStream:
    def __init__(self, primary, log_file):
        self.primary = primary
        self.log_file = log_file

    def write(self, text):
        self.primary.write(text)
        self.log_file.write(text)

    def flush(self):
        self.primary.flush()
        self.log_file.flush()


def ensure_model_path(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    os.makedirs(args.model_path, exist_ok=True)


def start_output_log(args):
    if getattr(args, "_run_log_started", False):
        return
    ensure_model_path(args)
    log_path = os.path.join(args.model_path, "train.log")
    log_file = open(log_path, "a", buffering=1)
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)
    args._run_log_started = True
    args.run_log_path = log_path
    # Keep the file alive for the process lifetime.
    args._run_log_file = log_file


class AsyncSaveWorker:
    """Offloads checkpoint & PLY file writes to a background thread.

    The main loop must pass CPU-captured snapshots (no GPU data sharing) to
    avoid data races with concurrent ``optimizer.step()``.
    """

    def __init__(self, maxsize=2):
        self._queue = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            task = self._queue.get()
            if task is None:
                break
            try:
                task()
            except Exception:
                import traceback
                traceback.print_exc()
            finally:
                self._queue.task_done()

    def enqueue(self, fn):
        """Enqueue a save task (callable with no args). Blocks if queue full."""
        self._queue.put(fn)

    def shutdown(self):
        """Drain queued saves and stop the background thread."""
        self._queue.put(None)
        self._thread.join(timeout=120)


def _capture_checkpoint(gaussians):
    """Snapshot gaussian state as CPU tensors for safe async checkpoint saving."""
    state = gaussians.capture()
    if isinstance(state, dict):
        return {k: _cpu_deep(v) for k, v in state.items()}
    # Legacy layout — state is a tuple of tensors + misc scalars
    return tuple(
        v.detach().cpu() if isinstance(v, torch.Tensor) else v
        for v in state
    )


def _cpu_deep(v):
    """Recursively move tensors to CPU; leave non-tensors as-is."""
    if isinstance(v, torch.Tensor):
        return v.detach().cpu()
    elif isinstance(v, dict):
        return {k: _cpu_deep(val) for k, val in v.items()}
    elif isinstance(v, (list, tuple)):
        return type(v)(_cpu_deep(x) for x in v)
    return v


def _snapshot_gaussians_for_ply(gaussians):
    """Capture gaussian parameters as CPU tensors (safe, no GPU sharing).

    Called in the training loop before handing off to the background save
    worker, avoiding data races with concurrent ``optimizer.step()``.
    """
    from collections import OrderedDict

    if hasattr(gaussians, "params") and isinstance(gaussians.params, (dict, OrderedDict)):
        # gsplat layout: params["opacities"] is (N,) → unsqueeze to (N, 1)
        params = gaussians.params
        return {
            "attrs": gaussians.construct_list_of_attributes(),
            "means": params["means"].detach().contiguous().cpu(),
            "sh0": params["sh0"].detach().contiguous().cpu(),
            "shN": params["shN"].detach().contiguous().cpu(),
            "opacities": params["opacities"].detach().unsqueeze(-1).contiguous().cpu(),
            "scales": params["scales"].detach().contiguous().cpu(),
            "quats": params["quats"].detach().contiguous().cpu(),
        }
    else:
        # legacy layout: _opacity is already (N, 1)
        return {
            "attrs": gaussians.construct_list_of_attributes(),
            "means": gaussians._xyz.detach().contiguous().cpu(),
            "sh0": gaussians._features_dc.detach().contiguous().cpu(),
            "shN": gaussians._features_rest.detach().contiguous().cpu(),
            "opacities": gaussians._opacity.detach().contiguous().cpu(),
            "scales": gaussians._scaling.detach().contiguous().cpu(),
            "quats": gaussians._rotation.detach().contiguous().cpu(),
        }


def _write_ply_from_snapshot(snap, out_path):
    """Build and write a PLY file from a CPU snapshot (worker thread)."""
    from plyfile import PlyData, PlyElement
    import numpy as np

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    xyz = snap["means"].numpy()
    normals = np.zeros_like(xyz)
    f_dc = snap["sh0"].transpose(1, 2).flatten(start_dim=1).contiguous().numpy()
    f_rest = snap["shN"].transpose(1, 2).flatten(start_dim=1).contiguous().numpy()
    opacities = snap["opacities"].numpy()
    scale = snap["scales"].numpy()
    rotation = snap["quats"].numpy()

    dtype_full = [(attr, "f4") for attr in snap["attrs"]]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, "vertex")]).write(out_path)


def public_namespace(args):
    return Namespace(**{k: v for k, v in vars(args).items() if not k.startswith("_")})


def apply_parallelism_profile(args):
    profile = getattr(args, "parallelism_profile", "off").lower()
    if profile == "safe":
        args.optimizer_type = "selective_adam"
        args.gsplat_sparse_grad = True
        args.sh_update_interval = 16
    elif profile != "off":
        raise ValueError(
            f"Unsupported --parallelism_profile '{profile}'. Expected 'off' or 'safe'."
        )

    args.optimizer_type = getattr(args, "optimizer_type", "adam").lower()
    if args.optimizer_type == "default":
        args.optimizer_type = "adam"
    if args.optimizer_type not in {"adam", "selective_adam"}:
        raise ValueError(
            f"Unsupported --optimizer_type '{args.optimizer_type}'. "
            "Expected 'adam' or 'selective_adam'."
        )
    if args.optimizer_type == "selective_adam":
        args.gsplat_sparse_grad = True

    args.sparse_policy = getattr(args, "sparse_policy", "active_set").lower()
    if args.sparse_policy != "active_set":
        raise ValueError(
            f"Unsupported --sparse_policy '{args.sparse_policy}'. Expected 'active_set'."
        )

    args.sh_backend = getattr(args, "sh_backend", "python").lower()
    if args.sh_backend not in {"python", "compiled_python", "gsplat"}:
        raise ValueError(
            f"Unsupported --sh_backend '{args.sh_backend}'. "
            "Expected 'python', 'compiled_python', or 'gsplat'."
        )

    args.model_layout = getattr(args, "model_layout", "gsplat").lower()
    if args.model_layout not in {"gsplat", "legacy"}:
        raise ValueError(
            f"Unsupported --model_layout '{args.model_layout}'. Expected 'gsplat' or 'legacy'."
        )
    args.densification_strategy = getattr(args, "densification_strategy", "gsplat_energy_mcmc").lower()
    gsplat_strategies = {"gsplat_mcmc", "gsplat_energy_mcmc"}
    if args.model_layout == "gsplat" and args.densification_strategy not in gsplat_strategies:
        raise ValueError(
            "--model_layout gsplat currently supports only --densification_strategy "
            "gsplat_mcmc or gsplat_energy_mcmc. "
            "Use --model_layout legacy for mcmc, hybrid, or taming."
        )
    if args.model_layout == "legacy" and args.densification_strategy in gsplat_strategies:
        raise ValueError(
            f"--densification_strategy {args.densification_strategy} requires --model_layout gsplat."
        )

    args.sh_update_interval = max(1, int(getattr(args, "sh_update_interval", 1)))
    args.parallelism_profile = profile


def start_benchmark_log(args):
    benchmark_dir = getattr(args, "benchmark_dir", "")
    if not benchmark_dir:
        return
    if benchmark_dir in {"model", "model_path"}:
        benchmark_dir = args.model_path
    os.makedirs(benchmark_dir, exist_ok=True)
    args.benchmark_dir = benchmark_dir
    args._benchmark_file = open(
        os.path.join(benchmark_dir, "timings.jsonl"),
        "a",
        buffering=1,
    )


def begin_stage_timer(sync=False):
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    stage_times = {}
    start = time.perf_counter()
    last = start

    def mark(name):
        nonlocal last
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        stage_times[name] = stage_times.get(name, 0.0) + now - last
        last = now

    def finish():
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        stage_times["total_wall"] = time.perf_counter() - start

    return stage_times, mark, finish


def log_stage_times(tb_writer, benchmark_file, iteration, stage_times, args, scene, densification_strategy, num_visible=0):
    scalar_log_interval = max(1, int(getattr(args, "scalar_log_interval", 10)))
    if tb_writer and (iteration % scalar_log_interval == 0 or iteration == getattr(args, "iterations", iteration)):
        for name, seconds in stage_times.items():
            tb_writer.add_scalar(f"timing/{name}_ms", seconds * 1000.0, iteration)
        tb_writer.add_scalar("parallelism/gsplat_sparse_grad", int(getattr(args, "gsplat_sparse_grad", False)), iteration)
        tb_writer.add_scalar("parallelism/sh_update_interval", int(getattr(args, "sh_update_interval", 1)), iteration)
        tb_writer.add_scalar("parallelism/selective_adam", int(getattr(args, "optimizer_type", "adam") == "selective_adam"), iteration)

    if benchmark_file is not None:
        row = {
            "iteration": iteration,
            "model_path": getattr(args, "model_path", ""),
            "strategy": densification_strategy,
            "optimizer_type": getattr(args, "optimizer_type", "adam"),
            "parallelism_profile": getattr(args, "parallelism_profile", "off"),
            "gsplat_sparse_grad": bool(getattr(args, "gsplat_sparse_grad", False)),
            "sh_update_interval": int(getattr(args, "sh_update_interval", 1)),
            "active_sh_degree": int(scene.gaussians.active_sh_degree),
            "num_gaussians": int(scene.gaussians.get_xyz.shape[0]),
            "git_branch": getattr(args, "git_branch", "unknown"),
            "git_commit": getattr(args, "git_commit", "unknown"),
            "timing_ms": {name: seconds * 1000.0 for name, seconds in stage_times.items()},
        }
        if num_visible > 0:
            row["num_visible"] = num_visible
            row["active_fraction"] = num_visible / max(row["num_gaussians"], 1)
        benchmark_file.write(json.dumps(row) + "\n")


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, sh_degree_schedule, run_args=None):
    if run_args is None:
        raise ValueError("training() requires run_args so feature flags are explicit and non-global.")
    args = run_args
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, run_args=run_args)
    model_layout = getattr(args, "model_layout", getattr(dataset, "model_layout", "gsplat")).lower()
    model_cls = GsplatGaussianModel if model_layout == "gsplat" else GaussianModel
    gaussians = model_cls(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    save_worker = AsyncSaveWorker()

    # Preload camera tensors to GPU to avoid per-render DeviceCopy overhead.
    from scene.cameras import prepare_camera_for_render
    for cam in scene.getTrainCameras():
        prepare_camera_for_render(cam, device="cuda")
    for cam in scene.getTestCameras():
        prepare_camera_for_render(cam, device="cuda")
    mcmc_cfg = MCMCScheduleConfig(
        start_iter=opt.densify_from_iter,
        stop_growth_iter=getattr(opt, "mcmc_stop_growth_iter", 12_000),
        stop_reloc_iter=opt.densify_until_iter,
        growth_factor_start=getattr(opt, "mcmc_growth_factor_start", 1.05),
        growth_factor_min=getattr(opt, "mcmc_growth_factor_min", 1.002),
        growth_factor_tau=getattr(opt, "mcmc_growth_factor_tau", 0.35),
        grow_interval_min=getattr(opt, "mcmc_grow_interval_min", 100),
        grow_interval_max=getattr(opt, "mcmc_grow_interval_max", 2000),
        grow_tau=getattr(opt, "mcmc_grow_tau", 0.35),
        relocate_interval_min=getattr(opt, "mcmc_relocate_interval_min", 50),
        relocate_interval_max=getattr(opt, "mcmc_relocate_interval_max", 500),
        relocate_tau=getattr(opt, "mcmc_relocate_tau", 0.65),
        cap_growth_power=getattr(opt, "mcmc_cap_growth_power", 2.0),
        cap_interval_strength=getattr(opt, "mcmc_cap_interval_strength", 4.0),
        cap_interval_power=getattr(opt, "mcmc_cap_interval_power", 2.0),
        cap_stop_ratio=getattr(opt, "mcmc_cap_stop_ratio", 0.98),
        dead_opacity_start=getattr(opt, "mcmc_dead_opacity_start", 0.003),
        dead_opacity_end=getattr(opt, "mcmc_dead_opacity_end", 0.010),
        dead_opacity_power=getattr(opt, "mcmc_dead_opacity_power", 1.5),
        use_target_deficit=getattr(opt, "mcmc_use_target_deficit", False),
        target_splat_end=getattr(opt, "mcmc_target_splat_end", 150000),
        target_q_start=getattr(opt, "mcmc_target_q_start", 0.05),
        target_q_end=getattr(opt, "mcmc_target_q_end", 0.85),
        target_tau=getattr(opt, "mcmc_target_tau", 0.45),
    )
    densification_strategy = getattr(opt, "densification_strategy", "gsplat_energy_mcmc").lower()
    valid_strategies = {"mcmc", "taming", "hybrid", "gsplat_mcmc", "gsplat_energy_mcmc"}
    if densification_strategy not in valid_strategies:
        raise ValueError(
            f"Unsupported --densification_strategy '{densification_strategy}'. "
            f"Expected one of: {sorted(valid_strategies)}"
        )
    if dataset.cap_max == -1 and not (
        densification_strategy == "taming" and getattr(args, "taming_budget", -1.0) > 0
    ):
        print("Please specify the maximum number of Gaussians using --cap_max.")
        exit()
    pipe.gsplat_sparse_grad = bool(getattr(opt, "gsplat_sparse_grad", False))
    mcmc_strategy = make_mcmc_strategy(densification_strategy, gaussians=gaussians, args=args)
    optimizer_type = getattr(opt, "optimizer_type", "adam").lower()
    sh_update_interval = max(1, int(getattr(opt, "sh_update_interval", 1)))
    benchmark_file = getattr(args, "_benchmark_file", None)
    benchmark_sync = benchmark_file is not None
    print(
        f"[parallelism] profile={getattr(opt, 'parallelism_profile', 'off')} "
        f"model_layout={model_layout} "
        f"optimizer={optimizer_type} sparse_grad={pipe.gsplat_sparse_grad} "
        f"sparse_policy={getattr(args, 'sparse_policy', 'active_set')} "
        f"sh_backend={getattr(pipe, 'sh_backend', 'python')} "
        f"sh_update_interval={sh_update_interval}",
        flush=True,
    )
    allow_dense_grads = bool(getattr(args, "selective_adam_allow_dense_grads", False))
    sparse_active_set = optimizer_type == "selective_adam" and not allow_dense_grads
    if sparse_active_set:
        print(
            "[sparse] active-set training enabled: sparse grads preserved, "
            "active-set regularizers, active-set energy losses, active-set MCMC noise",
            flush=True,
        )
    elif optimizer_type == "selective_adam" and allow_dense_grads:
        print("[sparse] selective_adam with dense grads fallback (allow_dense_grads=True)", flush=True)

    # Growth-aware compile activation: never compile while N is still changing.
    # MCMC stops growth at mcmc_stop_growth_iter; taming/hybrid at densify_until_iter.
    if getattr(args, "compile_mode", "off") != "off":
        if densification_strategy in {"mcmc", "gsplat_mcmc", "gsplat_energy_mcmc"}:
            growth_stop = int(getattr(opt, "mcmc_stop_growth_iter", 12000))
        else:
            growth_stop = int(getattr(opt, "densify_until_iter", 25000))
        margin = int(getattr(args, "compile_growth_margin", 500))
        old_after = int(getattr(args, "compile_after_iter", 0))
        args.compile_after_iter = max(old_after, growth_stop + margin)
        print(
            f"[compiled-kernels] growth-aware compile_after_iter={args.compile_after_iter} "
            f"(strategy={densification_strategy}, growth_stop={growth_stop}, margin={margin})",
            flush=True,
        )

    configure_torch_compile(args)
    taming_enabled = densification_strategy in {"taming", "hybrid"}
    use_energy_mcmc = args.energy_mcmc and densification_strategy in {"mcmc", "hybrid", "gsplat_energy_mcmc"}
    taming_weights = get_taming_score_weights(args) if taming_enabled else None
    taming_counts = None
    taming_densify_step = 0
    taming_stats_logged = False
    taming_edge_maps = {}
    strategy_log_interval = max(1, getattr(args, "strategy_log_interval", 500))
    mcmc_control_log_interval = max(1, getattr(args, "mcmc_control_log_interval", 500))

    def should_log_strategy(iteration):
        return iteration % strategy_log_interval == 0 or iteration == opt.iterations

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)
    mcmc_strategy.initialize_state(gaussians=gaussians, args=args)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_disabled = getattr(args, "disable_progress_bar", False) or getattr(args, "quiet", False)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", disable=progress_disabled)
    progress_log_last_iter = first_iter
    progress_log_last_time = time.perf_counter()
    first_iter += 1

    if taming_enabled:
        for cam in scene.getTrainCameras():
            taming_edge_maps[id(cam)] = compute_edge_map(cam.original_image).detach().cpu()

    # Persistent web viewer and media cache. The viewer may already be running
    # from __main__ so users can connect while scene loading is still in flight.
    web_viewer = _web_viewer_mod.get_web_viewer() if _web_viewer_mod is not None else None
    web_viewer_enabled = bool(getattr(args, "web_viewer_enabled", True))
    web_viewer_port = int(getattr(args, "web_viewer_port", 6010))
    web_viewer_image_interval = max(1, getattr(args, "web_viewer_image_interval", 100))
    web_viewer_fixed_camera = bool(getattr(args, "web_viewer_fixed_camera", False))
    web_viewer_cache_dir = getattr(args, "web_viewer_cache_dir", "") or os.path.join(scene.model_path, "web_viewer_cache")
    web_viewer_backend = str(getattr(args, "web_viewer_backend", "process")).lower()
    if (not web_viewer_enabled) or web_viewer_port <= 0:
        web_viewer_backend = "off"
    if web_viewer_backend != "off" and _web_viewer_mod is not None and web_viewer is None:
        _train_cams = scene.getTrainCameras()
        web_viewer = _web_viewer_mod.start_web_viewer(
            port=web_viewer_port,
            host=getattr(args, "web_viewer_host", "0.0.0.0"),
            image_interval=web_viewer_image_interval,
            total_cams=len(_train_cams),
            viewer_cam_idx=0,
            backend=web_viewer_backend,
            cache_dir=web_viewer_cache_dir,
            model_path=scene.model_path,
        )
    elif web_viewer is not None:
        web_viewer.update_camera_info(len(scene.getTrainCameras()), viewer_cam_idx=0)
    elif web_viewer_backend != "off":
        print("[web-viewer] Disabled - web viewer module failed to import.", flush=True)

    scene_cache_writer = None
    scene_cache_interval = max(0, int(getattr(args, "web_viewer_scene_cache_interval", 500)))
    scene_cache_keep = max(0, int(getattr(args, "web_viewer_scene_cache_keep", 3)))
    if web_viewer_enabled and scene_cache_interval > 0:
        scene_cache_writer = ScenePlyCacheWriter(web_viewer_cache_dir, keep=scene_cache_keep, viewer=web_viewer)

    video_recorder = None
    video_camera_indices = []
    if bool(getattr(args, "record_video", False)) and _web_viewer_mod is not None:
        video_camera_indices = parse_video_camera_indices(
            getattr(args, "record_video_cameras", ""),
            len(scene.getTrainCameras()),
        )
        if video_camera_indices:
            video_recorder = MultiCameraVideoRecorder(
                cache_dir=web_viewer_cache_dir,
                camera_indices=video_camera_indices,
                fps=max(1, int(getattr(args, "record_video_fps", 30))),
                crf=max(0, int(getattr(args, "record_video_crf", 23))),
                preset=str(getattr(args, "record_video_preset", "veryfast")),
                viewer=web_viewer,
            )
            print(
                f"[web-viewer-video] recording cameras {video_camera_indices} "
                f"every {max(1, int(getattr(args, 'record_video_interval', 100)))} iterations",
                flush=True,
            )
    elif bool(getattr(args, "record_video", False)):
        print("[web-viewer-video] Disabled - web viewer image helpers failed to import.", flush=True)

    _profile_path = getattr(run_args, "profile", None) if run_args else None
    if _profile_path:
        _prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=200, repeat=1),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        _prof.__enter__()
        print(f"[profile] Profiling enabled → {_profile_path}", flush=True)
    else:
        _prof = None

    # Persistent CUDA stream for async GPU→CPU viewer image copies.
    # Keeps the copy from blocking the main training stream.
    _viewer_stream = torch.cuda.Stream() if web_viewer is not None else None
    _video_stream = torch.cuda.Stream() if video_recorder is not None else None

    for iteration in range(first_iter, opt.iterations + 1):
        set_compile_iteration(iteration)
        stage_times, mark_stage, finish_stage = begin_stage_timer(sync=benchmark_sync)
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        xyz_lr = gaussians.update_learning_rate(iteration)

        # Increase SH degree at configured schedule (default: 1000, 2000, 3000)
        if iteration in sh_degree_schedule:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        update_sh_rest = (
            gaussians.active_sh_degree == 0
            or sh_update_interval <= 1
            or iteration % sh_update_interval == 0
        )
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            update_sh_rest=update_sh_rest,
        )
        image = render_pkg["render"]
        # Cache a copy on CPU for the live web viewer.
        # Every image_interval iterations, render from the viewer's chosen camera
        # so the view stays stable (the image shows the same camera every time).
        # Uses a dedicated CUDA stream + pinned memory for async GPU→CPU copy so
        # the main training stream is not blocked by the transfer.
        _viewer_image_arr = None
        _viewer_render_time_ms = 0.0
        _viewer_pinned_buf = None  # non-None when async copy is in-flight
        _video_frames = []
        if web_viewer is not None and web_viewer.image_interval > 0 and iteration % web_viewer_image_interval == 0:
            _viewer_cam = scene.getTrainCameras()[web_viewer.viewer_cam_idx]
            if viewpoint_cam is _viewer_cam or not web_viewer_fixed_camera:
                # Training happened to render from the viewer's camera — reuse it.
                # By default the viewer shows the current training-camera render
                # to avoid a second rasterization pass during training.
                # Record a CUDA event so the viewer stream waits for the render to complete.
                _render_ready = torch.cuda.Event()
                _render_ready.record()
                _viewer_pinned_buf = _web_viewer_mod.encode_render_image_async_start(
                    image, _viewer_stream, wait_event=_render_ready,
                )
            else:
                # Do an extra render pass from the viewer's fixed camera
                _viewer_render_start = torch.cuda.Event(enable_timing=True)
                _viewer_render_end = torch.cuda.Event(enable_timing=True)
                _viewer_render_start.record()
                with torch.no_grad():
                    _viewer_pkg = render(_viewer_cam, gaussians, pipe, bg, update_sh_rest=update_sh_rest)
                    # Record event on default stream after render is queued
                    _render_ready = torch.cuda.Event()
                    _render_ready.record()
                    # Start async GPU→CPU copy on viewer stream (waits for render first)
                    _viewer_pinned_buf = _web_viewer_mod.encode_render_image_async_start(
                        _viewer_pkg["render"], _viewer_stream, wait_event=_render_ready,
                    )
                _viewer_render_end.record()
                _viewer_render_end.synchronize()
                _viewer_render_time_ms = _viewer_render_start.elapsed_time(_viewer_render_end)
        if video_recorder is not None and iteration % max(1, int(getattr(args, "record_video_interval", 100))) == 0:
            for _video_cam_idx in video_camera_indices:
                _video_cam = scene.getTrainCameras()[_video_cam_idx]
                if viewpoint_cam is _video_cam:
                    _render_ready = torch.cuda.Event()
                    _render_ready.record()
                    _video_buf = _web_viewer_mod.encode_render_image_async_start(
                        image, _video_stream, wait_event=_render_ready, reuse_buffer=False,
                    )
                else:
                    with torch.no_grad():
                        _video_pkg = render(_video_cam, gaussians, pipe, bg, update_sh_rest=update_sh_rest)
                        _render_ready = torch.cuda.Event()
                        _render_ready.record()
                        _video_buf = _web_viewer_mod.encode_render_image_async_start(
                            _video_pkg["render"], _video_stream, wait_event=_render_ready, reuse_buffer=False,
                        )
                _video_stream.synchronize()
                _video_frames.append((_video_cam_idx, _web_viewer_mod.encode_render_image_finish(_video_buf).copy()))
        mark_stage("forward")

        # Loss — capture intermediate values for logging (no recompute)
        gt_image = viewpoint_cam.original_image
        Ll1 = l1_loss(image, gt_image)
        _ssim_val = ssim(image, gt_image)  # capture for logging (already computed for loss)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - _ssim_val)

        # Active-set regularization: in sparse mode, only regularize visible Gaussians.
        # This avoids dense all-Gaussian gradient traffic that defeats sparse training.
        from utils.compiled_kernels import active_reg_core
        if sparse_active_set:
            _active_reg = render_pkg["visibility_filter"].detach()
            _reg_loss = active_reg_core(
                gaussians.get_opacity[_active_reg],
                gaussians.get_scaling[_active_reg],
                w_opacity=args.opacity_reg,
                w_scale=args.scale_reg,
            )
        else:
            _reg_loss = active_reg_core(
                gaussians.get_opacity,
                gaussians.get_scaling,
                w_opacity=args.opacity_reg,
                w_scale=args.scale_reg,
            )
        loss = loss + _reg_loss

        # Energy-guided MCMC losses (Stage A: soft loss steering)
        # Backward compatibility: only active when --energy_mcmc is enabled (default)
        if use_energy_mcmc:
            # In sparse mode, run differentiable energy losses only periodically as
            # global maintenance steps. Normal sparse steps skip them to preserve
            # the active-set profile (dense global opacity gradients defeat the
            # selective-optimizer benefit). When they do run, they operate on the
            # FULL opacity tensor (dense, all Gaussians) since they are global
            # population-control signals, not per-view losses.
            _energy_global_interval = getattr(args, "sparse_energy_global_interval", 500)
            _run_energy = True
            if sparse_active_set and _energy_global_interval > 0:
                _run_energy = iteration % _energy_global_interval == 0

            if _run_energy:
                # Effective count loss: steer toward target population curve
                L_eff, N_eff, N_target = compute_effective_count_loss(
                    gaussians._opacity,  # raw, full (global maintenance)
                    iteration=iteration,
                    cap_max=args.cap_max,
                    max_iterations=opt.iterations,
                    target_splat_end=getattr(args, "mcmc_target_splat_end", None),
                    q_start=getattr(args, "mcmc_target_q_start", 0.05),
                    q_end=getattr(args, "mcmc_target_q_end", 0.85),
                    tau_N=getattr(args, "mcmc_target_tau", 0.45),
                )
                loss = loss + args.lambda_eff_count * L_eff

                # Opacity entropy loss: encourage decisive alive/dead opacities
                L_entropy = compute_opacity_entropy_loss(gaussians._opacity)  # raw, full
                loss = loss + args.lambda_opacity_entropy * L_entropy
            else:
                # Normal sparse step: skip differentiable energy losses
                L_eff = None
                L_entropy = None
                N_eff = None
                N_target = None
        else:
            L_eff = None
            L_entropy = None
            N_eff = None
            N_target = None

        # Per-iteration TB logging of loss components (captured values, no recompute)
        scalar_log_interval = max(1, int(getattr(args, "scalar_log_interval", 10)))
        should_log_scalars = tb_writer and (iteration % scalar_log_interval == 0 or iteration == opt.iterations)
        if should_log_scalars:
            tb_writer.add_scalar('train_loss_patches/ssim_loss', (1.0 - _ssim_val).item(), iteration)
            # Recompute separate reg terms for logging (no graph cost — not in backward path)
            with torch.no_grad():
                if sparse_active_set:
                    _active_reg = render_pkg["visibility_filter"].detach()
                    _opacity_reg = args.opacity_reg * torch.abs(gaussians.get_opacity[_active_reg]).mean()
                    _scale_reg = args.scale_reg * torch.abs(gaussians.get_scaling[_active_reg]).mean()
                else:
                    _opacity_reg = args.opacity_reg * torch.abs(gaussians.get_opacity).mean()
                    _scale_reg = args.scale_reg * torch.abs(gaussians.get_scaling).mean()
            tb_writer.add_scalar('train_loss_patches/opacity_reg_term', _opacity_reg.item(), iteration)
            tb_writer.add_scalar('train_loss_patches/scale_reg_term', _scale_reg.item(), iteration)
            if use_energy_mcmc and L_eff is not None:
                tb_writer.add_scalar('train_loss_patches/eff_count_loss', args.lambda_eff_count * L_eff.item(), iteration)
                tb_writer.add_scalar('train_loss_patches/opacity_entropy_loss', args.lambda_opacity_entropy * L_entropy.item(), iteration)
                tb_writer.add_scalar('train_count/effective_count_N', N_eff.item(), iteration)
                tb_writer.add_scalar('train_count/target_count_N', int(N_target), iteration)

        mark_stage("loss")
        mcmc_strategy.step_pre_backward(
            gaussians=gaussians,
            args=args,
            iteration=iteration,
            render_pkg=render_pkg,
            loss=loss,
        )
        loss.backward()
        # Sparse safety: zero invisible rows in strided grads so SelectiveAdam
        # never sees stale non-zero values for invisible Gaussians.
        # (Active-set regularizers / energy losses produce strided grads;
        #  PyTorch autograd converts sparse + strided → strided in accumulation.)
        if sparse_active_set and getattr(args, "selective_adam_zero_invisible_grads", True):
            _mask_grad = render_pkg["visibility_filter"].detach()
            for group in gaussians.optimizer.param_groups:
                p = group["params"][0]
                if p.grad is not None and getattr(p.grad, "layout", torch.strided) == torch.strided:
                    p.grad[~_mask_grad] = 0.0
        # One-line gradient-layout snapshot at first iteration
        if sparse_active_set and iteration == first_iter:
            _layouts = {
                group["name"]: str(getattr(group["params"][0].grad, "layout", "None"))
                for group in gaussians.optimizer.param_groups
            }
            print(f"[sparse] first-iter grad layouts: {_layouts}", flush=True)
        mark_stage("backward")

        # Update visibility EMA for geometry dashboard
        with torch.no_grad():
            update_visibility_ema(gaussians, render_pkg["is_used"])
            if taming_enabled and iteration < opt.densify_until_iter:
                visibility_filter = render_pkg["visibility_filter"]
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    render_pkg["radii"][visibility_filter].to(gaussians.max_radii2D.dtype),
                )

        iter_end.record()

        # Compute utility between backward and step (reads gradient norms)
        if use_energy_mcmc:
            utility = compute_gaussian_utility(
                gaussians=gaussians,
                render_pkg=render_pkg,
                iteration=iteration,
                w_alpha=args.energy_w_alpha,
                w_vis=args.energy_w_vis,
                w_grad=args.energy_w_grad,
                w_scale=args.energy_w_scale,
                w_dead=args.energy_w_dead,
                w_support=getattr(args, "energy_w_support", 2.0),
                beta_opacity=getattr(args, "energy_beta_opacity", 1.0),
                beta_scale=getattr(args, "energy_beta_scale", 0.5),
                alpha_dead=getattr(args, "energy_alpha_dead", 0.005),
            )

            # Temperature annealing for birth/death sampling
            u_temp = min(iteration / 30000.0, 1.0)
            tau_t = max(getattr(args, "energy_temp_tau", 0.4), 1e-6)
            denom_t = 1.0 - math.exp(-1.0 / tau_t)
            alpha_t = (1.0 - math.exp(-u_temp / tau_t)) / denom_t
            temperature = (
                args.energy_temp_min
                + (args.energy_temp_start - args.energy_temp_min)
                * (1.0 - alpha_t)
            )
            if should_log_scalars:
                tb_writer.add_scalar('mcmc/temperature', temperature, iteration)
        else:
            utility = None
            temperature = 1.0

        # Mask utility to active set: non-visible Gaussians keep only their
        # opacity-based score so MCMC decisions are driven by visible Gaussians.
        if sparse_active_set and utility is not None:
            _active_util = render_pkg["visibility_filter"].detach()
            utility[~_active_util] = (
                getattr(args, "energy_w_alpha", 1.0)
                * gaussians.get_opacity.squeeze(-1)[~_active_util]
            )
        mark_stage("utility")

        taming_scores = None
        run_taming_growth = (
            taming_enabled
            and opt.densify_from_iter < iteration < opt.densify_until_iter
            and iteration % max(1, getattr(args, "taming_score_interval", 0) or opt.densification_interval) == 0
        )
        if run_taming_growth:
            if taming_counts is None:
                taming_budget = get_taming_budget(args)
                taming_counts = get_taming_count_array(
                    start_count=gaussians.get_xyz.shape[0],
                    budget=taming_budget,
                    opt=args,
                    mode=getattr(args, "taming_budget_mode", "final_count"),
                )
                print(
                    f"[taming-budget] start={taming_counts[0]} "
                    f"final={taming_counts[-1]} "
                    f"steps={len(taming_counts) - 1} "
                    f"mode={getattr(args, 'taming_budget_mode', 'final_count')}",
                    flush=True,
                )

            if not taming_stats_logged:
                print(
                    "[taming-stats] requiring exact renderer accumulators; "
                    "scoring raises on approximation fallback",
                    flush=True,
                )
                taming_stats_logged = True

            camlist = sample_taming_cameras(scene.getTrainCameras(), getattr(args, "taming_cams", 3))
            edge_maps = [taming_edge_maps[id(cam)].to(device=gaussians.get_xyz.device) for cam in camlist]
            with torch.no_grad():
                taming_scores = compute_taming_scores(
                    scene=scene,
                    camlist=camlist,
                    edge_maps=edge_maps,
                    gaussians=gaussians,
                    pipe=pipe,
                    bg=bg,
                    weights=taming_weights,
                    opt=opt,
                )
        mark_stage("taming_scoring")

        # Optimizer step (must come BEFORE MCMC mutations)
        if iteration < opt.iterations:
            if optimizer_type == "selective_adam":
                visible = render_pkg["visibility_filter"].detach().to(dtype=torch.bool).contiguous()
                gaussians.prepare_selective_adam_step(allow_dense_grads=allow_dense_grads)
                mark_stage("selective_adam_prepare")
                gaussians.optimizer.step(visibility=visible)
                mark_stage("selective_adam_step")
            else:
                gaussians.optimizer.step()
            if pipe.gsplat_sparse_grad:
                if sparse_active_set:
                    gaussians.normalize_rotation_params(mask=visible)
                else:
                    gaussians.normalize_rotation_params()
                mark_stage("rotation_normalize")
            gaussians.optimizer.zero_grad(set_to_none=True)

            if densification_strategy in {"mcmc", "hybrid", "gsplat_mcmc", "gsplat_energy_mcmc"}:
                mcmc_strategy.inject_noise(
                    gaussians=gaussians,
                    args=args,
                    xyz_lr=xyz_lr,
                    visible=visible if optimizer_type == "selective_adam" else None,
                    sparse_active_set=sparse_active_set,
                    iteration=iteration,
                )
        mark_stage("optimizer")

        # Track active-set statistics
        _num_visible = int(render_pkg["visibility_filter"].sum().item())
        _active_fraction = _num_visible / max(gaussians.get_xyz.shape[0], 1)

        # Progress bar, logging, geometry dashboard
        with torch.no_grad():
            # Cache scalar values ONCE — all downstream code uses these (avoids redundant syncs)
            _Ll1_val = Ll1.item()
            _loss_val = loss.item()
            _iter_time_ms = iter_start.elapsed_time(iter_end)

            ema_loss_for_log = 0.4 * _loss_val + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            training_report(
                tb_writer,
                iteration,
                _Ll1_val,
                _loss_val,
                l1_loss,
                _iter_time_ms,
                testing_iterations,
                scene,
                render,
                (pipe, background),
                no_empty_cache=getattr(run_args, 'no_empty_cache', False),
                log_scalars=bool(should_log_scalars),
            )
            if progress_disabled and (iteration % 500 == 0 or iteration == opt.iterations):
                now = time.perf_counter()
                delta_iter = max(1, iteration - progress_log_last_iter)
                it_s = delta_iter / max(now - progress_log_last_time, 1e-9)
                print(
                    f"[progress] iter={iteration} loss={ema_loss_for_log:.7f} "
                    f"it/s={it_s:.2f} sh={gaussians.active_sh_degree}",
                    flush=True,
                )
                progress_log_last_iter = iteration
                progress_log_last_time = now
                if should_log_scalars:
                    tb_writer.add_scalar('iterations_per_sec', it_s, iteration)
                    tb_writer.add_scalar('total_points', gaussians.get_xyz.shape[0], iteration)
                    if optimizer_type == "selective_adam":
                        tb_writer.add_scalar('active_set/num_visible', _num_visible, iteration)
                        tb_writer.add_scalar('active_set/active_fraction', _active_fraction, iteration)
                        # Periodic grad-layout dashboard: track which parameters retain
                        # sparse COO layouts vs. dense strided after backward.
                        _sparse_count = 0
                        _none_count = 0
                        for group in gaussians.optimizer.param_groups:
                            name = group["name"]
                            p = group["params"][0]
                            g = p.grad
                            if g is None:
                                _none_count += 1
                                tb_writer.add_scalar(f"grad_layout/{name}", -1.0, iteration)
                            elif getattr(g, "layout", torch.strided) != torch.strided:
                                _sparse_count += 1
                                tb_writer.add_scalar(f"grad_layout/{name}", 1.0, iteration)
                            else:
                                tb_writer.add_scalar(f"grad_layout/{name}", 0.0, iteration)
                        tb_writer.add_scalar("grad_layout/sparse_count", _sparse_count, iteration)
                        tb_writer.add_scalar("grad_layout/none_count", _none_count, iteration)
                    tb_writer.add_scalar('train_loss_patches/psnr', psnr(image, gt_image).mean().item(), iteration)
            if (iteration in saving_iterations) or (args.save_interval > 0 and iteration % args.save_interval == 0):
                print("\n[ITER {}] Saving Gaussians".format(iteration), flush=True)
                _snap = _snapshot_gaussians_for_ply(gaussians)
                _ply_path = os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration), "point_cloud.ply")
                save_worker.enqueue(lambda s=_snap, p=_ply_path: _write_ply_from_snapshot(s, p))

            # Geometry failure dashboard — configurable to avoid sync collisions.
            geometry_log_interval = max(1, int(getattr(args, "geometry_log_interval", 500)))
            sfm_anchor_interval = max(1, int(getattr(args, "sfm_anchor_interval", 2000)))
            if iteration % geometry_log_interval == 0:
                with torch.no_grad():
                    # Load SfM points once, but only on the slower SfM-anchor cadence.
                    include_sfm_anchor = iteration % sfm_anchor_interval == 0
                    if include_sfm_anchor and not hasattr(scene, "_sfm_points"):
                        try:
                            import numpy as np
                            from scene.dataset_readers import fetchPly
                            sfm_pcd = fetchPly(os.path.join(dataset.source_path, "sparse", "0", "points3D.ply"))
                            scene._sfm_points = torch.tensor(np.asarray(sfm_pcd.points), dtype=torch.float32, device="cuda")
                        except Exception:
                            scene._sfm_points = None

                    geo = compute_geometry_dashboard(
                        gaussians,
                        sfm_points=getattr(scene, "_sfm_points", None) if include_sfm_anchor else None,
                    )
                    gaussians._last_geo = geo

                    for k, v in geo.items():
                        if should_log_scalars:
                            tb_writer.add_scalar(f"geometry/{k}", v, iteration)

                    print(
                        f"[geo] iter={iteration} "
                        f"N={int(geo['num_gaussians'])} "
                        f"LSOM={geo.get('low_support_opacity_mass', 0.0):.4f} "
                        f"OSF={geo.get('opacity_scale_floater_score', 0.0):.4f} "
                        f"mean_sup={geo.get('mean_support', 0.0):.4f}",
                        flush=True,
                    )
        mark_stage("reporting")

        # Log CUDA memory allocator stats every 50 iterations for debugging stutters
        if should_log_scalars and getattr(args, "log_memory", False) and torch.cuda.is_available():
            try:
                mem_stats = torch.cuda.memory_stats()
                tb_writer.add_scalar('memory/allocated_bytes', mem_stats.get('allocated_bytes.all.current', 0), iteration)
                tb_writer.add_scalar('memory/reserved_bytes', mem_stats.get('reserved_bytes.all.current', 0), iteration)
                tb_writer.add_scalar('memory/active_bytes', mem_stats.get('active_bytes.all.current', 0), iteration)
                tb_writer.add_scalar('memory/num_alloc_retries', mem_stats.get('num_alloc_retries', 0), iteration)
                tb_writer.add_scalar('memory/num_segments', mem_stats.get('num_segments', 0), iteration)
                tb_writer.add_scalar('memory/num_segments_reclaimed', mem_stats.get('num_segments_reclaimed', 0), iteration)
                tb_writer.add_scalar('memory/cached_events', mem_stats.get('cuda_events.max', 0), iteration)
                # Log # of GPU kernel launches in last iteration (from the profiler viewpoint)
                tb_writer.add_scalar('memory/oversize_allocations', mem_stats.get('oversize_allocations.current', 0), iteration)
            except Exception:
                pass  # memory stats may not be available on all ROCm versions

        # Push metrics and image to the live web viewer
        if web_viewer is not None:
            _it_s = 1000.0 / max(_iter_time_ms, 0.001) if _iter_time_ms > 0 else 0.0
            web_viewer.push_metrics({
                "iteration": iteration,
                "loss": _loss_val,
                "l1": _Ll1_val,
                "num_gaussians": gaussians.get_xyz.shape[0],
                "sh_degree": gaussians.active_sh_degree,
                "iter_time_ms": _iter_time_ms,
                "viewer_render_ms": _viewer_render_time_ms,
                "it_s": _it_s,
            })
            # Finalize async GPU→CPU copy (pinned buffer + separate stream avoids blocking main stream)
            if _viewer_pinned_buf is not None:
                _viewer_stream.synchronize()
                _viewer_image_arr = _web_viewer_mod.encode_render_image_finish(_viewer_pinned_buf)
                _viewer_pinned_buf = None
                if _viewer_image_arr is not None:
                    web_viewer.push_image(_viewer_image_arr, iteration)
        if video_recorder is not None and _video_frames:
            for _video_cam_idx, _video_arr in _video_frames:
                video_recorder.write(_video_cam_idx, _video_arr)

        # Compute MCMC schedule
        sched = get_mcmc_schedule(
            iteration=iteration,
            current_n=gaussians.get_xyz.shape[0],
            cap_max=args.cap_max,
            cfg=mcmc_cfg,
        )

        # Log schedule metrics to TensorBoard
        if should_log_scalars:
            tb_writer.add_scalar('mcmc/growth_factor', sched['growth_factor'], iteration)
            tb_writer.add_scalar('mcmc/capacity_ratio_rho', sched['rho'], iteration)
            tb_writer.add_scalar('mcmc/dead_opacity_threshold', sched['dead_opacity_threshold'], iteration)
            tb_writer.add_scalar('mcmc/relocate_interval', sched['relocate_interval'], iteration)
            tb_writer.add_scalar('mcmc/grow_interval', sched['grow_interval'], iteration)
            tb_writer.add_scalar('mcmc/allow_growth', int(sched['allow_growth']), iteration)
            tb_writer.add_scalar('mcmc/allow_relocation', int(sched['allow_relocation']), iteration)
            tb_writer.add_scalar('train_loss_patches/xyz_lr', xyz_lr, iteration)
            tb_writer.add_scalar('timing/viewer_render_ms', _viewer_render_time_ms, iteration)

        # Optional closed-loop: suppress growth if LSOM is rising
        if use_energy_mcmc and iteration > opt.densify_from_iter:
            geo = getattr(gaussians, "_last_geo", {})
            lsom = geo.get("low_support_opacity_mass", 0.0)
            if lsom > 0.15:
                # Too many unsupported splats: halve growth factor
                sched["growth_factor"] = max(1.0, sched["growth_factor"] * 0.5)
                if iteration % mcmc_control_log_interval == 0 or iteration == opt.iterations:
                    print(f"[mcmc-control] LSOM={lsom:.3f} > 0.15, suppressing growth", flush=True)
        mark_stage("schedule")

        # Strategy mutation phase. The adapter shape mirrors gsplat Strategy:
        # training owns loss/backward/optimizer, strategy owns population edits.
        mcmc_strategy.step_post_backward(
            gaussians=gaussians,
            args=args,
            sched=sched,
            iteration=iteration,
            utility=utility,
            temperature=temperature,
            use_energy_mcmc=use_energy_mcmc,
            tb_writer=tb_writer,
            should_log_strategy=should_log_strategy,
            render_pkg=render_pkg,
            lr=xyz_lr,
        )

        if taming_enabled and run_taming_growth and taming_scores is not None:
            with torch.no_grad():
                before = gaussians.get_xyz.shape[0]
                target_idx = min(taming_densify_step + 1, len(taming_counts) - 1)
                target_count = taming_counts[target_idx]
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                result = gaussians.densify_with_taming_scores(
                    scores=taming_scores,
                    target_count=target_count,
                    extent=scene.cameras_extent,
                    min_opacity=getattr(args, "taming_min_opacity", 0.005),
                    max_screen_size=size_threshold,
                    iteration=iteration,
                    prune_stop_iter=getattr(args, "taming_prune_stop_iter", 3200),
                    grad_threshold=opt.densify_grad_threshold,
                )
                taming_densify_step += 1
                after = gaussians.get_xyz.shape[0]

                if tb_writer:
                    tb_writer.add_scalar('mcmc/growth_delta_N', after - before, iteration)
                    tb_writer.add_scalar('taming/cloned_count', result['cloned'], iteration)
                    tb_writer.add_scalar('taming/split_count', result['split'], iteration)
                    tb_writer.add_scalar('taming/pruned_count', result['pruned'], iteration)
                    tb_writer.add_scalar('taming/target_count', target_count, iteration)

                if should_log_strategy(iteration):
                    print(
                        f"[{densification_strategy}-taming-grow] iter={iteration} "
                        f"target={target_count} "
                        f"clone={result['cloned']} split={result['split']} prune={result['pruned']} "
                        f"N={before}->{after}",
                        flush=True,
                    )
        mark_stage("mutation")

        if scene_cache_writer is not None and scene_cache_interval > 0 and iteration % scene_cache_interval == 0:
            _cache_start = time.perf_counter()
            scene_cache_writer.save(gaussians, iteration, final=False)
            stage_times["web_viewer_scene_cache"] = stage_times.get("web_viewer_scene_cache", 0.0) + (
                time.perf_counter() - _cache_start
            )

        if (iteration in checkpoint_iterations) or (args.checkpoint_interval > 0 and iteration % args.checkpoint_interval == 0):
            print("\n[ITER {}] Saving Checkpoint".format(iteration), flush=True)
            _chk_state = _capture_checkpoint(gaussians)
            _chk_iter = iteration
            _chk_path = scene.model_path + "/chkpnt" + str(iteration) + ".pth"
            save_worker.enqueue(lambda s=_chk_state, i=_chk_iter, p=_chk_path: torch.save((s, i), p))
        mark_stage("checkpoint")
        finish_stage()
        log_stage_times(tb_writer, benchmark_file, iteration, stage_times, args, scene, densification_strategy, num_visible=_num_visible)

        if _prof is not None:
            _prof.step()

    # --- Profiler export (after loop ends) ---
    if _prof is not None:
        _prof.__exit__(None, None, None)
        _prof.export_chrome_trace(_profile_path)
        print(f"[profile] Trace saved to {_profile_path}.", flush=True)
    if scene_cache_writer is not None:
        scene_cache_writer.save(gaussians, opt.iterations, final=True)
    save_worker.shutdown()
    if video_recorder is not None:
        video_recorder.close()
    if web_viewer is not None:
        web_viewer.set_status("finished", "training complete")
        web_viewer.close()
        if bool(getattr(args, "web_viewer_keep_alive", True)) and web_viewer_backend == "process":
            print(
                f"[web-viewer] Training complete. Viewer remains available at "
                f"http://127.0.0.1:{web_viewer_port}. Press Ctrl+C to stop this container/process.",
                flush=True,
            )
            try:
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                print("[web-viewer] Keep-alive stopped.", flush=True)

def prepare_output_and_logger(args, run_args=None):
    ensure_model_path(args)

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(public_namespace(args)))
    if run_args is not None:
        with open(os.path.join(args.model_path, "run_args"), 'w') as run_args_f:
            run_args_f.write(str(public_namespace(run_args)))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1_val, loss_val, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, no_empty_cache=False, log_scalars=True):
    if tb_writer and log_scalars:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1_val, iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss_val, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('active_sh_degree', scene.gaussians.active_sh_degree, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        if not no_empty_cache:
            torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        if not no_empty_cache:
            torch.cuda.empty_cache()

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def get_git_metadata():
    import subprocess

    metadata = {
        "git_branch": "unknown",
        "git_commit": "unknown",
        "git_dirty": "unknown",
    }
    try:
        metadata["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        metadata["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        metadata["git_dirty"] = bool(status.strip())
    except Exception:
        pass
    return metadata

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--early_output_iterations", nargs="+", type=int, default=[500, 1000],
                        help="Always save point-cloud outputs at these early iterations when they fit in the run.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_progress_bar", action="store_true",
                        help="Disable tqdm and print concise 500-iteration progress summaries instead.")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_interval", type=int, default=2000,
                        help="Save a checkpoint every N iterations (independent of --checkpoint_iterations).")
    parser.add_argument("--save_interval", type=int, default=2000,
                        help="Save a PLY point cloud every N iterations (independent of --save_iterations).")
    parser.add_argument("--sh_degree_schedule", nargs="+", type=int, default=[1000, 2000, 3000],
                        help="Iterations at which to increase SH degree (default: 1000 2000 3000).")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--profile", type=str, default=None,
                        help="Path to save PyTorch profiler trace (e.g. /tmp/profile.json). "
                             "Captures ~100 iters starting at iter 50 to catch the 100-iter stutter.")
    parser.add_argument("--no_empty_cache", action="store_true", default=False,
                        help="Skip torch.cuda.empty_cache() calls at test iterations for profiling.")
    parser.add_argument("--log_memory", action="store_true", default=False,
                        help="Log CUDA memory allocator stats every 50 iterations to TensorBoard.")
    args = parser.parse_args(sys.argv[1:])
    
    if args.config is not None:
        # Load the configuration file
        config = load_config(args.config)
        # Set the configuration parameters on args, if they are not already set by command line arguments
        for key, value in config.items():
            setattr(args, key, value)

    # Keep explicit negative CLI flags authoritative even when --config is used.

    apply_parallelism_profile(args)

    for key, value in get_git_metadata().items():
        setattr(args, key, value)

    args.save_iterations.extend(i for i in args.early_output_iterations if 0 < i <= args.iterations)
    args.save_iterations.append(args.iterations)
    args.save_iterations = sorted(set(args.save_iterations))

    ensure_model_path(args)
    start_benchmark_log(args)
    start_output_log(args)
    early_web_viewer = None
    if bool(getattr(args, "web_viewer_enabled", True)) and int(getattr(args, "web_viewer_port", 6010)) > 0 and _web_viewer_mod is not None:
        early_web_viewer = _web_viewer_mod.start_web_viewer(
            port=int(getattr(args, "web_viewer_port", 6010)),
            host=getattr(args, "web_viewer_host", "0.0.0.0"),
            image_interval=max(1, int(getattr(args, "web_viewer_image_interval", 100))),
            total_cams=1,
            viewer_cam_idx=0,
            backend=str(getattr(args, "web_viewer_backend", "process")).lower(),
            cache_dir=getattr(args, "web_viewer_cache_dir", "") or os.path.join(args.model_path, "web_viewer_cache"),
            model_path=args.model_path,
        )
        if early_web_viewer is not None:
            early_web_viewer.set_status("loading", "loading scene and cameras")
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.sh_degree_schedule, run_args=args)

    # All done
    print("\nTraining complete.")
