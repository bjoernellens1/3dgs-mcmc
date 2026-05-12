"""
Streaming-replay training loop for 3DGS-MCMC.

Activated by --streaming_replay.  Simulates real-time RGB-D input by
consuming an existing dataset (RGBDSequence / TUM / ScanNet) as an ordered
frame stream instead of loading all cameras at once.

Phase 1  – ordered windowed training
    Frames arrive every streaming_steps_per_frame iterations (deterministic)
    or at streaming_input_fps wall-clock rate.  The optimizer sees only the
    recent keyframe window plus an occasional older replay frame.

Phase 2  – incremental depth insertion
    Each new RGB-D frame backprojects its depth into world space, voxel-
    downsamples the result, filters away already-covered regions, and appends
    new Gaussians via gaussians.add_points_as_gaussians().

The offline training() path in train.py is untouched.
"""

from __future__ import annotations

import json
import math
import os
import queue
import threading
import time
from argparse import Namespace

import numpy as np
import torch
from tqdm import tqdm

from gaussian_renderer import render
from scene import GaussianModel
from scene.gsplat_model import GsplatGaussianModel
from scene.streaming_scene import StreamingScene
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.stream_scheduler import FrameScheduler
from utils.streaming_frames import make_frame_source

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB = True
except ImportError:
    _TB = False


# ---------------------------------------------------------------------------
# Async save worker (self-contained copy so train_streaming doesn't import
# from train.py, which is __main__ and cannot be cleanly imported).
# ---------------------------------------------------------------------------

class _AsyncSaveWorker:
    def __init__(self):
        self._queue: queue.Queue = queue.Queue(maxsize=4)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
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

    def enqueue(self, fn) -> None:
        self._queue.put(fn)

    def shutdown(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=120)


# ---------------------------------------------------------------------------
# Point-cloud insertion helpers (Phase 2)
# ---------------------------------------------------------------------------

def _voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float):
    """Keep one representative point per voxel cell (O(N log N))."""
    if points.shape[0] == 0:
        return points, colors
    mins = points.min(axis=0)
    cell = np.floor((points - mins) / max(voxel_size, 1e-6)).astype(np.int64)
    # Pack (ix, iy, iz) into a single integer for np.unique
    maxc = cell.max(axis=0) + 1
    stride = np.array([maxc[1] * maxc[2], maxc[2], 1], dtype=np.int64)
    keys = cell @ stride
    _, first = np.unique(keys, return_index=True)
    return points[first], colors[first]


def _filter_existing_coverage(
    new_pts: np.ndarray,
    new_cols: np.ndarray,
    existing_xyz: torch.Tensor,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove new points that fall within voxel_size of any existing Gaussian.
    Uses a voxel-grid occupancy set — O(N_exist + N_new).
    """
    if new_pts.shape[0] == 0 or existing_xyz.shape[0] == 0:
        return new_pts, new_cols

    exist_np = existing_xyz.detach().cpu().numpy()
    all_pts = np.concatenate([exist_np, new_pts], axis=0)
    mins = all_pts.min(axis=0)

    def to_voxel_key(pts):
        cell = np.floor((pts - mins) / max(voxel_size, 1e-6)).astype(np.int64)
        mx = np.floor((all_pts.max(axis=0) - mins) / max(voxel_size, 1e-6)).astype(np.int64) + 1
        stride = np.array([mx[1] * mx[2], mx[2], 1], dtype=np.int64)
        return cell @ stride

    occupied = set(to_voxel_key(exist_np).tolist())
    new_keys = to_voxel_key(new_pts)
    keep = np.array([k not in occupied for k in new_keys.tolist()])
    return new_pts[keep], new_cols[keep]


def insert_gaussians_from_frame(gaussians, frame, args) -> int:
    """
    Phase 2: backproject an RGB-D frame, voxel-filter, remove already-covered
    regions, and append new Gaussians.  Returns the count actually added.
    """
    from utils.rgbd_frames import depth_to_meters
    from PIL import Image as _Image

    if frame.depth_path is None or not os.path.exists(frame.depth_path):
        return 0

    depth_stride = getattr(args, "streaming_depth_stride", 8)
    min_depth = getattr(args, "streaming_min_depth", 0.1)
    max_depth = getattr(args, "streaming_max_depth", 8.0)
    voxel_size = getattr(args, "streaming_insert_voxel_size", 0.02)
    cover_voxel = getattr(args, "streaming_cover_voxel_size", 0.0)
    if cover_voxel <= 0:
        cover_voxel = voxel_size * 1.5
    max_new = getattr(args, "streaming_max_new_gaussians_per_frame", 2000)
    init_scale = getattr(args, "init_scale", 0.01)
    init_opacity = getattr(args, "streaming_insert_opacity", 0.3)
    edge_threshold = getattr(args, "streaming_depth_edge_threshold", 0.1)
    max_view_angle = getattr(args, "streaming_max_view_angle", 70.0)
    use_knn_scale = getattr(args, "streaming_insert_knn_scale", True)

    try:
        depth = np.array(_Image.open(frame.depth_path))
        rgb = np.array(_Image.open(frame.rgb_path).convert("RGB")).astype(np.float32) / 255.0
    except Exception:
        return 0

    z = depth_to_meters(depth, frame.depth_scale)
    h, w = z.shape
    ys, xs = np.mgrid[0:h:depth_stride, 0:w:depth_stride]
    z_v = z[ys, xs]
    valid = np.isfinite(z_v) & (z_v > min_depth) & (z_v < max_depth)

    # ---- Depth discontinuity masking -----------------------------------------
    # Reject pixels at depth edges (foreground/background boundaries) — the
    # primary source of mid-air floaters.  Gradient computed at stride spacing.
    if edge_threshold > 0:
        s = depth_stride
        xs_l = np.clip(xs - s, 0, w - 1)
        xs_r = np.clip(xs + s, 0, w - 1)
        ys_u = np.clip(ys - s, 0, h - 1)
        ys_d = np.clip(ys + s, 0, h - 1)
        dzdx = np.abs(z[ys, xs_r] - z[ys, xs_l])
        dzdy = np.abs(z[ys_d, xs] - z[ys_u, xs])
        valid = valid & ((dzdx + dzdy) < edge_threshold)

    if not valid.any():
        return 0

    xs_v = xs[valid].astype(np.float32)
    ys_v = ys[valid].astype(np.float32)
    z_v = z_v[valid].astype(np.float32)
    x_c = (xs_v - frame.cx) / frame.fx * z_v
    y_c = (ys_v - frame.cy) / frame.fy * z_v

    # ---- Grazing-angle rejection ---------------------------------------------
    # Surface normal in camera frame: n ≈ normalize([-dzdx/fx, -dzdy/fy, 1]).
    # cos(angle) = n_z / ||n|| — reject when angle to view ray exceeds threshold.
    if max_view_angle < 90.0:
        cos_thresh = float(np.cos(np.deg2rad(max_view_angle)))
        xs_vi = xs_v.astype(np.int32)
        ys_vi = ys_v.astype(np.int32)
        dzdx_v = (z[ys_vi, np.clip(xs_vi + depth_stride, 0, w - 1)] -
                  z[ys_vi, np.clip(xs_vi - depth_stride, 0, w - 1)])
        dzdy_v = (z[np.clip(ys_vi + depth_stride, 0, h - 1), xs_vi] -
                  z[np.clip(ys_vi - depth_stride, 0, h - 1), xs_vi])
        # Normal components (camera frame, unnormalised)
        nx = -dzdx_v / frame.fx
        ny = -dzdy_v / frame.fy
        nz = np.ones_like(nx)
        norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
        cos_angle = nz / norm  # dot with [0,0,1] view dir
        angle_ok = cos_angle > cos_thresh
        xs_v = xs_v[angle_ok]
        ys_v = ys_v[angle_ok]
        z_v = z_v[angle_ok]
        x_c = x_c[angle_ok]
        y_c = y_c[angle_ok]

    if xs_v.shape[0] == 0:
        return 0

    pts = ((frame.c2w[:3, :3] @ np.stack([x_c, y_c, z_v], axis=1).T).T + frame.c2w[:3, 3]).astype(np.float32)
    rgb_h, rgb_w = rgb.shape[:2]
    rx = np.clip((xs_v / max(w - 1, 1) * (rgb_w - 1)).round().astype(np.int32), 0, rgb_w - 1)
    ry = np.clip((ys_v / max(h - 1, 1) * (rgb_h - 1)).round().astype(np.int32), 0, rgb_h - 1)
    cols = rgb[ry, rx].astype(np.float32)

    pts, cols = _voxel_downsample(pts, cols, voxel_size)
    pts, cols = _filter_existing_coverage(pts, cols, gaussians.get_xyz, cover_voxel)

    if pts.shape[0] == 0:
        return 0
    if pts.shape[0] > max_new:
        rng = np.random.default_rng()
        idx = rng.choice(pts.shape[0], size=max_new, replace=False)
        pts, cols = pts[idx], cols[idx]

    pts_t = torch.from_numpy(pts).float()
    cols_t = torch.from_numpy(cols).float()
    added = gaussians.add_points_as_gaussians(
        pts_t, cols_t,
        init_scale=init_scale,
        init_opacity=init_opacity,
        use_knn_scale=use_knn_scale,
    )
    return added


# ---------------------------------------------------------------------------
# PLY snapshot helpers (self-contained)
# ---------------------------------------------------------------------------

def _snapshot_for_ply(gaussians):
    from collections import OrderedDict
    if hasattr(gaussians, "params") and isinstance(gaussians.params, (dict, OrderedDict)):
        p = gaussians.params
        return {
            "attrs": gaussians.construct_list_of_attributes(),
            "means": p["means"].detach().contiguous().cpu(),
            "sh0": p["sh0"].detach().contiguous().cpu(),
            "shN": p["shN"].detach().contiguous().cpu(),
            "opacities": p["opacities"].detach().unsqueeze(-1).contiguous().cpu(),
            "scales": p["scales"].detach().contiguous().cpu(),
            "quats": p["quats"].detach().contiguous().cpu(),
        }
    return {
        "attrs": gaussians.construct_list_of_attributes(),
        "means": gaussians._xyz.detach().contiguous().cpu(),
        "sh0": gaussians._features_dc.detach().contiguous().cpu(),
        "shN": gaussians._features_rest.detach().contiguous().cpu(),
        "opacities": gaussians._opacity.detach().contiguous().cpu(),
        "scales": gaussians._scaling.detach().contiguous().cpu(),
        "quats": gaussians._rotation.detach().contiguous().cpu(),
    }


def _write_ply(snap, out_path):
    from plyfile import PlyData, PlyElement
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
    elements[:] = list(map(tuple, np.concatenate(
        (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
    )))
    PlyData([PlyElement.describe(elements, "vertex")]).write(out_path)


# ---------------------------------------------------------------------------
# Main streaming training function
# ---------------------------------------------------------------------------

def streaming_training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    sh_degree_schedule,
    run_args=None,
):
    if run_args is None:
        raise ValueError("streaming_training() requires run_args.")
    args = run_args

    # --- Output / logger ---------------------------------------------------
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**{k: v for k, v in vars(args).items() if not k.startswith("_")})))

    tb_writer = None
    if _TB:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("[streaming] TensorBoard not available.")

    save_worker = _AsyncSaveWorker()

    # --- Build Gaussian model ----------------------------------------------
    model_layout = getattr(args, "model_layout", "gsplat").lower()
    model_cls = GsplatGaussianModel if model_layout == "gsplat" else GaussianModel
    gaussians = model_cls(dataset.sh_degree)

    # --- Build frame source and streaming scene ----------------------------
    frame_source = make_frame_source(dataset.source_path, args)
    if len(frame_source) == 0:
        raise RuntimeError("[streaming] Frame source is empty — check dataset path.")

    streaming_scene = StreamingScene(args, gaussians, frame_source)

    # Initialise Gaussians from first K frames BEFORE training_setup() so
    # the optimizer is built on the correct initial parameter tensors.
    n_init = max(1, getattr(args, "streaming_initial_frames", 5))
    streaming_scene.initialize_from_frames(n_init)
    gaussians.training_setup(opt)

    # Restore checkpoint if requested
    first_iter = 0
    if checkpoint:
        model_params, first_iter = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)

    # --- Frame scheduler ---------------------------------------------------
    scheduler = FrameScheduler(
        fps=getattr(args, "streaming_input_fps", 30.0),
        steps_per_frame=max(1, getattr(args, "streaming_steps_per_frame", 50)),
        wallclock=getattr(args, "streaming_wallclock", False),
    )

    # --- MCMC / strategy setup ---------------------------------------------
    from utils.mcmc_schedule import MCMCScheduleConfig, get_mcmc_schedule
    from utils.strategies import make_mcmc_strategy
    from utils.compiled_kernels import configure_torch_compile, set_compile_iteration

    densification_strategy = getattr(opt, "densification_strategy", "gsplat_energy_mcmc").lower()
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
    mcmc_strategy = make_mcmc_strategy(densification_strategy, gaussians=gaussians, args=args)
    mcmc_strategy.initialize_state(gaussians=gaussians, args=args)
    configure_torch_compile(args)

    optimizer_type = getattr(opt, "optimizer_type", "adam").lower()
    sh_update_interval = max(1, int(getattr(opt, "sh_update_interval", 1)))
    sparse_active_set = optimizer_type == "selective_adam" and not getattr(
        args, "selective_adam_allow_dense_grads", False
    )
    pipe.gsplat_sparse_grad = bool(getattr(opt, "gsplat_sparse_grad", False))

    # --- Background --------------------------------------------------------
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # --- Loop state --------------------------------------------------------
    ema_loss = 0.0
    n_frames_ingested = n_init  # already ingested during init
    total_inserted = 0
    streaming_mcmc_local = getattr(args, "streaming_mcmc_local_only", True)
    global_maint_interval = max(0, getattr(args, "streaming_global_maintenance_interval", 500))
    scalar_log_interval = max(1, int(getattr(args, "scalar_log_interval", 10)))
    save_interval = max(0, int(getattr(args, "save_interval", 2000)))
    chk_interval = max(0, int(getattr(args, "checkpoint_interval", 2000)))
    save_frame_interval = max(0, int(getattr(args, "streaming_save_frame_interval", 50)))
    # Track which frame milestone we last saved at (avoids repeated saves for same frame)
    _last_frame_save = n_init  # bootstrap frames already "processed"
    use_energy_mcmc = getattr(args, "energy_mcmc", True) and densification_strategy in {
        "mcmc", "hybrid", "gsplat_energy_mcmc"
    }

    print(
        f"[streaming] Starting: strategy={densification_strategy} "
        f"optimizer={optimizer_type} sparse={sparse_active_set} "
        f"steps_per_frame={scheduler.steps_per_frame} "
        f"wallclock={scheduler.wallclock}",
        flush=True,
    )

    progress_bar = tqdm(
        range(first_iter, opt.iterations),
        desc="Streaming training",
        disable=getattr(args, "quiet", False),
    )
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        set_compile_iteration(iteration)

        # ---- Frame ingestion -----------------------------------------------
        if streaming_scene.has_next_frame() and scheduler.should_release(iteration):
            result = streaming_scene.ingest_next_frame()
            if result is not None:
                new_cam, new_frame = result
                scheduler.mark_released()
                n_frames_ingested += 1

                # Phase 2: insert new Gaussians from depth
                if getattr(args, "streaming_insert_from_depth", True):
                    cap = getattr(args, "cap_max", -1)
                    current_n = gaussians.get_xyz.shape[0]
                    if cap <= 0 or current_n < cap:
                        added = insert_gaussians_from_frame(gaussians, new_frame, args)
                        total_inserted += added
                        if added > 0 and iteration % 100 == 0:
                            print(
                                f"[streaming] iter={iteration} frame={n_frames_ingested} "
                                f"inserted={added} total_inserted={total_inserted} N={gaussians.get_xyz.shape[0]}",
                                flush=True,
                            )

        # ---- Learning rate update -----------------------------------------
        xyz_lr = gaussians.update_learning_rate(iteration)

        # SH degree schedule
        if iteration in sh_degree_schedule:
            gaussians.oneupSHdegree()

        # ---- Camera selection ---------------------------------------------
        viewpoint_cam = streaming_scene.sample_training_camera()
        if viewpoint_cam is None:
            continue

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3,), device="cuda") if opt.random_background else background

        # ---- Forward pass -------------------------------------------------
        update_sh_rest = (
            gaussians.active_sh_degree == 0
            or sh_update_interval <= 1
            or iteration % sh_update_interval == 0
        )
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, update_sh_rest=update_sh_rest)
        image = render_pkg["render"]

        # ---- Loss ---------------------------------------------------------
        gt_image = viewpoint_cam.original_image
        Ll1 = l1_loss(image, gt_image)
        _ssim = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - _ssim)

        # Regularisation: local (visible) or global depending on active-set mode
        from utils.compiled_kernels import active_reg_core
        if sparse_active_set:
            _active = render_pkg["visibility_filter"].detach()
            loss = loss + active_reg_core(
                gaussians.get_opacity[_active],
                gaussians.get_scaling[_active],
                w_opacity=args.opacity_reg,
                w_scale=args.scale_reg,
            )
        else:
            loss = loss + active_reg_core(
                gaussians.get_opacity,
                gaussians.get_scaling,
                w_opacity=args.opacity_reg,
                w_scale=args.scale_reg,
            )

        # Energy-guided losses (full path, same as offline when enabled)
        if use_energy_mcmc:
            from utils.energy_mcmc import (
                compute_effective_count_loss,
                compute_opacity_entropy_loss,
                compute_gaussian_utility,
            )
            _energy_interval = getattr(args, "sparse_energy_global_interval", 500)
            _run_energy = not sparse_active_set or _energy_interval <= 0 or iteration % _energy_interval == 0
            if _run_energy:
                L_eff, N_eff, N_target = compute_effective_count_loss(
                    gaussians._opacity,
                    iteration=iteration,
                    cap_max=getattr(args, "cap_max", -1),
                    max_iterations=opt.iterations,
                    target_splat_end=getattr(args, "mcmc_target_splat_end", None),
                    q_start=getattr(args, "mcmc_target_q_start", 0.05),
                    q_end=getattr(args, "mcmc_target_q_end", 0.85),
                    tau_N=getattr(args, "mcmc_target_tau", 0.45),
                )
                loss = loss + args.lambda_eff_count * L_eff
                loss = loss + args.lambda_opacity_entropy * compute_opacity_entropy_loss(gaussians._opacity)

        # ---- Backward -----------------------------------------------------
        mcmc_strategy.step_pre_backward(
            gaussians=gaussians, args=args, iteration=iteration,
            render_pkg=render_pkg, loss=loss,
        )
        loss.backward()

        # Zero invisible grad rows in strided grads (active-set safety)
        if sparse_active_set and getattr(args, "selective_adam_zero_invisible_grads", True):
            _mask = render_pkg["visibility_filter"].detach()
            for group in gaussians.optimizer.param_groups:
                p = group["params"][0]
                if p.grad is not None and getattr(p.grad, "layout", torch.strided) == torch.strided:
                    p.grad[~_mask] = 0.0

        # ---- Optimizer step -----------------------------------------------
        visible = render_pkg["visibility_filter"].detach().to(dtype=torch.bool).contiguous()
        if iteration < opt.iterations:
            if optimizer_type == "selective_adam":
                gaussians.prepare_selective_adam_step(
                    allow_dense_grads=getattr(args, "selective_adam_allow_dense_grads", False)
                )
                gaussians.optimizer.step(visibility=visible)
            else:
                gaussians.optimizer.step()
            if pipe.gsplat_sparse_grad:
                gaussians.normalize_rotation_params(
                    mask=visible if sparse_active_set else None
                )
            gaussians.optimizer.zero_grad(set_to_none=True)

            # MCMC noise injection — local-only in streaming mode
            if densification_strategy in {"mcmc", "hybrid", "gsplat_mcmc", "gsplat_energy_mcmc"}:
                if streaming_mcmc_local and sparse_active_set:
                    mcmc_strategy.inject_noise(
                        gaussians=gaussians, args=args, xyz_lr=xyz_lr,
                        visible=visible, sparse_active_set=True, iteration=iteration,
                    )
                else:
                    mcmc_strategy.inject_noise(
                        gaussians=gaussians, args=args, xyz_lr=xyz_lr,
                        visible=None, sparse_active_set=False, iteration=iteration,
                    )

        # ---- Utility for energy MCMC --------------------------------------
        if use_energy_mcmc:
            from utils.energy_mcmc import compute_gaussian_utility
            utility = compute_gaussian_utility(
                gaussians=gaussians, render_pkg=render_pkg, iteration=iteration,
                w_alpha=args.energy_w_alpha, w_vis=args.energy_w_vis,
                w_grad=args.energy_w_grad, w_scale=args.energy_w_scale,
                w_dead=args.energy_w_dead,
                w_support=getattr(args, "energy_w_support", 2.0),
                beta_opacity=getattr(args, "energy_beta_opacity", 1.0),
                beta_scale=getattr(args, "energy_beta_scale", 0.5),
                alpha_dead=getattr(args, "energy_alpha_dead", 0.005),
            )
            u_temp = min(iteration / 30000.0, 1.0)
            tau_t = max(getattr(args, "energy_temp_tau", 0.4), 1e-6)
            denom_t = 1.0 - math.exp(-1.0 / tau_t)
            temperature = args.energy_temp_min + (args.energy_temp_start - args.energy_temp_min) * (
                1.0 - (1.0 - math.exp(-u_temp / tau_t)) / denom_t
            )
            # Restrict utility to active set in local mode
            if streaming_mcmc_local and sparse_active_set:
                utility[~visible] = (
                    getattr(args, "energy_w_alpha", 1.0) * gaussians.get_opacity.squeeze(-1)[~visible]
                )
        else:
            utility = None
            temperature = 1.0

        # ---- MCMC mutation ------------------------------------------------
        sched = get_mcmc_schedule(
            iteration=iteration,
            current_n=gaussians.get_xyz.shape[0],
            cap_max=getattr(args, "cap_max", -1),
            cfg=mcmc_cfg,
        )

        # In local-only mode, suppress global population growth entirely;
        # new geometry comes from depth insertion instead.
        if streaming_mcmc_local:
            sched["allow_growth"] = False
            # Global maintenance: allow full reloc/grow on configured cadence
            if global_maint_interval > 0 and iteration % global_maint_interval == 0:
                sched["allow_growth"] = True
                sched["allow_relocation"] = True

        mcmc_strategy.step_post_backward(
            gaussians=gaussians, args=args, sched=sched, iteration=iteration,
            utility=utility, temperature=temperature, use_energy_mcmc=use_energy_mcmc,
            tb_writer=tb_writer,
            should_log_strategy=lambda i: i % max(1, getattr(args, "strategy_log_interval", 500)) == 0,
            render_pkg=render_pkg, lr=xyz_lr,
        )

        # ---- Logging ------------------------------------------------------
        with torch.no_grad():
            _loss_val = loss.item()
            ema_loss = 0.4 * _loss_val + 0.6 * ema_loss
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss:.7f}",
                    "N": gaussians.get_xyz.shape[0],
                    "F": n_frames_ingested,
                })
                progress_bar.update(10)

            should_log = tb_writer and iteration % scalar_log_interval == 0
            if should_log:
                tb_writer.add_scalar("train/l1_loss", Ll1.item(), iteration)
                tb_writer.add_scalar("train/total_loss", _loss_val, iteration)
                tb_writer.add_scalar("train/psnr", psnr(image, gt_image).mean().item(), iteration)
                tb_writer.add_scalar("streaming/frames_ingested", n_frames_ingested, iteration)
                tb_writer.add_scalar("streaming/gaussians", gaussians.get_xyz.shape[0], iteration)
                tb_writer.add_scalar("streaming/total_inserted", total_inserted, iteration)
                tb_writer.add_scalar("streaming/keyframe_window",
                                     len(streaming_scene.get_local_cameras()), iteration)
                tb_writer.add_scalar("streaming/replay_buffer",
                                     len(streaming_scene._replay_buffer), iteration)

            if iteration == opt.iterations:
                progress_bar.close()

            # Test evaluation
            if iteration in testing_iterations:
                _run_test_eval(tb_writer, iteration, streaming_scene, gaussians, render, pipe, background, args)

        # ---- Saving -------------------------------------------------------
        # Frame-based PLY saves: fire when frame count crosses a new milestone
        if save_frame_interval > 0:
            _milestone = (n_frames_ingested // save_frame_interval) * save_frame_interval
            if _milestone > _last_frame_save and _milestone > 0:
                _last_frame_save = _milestone
                print(f"\n[FRAME {n_frames_ingested}] Saving Gaussians (iter={iteration})", flush=True)
                _snap = _snapshot_for_ply(gaussians)
                _ply_path = os.path.join(
                    args.model_path, f"point_cloud/frame_{n_frames_ingested}/point_cloud.ply"
                )
                save_worker.enqueue(lambda s=_snap, p=_ply_path: _write_ply(s, p))

        # Iteration-based PLY saves (explicit list or interval fallback)
        if (iteration in saving_iterations) or (save_interval > 0 and iteration % save_interval == 0):
            print(f"\n[ITER {iteration}] Saving Gaussians", flush=True)
            _snap = _snapshot_for_ply(gaussians)
            _ply_path = os.path.join(
                args.model_path, f"point_cloud/iteration_{iteration}/point_cloud.ply"
            )
            save_worker.enqueue(lambda s=_snap, p=_ply_path: _write_ply(s, p))

        if (iteration in checkpoint_iterations) or (chk_interval > 0 and iteration % chk_interval == 0):
            print(f"\n[ITER {iteration}] Saving Checkpoint", flush=True)
            _state = gaussians.capture()
            _chk = os.path.join(args.model_path, f"chkpnt{iteration}.pth")
            save_worker.enqueue(lambda s=_state, i=iteration, p=_chk: torch.save((s, i), p))

    save_worker.shutdown()
    print("\n[streaming] Training complete.", flush=True)


# ---------------------------------------------------------------------------
# Test evaluation helper
# ---------------------------------------------------------------------------

def _run_test_eval(tb_writer, iteration, streaming_scene, gaussians, render_fn, pipe, background, args):
    test_cams = streaming_scene.getTestCameras()
    if not test_cams:
        return
    torch.cuda.empty_cache()
    l1_total = 0.0
    psnr_total = 0.0
    for idx, cam in enumerate(test_cams):
        with torch.no_grad():
            img = torch.clamp(render_fn(cam, gaussians, pipe, background)["render"], 0.0, 1.0)
            gt = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)
        l1_total += l1_loss(img, gt).mean().double().item()
        psnr_total += psnr(img, gt).mean().double().item()
        if tb_writer and idx < 5:
            tb_writer.add_images(f"test_view_{cam.image_name}/render", img[None], global_step=iteration)
    n = max(len(test_cams), 1)
    print(f"\n[ITER {iteration}] Test L1={l1_total / n:.4f} PSNR={psnr_total / n:.2f}")
    if tb_writer:
        tb_writer.add_scalar("test/l1", l1_total / n, iteration)
        tb_writer.add_scalar("test/psnr", psnr_total / n, iteration)
    torch.cuda.empty_cache()
