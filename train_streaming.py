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
from typing import Optional

import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from gaussian_renderer import render
from scene import GaussianModel
from scene.gsplat_model import GsplatGaussianModel
from scene.streaming_scene import StreamingScene
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.stream_scheduler import FrameScheduler
from utils.streaming_frames import make_frame_source
from utils.general_utils import inverse_sigmoid

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

def _voxel_downsample_indices(points: np.ndarray, voxel_size: float):
    if points.shape[0] == 0:
        return points, np.array([], dtype=np.int64)
    mins = points.min(axis=0)
    cell = np.floor((points - mins) / max(voxel_size, 1e-6)).astype(np.int64)
    maxc = cell.max(axis=0) + 1
    stride = np.array([maxc[1] * maxc[2], maxc[2], 1], dtype=np.int64)
    keys = cell @ stride
    _, first = np.unique(keys, return_index=True)
    return points[first], first


def _voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float):
    """Keep one representative point per voxel cell (O(N log N))."""
    p, i = _voxel_downsample_indices(points, voxel_size)
    return p, colors[i]


def _filter_existing_coverage(
    new_pts: np.ndarray,
    new_cols: np.ndarray,
    existing_xyz: torch.Tensor,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove new points that fall within voxel_size of any existing Gaussian.
    (DEPRECATED: Use StreamingScene.check_occupancy instead)
    """
    return new_pts, new_cols


def _get_sensor_depth(cam, target_h: int, target_w: int) -> Optional[torch.Tensor]:
    """
    Return the sensor depth for a streaming camera as a [1, H, W] float32 CPU tensor,
    resized to (target_h, target_w) with nearest-neighbour interpolation.
    Lazy-loaded and cached on the camera object (cam._sensor_depth_cache).
    Returns None for non-streaming cameras or if depth is unavailable.
    """
    if not hasattr(cam, "_streaming_depth_path"):
        return None
    if cam._sensor_depth_cache is False:  # sentinel: previous load failed
        return None
    if cam._sensor_depth_cache is not None:
        return cam._sensor_depth_cache
    # First access: load from disk
    depth_path = cam._streaming_depth_path
    if depth_path is None or not os.path.exists(depth_path):
        cam._sensor_depth_cache = False
        return None
    try:
        from utils.rgbd_frames import depth_to_meters
        from PIL import Image as _Image
        depth_raw = np.array(_Image.open(depth_path))
        depth_m = depth_to_meters(depth_raw, cam._streaming_depth_scale).astype(np.float32)
    except Exception:
        cam._sensor_depth_cache = False
        return None
    d_t = torch.from_numpy(depth_m)[None, None]  # [1, 1, H_sensor, W_sensor]
    if d_t.shape[2] != target_h or d_t.shape[3] != target_w:
        d_t = torch.nn.functional.interpolate(d_t, (target_h, target_w), mode="nearest")
    cam._sensor_depth_cache = d_t[0]  # [1, H, W] on CPU
    return cam._sensor_depth_cache


def _sync_streaming_state_lengths(gaussians):
    """Keep streaming lifecycle tensors aligned with the current Gaussian count."""
    count = int(gaussians.get_xyz.shape[0])
    for name, fill_value in (
        ("birth_frame", 0),
        ("support_count", 0),
        ("provisional", False),
        ("anchor_iter", 0),
        ("lifecycle_state", 0),
        ("utility_ema", 0.0),
    ):
        tensor = getattr(gaussians, name, None)
        if tensor is None:
            continue
        if tensor.shape[0] == count:
            continue
        if tensor.shape[0] > count:
            setattr(gaussians, name, tensor[:count].contiguous())
            continue
        pad = torch.full(
            (count - tensor.shape[0],),
            fill_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        setattr(gaussians, name, torch.cat([tensor, pad], dim=0))
    # anchor_xyz needs per-point 3-vector; pad with current positions so anchor=birth position
    anchor_xyz = getattr(gaussians, "anchor_xyz", None)
    if anchor_xyz is not None and anchor_xyz.shape[0] != count:
        if anchor_xyz.shape[0] > count:
            gaussians.anchor_xyz = anchor_xyz[:count].contiguous()
        else:
            deficit = count - anchor_xyz.shape[0]
            new_means = gaussians.get_xyz[-deficit:].detach().clone()
            gaussians.anchor_xyz = torch.cat([anchor_xyz, new_means], dim=0)
    # anchor_scale_log / anchor_opacity_logit: pad with current param values
    for attr, param_key, ndim in (
        ("anchor_scale_log", "scales", 3),
        ("anchor_opacity_logit", "opacities", 1),
    ):
        buf = getattr(gaussians, attr, None)
        if buf is None:
            continue
        if buf.shape[0] == count:
            continue
        if buf.shape[0] > count:
            setattr(gaussians, attr, buf[:count].contiguous())
        else:
            deficit = count - buf.shape[0]
            if hasattr(gaussians, "params") and param_key in gaussians.params:
                new_vals = gaussians.params[param_key][-deficit:].detach().clone()
            else:
                new_vals = torch.zeros((deficit, ndim) if ndim > 1 else (deficit,), device=buf.device, dtype=buf.dtype)
            setattr(gaussians, attr, torch.cat([buf, new_vals], dim=0))


def _load_depth_meters(frame) -> Optional[np.ndarray]:
    """Load and convert a frame's depth image to float32 metres array, or None on failure."""
    from utils.rgbd_frames import depth_to_meters
    from PIL import Image as _Image
    if frame is None or frame.depth_path is None or not os.path.exists(frame.depth_path):
        return None
    try:
        depth = np.array(_Image.open(frame.depth_path))
        return depth_to_meters(depth, frame.depth_scale).astype(np.float32)
    except Exception:
        return None


def _quaternion_from_normal(normal: np.ndarray) -> np.ndarray:
    """Compute wxyz quaternions rotating [0,0,1] to the given normals."""
    z_axis = np.array([0, 0, 1.0])
    cross = np.cross(z_axis, normal)           # [N, 3]
    w = 1.0 + np.sum(z_axis * normal, axis=1)  # [N]
    q = np.stack([w, cross[:, 0], cross[:, 1], cross[:, 2]], axis=1)
    nrm = np.linalg.norm(q, axis=1, keepdims=True)
    # Degenerate case: normal ≈ [0,0,-1] → 180° rotation around x-axis
    bad = (nrm.squeeze(-1) < 1e-6)
    if bad.any():
        q[bad] = np.array([0.0, 1.0, 0.0, 0.0])
        nrm[bad] = 1.0
    return q / nrm


def insert_gaussians_from_frame(
    gaussians, frame, args,
    streaming_scene: "StreamingScene",
    prev_frame=None,
    prev_depth_meters: Optional[np.ndarray] = None,
    render_alpha: Optional[torch.Tensor] = None,
    render_depth: Optional[torch.Tensor] = None,
    current_frame_idx: int = 0,
) -> tuple:
    """
    Phase 2: backproject an RGB-D frame, voxel-filter, remove already-covered
    regions, and append new Gaussians as surface-aligned surfels.

    Returns (n_inserted, stats_dict). stats_dict tracks per-filter-step counts.
    """
    from utils.rgbd_frames import depth_to_meters
    from PIL import Image as _Image

    _debug = getattr(args, "streaming_insertion_debug", False)
    _stats: dict = {}

    if frame.depth_path is None or not os.path.exists(frame.depth_path):
        return 0, _stats

    depth_stride = getattr(args, "streaming_depth_stride", 8)
    min_depth = getattr(args, "streaming_min_depth", 0.1)
    max_depth = getattr(args, "streaming_max_depth", 8.0)
    voxel_size = getattr(args, "streaming_insert_voxel_size", 0.02)
    cover_voxel = getattr(args, "streaming_cover_voxel_size", 0.0)
    if cover_voxel <= 0:
        _cover_mult = getattr(args, "streaming_cover_voxel_multiplier", 1.0)
        cover_voxel = voxel_size * max(_cover_mult, 0.1)
    max_new = getattr(args, "streaming_max_new_gaussians_per_frame", 2000)
    init_opacity = getattr(args, "streaming_insert_opacity", 0.05)
    edge_threshold = getattr(args, "streaming_depth_edge_threshold", 0.02)
    max_view_angle = getattr(args, "streaming_max_view_angle", 70.0)
    use_knn_scale = getattr(args, "streaming_insert_knn_scale", True)

    try:
        depth = np.array(_Image.open(frame.depth_path))
        rgb = np.array(_Image.open(frame.rgb_path).convert("RGB")).astype(np.float32) / 255.0
    except Exception:
        return 0, _stats

    z = depth_to_meters(depth, frame.depth_scale)
    h, w = z.shape
    ys, xs = np.mgrid[0:h:depth_stride, 0:w:depth_stride]
    z_v = z[ys, xs]
    valid = np.isfinite(z_v) & (z_v > min_depth) & (z_v < max_depth)
    if _debug:
        _stats["raw_candidates"] = int(valid.sum())

    # ---- Alpha + Depth occupancy masking (Step 4) ----------------------------
    # Render may be at a different resolution than the raw sensor (--resolution
    # rescaling). Map sensor pixel coords into render pixel coords before indexing.
    #
    # Insert a depth candidate if ANY of these hold:
    #   (a) alpha < 0.3  → pixel is unoccupied in the current map
    #   (b) rendered depth = 0 → nothing rendered here at all
    #   (c) rendered depth > sensor depth + thresh → sensor found a closer surface
    #       the map hasn't captured yet (e.g. behind a floater)
    #
    # Previously only (a) was tested, which caused N to plateau once every
    # visible pixel reached alpha ≥ 0.3 — regardless of new geometry in-frame.
    if render_alpha is not None or render_depth is not None:
        sh2d = xs.shape  # (Nh, Nw) — all working arrays must stay 2D
        low_alpha = np.zeros(sh2d, dtype=bool)
        no_rend_depth = np.ones(sh2d, dtype=bool)   # default: treat as "no depth"
        sensor_closer = np.zeros(sh2d, dtype=bool)

        if render_alpha is not None:
            alpha_cpu = render_alpha.detach().cpu().numpy().squeeze()  # [Hr, Wr]
            Hr, Wr = alpha_cpu.shape
            xs_r = np.clip(np.round(xs / max(w - 1, 1) * (Wr - 1)).astype(np.int32), 0, Wr - 1)
            ys_r = np.clip(np.round(ys / max(h - 1, 1) * (Hr - 1)).astype(np.int32), 0, Hr - 1)
            alpha_v = alpha_cpu[ys_r, xs_r]
            low_alpha = alpha_v < 0.3

        if render_depth is not None:
            rend_d_cpu = render_depth.detach().cpu().numpy().squeeze()
            Hr_d, Wr_d = rend_d_cpu.shape
            xs_rd = np.clip(np.round(xs / max(w - 1, 1) * (Wr_d - 1)).astype(np.int32), 0, Wr_d - 1)
            ys_rd = np.clip(np.round(ys / max(h - 1, 1) * (Hr_d - 1)).astype(np.int32), 0, Hr_d - 1)
            rend_d_v = rend_d_cpu[ys_rd, xs_rd]
            consistency_thresh = getattr(args, "streaming_depth_consistency_thresh", 0.05)
            no_rend_depth = rend_d_v <= 0
            sensor_closer = rend_d_v - z_v > consistency_thresh

        valid = valid & (low_alpha | no_rend_depth | sensor_closer)
        if _debug:
            _stats["after_alpha_depth_mask"] = int(valid.sum())
            _stats["insert_path_low_alpha"] = int((valid & low_alpha).sum())
            _stats["insert_path_no_depth"] = int((valid & no_rend_depth & ~low_alpha).sum())
            _stats["insert_path_sensor_closer"] = int((valid & sensor_closer & ~low_alpha & ~no_rend_depth).sum())

    # ---- Relative depth discontinuity masking (Step 5) ----------------------
    if edge_threshold > 0:
        s = depth_stride
        dzdx = np.abs(z[ys, np.clip(xs + s, 0, w - 1)] - z[ys, np.clip(xs - s, 0, w - 1)])
        dzdy = np.abs(z[np.clip(ys + s, 0, h - 1), xs] - z[np.clip(ys - s, 0, h - 1), xs])
        edge_rel = (dzdx + dzdy) / np.maximum(z_v, 1e-6)
        valid = valid & (edge_rel < edge_threshold)
        if _debug:
            _stats["after_edge_filter"] = int(valid.sum())

    if not valid.any():
        return 0, _stats

    # Sampling for normals
    s = depth_stride
    xs_vi = xs[valid].astype(np.int32)
    ys_vi = ys[valid].astype(np.int32)
    z_vi = z_v[valid]
    
    dzdx_v = (z[ys_vi, np.clip(xs_vi + s, 0, w - 1)] - z[ys_vi, np.clip(xs_vi - s, 0, w - 1)])
    dzdy_v = (z[np.clip(ys_vi + s, 0, h - 1), xs_vi] - z[np.clip(ys_vi - s, 0, h - 1), xs_vi])
    
    # ---- Grazing-angle rejection ---------------------------------------------
    nx = -dzdx_v / (frame.fx * (2.0 * s / depth_stride)) # scaled by local stride
    ny = -dzdy_v / (frame.fy * (2.0 * s / depth_stride))
    nz = np.ones_like(nx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    nx /= norm; ny /= norm; nz /= norm
    
    cos_thresh = float(np.cos(np.deg2rad(max_view_angle)))
    angle_ok = nz > cos_thresh
    
    if _debug:
        _stats["after_grazing_filter"] = int(angle_ok.sum())
    if not angle_ok.any():
        return 0, _stats
        
    xs_v = xs_vi[angle_ok].astype(np.float32)
    ys_v = ys_vi[angle_ok].astype(np.float32)
    z_v = z_vi[angle_ok].astype(np.float32)
    nx = nx[angle_ok]; ny = ny[angle_ok]; nz = nz[angle_ok]
    
    x_c = (xs_v - frame.cx) / frame.fx * z_v
    y_c = (ys_v - frame.cy) / frame.fy * z_v
    pts = ((frame.c2w[:3, :3] @ np.stack([x_c, y_c, z_v], axis=1).T).T + frame.c2w[:3, 3]).astype(np.float32)
    
    rgb_h, rgb_w = rgb.shape[:2]
    rx = np.clip((xs_v / max(w - 1, 1) * (rgb_w - 1)).round().astype(np.int32), 0, rgb_w - 1)
    ry = np.clip((ys_v / max(h - 1, 1) * (rgb_h - 1)).round().astype(np.int32), 0, rgb_h - 1)
    cols = rgb[ry, rx].astype(np.float32)

    # ---- Persistent Voxel coverage filter (Step 7) --------------------------
    occupied = streaming_scene.check_occupancy(pts, cover_voxel, check_neighbors=True)
    pts = pts[~occupied]
    cols = cols[~occupied]
    nx = nx[~occupied]; ny = ny[~occupied]; nz = nz[~occupied]
    z_v = z_v[~occupied]
    if _debug:
        _stats["after_voxel_occupancy"] = pts.shape[0]

    if pts.shape[0] == 0:
        return 0, _stats

    # Downsample remaining
    pts, indices = _voxel_downsample_indices(pts, voxel_size)
    cols = cols[indices]
    nx = nx[indices]; ny = ny[indices]; nz = nz[indices]
    z_v = z_v[indices]

    # ---- KNN insertion dedup (Component E) ----------------------------------
    # Reject candidates whose nearest existing Gaussian is within depth*radius_factor.
    # Uses torch.cdist; subsamples existing set if too large for memory.
    use_knn_dedup = getattr(args, "streaming_insert_knn_dedup", False)
    if use_knn_dedup and pts.shape[0] > 0:
        n_existing = gaussians.get_xyz.shape[0]
        if n_existing > 0:
            knn_radius_factor = float(getattr(args, "streaming_insert_knn_radius_factor", 0.005))
            max_existing = int(getattr(args, "streaming_insert_knn_max_existing", 200_000))
            with torch.no_grad():
                cand_t = torch.from_numpy(pts).to(device="cuda", dtype=torch.float32)
                exist_t = gaussians.get_xyz.detach()
                if n_existing > max_existing:
                    # Subsample via uniform stride to keep memory manageable
                    stride = max(1, n_existing // max_existing)
                    exist_t = exist_t[::stride]
                # Chunk the distance computation to avoid OOM
                chunk = 1024
                min_dists = torch.full((cand_t.shape[0],), float("inf"), device="cuda")
                for c_start in range(0, cand_t.shape[0], chunk):
                    c_end = min(c_start + chunk, cand_t.shape[0])
                    d = torch.cdist(cand_t[c_start:c_end].float(), exist_t.float())  # [chunk, M]
                    min_dists[c_start:c_end] = d.min(dim=1).values
                per_pt_radius = torch.from_numpy(z_v * knn_radius_factor).to(device="cuda", dtype=torch.float32)
                too_close = min_dists < per_pt_radius
                keep = ~too_close
            if keep.any():
                keep_np = keep.cpu().numpy()
                pts = pts[keep_np]; cols = cols[keep_np]
                nx = nx[keep_np]; ny = ny[keep_np]; nz = nz[keep_np]
                z_v = z_v[keep_np]
                if _debug:
                    _stats["after_knn_dedup"] = pts.shape[0]
            else:
                return 0, _stats

    if max_new > 0 and pts.shape[0] > max_new:
        rng = np.random.default_rng(42)
        idx = rng.choice(pts.shape[0], size=max_new, replace=False)
        pts, cols, nx, ny, nz, z_v = pts[idx], cols[idx], nx[idx], ny[idx], nz[idx], z_v[idx]

    # ---- Surface-aligned initialisation (Step 6) ----------------------------
    # Align Gaussians to surface normals
    normals_cam = np.stack([nx, ny, nz], axis=1)
    normals_world = (frame.c2w[:3, :3] @ normals_cam.T).T
    q_world = _quaternion_from_normal(normals_world)
    
    # Scales: depth-derived per-axis pixel footprint in world space, with
    # an optional multiplier and hard upper clamp to avoid over-large splats.
    scale_mult  = getattr(args, "streaming_insert_scale_mult", 0.5)
    scale_max   = getattr(args, "streaming_insert_scale_max", 0.05)
    normal_ratio = getattr(args, "streaming_insert_normal_scale_ratio", 0.15)
    tx = np.clip(scale_mult * z_v / frame.fx * depth_stride, 1e-4, scale_max)
    ty = np.clip(scale_mult * z_v / frame.fy * depth_stride, 1e-4, scale_max)
    if getattr(args, "streaming_insert_isotropic_scale", False):
        # Isotropic sphere (H3): geometric mean of in-plane footprint.
        # Avoids edge-on streaking from flat surfels seen off-axis.
        t_iso = np.sqrt(tx * ty)
        log_scales = np.log(np.stack([t_iso, t_iso, t_iso], axis=1))
    else:
        tz = np.clip(normal_ratio * np.minimum(tx, ty), 1e-5, scale_max * normal_ratio)
        log_scales = np.log(np.stack([tx, ty, tz], axis=1))

    if _debug:
        _stats["after_per_frame_cap"] = pts.shape[0]

    added = gaussians.add_points_as_gaussians(
        torch.from_numpy(pts),
        torch.from_numpy(cols),
        init_opacity=init_opacity,
        scales=torch.from_numpy(log_scales),
        rotations=torch.from_numpy(q_world),
        is_provisional=True,
        birth_frame=current_frame_idx,
    )
    if added > 0:
        streaming_scene.add_to_occupancy_hash(gaussians.get_xyz[-added:])
    if _debug:
        _stats["inserted"] = added
    return added, _stats


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
# Streaming render snapshot (train + test views at frame milestones)
# ---------------------------------------------------------------------------

def _render_streaming_snapshot(
    gaussians, streaming_scene, render_fn, pipe, background, args, n_frames, tb_writer,
    save_worker: Optional[_AsyncSaveWorker] = None,
):
    """
    Render train keyframe window + held-out test cameras, save PNGs, log PSNR.
    Called at frame-save milestones when --streaming_render_at_saves is set.
    """
    train_cams = streaming_scene.get_local_cameras()
    test_cams = streaming_scene.getTestCameras()
    if not train_cams and not test_cams:
        return

    output_base = os.path.join(args.model_path, "streaming_renders", f"frame_{n_frames}")

    for split, cameras in [("train", train_cams), ("test", test_cams)]:
        if not cameras:
            continue
        render_dir = os.path.join(output_base, split, "renders")
        gt_dir = os.path.join(output_base, split, "gt")
        os.makedirs(render_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        psnrs = []
        render_data = []
        for cam in cameras:
            with torch.no_grad():
                img = torch.clamp(render_fn(cam, gaussians, pipe, background)["render"], 0.0, 1.0)
                gt = torch.clamp(cam.original_image[:3].cuda(), 0.0, 1.0)
            psnrs.append(psnr(img, gt).mean().item())
            render_data.append((
                img.cpu(), gt.cpu(),
                os.path.join(render_dir, f"{cam.image_name}.png"),
                os.path.join(gt_dir, f"{cam.image_name}.png"),
            ))
        if psnrs:
            mean_psnr = float(np.mean(psnrs))
            print(
                f"[streaming-render] frame={n_frames} {split} PSNR={mean_psnr:.2f}dB ({len(psnrs)} views)",
                flush=True,
            )
            if tb_writer:
                tb_writer.add_scalar(f"streaming_eval/{split}_psnr", mean_psnr, n_frames)

        # Save images via async worker to avoid blocking training
        def _save_batch(data=render_data):
            for img_cpu, gt_cpu, r_path, g_path in data:
                save_image(img_cpu, r_path)
                save_image(gt_cpu, g_path)

        if save_worker is not None:
            save_worker.enqueue(_save_batch)
        else:
            _save_batch()


# ---------------------------------------------------------------------------
# Main streaming training function
# ---------------------------------------------------------------------------

def _update_provisional_support(gaussians, cam, render_pkg, args):
    """
    Check visible provisional Gaussians against sensor depth and increment support.
    """
    if not hasattr(gaussians, "provisional") or not gaussians.provisional.any():
        return
    
    visible = render_pkg["visibility_filter"]
    to_check = gaussians.provisional & visible
    if not to_check.any():
        return
        
    sensor_d_cpu = _get_sensor_depth(cam, cam.image_height, cam.image_width)
    if sensor_d_cpu is None:
        return
        
    sensor_d = sensor_d_cpu.to(device="cuda", non_blocking=True)
    
    # Project points to get pixel coords
    xyz_v = gaussians.get_xyz[to_check]
    pts_h = torch.cat([xyz_v, torch.ones((xyz_v.shape[0], 1), device="cuda")], dim=1)
    
    # W2C
    w2c = cam.world_view_transform.transpose(0, 1)
    pts_cam = pts_h @ w2c.T
    z_p = pts_cam[:, 2]
    
    # Intrinsic projection
    W, H = cam.image_width, cam.image_height
    fx = getattr(cam, "fx", W / (2.0 * math.tan(cam.FoVx / 2.0)))
    fy = getattr(cam, "fy", H / (2.0 * math.tan(cam.FoVy / 2.0)))
    cx = getattr(cam, "cx", W / 2.0)
    cy = getattr(cam, "cy", H / 2.0)
    
    u = (pts_cam[:, 0] / torch.clamp(z_p, min=1e-6)) * fx + cx
    v = (pts_cam[:, 1] / torch.clamp(z_p, min=1e-6)) * fy + cy
    
    ui = u.round().long()
    vi = v.round().long()
    
    valid_px = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (z_p > 0)
    if not valid_px.any():
        return
        
    # Consistency check
    z_lookup = sensor_d[0, vi[valid_px], ui[valid_px]]
    consistent = (z_lookup > 0) & (torch.abs(z_p[valid_px] - z_lookup) < args.streaming_depth_consistency_thresh)
    
    # Increment support for consistent points
    # Need to map back from valid_px -> to_check -> global
    indices = torch.where(to_check)[0][valid_px][consistent]
    gaussians.support_count[indices] += 1



def _run_rolling_seed(
    gaussians,
    streaming_scene,
    opt,
    pipe,
    args,
    background,
    dataset,
    tb_writer,
    save_worker,
    testing_iterations,
    saving_iterations,
    sh_degree_schedule,
):
    """
    Diagnostic mode: Aggregates geometry by streaming through windows of frames,
    building depth point clouds, and inserting them into the global map via the
    occupancy grid. No training occurs during the streaming phase.
    Finally, performs global optimization for the specified iterations.
    """
    import torch
    import os
    import numpy as np
    from tqdm import tqdm
    from gaussian_renderer import render

    submap_frames = max(2, int(getattr(args, "streaming_submap_frames", 20)))
    refine_iters  = max(0, int(getattr(args, "streaming_global_refine_iters", 5000)))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print(f"[rolling_seed] mode: {submap_frames} frames/window, {refine_iters} global refine iters", flush=True)

    # Collect all frames
    all_frames = list(streaming_scene._all_frames)
    all_train_cameras = list(streaming_scene.train_cameras)
    while streaming_scene.has_next_frame():
        result = streaming_scene.ingest_next_frame()
        if result is not None:
            cam, frame, is_train = result
            if is_train:
                all_train_cameras.append(cam)

    eval_hold = getattr(args, "streaming_eval_hold", 0)
    train_frames = [f for f in all_frames if eval_hold <= 0 or f.index % eval_hold != 0]
    n_paired = min(len(train_frames), len(all_train_cameras))
    frame_cam_pairs = list(zip(train_frames[:n_paired], all_train_cameras[:n_paired]))

    windows = [
        frame_cam_pairs[i: i + submap_frames]
        for i in range(0, len(frame_cam_pairs), submap_frames)
    ]

    print(f"[rolling_seed] Aggregating geometry across {len(windows)} windows...", flush=True)
    
    for sm_idx, sm_pairs in enumerate(windows):
        if not sm_pairs:
            continue
            
        sm_frames, sm_cams = zip(*sm_pairs)
        
        # Build PCD from this window
        from scene.streaming_scene import StreamingScene
        _tmp_scene = StreamingScene.__new__(StreamingScene)
        _tmp_scene.args = args
        _tmp_scene.cameras_extent = streaming_scene.cameras_extent
        pcd = _tmp_scene._build_pcd_from_frames(list(sm_frames))
        
        if pcd is None or pcd.points.shape[0] == 0:
            continue
            
        # Filter points through global occupancy grid
        voxel_size = getattr(args, "streaming_insert_voxel_size", 0.02)
        occupied = streaming_scene.check_occupancy(pcd.points, voxel_size, check_neighbors=True)
        unoccupied = ~occupied
        
        if not unoccupied.any():
            continue
            
        # Extract new points
        new_pts = torch.tensor(pcd.points[unoccupied], dtype=torch.float32, device="cuda")
        new_cols = torch.tensor(pcd.colors[unoccupied], dtype=torch.float32, device="cuda")
        
        # Compute KNN scale with stricter clamp to prevent blobbing
        if new_pts.shape[0] > 1:
            from utils.rocm_knn_fallback import distCUDA2
            dist_sq = distCUDA2(new_pts)
            scales_1d = torch.clamp(dist_sq.sqrt() * 0.5, 1e-4, 0.02)  # max 2cm radius
            import math
            log_scales = torch.log(scales_1d).unsqueeze(-1).expand(-1, 3).contiguous()
        else:
            import math
            log_scales = torch.full((new_pts.shape[0], 3), math.log(0.01), device="cuda")
            
        # Add to global gaussians
        added = gaussians.add_points_as_gaussians(
            new_pts, new_cols,
            scales=log_scales,
            init_opacity=0.9,  # High opacity
            is_provisional=False,
            birth_frame=0,
        )
        
        # Update occupancy
        streaming_scene.add_to_occupancy_hash(new_pts)
        print(f"[rolling_seed] Window {sm_idx+1}/{len(windows)}: added {added} points. Total: {gaussians.get_xyz.shape[0]}", flush=True)

    print(f"[rolling_seed] Geometry aggregation complete. Total points: {gaussians.get_xyz.shape[0]}", flush=True)
    
    if refine_iters <= 0:
        print("[rolling_seed] No global refinement requested. Skipping to evaluation.", flush=True)
    else:
        print(f"[rolling_seed] Running {refine_iters} global refinement iters over {len(all_train_cameras)} cameras...", flush=True)
        
        # Scale optimization schedules to fit the shorter refine_iters window
        opt.iterations = refine_iters
        opt.position_lr_max_steps = refine_iters
        opt.densify_until_iter = int(refine_iters * 0.8)  # Stop MCMC at 80% to allow convergence
        opt.mcmc_stop_growth_iter = int(refine_iters * 0.8)
        
        import numpy as np
        sh_degree_schedule = [int(x) for x in np.linspace(refine_iters // 20, refine_iters // 2, dataset.sh_degree)]
        
        # Re-setup training to recreate optimizers for all points
        gaussians.training_setup(opt)
        _global_is_selective = getattr(gaussians, "optimizer_type", "adam") == "selective_adam"
        
        from utils.mcmc_schedule import MCMCScheduleConfig, get_mcmc_schedule
        from utils.strategies import make_mcmc_strategy
        from utils.compiled_kernels import configure_torch_compile, set_compile_iteration
        
        densification_strategy = getattr(opt, "densification_strategy", "gsplat_energy_mcmc").lower()
        mcmc_cfg = MCMCScheduleConfig(
            start_iter=opt.densify_from_iter,
            stop_growth_iter=getattr(opt, "mcmc_stop_growth_iter", 12_000),
            stop_reloc_iter=opt.densify_until_iter,
            growth_factor_start=getattr(opt, "mcmc_growth_factor_start", 1.05),
        )
        mcmc_strategy = make_mcmc_strategy(densification_strategy, gaussians=gaussians, args=args)
        mcmc_strategy.initialize_state(gaussians=gaussians, args=args)
        configure_torch_compile(args)
        
        from utils.loss_utils import l1_loss, ssim as _ssim_fn
        import random as _rnd
        
        for _it in tqdm(range(1, refine_iters + 1), desc="Global refine"):
            set_compile_iteration(_it)
            cam = _rnd.choice(all_train_cameras)
            
            xyz_lr = gaussians.update_learning_rate(_it)
            if _it in sh_degree_schedule:
                gaussians.oneupSHdegree()
                
            bg = torch.rand((3,), device="cuda") if opt.random_background else background
            
            pkg = render(cam, gaussians, pipe, bg)
            img = pkg["render"]
            gt = cam.original_image
            loss = (1.0 - opt.lambda_dssim) * l1_loss(img, gt) + opt.lambda_dssim * (1.0 - _ssim_fn(img, gt))
            
            # Regularization to prevent Gaussians from exploding
            from utils.compiled_kernels import active_reg_core
            if _global_is_selective:
                _active_reg = pkg["visibility_filter"].detach()
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
            
            mcmc_strategy.step_pre_backward(gaussians=gaussians, args=args, iteration=_it, render_pkg=pkg, loss=loss)
            loss.backward()
            
            if _it < refine_iters:
                if _global_is_selective:
                    gaussians.prepare_selective_adam_step()
                    _vis_all = pkg["visibility_filter"].detach()
                    
                    # Zero invisible grads (safety)
                    if getattr(args, "selective_adam_zero_invisible_grads", True):
                        for group in gaussians.optimizer.param_groups:
                            p = group["params"][0]
                            if p.grad is not None and getattr(p.grad, "layout", torch.strided) == torch.strided:
                                p.grad[~_vis_all] = 0.0
                                
                    gaussians.optimizer.step(visibility=_vis_all)
                    gaussians.normalize_rotation_params(mask=_vis_all)
                else:
                    gaussians.optimizer.step()
                    if pipe.gsplat_sparse_grad:
                        gaussians.normalize_rotation_params()
                        
                gaussians.optimizer.zero_grad(set_to_none=True)
                
                # MCMC
                mcmc_strategy.inject_noise(gaussians=gaussians, args=args, xyz_lr=xyz_lr, visible=pkg["visibility_filter"].detach() if _global_is_selective else None, sparse_active_set=_global_is_selective, iteration=_it)
                
                use_energy_mcmc = args.energy_mcmc and densification_strategy in {"mcmc", "hybrid", "gsplat_energy_mcmc"}
                if use_energy_mcmc:
                    from utils.energy_mcmc import compute_gaussian_utility
                    utility = compute_gaussian_utility(
                        gaussians=gaussians, render_pkg=pkg, iteration=_it,
                        w_alpha=args.energy_w_alpha, w_vis=args.energy_w_vis,
                        w_grad=args.energy_w_grad, w_scale=args.energy_w_scale,
                        w_dead=args.energy_w_dead,
                        w_support=getattr(args, "energy_w_support", 2.0),
                        beta_opacity=getattr(args, "energy_beta_opacity", 1.0),
                        beta_scale=getattr(args, "energy_beta_scale", 0.5),
                        alpha_dead=getattr(args, "energy_alpha_dead", 0.005),
                    )
                    import math
                    u_temp = min(_it / 30000.0, 1.0)
                    tau_t = max(getattr(args, "energy_temp_tau", 0.4), 1e-6)
                    denom_t = 1.0 - math.exp(-1.0 / tau_t)
                    temperature = args.energy_temp_min + (args.energy_temp_start - args.energy_temp_min) * (
                        1.0 - (1.0 - math.exp(-u_temp / tau_t)) / denom_t
                    )
                    if _global_is_selective:
                        _vis = pkg["visibility_filter"].detach()
                        utility[~_vis] = getattr(args, "energy_w_alpha", 1.0) * gaussians.get_opacity.squeeze(-1)[~_vis]
                else:
                    utility = None
                    temperature = 1.0
    
                sched = get_mcmc_schedule(_it, gaussians.get_xyz.shape[0], getattr(args, "cap_max", -1), mcmc_cfg)
                mcmc_strategy.step_post_backward(gaussians=gaussians, args=args, sched=sched, iteration=_it, utility=utility, temperature=temperature, use_energy_mcmc=use_energy_mcmc, tb_writer=tb_writer, should_log_strategy=lambda i: False, render_pkg=pkg, lr=xyz_lr)

    # Save final refined PLY
    refined_ply_dir = os.path.join(args.model_path, "point_cloud", "iteration_final")
    os.makedirs(refined_ply_dir, exist_ok=True)
    gaussians.save_ply(os.path.join(refined_ply_dir, "point_cloud.ply"))

    # Eval
    if streaming_scene.getTestCameras():
        try:
            from utils.comparison_report import write_post_training_report
            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            _eval_bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            write_post_training_report(
                args.model_path, refine_iters, gaussians, all_train_cameras, streaming_scene.getTestCameras(),
                render, pipe, _eval_bg, tb_writer=tb_writer, subdir="rolling_seed_final"
            )
        except Exception as _e:
            print(f"[rolling_seed] Evaluation failed: {_e}", flush=True)

    print("[rolling_seed] Training complete.", flush=True)

def _run_submap_stitch(
    gaussians,
    streaming_scene: "StreamingScene",
    opt,
    pipe,
    args,
    background,
    dataset,
    tb_writer,
    save_worker,
    testing_iterations,
    saving_iterations,
    sh_degree_schedule,
):
    """H11: Submap-stitching training mode.

    Divides the incoming frame stream into fixed-size windows (submaps). Each
    submap is independently bootstrapped from its own depth data, locally
    optimised, then merged into a growing global model. A final global
    refinement pass trains over all cameras.

    This provides a clean per-submap geometry baseline to compare against the
    sliding-window approach.
    """
    from utils.graphics_utils import BasicPointCloud
    from utils.sh_utils import RGB2SH
    from utils.general_utils import inverse_sigmoid

    model_layout = getattr(args, "model_layout", "gsplat").lower()
    model_cls = GsplatGaussianModel if model_layout == "gsplat" else GaussianModel

    submap_frames = max(2, int(getattr(args, "streaming_submap_frames", 20)))
    submap_iters  = max(100, int(getattr(args, "streaming_submap_iters", 3000)))
    refine_iters  = max(0, int(getattr(args, "streaming_global_refine_iters", 5000)))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print(f"[submap] mode: {submap_frames} frames/submap, "
          f"{submap_iters} iters/submap, {refine_iters} global refine iters", flush=True)

    # Collect all frames; ingest any remaining ones
    all_frames = list(streaming_scene._all_frames)  # full ordered frame list
    all_train_cameras = list(streaming_scene.train_cameras)  # bootstrap already ingested
    while streaming_scene.has_next_frame():
        result = streaming_scene.ingest_next_frame()
        if result is not None:
            cam, frame, is_train = result
            if is_train:
                all_train_cameras.append(cam)

    print(f"[submap] {len(all_train_cameras)} total train cameras collected.", flush=True)

    eval_hold = getattr(args, "streaming_eval_hold", 0)
    # Pair frames with train cameras (same ordering: non-holdout frames = train cams)
    train_frames = [f for f in all_frames if eval_hold <= 0 or f.index % eval_hold != 0]
    n_paired = min(len(train_frames), len(all_train_cameras))
    frame_cam_pairs = list(zip(train_frames[:n_paired], all_train_cameras[:n_paired]))

    # Split into submap windows
    windows = [
        frame_cam_pairs[i: i + submap_frames]
        for i in range(0, len(frame_cam_pairs), submap_frames)
    ]
    print(f"[submap] {len(windows)} submaps of up to {submap_frames} frames each.", flush=True)

    # --- Per-submap optimisation -------------------------------------------
    all_gaussian_params = []  # collect param dicts per submap

    for sm_idx, sm_pairs in enumerate(windows):
        if not sm_pairs:
            continue
        sm_frames, sm_cams = zip(*sm_pairs)
        sm_cams = list(sm_cams)
        print(f"[submap {sm_idx}] {len(sm_cams)} cameras, optimising {submap_iters} iters...", flush=True)

        # Build a fresh Gaussian model bootstrapped from this submap's depth
        sub_gaussians = model_cls(dataset.sh_degree)

        # Reuse StreamingScene's _build_pcd_from_frames with the actual frames
        _tmp_scene = StreamingScene.__new__(StreamingScene)
        _tmp_scene.args = args
        _tmp_scene.cameras_extent = 1.0
        pcd = _tmp_scene._build_pcd_from_frames(list(sm_frames))

        if pcd is None or pcd.points.shape[0] == 0:
            # Fallback: small random cloud in unit cube
            import numpy as _np
            from utils.graphics_utils import BasicPointCloud
            pcd_pts = _np.random.uniform(-0.5, 0.5, (200, 3)).astype(_np.float32)
            pcd_cols = _np.full((200, 3), 0.5, dtype=_np.float32)
            pcd = BasicPointCloud(points=pcd_pts, colors=pcd_cols, normals=_np.zeros_like(pcd_pts))

        sub_gaussians.create_from_pcd(pcd, spatial_lr_scale=1.0,
                                       init_scale_mode="fixed", init_scale=0.01)
        sub_gaussians.training_setup(opt)
        _sub_is_selective = getattr(sub_gaussians, "optimizer_type", "adam") == "selective_adam"

        # Local optimisation loop
        from utils.loss_utils import l1_loss, ssim as _ssim_fn
        import random as _rnd
        for _it in range(1, submap_iters + 1):
            cam = _rnd.choice(sm_cams)
            pkg = render(cam, sub_gaussians, pipe, background)
            img = pkg["render"]
            gt = cam.original_image
            loss = (1.0 - opt.lambda_dssim) * l1_loss(img, gt) + opt.lambda_dssim * (1.0 - _ssim_fn(img, gt))
            loss.backward()
            if _it < submap_iters:
                if _sub_is_selective:
                    sub_gaussians.prepare_selective_adam_step()
                    _vis_all = torch.ones(sub_gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")
                    sub_gaussians.optimizer.step(visibility=_vis_all)
                    sub_gaussians.normalize_rotation_params()
                else:
                    sub_gaussians.optimizer.step()
                sub_gaussians.optimizer.zero_grad(set_to_none=True)

        print(f"[submap {sm_idx}] done, {sub_gaussians.get_xyz.shape[0]} Gaussians.", flush=True)

        # Save the submap's final parameters (detached)
        if model_layout == "gsplat":
            all_gaussian_params.append({
                k: sub_gaussians.params[k].detach().cpu()
                for k in ("means", "sh0", "shN", "scales", "quats", "opacities")
            })
        else:
            all_gaussian_params.append({
                "means": sub_gaussians._xyz.detach().cpu(),
                "sh0": sub_gaussians._features_dc.detach().cpu(),
                "shN": sub_gaussians._features_rest.detach().cpu(),
                "opacities": sub_gaussians._opacity.detach().cpu(),
                "scales": sub_gaussians._scaling.detach().cpu(),
                "quats": sub_gaussians._rotation.detach().cpu(),
            })

        del sub_gaussians
        torch.cuda.empty_cache()

    if not all_gaussian_params:
        print("[submap] No submap params collected; aborting stitch.", flush=True)
        return

    # --- Merge submaps into global model -----------------------------------
    print(f"[submap] Merging {len(all_gaussian_params)} submaps into global model...", flush=True)

    # Use the pre-built gaussians object; extend it with all submap data
    for sm_params in all_gaussian_params:
        pts   = sm_params["means"].to("cuda")
        cols  = sm_params["sh0"].squeeze(1).to("cuda")  # (N,3) SH DC
        # Convert SH DC back to approximate RGB for add_points_as_gaussians API
        from utils.sh_utils import SH2RGB
        rgb_approx = SH2RGB(cols).clamp(0, 1)
        log_scales = sm_params["scales"].to("cuda")
        quats = sm_params["quats"].to("cuda")
        gaussians.add_points_as_gaussians(
            pts, rgb_approx,
            scales=log_scales, rotations=quats,
            init_opacity=0.3,
            is_provisional=False, birth_frame=0,
        )

    print(f"[submap] Global model: {gaussians.get_xyz.shape[0]} Gaussians total.", flush=True)

    # Save merged PLY
    merged_ply_dir = os.path.join(args.model_path, "point_cloud", "submap_merged")
    os.makedirs(merged_ply_dir, exist_ok=True)
    gaussians.save_ply(os.path.join(merged_ply_dir, "point_cloud.ply"))

    if refine_iters <= 0:
        print("[submap] No global refinement requested. Done.", flush=True)
        return

    # --- Global refinement over all cameras --------------------------------
    print(f"[submap] Running {refine_iters} global refinement iters over {len(all_train_cameras)} cameras...", flush=True)
    gaussians.training_setup(opt)
    _global_is_selective = getattr(gaussians, "optimizer_type", "adam") == "selective_adam"

    from utils.loss_utils import l1_loss, ssim as _ssim_fn
    import random as _rnd
    for _it in tqdm(range(1, refine_iters + 1), desc="Global refine"):
        cam = _rnd.choice(all_train_cameras)
        pkg = render(cam, gaussians, pipe, background)
        img = pkg["render"]
        gt = cam.original_image
        loss = (1.0 - opt.lambda_dssim) * l1_loss(img, gt) + opt.lambda_dssim * (1.0 - _ssim_fn(img, gt))
        loss.backward()
        if _it < refine_iters:
            if _global_is_selective:
                gaussians.prepare_selective_adam_step()
                _vis_all = torch.ones(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")
                gaussians.optimizer.step(visibility=_vis_all)
                gaussians.normalize_rotation_params()
            else:
                gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

    # Save final refined PLY
    refined_ply_dir = os.path.join(args.model_path, "point_cloud", "iteration_final")
    os.makedirs(refined_ply_dir, exist_ok=True)
    gaussians.save_ply(os.path.join(refined_ply_dir, "point_cloud.ply"))

    # Run evaluation if test cameras exist
    if streaming_scene.getTestCameras():
        try:
            from utils.comparison_report import write_post_training_report
            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            _eval_bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            write_post_training_report(
                args.model_path,
                refine_iters,
                gaussians,
                all_train_cameras,
                streaming_scene.getTestCameras(),
                render,
                pipe,
                _eval_bg,
                tb_writer=tb_writer,
                subdir="submap_final",
            )
        except Exception as _e:
            print(f"[submap] Evaluation failed: {_e}", flush=True)

    print("[submap] Stitch complete.", flush=True)


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

    # Write reproduce.sh (includes docker compose wrapper when running in container)
    import sys as _sys
    from utils.comparison_report import write_reproduce_sh
    write_reproduce_sh(os.path.join(args.model_path, "reproduce.sh"), _sys.argv)

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

    # Ensure SLAM lifecycle attributes exist for bootstrap Gaussians (birth_frame=0).
    # add_points_as_gaussians() creates these lazily on first insertion; initialise
    # them here so that --streaming_anchor_bootstrap works from iteration 1 even when
    # no depth insertion has yet occurred.
    if not hasattr(gaussians, "birth_frame"):
        _n_boot = gaussians.get_xyz.shape[0]
        _dev = gaussians.get_xyz.device
        gaussians.provisional = torch.zeros(_n_boot, dtype=torch.bool, device=_dev)
        gaussians.support_count = torch.zeros(_n_boot, dtype=torch.int32, device=_dev)
        gaussians.birth_frame = torch.zeros(_n_boot, dtype=torch.int32, device=_dev)

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
        ingestion_mode=getattr(args, "streaming_ingestion_mode", "iter_based"),
        fps_cap=getattr(args, "streaming_input_fps_cap", 30.0),
    )
    _iter_wall_start: float = time.perf_counter()  # for per-iter dt in dataset_fps mode

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
    mcmc_strategy.initialize_state(
        gaussians=gaussians, args=args,
        scene_scale=float(getattr(streaming_scene, "cameras_extent", 1.0)),
    )
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
    _prev_insert_frame = None       # previous frame for depth consistency check
    _prev_insert_depth_m = None     # depth map (metres) for previous frame
    depth_loss_weight = getattr(args, "streaming_depth_loss_weight", 0.0)
    use_depth_loss = depth_loss_weight > 0
    depth_loss_type = getattr(args, "streaming_depth_loss_type", "l1")
    streaming_mcmc_local = getattr(args, "streaming_mcmc_local_only", True)
    global_maint_interval = max(0, getattr(args, "streaming_global_maintenance_interval", 500))

    # Diagnostic training modes (H1/H2 ablation + H11 submap-stitch)
    _training_mode = getattr(args, "streaming_training_mode", "normal")
    _placement_only = _training_mode == "placement_only"   # H1: no backward/step/MCMC
    _colors_only    = _training_mode == "colors_only"       # H2: geometry frozen, SH trains
    _skip_mcmc      = _placement_only or _colors_only
    if _colors_only:
        for _pg in gaussians.optimizer.param_groups:
            if _pg.get("name") in ("means", "scales", "quats", "opacities"):
                _pg["lr"] = 0.0
        gaussians.xyz_scheduler_args = lambda _step: 0.0
    if _training_mode == "submap_stitch":
        # H11: dispatch to submap-stitching path before entering the main loop
        _run_submap_stitch(
            gaussians, streaming_scene, opt, pipe, args,
            background=None,  # built inside helper
            dataset=dataset, tb_writer=tb_writer, save_worker=save_worker,
            testing_iterations=testing_iterations, saving_iterations=saving_iterations,
            sh_degree_schedule=sh_degree_schedule,
        )
        save_worker.shutdown()
        return
    if _training_mode == "rolling_seed":
        _run_rolling_seed(
            gaussians, streaming_scene, opt, pipe, args,
            background=None,
            dataset=dataset, tb_writer=tb_writer, save_worker=save_worker,
            testing_iterations=testing_iterations, saving_iterations=saving_iterations,
            sh_degree_schedule=sh_degree_schedule,
        )
        save_worker.shutdown()
        return
    if _training_mode != "normal":
        print(f"[streaming] training_mode={_training_mode}: "
              f"placement_only={_placement_only} colors_only={_colors_only}", flush=True)
    scalar_log_interval = max(1, int(getattr(args, "scalar_log_interval", 10)))
    save_interval = max(0, int(getattr(args, "save_interval", 2000)))
    chk_interval = max(0, int(getattr(args, "checkpoint_interval", 2000)))
    save_frame_interval = max(0, int(getattr(args, "streaming_save_frame_interval", 50)))
    # Track which frame milestone we last saved at (avoids repeated saves for same frame)
    _last_frame_save = n_init  # bootstrap frames already "processed"

    # H7: old-geometry gradient freeze parameters
    _freeze_old = getattr(args, "streaming_freeze_old_geometry", False)
    _young_age_frames = max(0, int(getattr(args, "streaming_young_age_frames", 5)))
    _freeze_new_frame_steps = max(0, int(getattr(args, "streaming_freeze_new_frame_steps", 50)))
    _iter_since_new_frame = _freeze_new_frame_steps  # start without strict freeze

    # H9: anchor loss parameters
    _anchor_loss_weight = float(getattr(args, "streaming_anchor_loss_weight", 0.0))
    _anchor_decay_steps = max(1, int(getattr(args, "streaming_anchor_decay_steps", 500)))

    # Training-progress video setup
    _progress_video_interval = max(0, int(getattr(args, "progress_video_interval", 200)))
    _progress_video_fps = int(getattr(args, "progress_video_fps", 10))
    _progress_cams: list = []     # 3 equally-spaced fixed cameras, locked in once enough are available
    _progress_frames: list = []   # accumulated side-by-side uint8 HWC numpy frames

    # Async render: renders go on a side CUDA stream; CPU work (numpy/cv2) goes to a daemon thread.
    # This overlaps GPU renders and CPU image assembly with the next training iteration.
    import threading as _threading
    import queue as _queue_mod
    _pv_stream = torch.cuda.Stream()
    _pv_cpu_queue: _queue_mod.Queue = _queue_mod.Queue(maxsize=2)

    def _pv_cpu_worker():
        while True:
            item = _pv_cpu_queue.get()
            if item is None:
                break
            _gpu_panels, _pv_iter, _pv_total = item
            _panels = []
            for _t in _gpu_panels:
                _arr = _t.cpu().numpy()  # waits for _pv_stream to finish this tensor
                _arr = np.transpose(_arr, (1, 2, 0))
                _arr = (_arr * 255 + 0.5).astype(np.uint8)
                _panels.append(_arr)
            _sep = np.ones((_panels[0].shape[0], 2, 3), dtype=np.uint8) * 80
            _combined = np.concatenate([_panels[0], _sep, _panels[1], _sep, _panels[2]], axis=1)
            try:
                import cv2 as _cv2
                _lbl = f"iter {_pv_iter:>6d} / {_pv_total}"
                _cv2.putText(_combined, _lbl, (8, 20), _cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, _cv2.LINE_AA)
                _cv2.putText(_combined, _lbl, (8, 20), _cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, _cv2.LINE_AA)
            except Exception:
                pass
            _progress_frames.append(_combined)

    _pv_thread = _threading.Thread(target=_pv_cpu_worker, daemon=True)
    _pv_thread.start()

    # Four-state lifecycle parameters (Component B/C)
    _lifecycle_enabled = getattr(args, "streaming_lifecycle_enabled", False)
    _mature_age_frames = max(1, int(getattr(args, "streaming_mature_age_frames", 15)))
    _mature_min_utility = float(getattr(args, "streaming_mature_min_utility", 0.1))
    _freeze_age_frames = int(getattr(args, "streaming_freeze_age_frames", -1))
    _utility_ema_beta = float(getattr(args, "streaming_utility_ema_beta", 0.95))
    # Mature-Gaussian anchor loss weights (Component C)
    _mature_anchor_xyz_w = float(getattr(args, "streaming_mature_anchor_xyz_weight", 0.0))
    _mature_anchor_scale_w = float(getattr(args, "streaming_mature_anchor_scale_weight", 0.0))
    _mature_anchor_opacity_w = float(getattr(args, "streaming_mature_anchor_opacity_weight", 0.0))
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

    # Snapshot the bootstrap camera list so the end-of-training reporter can
    # render the SAME views with the trained Gaussians for a direct diff.
    _bootstrap_cams = list(streaming_scene.getTrainCameras())

    # Pre-training snapshot: render the post-bootstrap state through the
    # bootstrap cameras themselves (no holdout exists yet — those frames
    # are only added as cameras stream in).
    try:
        from utils.comparison_report import write_post_training_report
        print(
            f"[streaming] Bootstrap (post-init) report at iter_0 over "
            f"{len(_bootstrap_cams)} bootstrap views...",
            flush=True,
        )
        write_post_training_report(
            model_path=args.model_path,
            iteration=0,
            gaussians=gaussians,
            train_cams=_bootstrap_cams,
            test_cams=_bootstrap_cams,
            render_fn=render,
            pipe=pipe,
            background=background,
            tb_writer=tb_writer,
            log_prefix="streaming_report",
            subdir="iter_0_bootstrap_views",
        )
    except Exception as e:
        print(f"[streaming-report] pre-training report failed: {e}", flush=True)

    progress_bar = tqdm(
        range(first_iter, opt.iterations),
        desc="Streaming training",
        disable=getattr(args, "quiet", False),
    )
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        set_compile_iteration(iteration)

        # Measure per-iteration wall time for dataset_fps simulated clock
        _now = time.perf_counter()
        _iter_dt = _now - _iter_wall_start
        _iter_wall_start = _now

        # ---- Frame ingestion -----------------------------------------------
        if streaming_scene.has_next_frame() and scheduler.should_release(iteration, dt=_iter_dt):
            result = streaming_scene.ingest_next_frame()
            if result is not None:
                new_cam, new_frame, is_train = result
                scheduler.mark_released()
                n_frames_ingested += 1
                _iter_since_new_frame = 0  # H7: reset freeze counter on new frame

                # Phase 2: insert new Gaussians from depth
                if is_train and getattr(args, "streaming_insert_from_depth", True):
                    cap = getattr(args, "cap_max", -1)
                    current_n = gaussians.get_xyz.shape[0]
                    if cap <= 0 or current_n < cap:
                        # Step 4: render new_cam to get alpha/depth mask
                        with torch.no_grad():
                            pkg_new = render(new_cam, gaussians, pipe, background, render_depth=True)
                            alpha_new = pkg_new["alpha"]
                            depth_new = pkg_new.get("rendered_depth", None)
                        
                        added, _insert_stats = insert_gaussians_from_frame(
                            gaussians, new_frame, args,
                            streaming_scene=streaming_scene,
                            prev_frame=_prev_insert_frame,
                            prev_depth_meters=_prev_insert_depth_m,
                            render_alpha=alpha_new,
                            render_depth=depth_new,
                            current_frame_idx=n_frames_ingested,
                        )
                        # H9: record insertion iteration for anchor-loss decay
                        if added > 0 and hasattr(gaussians, "anchor_iter"):
                            gaussians.anchor_iter[-added:] = iteration
                        total_inserted += added
                        if added > 0 and (iteration % 100 == 0 or added > 1000):
                            print(
                                f"[streaming] iter={iteration} frame={n_frames_ingested} "
                                f"inserted={added} total_inserted={total_inserted} N={gaussians.get_xyz.shape[0]}",
                                flush=True,
                            )
                        # Log per-filter insertion telemetry
                        if tb_writer and _insert_stats:
                            for _k, _v in _insert_stats.items():
                                tb_writer.add_scalar(f"streaming/insertion/{_k}", _v, iteration)
                    # Cache this frame as the previous frame for next insertion
                    _prev_insert_frame = new_frame
                    _prev_insert_depth_m = _load_depth_meters(new_frame)

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
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, update_sh_rest=update_sh_rest,
                            render_depth=use_depth_loss)
        image = render_pkg["render"]
        
        _sync_streaming_state_lengths(gaussians)

        # Support update for provisional Gaussians (Step 8)
        _update_provisional_support(gaussians, viewpoint_cam, render_pkg, args)

        # SLAM Lifecycle: Promote / Prune (Step 8)
        # Skipped in placement_only — pruning would invalidate render_pkg tensors
        # before the loss computation reads visibility_filter.
        if not _placement_only and iteration % 100 == 0:
            with torch.no_grad():
                # Promote
                promote_mask = gaussians.provisional & (gaussians.support_count >= args.streaming_min_support_views)
                if promote_mask.any():
                    gaussians.provisional[promote_mask] = False
                    # Boost opacity to signal permanence
                    target_op = inverse_sigmoid(torch.tensor(args.streaming_promote_opacity, device="cuda"))
                    if model_layout == "gsplat":
                        gaussians.params["opacities"].data[promote_mask] = target_op
                    else:
                        gaussians._opacity.data[promote_mask] = target_op
                    print(f"[streaming] iter={iteration} promoted {promote_mask.sum().item()} points to permanent structure.", flush=True)

                # Prune stale low-support points
                age = n_frames_ingested - gaussians.birth_frame
                stale_mask = gaussians.provisional & (age > args.streaming_provisional_max_age)
                if stale_mask.any():
                    print(f"[streaming] iter={iteration} pruning {stale_mask.sum().item()} stale provisional points.", flush=True)
                    gaussians.prune_points(stale_mask)
                    # Force occupancy hash update after pruning
                    streaming_scene.maintain_occupancy_hash(getattr(args, "streaming_insert_voxel_size", 0.02))

                # Periodic rebuild regardless of pruning — frees voxels of relocated Gaussians
                _occ_rebuild_every = getattr(args, "streaming_occupancy_rebuild_interval", 200)
                if _occ_rebuild_every > 0 and iteration % _occ_rebuild_every == 0:
                    streaming_scene.maintain_occupancy_hash(getattr(args, "streaming_insert_voxel_size", 0.02))

        # If N changed this iteration (insertion or pruning), resize render_pkg visibility
        # so all downstream code (loss, MCMC, energy, grad-zeroing) sees a consistent size.
        _cur_n_after = gaussians.get_xyz.shape[0]
        _vf = render_pkg.get("visibility_filter")
        if _vf is not None and _vf.shape[0] != _cur_n_after:
            _vf_safe = torch.zeros(_cur_n_after, dtype=torch.bool, device=_vf.device)
            _vf_safe[:min(_vf.shape[0], _cur_n_after)] = _vf[:min(_vf.shape[0], _cur_n_after)]
            render_pkg["visibility_filter"] = _vf_safe

        # ---- Loss ---------------------------------------------------------
        gt_image = viewpoint_cam.original_image
        Ll1 = l1_loss(image, gt_image)
        _ssim = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - _ssim)

        # Regularisation: local (visible) or global depending on active-set mode
        from utils.compiled_kernels import active_reg_core
        if sparse_active_set:
            # Step 10: Refine active set to include provisional/new points
            _visible = render_pkg["visibility_filter"].detach()
            _provisional = gaussians.provisional.detach()
            _active = _visible | _provisional
            
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

        # ---- Depth supervision loss ----------------------------------------
        _depth_loss_val = None
        if use_depth_loss and "rendered_depth" in render_pkg:
            sensor_d_cpu = _get_sensor_depth(
                viewpoint_cam, image.shape[1], image.shape[2]
            )
            if sensor_d_cpu is not None:
                sensor_d = sensor_d_cpu.to(device="cuda", non_blocking=True)
                rend_d = render_pkg["rendered_depth"]  # [1, H, W]
                valid = (sensor_d > 0) & (rend_d.detach() > 0)
                if valid.any():
                    if depth_loss_type == "huber":
                        L_depth = torch.nn.functional.huber_loss(
                            rend_d[valid], sensor_d[valid], reduction="mean", delta=0.1
                        )
                    else:
                        L_depth = (rend_d[valid] - sensor_d[valid]).abs().mean()
                    loss = loss + depth_loss_weight * L_depth
                    _depth_loss_val = L_depth.item()

        # ---- H9: Anchor loss for young provisional splats ------------------
        # Penalise drift from the depth-insertion position while the splat is
        # young. Weight decays linearly to zero over anchor_decay_steps iters.
        _anchor_loss_val = None
        if _anchor_loss_weight > 0 and hasattr(gaussians, "anchor_xyz"):
            with torch.no_grad():
                age_iters = (iteration - gaussians.anchor_iter.long()).clamp(min=0)
                young_mask = (age_iters < _anchor_decay_steps) & gaussians.provisional
            if young_mask.any():
                decay = (1.0 - age_iters[young_mask].float() / _anchor_decay_steps).clamp(0.0, 1.0)
                dxyz = gaussians.get_xyz[young_mask] - gaussians.anchor_xyz[young_mask]
                L_anchor = (decay.unsqueeze(1) * dxyz.pow(2)).sum(dim=1).mean()
                loss = loss + _anchor_loss_weight * L_anchor
                _anchor_loss_val = L_anchor.item()

        # ---- Mature-Gaussian soft-anchor anti-fade losses (Component C) -----
        # Constant-weight (non-decaying) anchor losses on xyz/scale/opacity for
        # MATURE Gaussians. Prevents confirmed geometry from fading when old
        # frames leave the training window.
        _mature_anchor_val = None
        if _lifecycle_enabled and _mature_anchor_xyz_w + _mature_anchor_scale_w + _mature_anchor_opacity_w > 0 and hasattr(gaussians, "lifecycle_state"):
            with torch.no_grad():
                mature_mask = (gaussians.lifecycle_state >= 2)  # MATURE or FROZEN
            if mature_mask.any():
                L_ma = torch.tensor(0.0, device="cuda")
                if _mature_anchor_xyz_w > 0 and hasattr(gaussians, "anchor_xyz"):
                    dxyz = gaussians.get_xyz[mature_mask] - gaussians.anchor_xyz[mature_mask]
                    L_ma = L_ma + _mature_anchor_xyz_w * dxyz.pow(2).sum(dim=1).mean()
                if _mature_anchor_scale_w > 0 and hasattr(gaussians, "anchor_scale_log"):
                    if model_layout == "gsplat":
                        d_scale = gaussians.params["scales"][mature_mask] - gaussians.anchor_scale_log[mature_mask]
                    else:
                        d_scale = gaussians._scaling[mature_mask] - gaussians.anchor_scale_log[mature_mask]
                    L_ma = L_ma + _mature_anchor_scale_w * d_scale.pow(2).mean()
                if _mature_anchor_opacity_w > 0 and hasattr(gaussians, "anchor_opacity_logit"):
                    if model_layout == "gsplat":
                        d_op = gaussians.params["opacities"][mature_mask] - gaussians.anchor_opacity_logit[mature_mask]
                    else:
                        d_op = gaussians._opacity.squeeze(-1)[mature_mask] - gaussians.anchor_opacity_logit[mature_mask]
                    L_ma = L_ma + _mature_anchor_opacity_w * d_op.pow(2).mean()
                loss = loss + L_ma
                _mature_anchor_val = L_ma.item()

        # ---- Free-space / floater loss -------------------------------------
        # Penalise opacity that is rendered in front of the observed surface.
        # Targets mid-air splats that photometric loss would otherwise keep.
        _free_loss_val = None
        free_space_weight = getattr(args, "streaming_free_space_loss_weight", 0.0)
        if free_space_weight > 0 and use_depth_loss and "alpha" in render_pkg and "rendered_depth" in render_pkg:
            sensor_d_fs = _get_sensor_depth(viewpoint_cam, image.shape[1], image.shape[2])
            if sensor_d_fs is not None:
                sensor_d_fs = sensor_d_fs.to("cuda", non_blocking=True)
                rend_d_fs = render_pkg["rendered_depth"]         # [1, H, W]
                # render_alphas[0] from gsplat is [H, W, 1]; permute to [1, H, W]
                rend_a_raw = render_pkg["alpha"]
                if rend_a_raw.dim() == 3 and rend_a_raw.shape[-1] == 1:
                    rend_a_fs = rend_a_raw.permute(2, 0, 1)     # [1, H, W]
                else:
                    rend_a_fs = rend_a_raw.unsqueeze(0) if rend_a_raw.dim() == 2 else rend_a_raw
                # Floater: rendered alpha is significant AND rendered depth is
                # shallower than the sensor surface by at least 5 cm.
                floater = (
                    (sensor_d_fs > 0)
                    & (rend_d_fs.detach() > 0)
                    & (rend_a_fs.detach() > 0.2)
                    & (rend_d_fs.detach() < sensor_d_fs - 0.05)
                )
                if floater.any():
                    L_free = (rend_a_fs[floater] * (sensor_d_fs[floater] - rend_d_fs[floater]).detach()).mean()
                    loss = loss + free_space_weight * L_free
                    _free_loss_val = L_free.item()

        # ---- Stratified sampling state update --------------------------------
        _loss_val = loss.item()
        streaming_scene.update_stratified_state(viewpoint_cam, _loss_val)

        # ---- Backward + optimizer step (skipped in placement_only mode) -----
        visible = render_pkg["visibility_filter"].detach().to(dtype=torch.bool).contiguous()
        if not _placement_only:
            mcmc_strategy.step_pre_backward(
                gaussians=gaussians, args=args, iteration=iteration,
                render_pkg=render_pkg, loss=loss,
            )
            loss.backward()

            # Zero invisible grad rows in strided grads (active-set safety)
            if sparse_active_set and getattr(args, "selective_adam_zero_invisible_grads", True):
                _mask = visible  # already resized above
                for group in gaussians.optimizer.param_groups:
                    p = group["params"][0]
                    if p.grad is not None and getattr(p.grad, "layout", torch.strided) == torch.strided:
                        p.grad[~_mask] = 0.0

            # H7: freeze confirmed old geometry — zero geometry gradients for splats
            # that are old enough not to be "young" and are not provisional.
            # Prevents optimizer from dragging confirmed good splats to explain new views.
            _iter_since_new_frame += 1
            if _freeze_old and hasattr(gaussians, "birth_frame") and hasattr(gaussians, "provisional"):
                with torch.no_grad():
                    _frame_age = n_frames_ingested - gaussians.birth_frame.long()
                    _young = gaussians.provisional | (_frame_age <= _young_age_frames)
                    _old = ~_young
                    if _old.any():
                        if model_layout == "gsplat":
                            for _pg_name in ("means", "scales", "quats"):
                                _p = gaussians.params.get(_pg_name)
                                if (_p is not None and _p.grad is not None
                                        and getattr(_p.grad, "layout", torch.strided) == torch.strided):
                                    _p.grad[_old] = 0.0
                        else:
                            for _param in (gaussians._xyz, gaussians._scaling, gaussians._rotation):
                                if (_param.grad is not None
                                        and getattr(_param.grad, "layout", torch.strided) == torch.strided):
                                    _param.grad[_old] = 0.0

            # Lifecycle: zero all gradients for FROZEN Gaussians (lifecycle_state==3)
            if _lifecycle_enabled and hasattr(gaussians, "lifecycle_state"):
                _frozen_mask = gaussians.lifecycle_state == 3
                if _frozen_mask.any():
                    with torch.no_grad():
                        if model_layout == "gsplat":
                            for _pg_name in ("means", "scales", "quats", "opacities", "sh0", "shN"):
                                _p = gaussians.params.get(_pg_name)
                                if (_p is not None and _p.grad is not None
                                        and getattr(_p.grad, "layout", torch.strided) == torch.strided):
                                    _p.grad[_frozen_mask] = 0.0
                        else:
                            for _param in (gaussians._xyz, gaussians._scaling, gaussians._rotation,
                                           gaussians._opacity, gaussians._features_dc, gaussians._features_rest):
                                if (_param.grad is not None
                                        and getattr(_param.grad, "layout", torch.strided) == torch.strided):
                                    _param.grad[_frozen_mask] = 0.0

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

                # MCMC noise injection — skipped in placement_only / colors_only modes
                if not _skip_mcmc:
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

        # ---- Lifecycle: update utility_ema and advance states ---------------
        if _lifecycle_enabled and iteration % 100 == 0 and hasattr(gaussians, "lifecycle_state"):
            with torch.no_grad():
                # Update per-Gaussian utility EMA using last computed utility score
                if utility is not None and hasattr(gaussians, "utility_ema"):
                    beta = _utility_ema_beta
                    u_score = utility.detach().clamp(min=0.0)
                    gaussians.utility_ema.mul_(beta).add_(u_score, alpha=(1.0 - beta))

                lc = gaussians.lifecycle_state  # [N] int8
                frame_age = (n_frames_ingested - gaussians.birth_frame.long()).clamp(min=0)

                # PROVISIONAL (0) → YOUNG (1): existing promote logic already sets provisional=False;
                # keep in sync by upgrading lifecycle_state to YOUNG for newly promoted.
                newly_promoted = (~gaussians.provisional) & (lc == 0)
                if newly_promoted.any():
                    gaussians.lifecycle_state[newly_promoted] = 1  # YOUNG

                # YOUNG (1) → MATURE (2)
                young_mask = lc == 1
                if young_mask.any():
                    age_ok = frame_age >= _mature_age_frames
                    if hasattr(gaussians, "utility_ema") and _mature_min_utility > 0:
                        util_ok = gaussians.utility_ema >= _mature_min_utility
                    else:
                        util_ok = torch.ones_like(young_mask)
                    to_mature = young_mask & age_ok & util_ok
                    if to_mature.any():
                        gaussians.lifecycle_state[to_mature] = 2  # MATURE
                        # Re-snapshot anchors at the mature pose (lock-in target)
                        if hasattr(gaussians, "anchor_scale_log"):
                            gaussians.anchor_scale_log[to_mature] = gaussians.params["scales"][to_mature].detach().clone()
                        if hasattr(gaussians, "anchor_opacity_logit"):
                            gaussians.anchor_opacity_logit[to_mature] = gaussians.params["opacities"][to_mature].detach().clone()

                # MATURE (2) → FROZEN (3): only when freeze_age_frames >= 0
                if _freeze_age_frames >= 0:
                    mature_mask = lc == 2
                    if mature_mask.any():
                        to_freeze = mature_mask & (frame_age >= _freeze_age_frames)
                        if to_freeze.any():
                            gaussians.lifecycle_state[to_freeze] = 3  # FROZEN

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

        # Anchor bootstrap Gaussians: save positions before step_post_backward
        # (which includes noise injection) and restore them after, so that MCMC
        # noise never displaces the initial geometry.
        _anchor_bootstrap = getattr(args, "streaming_anchor_bootstrap", False)
        _anchor_mask = None
        _anchor_pos = None
        if _anchor_bootstrap and hasattr(gaussians, "birth_frame"):
            _anchor_mask = (gaussians.birth_frame == 0)
            if _anchor_mask.any():
                _anchor_pos = gaussians.get_xyz[_anchor_mask].detach().clone()

        if not _skip_mcmc:
            mcmc_strategy.step_post_backward(
                gaussians=gaussians, args=args, sched=sched, iteration=iteration,
                utility=utility, temperature=temperature, use_energy_mcmc=use_energy_mcmc,
                tb_writer=tb_writer,
                should_log_strategy=lambda i: i % max(1, getattr(args, "strategy_log_interval", 500)) == 0,
                render_pkg=render_pkg, lr=xyz_lr,
            )

            # Restore bootstrap positions after noise injection
            if _anchor_mask is not None and _anchor_pos is not None:
                if gaussians.get_xyz.shape[0] == _anchor_mask.shape[0]:
                    if hasattr(gaussians, "params"):
                        gaussians.params["means"].data[_anchor_mask] = _anchor_pos
                    else:
                        gaussians._xyz.data[_anchor_mask] = _anchor_pos

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

            # ---- Training-progress video capture (async) --------------------
            if _progress_video_interval > 0 and iteration % _progress_video_interval == 0:
                # Lock in 3 cameras once enough trajectory is covered.
                # Require ≥3 test cameras AND ≥20% of frames ingested so the
                # selected cameras span meaningfully different viewpoints.
                if not _progress_cams:
                    _total_frames_avail = len(streaming_scene._all_frames)
                    # Wait until 20% of the trajectory has been ingested so that
                    # the first, middle, and last of the pool span genuinely
                    # different viewpoints (not just the same position 3×).
                    _lock_at = max(30, _total_frames_avail // 5)
                    if n_frames_ingested >= _lock_at:
                        _tcams = list(streaming_scene.getTestCameras())
                        _trcams = list(streaming_scene.train_cameras)
                        _pool = _tcams if len(_tcams) >= 3 else (_trcams if len(_trcams) >= 3 else None)
                        if _pool is not None:
                            _n = len(_pool)
                            _progress_cams = [_pool[0], _pool[(_n - 1) // 2], _pool[_n - 1]]
                if _progress_cams:
                    # Submit 3 renders to side CUDA stream (non-blocking for main stream)
                    _gpu_panels = []
                    with torch.cuda.stream(_pv_stream):
                        _pv_stream.wait_stream(torch.cuda.current_stream())
                        with torch.no_grad():
                            for _pvc in _progress_cams:
                                _pv_pkg = render(_pvc, gaussians, pipe, background)
                                _gpu_panels.append(_pv_pkg["render"].detach().clamp(0, 1))
                    # Hand off to CPU worker; drop frame if worker is still busy
                    try:
                        _pv_cpu_queue.put_nowait((_gpu_panels, iteration, opt.iterations))
                    except _queue_mod.Full:
                        pass

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
                if scheduler.frames_dropped > 0:
                    tb_writer.add_scalar("streaming/frames_dropped", scheduler.frames_dropped, iteration)
                if _depth_loss_val is not None:
                    tb_writer.add_scalar("train/depth_loss", _depth_loss_val, iteration)
                if _free_loss_val is not None:
                    tb_writer.add_scalar("train/free_space_loss", _free_loss_val, iteration)
                if _anchor_loss_val is not None:
                    tb_writer.add_scalar("train/anchor_loss", _anchor_loss_val, iteration)
                if _mature_anchor_val is not None:
                    tb_writer.add_scalar("train/mature_anchor_loss", _mature_anchor_val, iteration)
                # Lifecycle state counts (when enabled)
                if _lifecycle_enabled and hasattr(gaussians, "lifecycle_state"):
                    lc = gaussians.lifecycle_state
                    tb_writer.add_scalar("streaming/lifecycle/n_provisional", int((lc == 0).sum()), iteration)
                    tb_writer.add_scalar("streaming/lifecycle/n_young", int((lc == 1).sum()), iteration)
                    tb_writer.add_scalar("streaming/lifecycle/n_mature", int((lc == 2).sum()), iteration)
                    tb_writer.add_scalar("streaming/lifecycle/n_frozen", int((lc == 3).sum()), iteration)

            if iteration == opt.iterations:
                progress_bar.close()

            # Test evaluation: full comparison report (renders + contact sheet + MP4)
            if iteration in testing_iterations:
                try:
                    from utils.comparison_report import write_post_training_report
                    write_post_training_report(
                        model_path=args.model_path,
                        iteration=iteration,
                        gaussians=gaussians,
                        train_cams=list(streaming_scene.getTrainCameras()),
                        test_cams=list(streaming_scene.getTestCameras()),
                        render_fn=render,
                        pipe=pipe,
                        background=background,
                        tb_writer=tb_writer,
                        log_prefix="streaming_report",
                    )
                except Exception as _e:
                    print(f"[streaming-report] mid-training report failed iter={iteration}: {_e}", flush=True)
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
                if getattr(args, "streaming_render_at_saves", False):
                    _render_streaming_snapshot(
                        gaussians, streaming_scene, render, pipe, background,
                        args, n_frames_ingested, tb_writer, save_worker=save_worker,
                    )

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

    # Mandatory post-training report: test PSNR + side-by-side PNGs +
    # contact sheet + trajectory MP4. Always runs (independent of the
    # opt-in --streaming_render_at_saves milestone snapshots).
    try:
        from utils.comparison_report import write_post_training_report
        write_post_training_report(
            model_path=args.model_path,
            iteration=opt.iterations,
            gaussians=gaussians,
            train_cams=list(streaming_scene.getTrainCameras()),
            test_cams=list(streaming_scene.getTestCameras()),
            render_fn=render,
            pipe=pipe,
            background=background,
            tb_writer=tb_writer,
            log_prefix="streaming_report",
        )
        # Re-render the bootstrap views with the trained Gaussians so the
        # iter_0 vs end-of-training comparison is over the same viewpoints.
        if _bootstrap_cams:
            write_post_training_report(
                model_path=args.model_path,
                iteration=opt.iterations,
                gaussians=gaussians,
                train_cams=_bootstrap_cams,
                test_cams=_bootstrap_cams,
                render_fn=render,
                pipe=pipe,
                background=background,
                tb_writer=tb_writer,
                log_prefix="streaming_report",
                subdir=f"iter_{opt.iterations}_bootstrap_views",
            )
    except Exception as e:
        print(f"[streaming-report] post-training report failed: {e}", flush=True)

    # ---- Training-progress video write ---------------------------------------
    # Signal the CPU worker to stop, then wait for it to finish any in-flight frames.
    _pv_cpu_queue.put(None)
    _pv_thread.join(timeout=120)
    if _progress_frames:
        from utils.comparison_report import _write_mp4
        _prog_path = os.path.join(args.model_path, "training_progress.mp4")
        print(f"[streaming] Writing training_progress.mp4 ({len(_progress_frames)} frames @ {_progress_video_fps}fps)…",
              flush=True)
        _err = _write_mp4(_prog_path, iter(_progress_frames), fps=_progress_video_fps)
        if _err is None:
            print(f"[streaming] training_progress.mp4 → {_prog_path}", flush=True)
        else:
            print(f"[streaming] training_progress.mp4 FAILED: {_err}", flush=True)

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
