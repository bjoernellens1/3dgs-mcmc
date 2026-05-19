"""
StreamingScene: manages arriving RGB-D frames for streaming-replay training.

Replaces the full-dataset Scene for --streaming_replay mode. Frames arrive
one at a time (controlled by FrameScheduler in train_streaming.py), each
converting to a Camera object and appending to the local keyframe window and
replay buffer. The Gaussian model is initialised from the first
streaming_initial_frames frames only; later frames insert Gaussians
incrementally via add_points_as_gaussians() (Phase 2).

Thread-safety: not re-entrant. All methods called from the training loop only.
"""
from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch

if TYPE_CHECKING:
    from utils.streaming_frames import StreamingRGBDFrame


class StreamingScene:
    def __init__(self, args, gaussians, frame_source, resolution_scale: float = 1.0):
        self.args = args
        self.gaussians = gaussians
        self.resolution_scale = resolution_scale
        self.model_path = args.model_path

        # All cameras that have "arrived" so far (in order)
        self.train_cameras: List = []
        # Hold-out cameras for test evaluation (never used in training)
        self._test_cameras: List = []
        # Ring buffer of the last streaming_replay_buffer cameras
        self._replay_buffer: List = []
        self.current_camera = None
        self.cameras_extent = 1.0  # updated during initialize_from_frames
        self._source_idx = 0
        self._frame_source = frame_source
        self._lazy_frames = bool(getattr(frame_source, "lazy_frames", False))
        self._all_frames = None if self._lazy_frames else list(frame_source)
        self._frame_count = len(frame_source) if self._lazy_frames else len(self._all_frames)
        self._keyframes_admitted = 0
        self._keyframes_rejected = 0

        # Persistent occupancy hash for incremental insertion
        self._occupancy_keys: np.ndarray = np.empty(0, dtype=np.int64)
        self.occupancy_voxel_size = 0.02

        # H8: warmup counter — steps since the last new frame arrived
        self._steps_since_new_frame: int = 0

        # H10: permanent global keyframe reservoir (never evicted, sampled for replay)
        self._global_reservoir: List = []
        self._n_train_frames_ingested: int = 0  # count of train frames (excl. holdout)

        # Stratified sampling support (Component D)
        # Hard-frame heap: list of (loss, camera) — capped at streaming_hard_frame_history
        self._hard_frames: List = []
        # Covisible sets: {cam_uid -> List[cam]} of recently-covisible cameras
        self._last_gaussian_ids: Optional[torch.Tensor] = None  # gaussian_ids from last render
        self._covisible_cache: List = []  # flat list of covisible cameras (refreshed periodically)

    def update_cameras_extent(self) -> None:
        """Grow cameras_extent to encompass all arrived camera positions (never shrinks)."""
        if not self.train_cameras:
            return
        try:
            centers = np.stack([c.camera_center.cpu().numpy() for c in self.train_cameras])
            centroid = centers.mean(axis=0)
            new_extent = float(np.max(np.linalg.norm(centers - centroid, axis=1)))
            self.cameras_extent = max(self.cameras_extent, new_extent, 1.0)
        except Exception:
            pass

    def _xyz_to_keys(self, xyz_np: np.ndarray, voxel_size: float) -> np.ndarray:
        offset = 1000.0
        coords = np.floor((xyz_np + offset) / max(voxel_size, 1e-6)).astype(np.int64)
        return (coords[:, 0] << 42) | (coords[:, 1] << 21) | coords[:, 2]

    def maintain_occupancy_hash(self, voxel_size: float = 0.02):
        """Rebuild the occupancy array from current Gaussian positions."""
        self.occupancy_voxel_size = voxel_size
        xyz = self.gaussians.get_xyz.detach().cpu().numpy()
        if xyz.shape[0] == 0:
            self._occupancy_keys = np.empty(0, dtype=np.int64)
            return
        keys = self._xyz_to_keys(xyz, voxel_size)
        self._occupancy_keys = np.sort(np.unique(keys))

    def add_to_occupancy_hash(self, xyz: torch.Tensor):
        """Incrementally merge new points into the sorted occupancy array."""
        if xyz.shape[0] == 0:
            return
        new_keys = np.sort(np.unique(self._xyz_to_keys(xyz.detach().cpu().numpy(), self.occupancy_voxel_size)))
        if not hasattr(self, "_occupancy_keys") or self._occupancy_keys.size == 0:
            self._occupancy_keys = new_keys
        else:
            self._occupancy_keys = np.union1d(self._occupancy_keys, new_keys)

    def check_occupancy(self, xyz_np: np.ndarray, voxel_size: float, check_neighbors: bool = True) -> np.ndarray:
        """Check whether points fall into occupied voxels.

        voxel_size must equal self.occupancy_voxel_size; the hash is keyed by that
        size so a mismatched query would silently return wrong results.
        """
        assert abs(voxel_size - self.occupancy_voxel_size) < 1e-9, (
            f"check_occupancy voxel_size={voxel_size} must equal "
            f"occupancy_voxel_size={self.occupancy_voxel_size}"
        )
        keys_arr = getattr(self, "_occupancy_keys", None)
        if keys_arr is None or keys_arr.size == 0 or xyz_np.shape[0] == 0:
            return np.zeros(xyz_np.shape[0], dtype=bool)

        offset = 1000.0
        coords = np.floor((xyz_np + offset) / max(voxel_size, 1e-6)).astype(np.int64)
        keys = (coords[:, 0] << 42) | (coords[:, 1] << 21) | coords[:, 2]

        occupied = np.isin(keys, keys_arr)

        if check_neighbors and not occupied.all():
            # Pre-generate all 26 neighbour direction vectors once
            dirs = np.array(
                [[dx, dy, dz]
                 for dx in (-1, 0, 1)
                 for dy in (-1, 0, 1)
                 for dz in (-1, 0, 1)
                 if not (dx == 0 and dy == 0 and dz == 0)],
                dtype=np.int64,
            )
            for d in dirs:
                remaining = ~occupied
                if not remaining.any():
                    break
                n_keys = ((coords[remaining, 0] + d[0]) << 42
                          | (coords[remaining, 1] + d[1]) << 21
                          | (coords[remaining, 2] + d[2]))
                occupied[remaining] |= np.isin(n_keys, keys_arr)
        return occupied

    def iter_all_frames(self):
        """Iterate all source frames regardless of whether the source is lazy or eager."""
        for i in range(self._frame_count):
            yield self._get_source_frame(i)

    def drain_pending_odometry(self) -> None:
        """Force the live-odom producer to compute every source frame."""
        drain = getattr(self._frame_source, "drain_pending", None)
        if callable(drain):
            drain()

    def stop_pending_odometry(self) -> None:
        """Stop background live odometry without computing unseen frames."""
        stop = getattr(self._frame_source, "stop_pending", None)
        if callable(stop):
            stop()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _get_source_frame(self, index: int) -> "StreamingRGBDFrame":
        if self._lazy_frames:
            return self._frame_source.get_frame(index)
        return self._all_frames[index]

    def initialize_from_frames(self, n_frames: int) -> None:
        """
        Ingest the first n_frames as initial cameras and build the initial
        Gaussian point cloud from their depth data.

        Must be called BEFORE gaussians.training_setup() so the optimizer is
        built on the correct initial parameter tensors.
        """
        from scene.cameras import prepare_camera_for_render

        n_frames = min(n_frames, self._frame_count)
        if n_frames == 0:
            raise RuntimeError("[streaming] No frames available for initialisation.")

        print(f"[streaming] Bootstrapping from {n_frames} frame(s)...", flush=True)
        initial_frames = [self._get_source_frame(i) for i in range(n_frames)]

        for frame in initial_frames:
            cam = self._frame_to_camera(frame)
            self.train_cameras.append(cam)
            self._replay_buffer.append(cam)
            prepare_camera_for_render(cam, device="cuda")

        self._source_idx = n_frames
        self.current_camera = self.train_cameras[-1]

        # Estimate scene extent from camera positions
        camera_extent = self._estimate_cameras_extent(initial_frames)

        # Build initial point cloud from depth images
        from utils.graphics_utils import BasicPointCloud
        pcd = self._build_pcd_from_frames(initial_frames)

        # Incorporate PCD bbox into extent
        pcd_extent = 0.0
        if pcd.points.shape[0] > 0:
            mins = np.min(pcd.points, axis=0)
            maxs = np.max(pcd.points, axis=0)
            pcd_extent = np.linalg.norm(maxs - mins) * 0.5

        # Metric scale: enforce a minimum radius of 1.0 for stability
        self.cameras_extent = max(camera_extent, pcd_extent, 1.0)

        self.gaussians.create_from_pcd(
            pcd,
            self.cameras_extent,
            init_scale_mode=getattr(self.args, "init_scale_mode", "fixed"),
            init_scale=getattr(self.args, "init_scale", 0.01),
            voxel_size=getattr(self.args, "pcd_voxel_size", 0.02),
        )
        self._export_initial_seed_pointcloud(pcd)
        self.maintain_occupancy_hash(voxel_size=getattr(self.args, "streaming_insert_voxel_size", 0.02))
        print(
            f"[streaming] Initialised {self.gaussians.get_xyz.shape[0]} Gaussians "
            f"from {n_frames} frames, scene radius={self.cameras_extent:.3f}",
            flush=True,
        )

    def _export_initial_seed_pointcloud(self, pcd) -> None:
        """Write the bootstrap RGB-D point cloud into the run output for inspection."""
        try:
            from scene.readers.common import storePly

            seed_dir = os.path.join(self.model_path, "point_cloud", "initial_seed")
            os.makedirs(seed_dir, exist_ok=True)
            points = np.asarray(pcd.points, dtype=np.float32)
            colors = np.clip(np.asarray(pcd.colors, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
            normals = np.asarray(getattr(pcd, "normals", np.zeros_like(points)), dtype=np.float32)
            storePly(os.path.join(seed_dir, "point_cloud.ply"), points, colors, normals=normals)
            storePly(os.path.join(self.model_path, "input_seed.ply"), points, colors, normals=normals)
            print(f"[streaming] Initial seed point cloud exported ({points.shape[0]} points)", flush=True)
        except Exception as exc:
            print(f"[streaming] Warning: initial seed point cloud export failed: {exc}", flush=True)

    def _estimate_cameras_extent(self, frames: List["StreamingRGBDFrame"]) -> float:
        """NeRF++ normalisation radius from a list of frames."""
        try:
            from scene.readers.common import CameraInfo, getNerfppNorm
            from utils.graphics_utils import focal2fov
            from utils.rgbd_frames import c2w_to_camera_rt

            infos = []
            for f in frames:
                R, T = c2w_to_camera_rt(f.c2w)
                infos.append(CameraInfo(
                    uid=f.index, R=R, T=T,
                    FovY=focal2fov(f.fy, f.height),
                    FovX=focal2fov(f.fx, f.width),
                    image=None, image_path="", image_name="",
                    width=f.width, height=f.height,
                ))
            norm = getNerfppNorm(infos)
            return max(float(norm["radius"]), 0.1)
        except Exception:
            return 1.0

    @staticmethod
    def _resize_nearest_np(arr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        target_h, target_w = int(target_hw[0]), int(target_hw[1])
        if arr.shape[:2] == (target_h, target_w):
            return arr
        src_h, src_w = arr.shape[:2]
        ys = np.clip(np.round(np.linspace(0, src_h - 1, target_h)).astype(np.int32), 0, src_h - 1)
        xs = np.clip(np.round(np.linspace(0, src_w - 1, target_w)).astype(np.int32), 0, src_w - 1)
        return arr[ys[:, None], xs[None, :]]

    @staticmethod
    def _stable_depth_samples(
        z: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        min_depth: float,
        max_depth: float,
        args,
    ) -> np.ndarray:
        if not getattr(args, "streaming_depth_filter_enabled", True):
            return np.ones(xs.shape, dtype=bool)
        h, w = z.shape
        erode_px = max(0, int(getattr(args, "streaming_depth_filter_erode_px", 2)))
        window = max(1, int(getattr(args, "streaming_depth_filter_window", 5)))
        min_valid_ratio = float(getattr(args, "streaming_depth_filter_min_valid_ratio", 0.75))
        max_range = float(getattr(args, "streaming_depth_filter_max_range", 0.08))
        median_thresh = float(getattr(args, "streaming_depth_filter_median_thresh", 0.04))
        bottom_margin = float(getattr(args, "streaming_depth_filter_bottom_margin", 0.0))

        radius = window // 2
        local_depths = []
        for dy in range(-radius, radius + 1):
            yy = np.clip(ys + dy, 0, h - 1)
            for dx in range(-radius, radius + 1):
                xx = np.clip(xs + dx, 0, w - 1)
                local_depths.append(z[yy, xx])
        local = np.stack(local_depths, axis=0).astype(np.float32, copy=False)
        local_valid = np.isfinite(local) & (local > min_depth) & (local < max_depth)
        stable = local_valid.mean(axis=0) >= min_valid_ratio

        if max_range > 0:
            local_min = np.where(local_valid, local, np.inf).min(axis=0)
            local_max = np.where(local_valid, local, -np.inf).max(axis=0)
            stable &= np.isfinite(local_min) & np.isfinite(local_max) & ((local_max - local_min) <= max_range)

        if median_thresh > 0:
            import warnings
            local_nan = np.where(local_valid, local, np.nan)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                local_median = np.nanmedian(local_nan, axis=0)
            stable &= np.isfinite(local_median) & (np.abs(z[ys, xs] - local_median) <= median_thresh)

        if erode_px > 0:
            valid_hw = np.isfinite(z) & (z > min_depth) & (z < max_depth)
            eroded = np.ones(xs.shape, dtype=bool)
            for dy in range(-erode_px, erode_px + 1):
                yy = np.clip(ys + dy, 0, h - 1)
                for dx in range(-erode_px, erode_px + 1):
                    xx = np.clip(xs + dx, 0, w - 1)
                    eroded &= valid_hw[yy, xx]
            stable &= eroded

        if bottom_margin > 0:
            stable &= ys < int(round(h * (1.0 - bottom_margin)))
        return stable

    @staticmethod
    def _temporal_depth_samples(
        z_v: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        frame,
        other_frame,
        other_depth_m: np.ndarray,
        d_fx: float,
        d_fy: float,
        d_cx: float,
        d_cy: float,
        thresh: float,
    ) -> Optional[np.ndarray]:
        if other_frame is None or other_depth_m is None or thresh <= 0:
            return None
        try:
            c2w = np.asarray(frame.c2w, dtype=np.float32)
            other_c2w = np.asarray(other_frame.c2w, dtype=np.float32)
            if c2w.shape != (4, 4) or other_c2w.shape != (4, 4):
                return None
            if not np.isfinite(c2w).all() or not np.isfinite(other_c2w).all():
                return None

            other_h, other_w = int(other_frame.height), int(other_frame.width)
            other_depth = StreamingScene._resize_nearest_np(
                np.asarray(other_depth_m, dtype=np.float32),
                (other_h, other_w),
            )

            x_c = (xs.astype(np.float32) - float(d_cx)) / float(d_fx) * z_v.astype(np.float32)
            y_c = (ys.astype(np.float32) - float(d_cy)) / float(d_fy) * z_v.astype(np.float32)
            pts_cam = np.stack([x_c, y_c, z_v.astype(np.float32)], axis=1)
            pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]

            other_w2c = np.linalg.inv(other_c2w)
            pts_other = (other_w2c[:3, :3] @ pts_world.T).T + other_w2c[:3, 3]
            z_other = pts_other[:, 2]

            o_fx = other_frame.depth_fx if other_frame.depth_fx is not None else other_frame.fx
            o_fy = other_frame.depth_fy if other_frame.depth_fy is not None else other_frame.fy
            o_cx = other_frame.depth_cx if other_frame.depth_cx is not None else other_frame.cx
            o_cy = other_frame.depth_cy if other_frame.depth_cy is not None else other_frame.cy
            u = np.round((pts_other[:, 0] / np.maximum(z_other, 1e-6)) * float(o_fx) + float(o_cx)).astype(np.int32)
            v = np.round((pts_other[:, 1] / np.maximum(z_other, 1e-6)) * float(o_fy) + float(o_cy)).astype(np.int32)

            ok = (z_other > 0) & (u >= 0) & (u < other_w) & (v >= 0) & (v < other_h)
            out = np.zeros(z_v.shape[0], dtype=bool)
            if ok.any():
                lookup = other_depth[v[ok], u[ok]]
                out[ok] = np.isfinite(lookup) & (lookup > 0) & (np.abs(z_other[ok] - lookup) <= float(thresh))
            return out
        except Exception:
            return None

    def _build_pcd_from_frames(self, frames: List["StreamingRGBDFrame"]):
        """Backproject depth into a BasicPointCloud for create_from_pcd()."""
        from utils.graphics_utils import BasicPointCloud
        from utils.rgbd_frames import depth_to_meters, backproject_depth_pixels
        from PIL import Image as _Image

        depth_stride = getattr(self.args, "streaming_depth_stride", 8)
        min_depth = getattr(self.args, "streaming_min_depth", 0.1)
        max_depth = getattr(self.args, "streaming_max_depth", 8.0)
        max_pts = getattr(self.args, "rgbd_max_init_points", 250000)

        loaded = []
        for frame in frames:
            if frame.depth_path is None and frame._depth_bytes is None:
                loaded.append(None)
                continue
            if frame.depth_path is not None and not os.path.exists(frame.depth_path) and frame._depth_bytes is None:
                loaded.append(None)
                continue
            try:
                from utils.streaming_frames import load_frame_rgb, load_frame_depth_np
                depth_raw = load_frame_depth_np(frame)
                if depth_raw is None:
                    loaded.append(None)
                    continue
                depth = np.asarray(depth_raw)
                _target = (int(frame.height), int(frame.width))
                if depth.shape[:2] != _target:
                    depth = np.array(_Image.fromarray(depth).resize(
                        (_target[1], _target[0]), _Image.NEAREST))
                rgb_pil = load_frame_rgb(frame)
                if rgb_pil.size != (_target[1], _target[0]):
                    rgb_pil = rgb_pil.resize((_target[1], _target[0]), _Image.BILINEAR)
                rgb = np.array(rgb_pil).astype(np.float32) / 255.0
                loaded.append((frame, depth, rgb, depth_to_meters(depth, frame.depth_scale)))
            except Exception:
                loaded.append(None)
                continue

        points_all, colors_all = [], []
        seed_temporal = bool(getattr(self.args, "streaming_depth_temporal_require_for_seed", True))
        temporal_thresh = float(getattr(self.args, "streaming_depth_temporal_thresh", 0.07))
        for idx, item in enumerate(loaded):
            if item is None:
                continue
            frame, depth, rgb, z = item
            h, w = z.shape
            ys, xs = np.mgrid[0:h:depth_stride, 0:w:depth_stride]
            z_v = z[ys, xs]
            valid = np.isfinite(z_v) & (z_v > min_depth) & (z_v < max_depth)
            valid &= self._stable_depth_samples(z, xs, ys, min_depth, max_depth, self.args)
            if not valid.any():
                continue
            d_fx = frame.depth_fx if frame.depth_fx is not None else frame.fx
            d_fy = frame.depth_fy if frame.depth_fy is not None else frame.fy
            d_cx = frame.depth_cx if frame.depth_cx is not None else frame.cx
            d_cy = frame.depth_cy if frame.depth_cy is not None else frame.cy
            if seed_temporal and len(loaded) > 1:
                other_item = None
                for j in (idx - 1, idx + 1):
                    if 0 <= j < len(loaded) and loaded[j] is not None:
                        other_item = loaded[j]
                        break
                if other_item is not None:
                    other_frame, _, _, other_z = other_item
                    temporal = self._temporal_depth_samples(
                        z_v[valid].astype(np.float32),
                        xs[valid].astype(np.int32),
                        ys[valid].astype(np.int32),
                        frame,
                        other_frame,
                        other_z,
                        d_fx,
                        d_fy,
                        d_cx,
                        d_cy,
                        temporal_thresh,
                    )
                    if temporal is not None:
                        full_temporal = np.zeros_like(valid, dtype=bool)
                        full_temporal[valid] = temporal
                        valid &= full_temporal
                    else:
                        valid &= False
                else:
                    valid &= False
            if not valid.any():
                continue

            xs_v = xs[valid].astype(np.float32)
            ys_v = ys[valid].astype(np.float32)
            z_v = z_v[valid].astype(np.float32)
            pts_world, cols = backproject_depth_pixels(
                z_v, xs_v, ys_v, d_fx, d_fy, d_cx, d_cy, frame.c2w, rgb, (h, w))
            points_all.append(pts_world)
            colors_all.append(cols)

        if not points_all:
            # Fall back to random scatter inside the estimated scene extent
            r = self.cameras_extent
            n = max(1000, getattr(self.args, "cap_max", 10000) // 10)
            pts = np.random.uniform(-r, r, (n, 3)).astype(np.float32)
            cols = np.random.uniform(0.3, 0.7, (n, 3)).astype(np.float32)
            print("[streaming] Warning: no depth available for init, falling back to random point cloud.", flush=True)
            return BasicPointCloud(points=pts, colors=cols, normals=np.zeros_like(pts))

        pts = np.concatenate(points_all, axis=0)
        cols = np.concatenate(colors_all, axis=0)
        if max_pts and pts.shape[0] > max_pts:
            rng = np.random.default_rng(42)
            idx = rng.choice(pts.shape[0], size=max_pts, replace=False)
            pts, cols = pts[idx], cols[idx]
        return BasicPointCloud(points=pts, colors=cols, normals=np.zeros_like(pts, dtype=np.float32))

    # ------------------------------------------------------------------
    # Frame ingestion
    # ------------------------------------------------------------------

    def has_next_frame(self) -> bool:
        return self._source_idx < self._frame_count

    def ingest_next_frame(self, admission_fn=None) -> Optional[Tuple]:
        """
        Load the next frame as a Camera and add it to the active window and
        replay buffer.  Returns (camera, StreamingRGBDFrame, is_train) or None
        when the source is exhausted.
        """
        if not self.has_next_frame():
            return None
        from scene.cameras import prepare_camera_for_render

        frame = self._get_source_frame(self._source_idx)
        self._source_idx += 1

        # Skip frames where live odometry failed — inserting depth from a copied
        # (wrong) pose corrupts the Gaussian map.
        if getattr(frame, "_odom_valid", None) is False:
            self._keyframes_rejected += 1
            return None, frame, False

        cam = self._frame_to_camera(frame)

        eval_hold = getattr(self.args, "streaming_eval_hold", 0)
        is_train = True
        if eval_hold > 0 and frame.index % eval_hold == 0:
            if admission_fn is not None and not getattr(self.args, "streaming_keyframe_admit_eval_holdouts", False):
                self._keyframes_rejected += 1
                return None, frame, False
            # Hold-out frame: add to test set only, skip training/replay
            self._test_cameras.append(cam)
            is_train = False
        else:
            if admission_fn is not None:
                admit, admission_stats, admission_render_pkg = admission_fn(cam, frame)
                frame._streaming_admission_stats = admission_stats
                if not admit:
                    self._keyframes_rejected += 1
                    return None, frame, False
                cam._streaming_admission_stats = admission_stats
                cam._streaming_admission_render_pkg = admission_render_pkg
            self._keyframes_admitted += 1
            self.train_cameras.append(cam)
            replay_size = getattr(self.args, "streaming_replay_buffer", 32)
            self._replay_buffer.append(cam)
            if len(self._replay_buffer) > replay_size:
                self._replay_buffer.pop(0)

            # H10: update global reservoir (every Nth train frame kept permanently)
            self._n_train_frames_ingested += 1
            reservoir_stride = getattr(self.args, "streaming_global_reservoir_stride", 0)
            if reservoir_stride > 0 and self._n_train_frames_ingested % reservoir_stride == 0:
                self._global_reservoir.append(cam)

            # H8: reset warmup counter whenever a new train frame arrives
            self._steps_since_new_frame = 0
            self.update_cameras_extent()

        self.current_camera = cam
        prepare_camera_for_render(cam, device="cuda")
        return cam, frame, is_train

    # ------------------------------------------------------------------
    # Camera sampling
    # ------------------------------------------------------------------

    def get_local_cameras(self) -> List:
        k = getattr(self.args, "streaming_keyframe_window", 8)
        return self.train_cameras[-k:] if self.train_cameras else []

    def sample_training_camera(self):
        """Sample a camera for one training step."""
        # H8: warmup — force current frame for K steps after each new arrival
        warmup_steps = getattr(self.args, "streaming_new_frame_warmup_steps", 0)
        if warmup_steps > 0 and self._steps_since_new_frame < warmup_steps and self.current_camera is not None:
            self._steps_since_new_frame += 1
            return self.current_camera
        self._steps_since_new_frame += 1

        sampling_mode = getattr(self.args, "streaming_sampling_mode", "legacy")
        if sampling_mode == "stratified":
            return self._sample_stratified()
        return self._sample_legacy()

    def _sample_legacy(self):
        """Original ring-buffer + reservoir sampling."""
        replay_ratio = getattr(self.args, "streaming_global_replay_ratio", 0.1)
        r = random.random()
        if replay_ratio > 0 and r < replay_ratio:
            reservoir_stride = getattr(self.args, "streaming_global_reservoir_stride", 0)
            if reservoir_stride > 0 and self._global_reservoir:
                return random.choice(self._global_reservoir)
            if self._replay_buffer:
                return random.choice(self._replay_buffer)
        local = self.get_local_cameras()
        return random.choice(local) if local else self.current_camera

    def _sample_stratified(self):
        """Four-strata sampling: recent / covisible / global-reservoir / hard-frames."""
        ratios_str = getattr(self.args, "streaming_sampling_ratios", "0.70,0.15,0.10,0.05")
        try:
            ratios = [float(x) for x in ratios_str.split(",")]
            if len(ratios) != 4:
                ratios = [0.70, 0.15, 0.10, 0.05]
        except Exception:
            ratios = [0.70, 0.15, 0.10, 0.05]
        r_recent, r_covis, r_reservoir, r_hard = ratios

        r = random.random()
        local = self.get_local_cameras()

        if r < r_recent:
            return random.choice(local) if local else self.current_camera

        if r < r_recent + r_covis:
            if self._covisible_cache:
                return random.choice(self._covisible_cache)
            return random.choice(local) if local else self.current_camera

        if r < r_recent + r_covis + r_reservoir:
            reservoir_stride = getattr(self.args, "streaming_global_reservoir_stride", 0)
            if reservoir_stride > 0 and self._global_reservoir:
                return random.choice(self._global_reservoir)
            if self._replay_buffer:
                return random.choice(self._replay_buffer)
            return random.choice(local) if local else self.current_camera

        # Hard frames stratum
        if self._hard_frames:
            return random.choice(self._hard_frames)[1]
        return random.choice(local) if local else self.current_camera

    def update_stratified_state(self, cam, loss_val: float, gaussian_ids=None):
        """Update hard-frame list and covisibility cache. Call once per training step."""
        if getattr(self.args, "streaming_sampling_mode", "legacy") != "stratified":
            return

        # Hard frames: keep the K highest-loss cameras
        max_hard = max(1, int(getattr(self.args, "streaming_hard_frame_history", 8)))
        self._hard_frames.append((loss_val, cam))
        self._hard_frames.sort(key=lambda x: -x[0])
        self._hard_frames = self._hard_frames[:max_hard]

        # Covisibility: cameras that share many Gaussians with the current view.
        # We build a flat covisible list from the replay buffer cameras closest
        # in index to the current train camera (cheap proxy for covisibility).
        if len(self.train_cameras) > 0 and cam in self.train_cameras:
            try:
                cam_idx = self.train_cameras.index(cam)
            except ValueError:
                cam_idx = len(self.train_cameras) - 1
            k_win = getattr(self.args, "streaming_keyframe_window", 8)
            # Covisible = cameras within 2×window of current cam, excluding local window
            lo = max(0, cam_idx - 2 * k_win)
            hi = max(0, cam_idx - k_win)
            self._covisible_cache = self.train_cameras[lo:hi] if lo < hi else []

    # ------------------------------------------------------------------
    # Compatibility shims for code that calls scene.getTrainCameras()
    # ------------------------------------------------------------------

    def getTrainCameras(self, scale: float = 1.0) -> List:
        return self.train_cameras

    def getTestCameras(self, scale: float = 1.0) -> List:
        return self._test_cameras

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _frame_to_camera(self, frame: "StreamingRGBDFrame"):
        from utils.camera_utils import loadCam
        from utils.graphics_utils import focal2fov
        from utils.rgbd_frames import c2w_to_camera_rt
        from scene.readers.common import CameraInfo
        from PIL import Image

        from utils.streaming_frames import load_frame_rgb
        R, T = c2w_to_camera_rt(frame.c2w)
        image = load_frame_rgb(frame)
        _target = (int(frame.width), int(frame.height))
        if image.size != _target:
            image = image.resize(_target, Image.BILINEAR)
        orig_w, orig_h = image.size
        cam_info = CameraInfo(
            uid=frame.index,
            R=R, T=T,
            FovY=focal2fov(frame.fy, orig_h),
            FovX=focal2fov(frame.fx, orig_w),
            image=image,
            image_path=frame.rgb_path,
            image_name=f"{frame.index:06d}",
            width=orig_w, height=orig_h,
            fx=float(frame.fx), fy=float(frame.fy),
            cx=float(frame.cx), cy=float(frame.cy),
        )
        cam = loadCam(self.args, frame.index, cam_info, self.resolution_scale)
        # Attach depth source for streaming depth loss (loaded lazily at training time)
        cam._streaming_depth_path = frame.depth_path
        cam._streaming_depth_scale = frame.depth_scale
        cam._sensor_depth_cache = None  # populated on first access
        cam._streaming_timestamp = float(frame.timestamp)
        # Keep reference to frame for in-memory depth access (.sens source)
        cam._streaming_frame = frame
        return cam
