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
        self._all_frames = list(frame_source)

        # Persistent occupancy hash for incremental insertion
        self.occupied_voxels = set()
        self.occupancy_voxel_size = 0.02

        # H8: warmup counter — steps since the last new frame arrived
        self._steps_since_new_frame: int = 0

        # H10: permanent global keyframe reservoir (never evicted, sampled for replay)
        self._global_reservoir: List = []
        self._n_train_frames_ingested: int = 0  # count of train frames (excl. holdout)

    def maintain_occupancy_hash(self, voxel_size: float = 0.02):
        """Update the occupancy hash from the current Gaussians (expensive)."""
        self.occupancy_voxel_size = voxel_size
        xyz = self.gaussians.get_xyz.detach().cpu().numpy()
        if xyz.shape[0] == 0:
            self.occupied_voxels = set()
            return
        
        # Use a fixed large offset to keep keys positive and avoid floating precision issues
        offset = 1000.0
        coords = np.floor((xyz + offset) / max(voxel_size, 1e-6)).astype(np.int64)
        
        # Pack into 64-bit keys: 21 bits per dimension (covers +/- 1000m at 1mm res)
        keys = (coords[:, 0] << 42) | (coords[:, 1] << 21) | coords[:, 2]
        self.occupied_voxels = set(keys.tolist())

    def add_to_occupancy_hash(self, xyz: torch.Tensor):
        """Incrementally add new points to the hash."""
        if xyz.shape[0] == 0:
            return
        xyz_np = xyz.detach().cpu().numpy()
        offset = 1000.0
        coords = np.floor((xyz_np + offset) / max(self.occupancy_voxel_size, 1e-6)).astype(np.int64)
        keys = (coords[:, 0] << 42) | (coords[:, 1] << 21) | coords[:, 2]
        self.occupied_voxels.update(keys.tolist())

    def check_occupancy(self, xyz_np: np.ndarray, voxel_size: float, check_neighbors: bool = True) -> np.ndarray:
        """Check whether points fall into occupied voxels."""
        if not self.occupied_voxels or xyz_np.shape[0] == 0:
            return np.zeros(xyz_np.shape[0], dtype=bool)
        
        offset = 1000.0
        coords = np.floor((xyz_np + offset) / max(voxel_size, 1e-6)).astype(np.int64)
        
        keys = (coords[:, 0] << 42) | (coords[:, 1] << 21) | coords[:, 2]
        occupied = np.array([k in self.occupied_voxels for k in keys.tolist()])
        
        if check_neighbors and not occupied.all():
            # Check 26 neighbors for points not already marked occupied
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        remaining = ~occupied
                        if not remaining.any():
                            break
                        n_coords = coords[remaining] + np.array([dx, dy, dz])
                        n_keys = (n_coords[:, 0] << 42) | (n_coords[:, 1] << 21) | n_coords[:, 2]
                        n_occupied = np.array([k in self.occupied_voxels for k in n_keys.tolist()])
                        occupied[remaining] |= n_occupied
        return occupied
        self._all_frames: List["StreamingRGBDFrame"] = frame_source.get_all()
        self._source_idx: int = 0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_from_frames(self, n_frames: int) -> None:
        """
        Ingest the first n_frames as initial cameras and build the initial
        Gaussian point cloud from their depth data.

        Must be called BEFORE gaussians.training_setup() so the optimizer is
        built on the correct initial parameter tensors.
        """
        from scene.cameras import prepare_camera_for_render

        n_frames = min(n_frames, len(self._all_frames))
        if n_frames == 0:
            raise RuntimeError("[streaming] No frames available for initialisation.")

        print(f"[streaming] Bootstrapping from {n_frames} frame(s)...", flush=True)
        initial_frames = self._all_frames[:n_frames]

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
        self.maintain_occupancy_hash(voxel_size=getattr(self.args, "streaming_insert_voxel_size", 0.02))
        print(
            f"[streaming] Initialised {self.gaussians.get_xyz.shape[0]} Gaussians "
            f"from {n_frames} frames, scene radius={self.cameras_extent:.3f}",
            flush=True,
        )

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

    def _build_pcd_from_frames(self, frames: List["StreamingRGBDFrame"]):
        """Backproject depth into a BasicPointCloud for create_from_pcd()."""
        from utils.graphics_utils import BasicPointCloud
        from utils.rgbd_frames import depth_to_meters
        from PIL import Image as _Image

        depth_stride = getattr(self.args, "streaming_depth_stride", 8)
        min_depth = getattr(self.args, "streaming_min_depth", 0.1)
        max_depth = getattr(self.args, "streaming_max_depth", 8.0)
        max_pts = getattr(self.args, "rgbd_max_init_points", 250000)

        points_all, colors_all = [], []
        for frame in frames:
            if frame.depth_path is None or not os.path.exists(frame.depth_path):
                continue
            try:
                depth = np.array(_Image.open(frame.depth_path))
                rgb = np.array(_Image.open(frame.rgb_path).convert("RGB")).astype(np.float32) / 255.0
            except Exception:
                continue
            z = depth_to_meters(depth, frame.depth_scale)
            h, w = z.shape
            ys, xs = np.mgrid[0:h:depth_stride, 0:w:depth_stride]
            z_v = z[ys, xs]
            valid = np.isfinite(z_v) & (z_v > min_depth) & (z_v < max_depth)
            if not valid.any():
                continue
            xs_v = xs[valid].astype(np.float32)
            ys_v = ys[valid].astype(np.float32)
            z_v = z_v[valid].astype(np.float32)
            x_c = (xs_v - frame.cx) / frame.fx * z_v
            y_c = (ys_v - frame.cy) / frame.fy * z_v
            pts_cam = np.stack([x_c, y_c, z_v], axis=1)
            pts_world = (frame.c2w[:3, :3] @ pts_cam.T).T + frame.c2w[:3, 3]
            rgb_h, rgb_w = rgb.shape[:2]
            rx = np.clip((xs_v / max(w - 1, 1) * (rgb_w - 1)).round().astype(np.int32), 0, rgb_w - 1)
            ry = np.clip((ys_v / max(h - 1, 1) * (rgb_h - 1)).round().astype(np.int32), 0, rgb_h - 1)
            points_all.append(pts_world.astype(np.float32))
            colors_all.append(rgb[ry, rx].astype(np.float32))

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
        return self._source_idx < len(self._all_frames)

    def ingest_next_frame(self) -> Optional[Tuple]:
        """
        Load the next frame as a Camera and add it to the active window and
        replay buffer.  Returns (camera, StreamingRGBDFrame, is_train) or None
        when the source is exhausted.
        """
        if not self.has_next_frame():
            return None
        from scene.cameras import prepare_camera_for_render

        frame = self._all_frames[self._source_idx]
        self._source_idx += 1

        cam = self._frame_to_camera(frame)

        eval_hold = getattr(self.args, "streaming_eval_hold", 0)
        is_train = True
        if eval_hold > 0 and frame.index % eval_hold == 0:
            # Hold-out frame: add to test set only, skip training/replay
            self._test_cameras.append(cam)
            is_train = False
        else:
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
        """Sample a camera for one training step.

        Priority order:
        1. H8 warmup: return current_camera exclusively for the first
           streaming_new_frame_warmup_steps steps after each new frame arrives.
        2. Replay (global reservoir if H10 enabled, else recent ring buffer):
           sampling probability = streaming_global_replay_ratio.
        3. Local keyframe window (default).
        """
        # H8: warmup — force current frame for K steps after each new arrival
        warmup_steps = getattr(self.args, "streaming_new_frame_warmup_steps", 0)
        if warmup_steps > 0 and self._steps_since_new_frame < warmup_steps and self.current_camera is not None:
            self._steps_since_new_frame += 1
            return self.current_camera
        self._steps_since_new_frame += 1

        replay_ratio = getattr(self.args, "streaming_global_replay_ratio", 0.1)
        r = random.random()

        if replay_ratio > 0 and r < replay_ratio:
            # H10: prefer global reservoir when available; fall back to ring buffer
            reservoir_stride = getattr(self.args, "streaming_global_reservoir_stride", 0)
            if reservoir_stride > 0 and self._global_reservoir:
                return random.choice(self._global_reservoir)
            if self._replay_buffer:
                return random.choice(self._replay_buffer)

        local = self.get_local_cameras()
        return random.choice(local) if local else self.current_camera

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

        R, T = c2w_to_camera_rt(frame.c2w)
        image = Image.open(frame.rgb_path).convert("RGB")
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
        return cam
