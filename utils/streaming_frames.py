"""
Ordered RGB-D frame sources for streaming replay simulation.

Each source reads dataset metadata upfront (file paths + poses + intrinsics,
no image pixels) and yields StreamingRGBDFrame records in temporal order.
Images are loaded lazily by StreamingScene when a frame "arrives".

Use make_frame_source() to auto-detect the dataset type and return the
appropriate source for a given dataset path.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional


@dataclass
class StreamingRGBDFrame:
    """Lightweight per-frame record: paths + pose + intrinsics, no pixel data.

    fx/fy/cx/cy + width/height describe the *color* camera used for rendering.
    depth_fx/depth_fy/depth_cx/depth_cy + depth_width/depth_height describe the
    *depth* sensor (may differ, e.g. ScanNet 640×480 vs 1296×968 color).
    When depth_fx is None the color intrinsics are used for backprojection too.

    _rgb_bytes / _depth_bytes / _sens_header are used by ScanNetSensFrameSource
    to store images in-memory instead of on disk (file paths remain empty).
    """
    index: int
    timestamp: float
    rgb_path: str
    depth_path: Optional[str]
    c2w: "np.ndarray"      # (4, 4) float32 camera-to-world
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    depth_scale: float = 1000.0
    # Optional separate depth-sensor intrinsics (populated by ScanNetFrameSource)
    depth_fx: Optional[float] = None
    depth_fy: Optional[float] = None
    depth_cx: Optional[float] = None
    depth_cy: Optional[float] = None
    depth_width: Optional[int] = None
    depth_height: Optional[int] = None
    # In-memory image data (populated by ScanNetSensFrameSource; None → use paths)
    _rgb_bytes: Optional[bytes] = field(default=None, repr=False, compare=False)
    _depth_bytes: Optional[bytes] = field(default=None, repr=False, compare=False)
    _sens_header: Optional[dict] = field(default=None, repr=False, compare=False)


def load_frame_rgb(frame: "StreamingRGBDFrame"):
    """Return a PIL RGB image for this frame, from bytes or disk."""
    from PIL import Image
    if frame._rgb_bytes is not None:
        return Image.open(io.BytesIO(frame._rgb_bytes)).convert("RGB")
    return Image.open(frame.rgb_path).convert("RGB")


def load_frame_depth_np(frame: "StreamingRGBDFrame"):
    """Return raw depth numpy array (uint16 or float32), or None on failure."""
    import numpy as np
    if frame._depth_bytes is not None:
        if frame._sens_header is not None:
            from scene.readers.scannet import _decode_sens_depth
            return _decode_sens_depth(frame._depth_bytes, frame._sens_header)
        # Plain PNG bytes (e.g. Replica uint16 depth, HyperSim uint16 depth)
        from PIL import Image
        return np.array(Image.open(io.BytesIO(frame._depth_bytes)))
    if frame.depth_path is None or not os.path.exists(frame.depth_path):
        return None
    from PIL import Image
    return np.array(Image.open(frame.depth_path))


def _rot_to_quat(R):
    import numpy as np
    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        return np.array([
            0.25 * s,
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
        ], dtype=np.float64)
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = [((R[2, 1] - R[1, 2]) / s), 0.25 * s, ((R[0, 1] + R[1, 0]) / s), ((R[0, 2] + R[2, 0]) / s)]
    elif i == 1:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = [((R[0, 2] - R[2, 0]) / s), ((R[0, 1] + R[1, 0]) / s), 0.25 * s, ((R[1, 2] + R[2, 1]) / s)]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = [((R[1, 0] - R[0, 1]) / s), ((R[0, 2] + R[2, 0]) / s), ((R[1, 2] + R[2, 1]) / s), 0.25 * s]
    q = np.asarray(q, dtype=np.float64)
    return q / max(np.linalg.norm(q), 1e-12)

def _quat_to_rot(q):
    import numpy as np
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)

def _slerp(q0, q1, t):
    import numpy as np
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / max(np.linalg.norm(out), 1e-12)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * t
    return (
        np.sin(theta_0 - theta) * q0 + np.sin(theta) * q1
    ) / np.sin(theta_0)

class RGBDSequenceFrameSource:
    """
    Ordered frames from a generic RGB-D sequence (frames.jsonl + intrinsics.json).
    Reuses read_frame_records / read_intrinsics from utils.rgbd_frames.
    """

    def __init__(self, path: str, max_frames: int = 0, frame_stride: int = 1):
        import numpy as np
        from utils.rgbd_frames import read_frame_records, read_intrinsics

        intr = read_intrinsics(os.path.join(path, "intrinsics.json"))
        records = read_frame_records(os.path.join(path, "frames.jsonl"))
        if frame_stride > 1:
            records = records[::frame_stride]
        if max_frames > 0:
            records = records[:max_frames]

        self._frames: List[StreamingRGBDFrame] = []
        for rec in records:
            rgb_path = os.path.join(path, rec["rgb"])
            if not os.path.exists(rgb_path):
                continue
            c2w = np.asarray(rec["c2w"], dtype=np.float32)
            if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
                continue
            depth_rel = rec.get("depth")
            depth_path = os.path.join(path, depth_rel) if depth_rel else None
            self._frames.append(StreamingRGBDFrame(
                index=int(rec.get("id", len(self._frames))),
                timestamp=float(rec.get("timestamp", len(self._frames))),
                rgb_path=rgb_path,
                depth_path=depth_path,
                c2w=c2w,
                fx=intr.fx, fy=intr.fy, cx=intr.cx, cy=intr.cy,
                width=intr.width, height=intr.height,
                depth_scale=intr.depth_scale,
            ))
        print(f"[streaming] RGBDSequence: {len(self._frames)} frames loaded from {path}")

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[StreamingRGBDFrame]:
        yield from self._frames

    def get_all(self) -> List[StreamingRGBDFrame]:
        return self._frames


class TUMFrameSource:
    """
    Ordered frames from a TUM RGB-D dataset (rgb.txt + depth.txt + groundtruth.txt).
    Reuses private helpers from scene.readers.tum.
    """

    def __init__(
        self,
        path: str,
        max_frames: int = 0,
        frame_stride: int = 1,
        association_max_dt: float = 0.03,
        sequence: str = "",
    ):
        import numpy as np
        from scene.readers.tum import (
            _read_tum_list,
            _read_tum_poses,
            _associate_tum,
            _tum_default_intrinsics,
            _tum_pose_to_c2w,
        )

        rgb_entries = _read_tum_list(os.path.join(path, "rgb.txt"))
        depth_entries = _read_tum_list(os.path.join(path, "depth.txt"))
        pose_entries = _read_tum_poses(os.path.join(path, "groundtruth.txt"))
        assocs = _associate_tum(rgb_entries, depth_entries, pose_entries, max_dt=association_max_dt)
        if frame_stride > 1:
            assocs = assocs[::frame_stride]
        if max_frames > 0:
            assocs = assocs[:max_frames]

        fx, fy, cx, cy = _tum_default_intrinsics(path, sequence=sequence)

        self._frames: List[StreamingRGBDFrame] = []
        for i, assoc in enumerate(assocs):
            rgb_path = os.path.join(path, assoc["rgb_rel"])
            if not os.path.exists(rgb_path):
                continue
            c2w = _tum_pose_to_c2w(assoc["tvec"], assoc["qxyzw"])
            if not np.isfinite(c2w).all():
                continue
            depth_path = os.path.join(path, assoc["depth_rel"])
            self._frames.append(StreamingRGBDFrame(
                index=i,
                timestamp=float(assoc["rgb_t"]),
                rgb_path=rgb_path,
                depth_path=depth_path if os.path.exists(depth_path) else None,
                c2w=c2w,
                fx=fx, fy=fy, cx=cx, cy=cy,
                width=640, height=480,  # TUM RGB-D is always 640×480
                depth_scale=5000.0,
            ))
        print(f"[streaming] TUM: {len(self._frames)} frames loaded from {path}")

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[StreamingRGBDFrame]:
        yield from self._frames

    def get_all(self) -> List[StreamingRGBDFrame]:
        return self._frames


class ScanNetFrameSource:
    """
    Ordered frames from a ScanNet scene (color/ + pose/ + intrinsic/).
    Reuses private helpers from scene.readers.scannet.
    """

    def __init__(
        self,
        path: str,
        max_frames: int = 0,
        frame_stride: int = 1,
        depth_scale: float = 1000.0,
    ):
        from scene.readers.scannet import (
            _list_scannet_color_frames,
            _scannet_frame_id_from_color,
            _find_scannet_intrinsics,
            _read_scannet_pose,
        )
        from PIL import Image

        intrinsics = _find_scannet_intrinsics(path)
        fx, fy, cx, cy = intrinsics["color"]
        dfx, dfy, dcx, dcy = intrinsics["depth"]

        color_files = _list_scannet_color_frames(path)
        if frame_stride > 1:
            color_files = color_files[::frame_stride]
        if max_frames > 0:
            color_files = color_files[:max_frames]

        pose_dir = os.path.join(path, "pose")
        depth_dir = os.path.join(path, "depth")

        # Detect color image dimensions from first existing frame
        width, height = 1296, 968
        for cf in color_files:
            try:
                img = Image.open(cf)
                width, height = img.size
                break
            except Exception:
                pass

        # Detect depth image dimensions from first existing depth frame
        depth_width, depth_height = 640, 480
        for cf in color_files:
            fid = _scannet_frame_id_from_color(cf)
            df = os.path.join(depth_dir, f"{fid}.png")
            if os.path.exists(df):
                try:
                    img = Image.open(df)
                    depth_width, depth_height = img.size
                except Exception:
                    pass
                break

        # Warn if color and depth intrinsics differ (common in ScanNet)
        if abs(fx - dfx) > 1.0 or abs(fy - dfy) > 1.0:
            print(
                f"[streaming] ScanNet: color intrinsics (fx={fx:.1f}) differ from "
                f"depth intrinsics (fx={dfx:.1f}). Using depth intrinsics for "
                f"backprojection, color intrinsics for rendering."
            )

        self._frames: List[StreamingRGBDFrame] = []
        for color_file in color_files:
            fid = _scannet_frame_id_from_color(color_file)
            pose_file = os.path.join(pose_dir, f"{fid}.txt")
            c2w = _read_scannet_pose(pose_file)
            if c2w is None:
                continue
            depth_file = os.path.join(depth_dir, f"{fid}.png")
            self._frames.append(StreamingRGBDFrame(
                index=len(self._frames),
                timestamp=float(fid) if fid.isdigit() else float(len(self._frames)),
                rgb_path=color_file,
                depth_path=depth_file if os.path.exists(depth_file) else None,
                c2w=c2w,
                fx=fx, fy=fy, cx=cx, cy=cy,
                width=width, height=height,
                depth_scale=depth_scale,
                depth_fx=dfx, depth_fy=dfy, depth_cx=dcx, depth_cy=dcy,
                depth_width=depth_width, depth_height=depth_height,
            ))
        print(f"[streaming] ScanNet: {len(self._frames)} frames loaded from {path}")

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[StreamingRGBDFrame]:
        yield from self._frames

    def get_all(self) -> List[StreamingRGBDFrame]:
        return self._frames


class ScanNetSensFrameSource:
    """
    Ordered frames streamed directly from a ScanNet .sens binary file.
    No pre-extraction needed — images are stored in-memory as compressed bytes.
    Memory: ~200KB/frame (JPEG) + ~75KB/frame (depth) ≈ 275KB/frame.
    At stride=5 and 2000 frames → ~110MB.  At stride=1 and 2000 frames → ~550MB.
    """

    def __init__(
        self,
        path: str,
        max_frames: int = 0,
        frame_stride: int = 10,
        depth_scale: float = 1000.0,
    ):
        import numpy as np
        from scene.readers.scannet import (
            _find_scannet_sens,
            _read_sens_header,
            _read_sens_frame_record,
        )

        sens_path = _find_scannet_sens(path)
        if sens_path is None:
            raise FileNotFoundError(f"No .sens file found in {path}")

        self._frames: List[StreamingRGBDFrame] = []

        with open(sens_path, "rb") as f:
            header = _read_sens_header(f)
            num_source = header["num_frames"]
            fx = float(header["intrinsic_color"][0, 0])
            fy = float(header["intrinsic_color"][1, 1])
            cx = float(header["intrinsic_color"][0, 2])
            cy = float(header["intrinsic_color"][1, 2])
            dfx = float(header["intrinsic_depth"][0, 0])
            dfy = float(header["intrinsic_depth"][1, 1])
            dcx = float(header["intrinsic_depth"][0, 2])
            dcy = float(header["intrinsic_depth"][1, 2])
            color_w = header["color_width"]
            color_h = header["color_height"]
            depth_w = header["depth_width"]
            depth_h = header["depth_height"]
            ds = depth_scale if depth_scale > 0 else float(header.get("depth_shift", 1000.0))

            if abs(fx - dfx) > 1.0 or abs(fy - dfy) > 1.0:
                print(
                    f"[streaming] ScanNet .sens: color intrinsics (fx={fx:.1f}) differ from "
                    f"depth intrinsics (fx={dfx:.1f}). Using depth intrinsics for backprojection."
                )

            emitted = 0
            for src_idx in range(num_source):
                c2w, ts_color, ts_depth, color_data, depth_data = _read_sens_frame_record(f)
                if frame_stride > 1 and src_idx % frame_stride != 0:
                    continue
                if max_frames > 0 and emitted >= max_frames:
                    break
                if not np.isfinite(c2w).all() or abs(np.linalg.det(c2w[:3, :3])) < 1e-6:
                    continue
                self._frames.append(StreamingRGBDFrame(
                    index=emitted,
                    timestamp=float(ts_color),
                    rgb_path="",
                    depth_path=None,
                    c2w=c2w.astype(np.float32),
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    width=color_w, height=color_h,
                    depth_scale=ds,
                    depth_fx=dfx, depth_fy=dfy, depth_cx=dcx, depth_cy=dcy,
                    depth_width=depth_w, depth_height=depth_h,
                    _rgb_bytes=color_data if color_data else None,
                    _depth_bytes=depth_data if depth_data else None,
                    _sens_header=header,
                ))
                emitted += 1

        print(
            f"[streaming] ScanNet .sens: {len(self._frames)} frames loaded "
            f"(stride={frame_stride}, source={num_source} total) from {sens_path}"
        )

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[StreamingRGBDFrame]:
        yield from self._frames

    def get_all(self) -> List[StreamingRGBDFrame]:
        return self._frames


class ReplicaFrameSource:
    """
    Ordered frames synthesised from a Replica scene mesh (mesh.ply).

    Generates a virtual camera trajectory around the mesh using the same
    `_replica_camera_path` as the offline Replica reader, then renders RGB
    and depth images using Open3D raycasting over a triangulated Replica mesh.

    depth_scale=1000.0 — depth stored as uint16 mm in _depth_bytes.
    """

    def __init__(
        self,
        path: str,
        num_views: int = 120,
        width: int = 640,
        height: int = 480,
        fov_degrees: float = 70.0,
        render_points: int = 300000,   # kept for API compat; unused
        splat_radius: int = 1,         # kept for API compat; unused
        max_frames: int = 0,
        frame_stride: int = 1,
    ):
        import math

        import numpy as np

        try:
            import sys as _sys
            if not hasattr(_sys.stdout, "isatty"):
                _sys.stdout.isatty = lambda: False  # type: ignore[attr-defined]
            if not hasattr(_sys.stderr, "isatty"):
                _sys.stderr.isatty = lambda: False  # type: ignore[attr-defined]
            import open3d as o3d
            import open3d.t.geometry as o3tg
        except ImportError:
            raise ImportError(
                "open3d is required for ReplicaFrameSource. "
                "Install it with: pip install open3d"
            )
        from scene.readers.replica import (
            _load_replica_raycast_mesh,
            _raycast_replica_frame,
            _replica_raycast_camera_path,
        )

        mesh_path = os.path.join(path, "mesh.ply")
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Replica mesh not found: {mesh_path}")

        vertices_np, mesh_t = _load_replica_raycast_mesh(mesh_path)
        scene = o3tg.RaycastingScene()
        scene.add_triangles(mesh_t)

        c2ws = _replica_raycast_camera_path(scene, vertices_np, int(num_views), fov_degrees=fov_degrees)
        if frame_stride > 1:
            c2ws = c2ws[::frame_stride]
        if max_frames > 0:
            c2ws = c2ws[:max_frames]

        fx = width / (2.0 * math.tan(math.radians(fov_degrees) * 0.5))
        fy = fx
        cx = width * 0.5
        cy = height * 0.5

        self._frames: List[StreamingRGBDFrame] = []
        for idx, c2w in enumerate(c2ws):
            rgb_bytes, depth_bytes = _raycast_replica_frame(
                scene, mesh_t, c2w, width, height, fx, fy, cx, cy
            )
            self._frames.append(StreamingRGBDFrame(
                index=idx,
                timestamp=float(idx) / 30.0,
                rgb_path="",
                depth_path=None,
                c2w=c2w.astype(np.float32),
                fx=fx, fy=fy, cx=cx, cy=cy,
                width=int(width), height=int(height),
                depth_scale=1000.0,
                _rgb_bytes=rgb_bytes,
                _depth_bytes=depth_bytes,
            ))

        print(
            f"[streaming] Replica: {len(self._frames)} raycasted frames from triangulated {mesh_path} "
            f"(vertices={len(vertices_np)})"
        )

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[StreamingRGBDFrame]:
        yield from self._frames

    def get_all(self) -> List[StreamingRGBDFrame]:
        return self._frames


class HyperSimFrameSource:
    """
    Ordered frames from an ML-HyperSim scene (ai_NNN_NNN/_detail/cam_00/).

    Reads camera poses from HDF5 files and lazily loads color + depth at
    __init__ time (all frames stored in-memory as JPEG/PNG bytes).
    Requires h5py (pip install h5py).

    depth_scale=1000.0 — depth stored as uint16 mm in _depth_bytes.
    """

    def __init__(
        self,
        path: str,
        cam_id: str = "cam_00",
        max_frames: int = 0,
        frame_stride: int = 1,
    ):
        import numpy as np
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for HyperSim streaming. "
                "Install it with: pip install h5py"
            )

        cam_dir = os.path.join(path, "_detail", cam_id)
        final_hdf5_dir = os.path.join(path, "images", f"scene_{cam_id}_final_hdf5")
        geom_hdf5_dir = os.path.join(path, "images", f"scene_{cam_id}_geometry_hdf5")

        for d in (cam_dir, final_hdf5_dir, geom_hdf5_dir):
            if not os.path.isdir(d):
                raise FileNotFoundError(f"HyperSim directory not found: {d}")

        # Read scene scale: positions are in asset units; scale to meters
        scene_meta = os.path.join(path, "_detail", "metadata_scene.csv")
        meters_per_unit = 0.0254  # default: 1 inch
        if os.path.exists(scene_meta):
            import csv
            with open(scene_meta) as f:
                for row in csv.DictReader(f):
                    if row.get("parameter_name") == "meters_per_asset_unit":
                        meters_per_unit = float(row["parameter_value"])

        # Camera poses
        with h5py.File(os.path.join(cam_dir, "camera_keyframe_positions.hdf5"), "r") as f:
            positions = f["dataset"][:].astype(np.float64)  # [N, 3] in asset units
        with h5py.File(os.path.join(cam_dir, "camera_keyframe_orientations.hdf5"), "r") as f:
            orientations = f["dataset"][:].astype(np.float64)  # [N, 3, 3]

        n_keyframes = positions.shape[0]

        # Map keyframe index → actual HDF5 file frame number
        fi_hdf5 = os.path.join(cam_dir, "camera_keyframe_frame_indices.hdf5")
        if os.path.exists(fi_hdf5):
            with h5py.File(fi_hdf5, "r") as f:
                file_frame_indices = f["dataset"][:].astype(int)  # [N]
        else:
            file_frame_indices = np.arange(n_keyframes, dtype=int)

        # Intrinsics: HyperSim default is 1024×768, fov=60° horizontal
        # (V-Ray renderer with standard settings; we use a fixed approximation)
        import math
        width, height = 1024, 768
        fov_h_deg = 60.0
        fx = width / (2.0 * math.tan(math.radians(fov_h_deg) * 0.5))
        fy = fx
        cx_f = width * 0.5
        cy_f = height * 0.5

        # Build keyframe index list (into positions/orientations arrays)
        kf_indices = list(range(n_keyframes))
        if frame_stride > 1:
            kf_indices = kf_indices[::frame_stride]
        if max_frames > 0:
            kf_indices = kf_indices[:max_frames]

        self._frames: List[StreamingRGBDFrame] = []
        import io as _io
        from PIL import Image

        for out_idx, kfi in enumerate(kf_indices):
            # kfi = index into positions/orientations; fi = HDF5 file frame number
            fi = int(file_frame_indices[kfi])
            # Build c2w from position + orientation
            # HyperSim orientation: columns = [right, up, backward] in world space
            # We use RDF convention: x=right, y=down, z=forward
            ori = orientations[kfi]  # [3, 3]
            pos = positions[kfi] * meters_per_unit  # convert to meters
            right = ori[:, 0]
            up = ori[:, 1]
            backward = ori[:, 2]
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, 0] = right.astype(np.float32)
            c2w[:3, 1] = (-up).astype(np.float32)       # down = -up
            c2w[:3, 2] = (-backward).astype(np.float32) # forward = -backward
            c2w[:3, 3] = pos.astype(np.float32)

            if not np.isfinite(c2w).all():
                continue

            # Color: float32 linear HDR → uint8 gamma-corrected
            color_path = os.path.join(final_hdf5_dir, f"frame.{fi:04d}.color.hdf5")
            depth_path = os.path.join(geom_hdf5_dir, f"frame.{fi:04d}.depth_meters.hdf5")

            if not os.path.exists(color_path) or not os.path.exists(depth_path):
                continue

            try:
                with h5py.File(color_path, "r") as f:
                    color_data = f["dataset"][:].astype(np.float32)  # [H, W, 3] linear
                # Gamma correction + clip
                color_uint8 = np.clip(np.power(np.maximum(color_data, 0.0), 1.0 / 2.2) * 255.0, 0, 255).astype(np.uint8)
                rgb_img = Image.fromarray(color_uint8, mode="RGB")
                rgb_buf = _io.BytesIO()
                rgb_img.save(rgb_buf, format="JPEG", quality=90)
                rgb_bytes = rgb_buf.getvalue()

                with h5py.File(depth_path, "r") as f:
                    depth_m = f["dataset"][:].astype(np.float32)  # [H, W] in meters
                # Convert to uint16 mm; cap at 65.535m
                depth_mm = np.round(depth_m * 1000.0).clip(0, 65535).astype(np.uint16)
                depth_img = Image.fromarray(depth_mm, mode="I;16")
                depth_buf = _io.BytesIO()
                depth_img.save(depth_buf, format="PNG")
                depth_bytes = depth_buf.getvalue()

            except Exception as e:
                print(f"[streaming] HyperSim: skipping frame {fi}: {e}", flush=True)
                continue

            self._frames.append(StreamingRGBDFrame(
                index=out_idx,
                timestamp=float(fi),
                rgb_path="",
                depth_path=None,
                c2w=c2w,
                fx=fx, fy=fy, cx=cx_f, cy=cy_f,
                width=width, height=height,
                depth_scale=1000.0,
                _rgb_bytes=rgb_bytes,
                _depth_bytes=depth_bytes,
            ))

        print(f"[streaming] HyperSim: {len(self._frames)} frames loaded from {path} (cam={cam_id})")

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[StreamingRGBDFrame]:
        yield from self._frames

    def get_all(self) -> List[StreamingRGBDFrame]:
        return self._frames


def _parse_icl_poses(poses_path: str):
    """Parse poses.gt.sim: blank-separated 3×4 c2w blocks → list of (4,4) float32 arrays."""
    import numpy as np
    with open(poses_path) as f:
        content = f.read()
    poses = []
    for block in content.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        rows = [list(map(float, r.split())) for r in block.split("\n") if r.strip()]
        if len(rows) == 3 and all(len(r) == 4 for r in rows):
            mat34 = np.array(rows, dtype=np.float32)
            mat44 = np.eye(4, dtype=np.float32)
            mat44[:3, :] = mat34
            poses.append(mat44)
    return poses


def _icl_frame_index(filename: str) -> int:
    """Extract sequential index N from ICL filenames like frame_TS_N.jpg or TS_N.png."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    return int(stem.rsplit("_", 1)[-1])


class OrbbecExportFrameSource:
    """Frames from an Orbbec Femto Bolt ICL export (rgbd_export2 tool).

    Directory layout::
        icl.yaml           — camera intrinsics + depth scale
        poses.gt.sim       — blank-separated 3×4 c2w matrices (one per frame)
        rgb/               — frame_TS_N.jpg  (JPEG colour)
        depth/             — TS_N.png        (uint16 PNG, mm)
    """

    def __init__(
        self,
        path: str,
        max_frames: int = 0,
        frame_stride: int = 1,
    ):
        import yaml
        import numpy as np

        icl_yaml = os.path.join(path, "icl.yaml")
        poses_file = os.path.join(path, "poses.gt.sim")
        rgb_dir = os.path.join(path, "rgb")
        depth_dir = os.path.join(path, "depth")

        with open(icl_yaml) as f:
            cfg = yaml.safe_load(f)
        cam = cfg["camera_params"]
        fx = float(cam["fx"])
        fy = float(cam["fy"])
        cx = float(cam["cx"])
        cy = float(cam["cy"])
        width = int(cam["image_width"])
        height = int(cam["image_height"])
        depth_scale = float(cam.get("png_depth_scale", 1000.0))

        poses = _parse_icl_poses(poses_file)

        rgb_files = sorted(
            [f for f in os.listdir(rgb_dir) if f.lower().endswith((".jpg", ".jpeg"))],
            key=_icl_frame_index,
        )
        depth_files = sorted(
            [f for f in os.listdir(depth_dir) if f.lower().endswith(".png")],
            key=_icl_frame_index,
        )

        # Match by sequential index
        rgb_by_idx = {_icl_frame_index(f): f for f in rgb_files}
        depth_by_idx = {_icl_frame_index(f): f for f in depth_files}
        common = sorted(set(rgb_by_idx) & set(depth_by_idx) & set(range(len(poses))))

        self._frames: List[StreamingRGBDFrame] = []
        for pose_idx in common[::frame_stride]:
            self._frames.append(StreamingRGBDFrame(
                index=len(self._frames),
                timestamp=float(pose_idx),
                rgb_path=os.path.join(rgb_dir, rgb_by_idx[pose_idx]),
                depth_path=os.path.join(depth_dir, depth_by_idx[pose_idx]),
                c2w=poses[pose_idx],
                fx=fx, fy=fy, cx=cx, cy=cy,
                width=width, height=height,
                depth_scale=depth_scale,
            ))
            if max_frames > 0 and len(self._frames) >= max_frames:
                break

        print(f"[streaming] OrbbecExport: {len(self._frames)} frames from {path}")

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[StreamingRGBDFrame]:
        yield from self._frames

    def get_all(self) -> List[StreamingRGBDFrame]:
        return self._frames


class OrbbecRosBagFrameSource:
    """Frames from an Orbbec Femto Bolt slam MCAP rosbag (rosbag2 format).

    Requires a slam bag that publishes ``/camera_pose``
    (``geometry_msgs/msg/PoseStamped``). Pure navigation bags (no
    ``/camera_pose``) are not supported — raise a clear error instead.

    Topic conventions (Orbbec ROS2 nodes, default namespace)::
        /camera_pose                         — PoseStamped (map frame, c2w)
        /camera/color/image_raw/compressed   — CompressedImage (JPEG)
        /camera/depth/image_raw/compressed   — CompressedImage (16UC1 PNG, mm)
        /camera/color/camera_info            — CameraInfo
    """

    def __init__(
        self,
        path: str,
        color_topic: str = "/camera/color/image_raw/compressed",
        depth_topic: str = "/camera/depth/image_raw/compressed",
        pose_topic: str = "/camera_pose",
        pose_source: str = "camera_pose",
        camera_info_topic: str = "/camera/color/camera_info",
        sync_threshold_ms: float = 33.0,
        max_frames: int = 0,
        frame_stride: int = 1,
        open3d_odom_max_failure_ratio: float = 0.25,
        open3d_odom_cache: bool = True,
        open3d_odom_cache_dir: str = "",
        open3d_odom_stride: int = 1,
        open3d_odom_downscale: int = 1,
        open3d_odom_max_trans_per_edge: float = 0.15,
        open3d_odom_max_rot_deg_per_edge: float = 8.0,
    ):
        try:
            from rosbags.rosbag2 import Reader
            from rosbags.typesys import Stores, get_typestore
        except ImportError:
            raise ImportError(
                "rosbags is required for OrbbecRosBagFrameSource. "
                "Install it with: uv pip install rosbags"
            )
        import numpy as np

        typestore = get_typestore(Stores.ROS2_HUMBLE)
        sync_ns = int(sync_threshold_ms * 1e6)
        pose_source = str(pose_source or "camera_pose")
        open3d_odom_stride = max(1, int(open3d_odom_stride))
        open3d_odom_downscale = max(1, int(open3d_odom_downscale))
        if pose_source not in {"camera_pose", "auto", "open3d_odometry", "open3d_odometry_live"}:
            raise ValueError(
                "orbbec_pose_source must be one of: camera_pose, auto, open3d_odometry, "
                "open3d_odometry_live "
                f"(got '{pose_source}')"
            )

        # First pass: inspect topics and choose the pose source.
        with Reader(path) as reader:
            topics = {c.topic for c in reader.connections}
        has_pose_topic = pose_topic in topics
        if pose_source == "auto":
            pose_source = "camera_pose" if has_pose_topic else "open3d_odometry_live"
        if pose_source == "camera_pose" and not has_pose_topic:
            raise ValueError(
                f"[streaming] OrbbecRosBagFrameSource: no '{pose_topic}' topic found in {path}.\n"
                f"  Available topics: {sorted(topics)}\n"
                "  Pass --orbbec_pose_source open3d_odometry_live to estimate poses from RGB-D, "
                "or --orbbec_pose_source auto to use /camera_pose when present and Open3D otherwise."
            )

        # Read all messages in one pass
        pose_msgs: list = []     # (ts_ns, c2w_4x4)
        color_msgs: list = []    # (ts_ns, bytes)
        depth_msgs: list = []    # (ts_ns, bytes)
        intrinsics: Optional[tuple] = None  # (fx, fy, cx, cy, W, H)

        _msg_ns_zero_warned: set = set()

        def _msg_ns(msg, fallback_ns, topic=""):
            header = getattr(msg, "header", None)
            stamp = getattr(header, "stamp", None)
            if stamp is None:
                return fallback_ns
            sec = int(getattr(stamp, "sec", getattr(stamp, "secs", 0)))
            nsec = int(getattr(stamp, "nanosec", getattr(stamp, "nsecs", 0)))
            if sec == 0 and nsec == 0:
                if topic not in _msg_ns_zero_warned:
                    _msg_ns_zero_warned.add(topic)
                    print(
                        f"[streaming] Warning: {topic!r} has unset header.stamp (sec=0, nsec=0); "
                        "falling back to bag receive timestamp for this topic.",
                        flush=True,
                    )
                return fallback_ns
            return sec * 1_000_000_000 + nsec

        with Reader(path) as reader:
            want = [color_topic, depth_topic, camera_info_topic]
            if pose_source == "camera_pose":
                want.append(pose_topic)
            conns = [c for c in reader.connections if c.topic in want]
            for conn, ts, data in reader.messages(connections=conns):
                topic = conn.topic
                if topic == pose_topic:
                    msg = typestore.deserialize_cdr(data, conn.msgtype)
                    msg_ts = _msg_ns(msg, ts, topic)
                    p = msg.pose.position
                    o = msg.pose.orientation
                    qx, qy, qz, qw = o.x, o.y, o.z, o.w
                    R = np.array([
                        [1-2*(qy*qy+qz*qz),  2*(qx*qy-qz*qw),  2*(qx*qz+qy*qw)],
                        [2*(qx*qy+qz*qw),  1-2*(qx*qx+qz*qz),  2*(qy*qz-qx*qw)],
                        [2*(qx*qz-qy*qw),    2*(qy*qz+qx*qw),  1-2*(qx*qx+qy*qy)],
                    ], dtype=np.float32)
                    c2w = np.eye(4, dtype=np.float32)
                    c2w[:3, :3] = R
                    c2w[:3, 3] = [p.x, p.y, p.z]
                    pose_msgs.append((msg_ts, c2w))
                elif topic == color_topic:
                    msg = typestore.deserialize_cdr(data, conn.msgtype)
                    msg_ts = _msg_ns(msg, ts, topic)
                    color_msgs.append((msg_ts, bytes(msg.data)))
                elif topic == depth_topic:
                    msg = typestore.deserialize_cdr(data, conn.msgtype)
                    msg_ts = _msg_ns(msg, ts, topic)
                    fmt = getattr(msg, "format", "")
                    if fmt and "png" not in fmt.lower() and "16uc1" not in fmt.lower():
                        raise RuntimeError(
                            f"Depth topic '{depth_topic}' has format '{fmt}' — "
                            "expected lossless PNG/16UC1, not JPEG. "
                            "JPEG-compressed depth corrupts metric values."
                        )
                    depth_msgs.append((msg_ts, bytes(msg.data)))
                elif topic == camera_info_topic and intrinsics is None:
                    msg = typestore.deserialize_cdr(data, conn.msgtype)
                    K = msg.k  # row-major 3×3
                    intrinsics = (
                        float(K[0]), float(K[4]),   # fx, fy
                        float(K[2]), float(K[5]),   # cx, cy
                        int(msg.width), int(msg.height),
                    )

        if intrinsics is None:
            raise RuntimeError(f"No '{camera_info_topic}' messages found in {path}")
        fx, fy, cx, cy, width, height = intrinsics
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self._width = width
        self._height = height
        self._depth_scale = 1000.0

        # Build sorted timestamp arrays for O(log n) nearest-neighbour sync
        pose_ts = np.array([m[0] for m in pose_msgs], dtype=np.int64)
        depth_ts = np.array([m[0] for m in depth_msgs], dtype=np.int64)

        synced: list = []
        for i, (color_ts, rgb_bytes) in enumerate(color_msgs[::frame_stride]):
            c2w = None
            if pose_source == "camera_pose":
                # Find closest pose
                pi = int(np.searchsorted(pose_ts, color_ts))
                pi = min(pi, len(pose_ts) - 1)
                if pi > 0 and abs(pose_ts[pi - 1] - color_ts) < abs(pose_ts[pi] - color_ts):
                    pi -= 1
                if abs(pose_ts[pi] - color_ts) > sync_ns:
                    continue
                c2w = pose_msgs[pi][1]

            # Find closest depth
            di = int(np.searchsorted(depth_ts, color_ts))
            di = min(di, len(depth_ts) - 1)
            if di > 0 and abs(depth_ts[di - 1] - color_ts) < abs(depth_ts[di] - color_ts):
                di -= 1
            depth_bytes: Optional[bytes] = None
            if abs(depth_ts[di] - color_ts) <= sync_ns:
                depth_bytes = depth_msgs[di][1]
            if depth_bytes is None:
                continue
            synced.append((color_ts, rgb_bytes, depth_bytes, c2w))
            if max_frames > 0 and len(synced) >= max_frames:
                break

        if pose_source == "open3d_odometry":
            cache_key = self._open3d_odom_cache_key(
                path=path,
                color_topic=color_topic,
                depth_topic=depth_topic,
                camera_info_topic=camera_info_topic,
                sync_threshold_ms=sync_threshold_ms,
                frame_stride=frame_stride,
                max_frames=max_frames,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                width=width,
                height=height,
                synced=synced,
                odom_stride=open3d_odom_stride,
                odom_downscale=open3d_odom_downscale,
                mode="precompute",
            )
            cache_path = self._open3d_odom_cache_path(
                bag_path=path,
                cache_dir=open3d_odom_cache_dir,
                cache_key=cache_key,
            )
            odom_poses = None
            odom_stats = None
            if open3d_odom_cache:
                odom_poses, odom_stats = self._load_open3d_odom_cache(
                    cache_path,
                    cache_key,
                    expected_frames=len(synced),
                )
            if odom_poses is None:
                odom_poses, odom_stats = self._estimate_open3d_odometry(
                    synced,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    width=width,
                    height=height,
                    depth_scale=1000.0,
                    max_failure_ratio=open3d_odom_max_failure_ratio,
                    odom_stride=open3d_odom_stride,
                    odom_downscale=open3d_odom_downscale,
                )
                if open3d_odom_cache:
                    self._save_open3d_odom_cache(cache_path, cache_key, odom_poses, odom_stats)
            synced = [
                (color_ts, rgb_bytes, depth_bytes, odom_poses[idx])
                for idx, (color_ts, rgb_bytes, depth_bytes, _) in enumerate(synced)
            ]
            path_len = odom_stats["path_length"]
            bbox_min = odom_stats["bbox_min"]
            bbox_max = odom_stats["bbox_max"]
            print(
                "[streaming] OrbbecRosBag: pose_source=open3d_odometry "
                f"success={odom_stats['successes']} failures={odom_stats['failures']} "
                f"path_length={path_len:.3f}m camera_bbox={bbox_min}->{bbox_max} "
                f"stride={open3d_odom_stride} downscale={open3d_odom_downscale}"
            )

        self._frames: List[StreamingRGBDFrame] = []
        for color_ts, rgb_bytes, depth_bytes, c2w in synced:
            self._frames.append(StreamingRGBDFrame(
                index=len(self._frames),
                timestamp=float(color_ts) * 1e-9,
                rgb_path="",
                depth_path=None,
                c2w=c2w,
                fx=fx, fy=fy, cx=cx, cy=cy,
                width=width, height=height,
                depth_scale=1000.0,
                _rgb_bytes=rgb_bytes,
                _depth_bytes=depth_bytes,
            ))

        self._pose_source = pose_source
        self._live_odom_enabled = pose_source == "open3d_odometry_live"
        self._live_odom_stride = open3d_odom_stride
        self._live_odom_downscale = open3d_odom_downscale
        self._live_odom_max_failure_ratio = float(open3d_odom_max_failure_ratio)
        self._live_odom_max_trans = float(open3d_odom_max_trans_per_edge) if open3d_odom_max_trans_per_edge > 0 else float("inf")
        self._live_odom_max_rot = float(open3d_odom_max_rot_deg_per_edge) if open3d_odom_max_rot_deg_per_edge > 0 else float("inf")
        self._live_odom_successes = 0
        self._live_odom_failures = 0
        self._live_odom_pairs = 0
        self._live_odom_computed_idx = -1
        self._live_odom_prev_key_idx = 0
        self._live_odom_prev_key_rgbd = None
        # Constant-velocity prior: reuse last successful relative transform as
        # the init guess for the next frame instead of identity.  Reset to None
        # on failure so a bad estimate doesn't compound.
        self._live_odom_last_trans: np.ndarray | None = None
        self._live_odom_last_n_edges: int = 1
        self._live_odom_cache_path = None
        self._live_odom_cache_key = None
        self._live_odom_cache_enabled = bool(open3d_odom_cache)
        self._live_odom_cached_full = False
        if self._live_odom_enabled:
            live_cache_key = self._open3d_odom_cache_key(
                path=path,
                color_topic=color_topic,
                depth_topic=depth_topic,
                camera_info_topic=camera_info_topic,
                sync_threshold_ms=sync_threshold_ms,
                frame_stride=frame_stride,
                max_frames=max_frames,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                width=width,
                height=height,
                synced=synced,
                odom_stride=open3d_odom_stride,
                odom_downscale=open3d_odom_downscale,
                mode="live",
            )
            live_cache_path = self._open3d_odom_cache_path(
                bag_path=path,
                cache_dir=open3d_odom_cache_dir,
                cache_key=live_cache_key,
            )
            self._live_odom_cache_key = live_cache_key
            self._live_odom_cache_path = live_cache_path
            if open3d_odom_cache:
                cached_poses, cached_stats = self._load_open3d_odom_cache(
                    live_cache_path,
                    live_cache_key,
                    expected_frames=len(self._frames),
                )
                if cached_poses is not None:
                    for idx, pose in enumerate(cached_poses):
                        self._frames[idx].c2w = pose
                    self._live_odom_cached_full = True
                    self._live_odom_computed_idx = len(self._frames) - 1
                    self._live_odom_successes = int(cached_stats.get("successes", 0))
                    self._live_odom_failures = int(cached_stats.get("failures", 0))
                    self._live_odom_pairs = self._live_odom_successes + self._live_odom_failures
            if not self._live_odom_cached_full and self._frames:
                self._frames[0].c2w = np.eye(4, dtype=np.float32)

        print(
            f"[streaming] OrbbecRosBag: {len(self._frames)} synced frames from {path} "
            f"(color={len(color_msgs)}, depth={len(depth_msgs)}, pose={len(pose_msgs)}, "
            f"pose_source={pose_source})"
        )

    @staticmethod
    def _open3d_odom_cache_key(
        *,
        path: str,
        color_topic: str,
        depth_topic: str,
        camera_info_topic: str,
        sync_threshold_ms: float,
        frame_stride: int,
        max_frames: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        synced,
        odom_stride: int,
        odom_downscale: int,
        mode: str = "precompute",
    ) -> str:
        import hashlib
        import json

        metadata_path = os.path.join(path, "metadata.yaml")
        try:
            stat = os.stat(metadata_path)
            metadata_sig = [int(stat.st_mtime_ns), int(stat.st_size)]
        except OSError:
            metadata_sig = [0, 0]
        timestamps = [int(item[0]) for item in synced]
        timestamps_digest = hashlib.sha256(
            ",".join(str(ts) for ts in timestamps).encode("ascii")
        ).hexdigest()[:16]
        payload = {
            "version": 1,
            "mode": str(mode),
            "path": os.path.abspath(path),
            "metadata": metadata_sig,
            "topics": [color_topic, depth_topic, camera_info_topic],
            "sync_threshold_ms": float(sync_threshold_ms),
            "frame_stride": int(frame_stride),
            "max_frames": int(max_frames),
            "intrinsics": [
                round(float(fx), 6),
                round(float(fy), 6),
                round(float(cx), 6),
                round(float(cy), 6),
                int(width),
                int(height),
            ],
            "odom_stride": int(odom_stride),
            "odom_downscale": int(odom_downscale),
            "n_frames": len(timestamps),
            "first_ts": timestamps[0] if timestamps else None,
            "last_ts": timestamps[-1] if timestamps else None,
            "timestamps_digest": timestamps_digest,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return digest[:24]

    @staticmethod
    def _open3d_odom_cache_path(bag_path: str, cache_dir: str, cache_key: str) -> Path:
        if cache_dir:
            root = Path(cache_dir).expanduser()
        else:
            root = Path(bag_path) / ".open3d_odometry_cache"
        return root / f"{cache_key}.npz"

    @staticmethod
    def _load_open3d_odom_cache(cache_path: Path, cache_key: str, expected_frames: int):
        if not cache_path.exists():
            return None, None
        try:
            import json
            import numpy as np

            data = np.load(cache_path, allow_pickle=False)
            if str(data["cache_key"]) != cache_key:
                return None, None
            poses = data["poses"].astype(np.float32)
            if poses.shape != (expected_frames, 4, 4):
                return None, None
            stats = json.loads(str(data["stats_json"]))
            stats["cache_hit"] = True
            print(f"[streaming] OrbbecRosBag: loaded Open3D odometry cache {cache_path}", flush=True)
            return [poses[i] for i in range(poses.shape[0])], stats
        except Exception as exc:
            print(f"[streaming] OrbbecRosBag: ignoring bad Open3D odometry cache {cache_path}: {exc}", flush=True)
            return None, None

    @staticmethod
    def _save_open3d_odom_cache(cache_path: Path, cache_key: str, poses, stats) -> None:
        try:
            import json
            import numpy as np

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                tmp_path,
                cache_key=np.asarray(cache_key),
                poses=np.stack(poses, axis=0).astype(np.float32),
                stats_json=np.asarray(json.dumps(stats, sort_keys=True)),
            )
            os.replace(tmp_path, cache_path)
            print(f"[streaming] OrbbecRosBag: saved Open3D odometry cache {cache_path}", flush=True)
        except Exception as exc:
            print(f"[streaming] OrbbecRosBag: failed to save Open3D odometry cache: {exc}", flush=True)

    @staticmethod
    def _estimate_open3d_odometry(
        synced_frames,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        depth_scale: float,
        max_failure_ratio: float,
        odom_stride: int = 1,
        odom_downscale: int = 1,
    ):
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError(
                "--orbbec_pose_source open3d_odometry requires Open3D in the project container."
            ) from exc
        import numpy as np

        if not synced_frames:
            raise RuntimeError("Cannot estimate Open3D odometry: no synchronized RGB-D frames.")

        odom_stride = max(1, int(odom_stride))
        odom_downscale = max(1, int(odom_downscale))
        odom_width = max(1, int(width) // odom_downscale)
        odom_height = max(1, int(height) // odom_downscale)
        odom_fx = float(fx) / odom_downscale
        odom_fy = float(fy) / odom_downscale
        odom_cx = float(cx) / odom_downscale
        odom_cy = float(cy) / odom_downscale

        key_indices = list(range(0, len(synced_frames), odom_stride))
        if key_indices[-1] != len(synced_frames) - 1:
            key_indices.append(len(synced_frames) - 1)

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(odom_width),
            int(odom_height),
            odom_fx,
            odom_fy,
            odom_cx,
            odom_cy,
        )
        option = o3d.pipelines.odometry.OdometryOption(
            depth_min=0.3,
            depth_max=5.0,
            depth_diff_max=0.07,
        )
        jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

        def to_rgbd(rgb_bytes, depth_bytes):
            from PIL import Image

            color_img = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
            depth_img = Image.open(io.BytesIO(depth_bytes))
            if color_img.size != (odom_width, odom_height):
                color_img = color_img.resize((odom_width, odom_height), Image.Resampling.BILINEAR)
            if depth_img.size != (odom_width, odom_height):
                depth_img = depth_img.resize((odom_width, odom_height), Image.Resampling.NEAREST)
            color_np = np.asarray(color_img)
            depth_np = np.asarray(depth_img)
            color = o3d.geometry.Image(np.ascontiguousarray(color_np))
            depth = o3d.geometry.Image(np.ascontiguousarray(depth_np))
            return o3d.geometry.RGBDImage.create_from_color_and_depth(
                color,
                depth,
                depth_scale=float(depth_scale),
                depth_trunc=5.0,
                convert_rgb_to_intensity=True,
            )

        def rot_to_quat(R):
            return _rot_to_quat(R)

        def quat_to_rot(q):
            return _quat_to_rot(q)

        def slerp(q0, q1, t):
            return _slerp(q0, q1, t)

        key_poses = [np.eye(4, dtype=np.float32)]
        prev_rgbd = to_rgbd(synced_frames[key_indices[0]][1], synced_frames[key_indices[0]][2])
        failures = 0
        successes = 0
        total_pairs = max(1, len(key_indices) - 1)
        for pair_idx, frame_idx in enumerate(key_indices[1:], start=1):
            _, rgb_bytes, depth_bytes, _ = synced_frames[frame_idx]
            curr_rgbd = to_rgbd(rgb_bytes, depth_bytes)
            success, trans_prev_to_curr, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                prev_rgbd,
                curr_rgbd,
                intrinsic,
                np.eye(4, dtype=np.float64),
                jacobian,
                option,
            )
            if success:
                key_poses.append((key_poses[-1] @ np.linalg.inv(trans_prev_to_curr)).astype(np.float32))
                prev_rgbd = curr_rgbd
                successes += 1
            else:
                key_poses.append(key_poses[-1].copy())
                failures += 1
            if pair_idx == 1 or pair_idx == total_pairs or pair_idx % 50 == 0:
                print(
                    "[streaming] Open3D odometry "
                    f"{pair_idx}/{total_pairs} keyframe pairs "
                    f"(source_frame={frame_idx}, successes={successes}, failures={failures})",
                    flush=True,
                )

        failure_ratio = failures / total_pairs
        if failure_ratio > float(max_failure_ratio):
            raise RuntimeError(
                "Open3D odometry failed too often: "
                f"{failures}/{total_pairs} keyframe pairs "
                f"({failure_ratio:.1%}) exceeds --orbbec_open3d_odom_max_failure_ratio={max_failure_ratio}"
            )

        poses = [None] * len(synced_frames)
        for seg_idx in range(len(key_indices) - 1):
            a = key_indices[seg_idx]
            b = key_indices[seg_idx + 1]
            pose_a = key_poses[seg_idx].astype(np.float64)
            pose_b = key_poses[seg_idx + 1].astype(np.float64)
            qa = rot_to_quat(pose_a[:3, :3])
            qb = rot_to_quat(pose_b[:3, :3])
            for frame_idx in range(a, b):
                alpha = 0.0 if b == a else (frame_idx - a) / float(b - a)
                pose = np.eye(4, dtype=np.float64)
                pose[:3, :3] = quat_to_rot(slerp(qa, qb, alpha))
                pose[:3, 3] = (1.0 - alpha) * pose_a[:3, 3] + alpha * pose_b[:3, 3]
                poses[frame_idx] = pose.astype(np.float32)
        poses[key_indices[-1]] = key_poses[-1].astype(np.float32)
        for i in range(len(poses)):
            if poses[i] is None:
                poses[i] = poses[i - 1].copy() if i > 0 else np.eye(4, dtype=np.float32)

        centers = np.stack([pose[:3, 3] for pose in poses], axis=0)
        deltas = centers[1:] - centers[:-1]
        path_length = float(np.linalg.norm(deltas, axis=1).sum()) if len(centers) > 1 else 0.0
        stats = {
            "successes": successes,
            "failures": failures,
            "path_length": path_length,
            "bbox_min": centers.min(axis=0).round(4).tolist(),
            "bbox_max": centers.max(axis=0).round(4).tolist(),
            "odom_stride": odom_stride,
            "odom_downscale": odom_downscale,
            "keyframes": len(key_indices),
            "cache_hit": False,
        }
        return poses, stats

    def _live_odom_to_rgbd(self, frame: StreamingRGBDFrame):
        try:
            import sys as _sys
            if not hasattr(_sys.stdout, "isatty"):
                _sys.stdout.isatty = lambda: False  # type: ignore[attr-defined]
            if not hasattr(_sys.stderr, "isatty"):
                _sys.stderr.isatty = lambda: False  # type: ignore[attr-defined]
            import open3d as o3d
        except ImportError as exc:
            raise ImportError(
                "--orbbec_pose_source open3d_odometry_live requires Open3D in the project container."
            ) from exc
        import numpy as np
        from PIL import Image

        odom_width = max(1, int(self._width) // self._live_odom_downscale)
        odom_height = max(1, int(self._height) // self._live_odom_downscale)
        color_img = Image.open(io.BytesIO(frame._rgb_bytes)).convert("RGB")
        depth_img = Image.open(io.BytesIO(frame._depth_bytes))
        if color_img.size != (odom_width, odom_height):
            color_img = color_img.resize((odom_width, odom_height), Image.Resampling.BILINEAR)
        if depth_img.size != (odom_width, odom_height):
            depth_img = depth_img.resize((odom_width, odom_height), Image.Resampling.NEAREST)
        color = o3d.geometry.Image(np.ascontiguousarray(np.asarray(color_img)))
        depth = o3d.geometry.Image(np.ascontiguousarray(np.asarray(depth_img)))
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=float(self._depth_scale),
            depth_trunc=5.0,
            convert_rgb_to_intensity=True,
        )

    def _save_live_odom_cache(self) -> None:
        if not self._live_odom_cache_enabled or self._live_odom_cache_path is None:
            return
        if self._live_odom_computed_idx < len(self._frames) - 1:
            return
        import numpy as np

        poses = [f.c2w if f.c2w is not None else np.eye(4, dtype=np.float32) for f in self._frames]
        centers = np.stack([pose[:3, 3] for pose in poses], axis=0)
        deltas = centers[1:] - centers[:-1]
        stats = {
            "successes": self._live_odom_successes,
            "failures": self._live_odom_failures,
            "path_length": float(np.linalg.norm(deltas, axis=1).sum()) if len(centers) > 1 else 0.0,
            "bbox_min": centers.min(axis=0).round(4).tolist(),
            "bbox_max": centers.max(axis=0).round(4).tolist(),
            "odom_stride": self._live_odom_stride,
            "odom_downscale": self._live_odom_downscale,
            "keyframes": self._live_odom_pairs + 1,
            "cache_hit": False,
        }
        self._save_open3d_odom_cache(
            self._live_odom_cache_path,
            self._live_odom_cache_key,
            poses,
            stats,
        )

    def _ensure_live_odom_until(self, index: int) -> None:
        if not self._live_odom_enabled or self._live_odom_cached_full:
            return
        if index <= self._live_odom_computed_idx:
            return
        import sys as _sys
        if not hasattr(_sys.stdout, "isatty"):
            _sys.stdout.isatty = lambda: False  # type: ignore[attr-defined]
        if not hasattr(_sys.stderr, "isatty"):
            _sys.stderr.isatty = lambda: False  # type: ignore[attr-defined]
        import numpy as np
        import open3d as o3d

        if not self._frames:
            return
        if self._live_odom_computed_idx < 0:
            self._frames[0].c2w = np.eye(4, dtype=np.float32)
            self._live_odom_prev_key_idx = 0
            self._live_odom_prev_key_rgbd = self._live_odom_to_rgbd(self._frames[0])
            self._live_odom_computed_idx = 0

        odom_width = max(1, int(self._width) // self._live_odom_downscale)
        odom_height = max(1, int(self._height) // self._live_odom_downscale)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(odom_width),
            int(odom_height),
            float(self._fx) / self._live_odom_downscale,
            float(self._fy) / self._live_odom_downscale,
            float(self._cx) / self._live_odom_downscale,
            float(self._cy) / self._live_odom_downscale,
        )
        option = o3d.pipelines.odometry.OdometryOption(
            depth_min=0.3,
            depth_max=5.0,
            depth_diff_max=0.07,
        )
        jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

        target = index
        if target > self._live_odom_stride and target % self._live_odom_stride != 0:
            target = ((target + self._live_odom_stride - 1) // self._live_odom_stride) * self._live_odom_stride
        target = min(target, len(self._frames) - 1)

        for idx in range(self._live_odom_computed_idx + 1, target + 1):
            should_estimate = (
                idx <= self._live_odom_stride
                or (idx % self._live_odom_stride) == 0
                or idx == len(self._frames) - 1
            )
            if not should_estimate:
                continue

            curr_rgbd = self._live_odom_to_rgbd(self._frames[idx])
            # Constant-velocity prior: use the last successful relative transform
            # as the initial guess instead of identity.  This dramatically improves
            # convergence when the camera starts moving from rest, reducing the
            # "double edge" artifact at depth discontinuities during motion onset.
            # Scale by the number of elapsed frames so the prediction is correct
            # for variable stride (for stride=1 last_trans is used as-is).
            if self._live_odom_last_trans is not None:
                n_prev = max(1, self._live_odom_last_n_edges)
                n_curr = max(1, idx - self._live_odom_prev_key_idx)
                if n_curr == n_prev:
                    init_guess = self._live_odom_last_trans.copy()
                else:
                    # Scale translation proportionally; rotation stays same direction
                    T = self._live_odom_last_trans.copy()
                    T[:3, 3] *= float(n_curr) / float(n_prev)
                    init_guess = T
            else:
                init_guess = np.eye(4, dtype=np.float64)
            success, trans_prev_to_curr, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                self._live_odom_prev_key_rgbd,
                curr_rgbd,
                intrinsic,
                init_guess,
                jacobian,
                option,
            )
            self._live_odom_pairs += 1
            key_pose = self._frames[self._live_odom_prev_key_idx].c2w

            # Phase 2.3: sanity-gate on estimated per-edge motion
            if success:
                delta = np.linalg.inv(trans_prev_to_curr)
                edge_t = float(np.linalg.norm(delta[:3, 3]))
                cos_r = float(np.clip((np.trace(delta[:3, :3]) - 1.0) * 0.5, -1.0, 1.0))
                edge_r = float(np.degrees(np.arccos(cos_r)))
                n_edges = max(1, idx - self._live_odom_prev_key_idx)
                if (edge_t > self._live_odom_max_trans * n_edges or
                        edge_r > self._live_odom_max_rot * n_edges):
                    print(
                        f"[live-o3d] idx={idx} sanity-gate reject: "
                        f"dt={edge_t:.3f}m dr={edge_r:.1f}° over {n_edges} edges "
                        f"(limits {self._live_odom_max_trans * n_edges:.3f}m "
                        f"/ {self._live_odom_max_rot * n_edges:.1f}°)",
                        flush=True,
                    )
                    success = False

            if success:
                curr_pose = (key_pose @ np.linalg.inv(trans_prev_to_curr)).astype(np.float32)
                self._frames[idx]._odom_valid = True
                self._live_odom_successes += 1
                # Update constant-velocity prior with the successful transform
                self._live_odom_last_trans = trans_prev_to_curr.copy()
                self._live_odom_last_n_edges = max(1, idx - self._live_odom_prev_key_idx)
            else:
                # Phase 2.2: on failure, keep the previous key pose but do NOT advance
                # the odometry reference — the next estimate will still compare against
                # the last good keyframe, preventing one bad edge from corrupting
                # the rest of the trajectory.
                curr_pose = key_pose.copy()
                self._frames[idx].c2w = curr_pose
                self._frames[idx]._odom_valid = False
                self._live_odom_failures += 1
                # Reset the velocity prior so a bad estimate doesn't compound
                self._live_odom_last_trans = None
                failure_ratio = self._live_odom_failures / max(1, self._live_odom_pairs)
                if failure_ratio > self._live_odom_max_failure_ratio:
                    raise RuntimeError(
                        "Open3D live odometry failed too often: "
                        f"{self._live_odom_failures}/{self._live_odom_pairs} keyframes "
                        f"({failure_ratio:.1%}) exceeds --orbbec_open3d_odom_max_failure_ratio="
                        f"{self._live_odom_max_failure_ratio}"
                    )
                self._live_odom_computed_idx = idx
                continue  # skip reference advance

            self._frames[idx].c2w = curr_pose

            if idx > self._live_odom_prev_key_idx + 1:
                qa = _rot_to_quat(key_pose[:3, :3])
                qb = _rot_to_quat(curr_pose[:3, :3])
                for i in range(self._live_odom_prev_key_idx + 1, idx):
                    alpha = (i - self._live_odom_prev_key_idx) / float(idx - self._live_odom_prev_key_idx)
                    pose = np.eye(4, dtype=np.float64)
                    pose[:3, :3] = _quat_to_rot(_slerp(qa, qb, alpha))
                    pose[:3, 3] = (1.0 - alpha) * key_pose[:3, 3] + alpha * curr_pose[:3, 3]
                    self._frames[i].c2w = pose.astype(np.float32)
                    self._frames[i]._odom_valid = True

            self._live_odom_prev_key_idx = idx
            self._live_odom_prev_key_rgbd = curr_rgbd

            failure_ratio = self._live_odom_failures / max(1, self._live_odom_pairs)
            if failure_ratio > self._live_odom_max_failure_ratio:
                raise RuntimeError(
                    "Open3D live odometry failed too often: "
                    f"{self._live_odom_failures}/{self._live_odom_pairs} keyframes "
                    f"({failure_ratio:.1%}) exceeds --orbbec_open3d_odom_max_failure_ratio="
                    f"{self._live_odom_max_failure_ratio}"
                )
            if self._live_odom_pairs == 1 or self._live_odom_pairs % 50 == 0:
                print(
                    "[streaming] live Open3D odometry "
                    f"source_frame={idx} successes={self._live_odom_successes} "
                    f"failures={self._live_odom_failures}",
                    flush=True,
                )
            self._live_odom_computed_idx = idx

        self._save_live_odom_cache()

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[StreamingRGBDFrame]:
        for idx in range(len(self._frames)):
            yield self.get_frame(idx)

    def get_all(self) -> List[StreamingRGBDFrame]:
        if self._live_odom_enabled:
            self._ensure_live_odom_until(len(self._frames) - 1)
        return self._frames

    @property
    def lazy_frames(self) -> bool:
        return self._live_odom_enabled and not self._live_odom_cached_full

    def get_frame(self, index: int) -> StreamingRGBDFrame:
        if index < 0 or index >= len(self._frames):
            raise IndexError(index)
        if self._live_odom_enabled:
            self._ensure_live_odom_until(index)
        return self._frames[index]


def make_frame_source(source_path: str, args) -> "RGBDSequenceFrameSource | TUMFrameSource | ScanNetFrameSource | ScanNetSensFrameSource | ReplicaFrameSource | HyperSimFrameSource | OrbbecExportFrameSource | OrbbecRosBagFrameSource":
    """Auto-detect the RGB-D dataset layout and return an ordered frame source."""
    path = source_path
    if (
        os.path.exists(os.path.join(path, "frames.jsonl"))
        and os.path.exists(os.path.join(path, "intrinsics.json"))
    ):
        return RGBDSequenceFrameSource(
            path,
            max_frames=getattr(args, "streaming_max_frames", 0),
            frame_stride=getattr(args, "streaming_frame_stride", 1),
        )
    if (
        os.path.exists(os.path.join(path, "rgb.txt"))
        and os.path.exists(os.path.join(path, "depth.txt"))
        and os.path.exists(os.path.join(path, "groundtruth.txt"))
    ):
        return TUMFrameSource(
            path,
            max_frames=getattr(args, "streaming_max_frames", 0),
            frame_stride=getattr(args, "tum_frame_stride", 1),
            association_max_dt=getattr(args, "tum_association_max_dt", 0.03),
            sequence=getattr(args, "tum_sequence", ""),
        )
    # Prefer .sens when present — it contains all frames; extracted dirs may be subsampled
    from scene.readers.scannet import _find_scannet_sens
    if _find_scannet_sens(path) is not None:
        return ScanNetSensFrameSource(
            path,
            max_frames=getattr(args, "streaming_max_frames", 0),
            frame_stride=getattr(args, "scannet_frame_stride", 10),
            depth_scale=getattr(args, "scannet_depth_scale", 1000.0),
        )
    if (
        os.path.exists(os.path.join(path, "color"))
        and os.path.exists(os.path.join(path, "pose"))
        and os.path.exists(os.path.join(path, "intrinsic"))
    ):
        return ScanNetFrameSource(
            path,
            max_frames=getattr(args, "streaming_max_frames", 0),
            frame_stride=getattr(args, "scannet_frame_stride", 10),
            depth_scale=getattr(args, "scannet_depth_scale", 1000.0),
        )
    # HyperSim: _detail/ directory with at least one cam_XX subdirectory
    _detail_dir = os.path.join(path, "_detail")
    if os.path.isdir(_detail_dir):
        cam_id = getattr(args, "hypersim_cam_id", "cam_00")
        if os.path.isdir(os.path.join(_detail_dir, cam_id)):
            return HyperSimFrameSource(
                path,
                cam_id=cam_id,
                max_frames=getattr(args, "streaming_max_frames", 0),
                frame_stride=getattr(args, "hypersim_frame_stride", 1),
            )
    # Replica: mesh.ply present (synthetic point-splat rendering)
    if os.path.exists(os.path.join(path, "mesh.ply")):
        return ReplicaFrameSource(
            path,
            num_views=getattr(args, "replica_num_views", 120),
            width=getattr(args, "replica_width", 640),
            height=getattr(args, "replica_height", 480),
            fov_degrees=getattr(args, "replica_fov", 70.0),
            render_points=getattr(args, "replica_render_points", 300000),
            splat_radius=getattr(args, "replica_splat_radius", 1),
            max_frames=getattr(args, "streaming_max_frames", 0),
            frame_stride=getattr(args, "streaming_frame_stride", 1),
        )
    # Orbbec ICL export: icl.yaml + poses.gt.sim
    if (
        os.path.exists(os.path.join(path, "icl.yaml"))
        and os.path.exists(os.path.join(path, "poses.gt.sim"))
    ):
        return OrbbecExportFrameSource(
            path,
            max_frames=getattr(args, "streaming_max_frames", 0),
            frame_stride=getattr(args, "streaming_frame_stride", 1),
        )
    # Orbbec ROS2 slam bag: metadata.yaml (rosbag2 format)
    if os.path.exists(os.path.join(path, "metadata.yaml")):
        return OrbbecRosBagFrameSource(
            path,
            color_topic=getattr(args, "orbbec_color_topic", "/camera/color/image_raw/compressed"),
            depth_topic=getattr(args, "orbbec_depth_topic", "/camera/depth/image_raw/compressed"),
            pose_topic=getattr(args, "orbbec_pose_topic", "/camera_pose"),
            pose_source=getattr(args, "orbbec_pose_source", "camera_pose"),
            camera_info_topic=getattr(args, "orbbec_camera_info_topic", "/camera/color/camera_info"),
            sync_threshold_ms=getattr(args, "orbbec_sync_threshold_ms", 33.0),
            max_frames=getattr(args, "streaming_max_frames", 0),
            frame_stride=getattr(args, "streaming_frame_stride", 1),
            open3d_odom_max_failure_ratio=getattr(args, "orbbec_open3d_odom_max_failure_ratio", 0.25),
            open3d_odom_cache=getattr(args, "orbbec_open3d_odom_cache", True),
            open3d_odom_cache_dir=getattr(args, "orbbec_open3d_odom_cache_dir", ""),
            open3d_odom_stride=getattr(args, "orbbec_open3d_odom_stride", 1),
            open3d_odom_downscale=getattr(args, "orbbec_open3d_odom_downscale", 1),
            open3d_odom_max_trans_per_edge=getattr(args, "orbbec_open3d_odom_max_trans_per_edge", 0.15),
            open3d_odom_max_rot_deg_per_edge=getattr(args, "orbbec_open3d_odom_max_rot_deg_per_edge", 8.0),
        )
    raise ValueError(
        f"[streaming] No supported RGB-D dataset layout found at: {path}\n"
        "Supported: generic RGBDSequence (frames.jsonl + intrinsics.json), "
        "TUM (rgb.txt + depth.txt + groundtruth.txt), "
        "ScanNet (color/ + pose/ + intrinsic/ or .sens file), "
        "HyperSim (_detail/cam_00/ with HDF5 frames), "
        "Replica (mesh.ply), "
        "OrbbecExport (icl.yaml + poses.gt.sim), "
        "OrbbecRosBag (metadata.yaml with /camera_pose topic)."
    )
