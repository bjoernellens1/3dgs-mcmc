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
    and depth images using software point-splatting so no GPU / habitat-sim
    is required at dataset-loading time.

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
            import open3d as o3d
            import open3d.t.geometry as o3tg
        except ImportError:
            raise ImportError(
                "open3d is required for ReplicaFrameSource. "
                "Install it with: pip install open3d"
            )
        from scene.readers.replica import _raycast_replica_frame, _replica_camera_path

        mesh_path = os.path.join(path, "mesh.ply")
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Replica mesh not found: {mesh_path}")

        mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
        mesh_t = o3tg.TriangleMesh.from_legacy(mesh_o3d)
        scene = o3tg.RaycastingScene()
        scene.add_triangles(mesh_t)

        vertices_np = np.asarray(mesh_o3d.vertices)
        c2ws = _replica_camera_path(vertices_np, int(num_views))
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
                timestamp=float(idx),
                rgb_path="",
                depth_path=None,
                c2w=c2w.astype(np.float32),
                fx=fx, fy=fy, cx=cx, cy=cy,
                width=int(width), height=int(height),
                depth_scale=1000.0,
                _rgb_bytes=rgb_bytes,
                _depth_bytes=depth_bytes,
            ))

        print(f"[streaming] Replica: {len(self._frames)} raycasted frames from {mesh_path}")

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


def make_frame_source(source_path: str, args) -> "RGBDSequenceFrameSource | TUMFrameSource | ScanNetFrameSource | ScanNetSensFrameSource | ReplicaFrameSource | HyperSimFrameSource":
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
    raise ValueError(
        f"[streaming] No supported RGB-D dataset layout found at: {path}\n"
        "Supported: generic RGBDSequence (frames.jsonl + intrinsics.json), "
        "TUM (rgb.txt + depth.txt + groundtruth.txt), "
        "ScanNet (color/ + pose/ + intrinsic/ or .sens file), "
        "HyperSim (_detail/cam_00/ with HDF5 frames), "
        "Replica (mesh.ply)."
    )
