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
    """Return raw uint16 depth numpy array, or None on failure."""
    import numpy as np
    if frame._depth_bytes is not None and frame._sens_header is not None:
        from scene.readers.scannet import _decode_sens_depth
        return _decode_sens_depth(frame._depth_bytes, frame._sens_header)
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


def make_frame_source(source_path: str, args) -> "RGBDSequenceFrameSource | TUMFrameSource | ScanNetFrameSource | ScanNetSensFrameSource":
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
    raise ValueError(
        f"[streaming] No supported RGB-D dataset layout found at: {path}\n"
        "Supported: generic RGBDSequence (frames.jsonl + intrinsics.json), "
        "TUM (rgb.txt + depth.txt + groundtruth.txt), "
        "ScanNet (color/ + pose/ + intrinsic/ or .sens file)."
    )
