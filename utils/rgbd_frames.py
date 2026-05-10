import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
from PIL import Image


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 1000.0
    camera_frame: str = ""
    world_frame: str = ""


@dataclass
class RGBDFrame:
    index: int
    timestamp: float
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: CameraIntrinsics
    c2w: np.ndarray
    rgb_path: Optional[str] = None
    depth_path: Optional[str] = None


class RGBDFrameSource:
    def __iter__(self) -> Iterator[RGBDFrame]:
        raise NotImplementedError


def read_intrinsics(path):
    with open(path, "r") as f:
        data = json.load(f)
    return CameraIntrinsics(
        width=int(data["width"]),
        height=int(data["height"]),
        fx=float(data["fx"]),
        fy=float(data["fy"]),
        cx=float(data["cx"]),
        cy=float(data["cy"]),
        depth_scale=float(data.get("depth_scale", 1000.0)),
        camera_frame=str(data.get("camera_frame", "")),
        world_frame=str(data.get("world_frame", "")),
    )


def write_intrinsics(path, intrinsics):
    data = asdict(intrinsics)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def read_metadata(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def write_metadata(path, metadata):
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


def read_frame_records(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def append_frame_record(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record, separators=(",", ":")))
        f.write("\n")


def ensure_sequence_dirs(path):
    root = Path(path)
    (root / "rgb").mkdir(parents=True, exist_ok=True)
    (root / "depth").mkdir(parents=True, exist_ok=True)
    return root


def write_sequence_frame(root, frame, depth_scale=None):
    root = ensure_sequence_dirs(root)
    depth_scale = float(depth_scale if depth_scale is not None else frame.intrinsics.depth_scale)
    rgb_rel = f"rgb/{frame.index:06d}.png"
    depth_rel = f"depth/{frame.index:06d}.png"

    Image.fromarray(np.asarray(frame.rgb, dtype=np.uint8)).save(root / rgb_rel)
    Image.fromarray(depth_to_uint16_png(frame.depth, depth_scale)).save(root / depth_rel)

    record = {
        "id": int(frame.index),
        "timestamp": float(frame.timestamp),
        "rgb": rgb_rel,
        "depth": depth_rel,
        "c2w": np.asarray(frame.c2w, dtype=np.float32).tolist(),
    }
    append_frame_record(root / "frames.jsonl", record)
    return record


def decode_color_image(data, encoding):
    enc = str(encoding).lower()
    arr = np.asarray(data)
    if enc in {"rgb8", "8uc3"}:
        rgb = arr
    elif enc == "bgr8":
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    elif enc == "rgba8":
        rgb = arr[..., :3]
    elif enc == "bgra8":
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    elif enc in {"mono8", "8uc1"}:
        rgb = np.repeat(arr[..., None], 3, axis=2)
    else:
        raise ValueError(f"Unsupported RGB image encoding: {encoding}")
    return np.ascontiguousarray(rgb.astype(np.uint8))


def decode_depth_image(data, encoding):
    enc = str(encoding).lower()
    arr = np.asarray(data)
    if enc in {"16uc1", "mono16", "uint16"}:
        return arr.astype(np.uint16, copy=False)
    if enc in {"32fc1", "float32"}:
        return arr.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported depth image encoding: {encoding}")


def depth_to_meters(depth, depth_scale):
    arr = np.asarray(depth)
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        return arr.astype(np.float32)
    return arr.astype(np.float32) / float(depth_scale)


def depth_to_uint16_png(depth, depth_scale):
    arr = np.asarray(depth)
    if arr.dtype == np.uint16:
        return arr
    meters = depth_to_meters(arr, depth_scale=1.0)
    scaled = np.nan_to_num(meters, nan=0.0, posinf=0.0, neginf=0.0) * float(depth_scale)
    return np.clip(np.rint(scaled), 0, np.iinfo(np.uint16).max).astype(np.uint16)


def c2w_to_camera_rt(c2w):
    c2w = np.asarray(c2w, dtype=np.float32)
    if c2w.shape != (4, 4):
        raise ValueError("Expected c2w to have shape 4x4")
    if not np.isfinite(c2w).all():
        raise ValueError("Expected c2w to contain only finite values")
    w2c = np.linalg.inv(c2w)
    return w2c[:3, :3].T, w2c[:3, 3]


def reproject_depth_to_color(
    depth,
    depth_intrinsics,
    color_intrinsics,
    depth_to_color,
    min_depth=0.1,
    max_depth=8.0,
):
    """Align a depth image into the color optical frame with nearest-pixel z-buffering."""
    z = depth_to_meters(depth, depth_intrinsics.depth_scale)
    h, w = z.shape
    ys, xs = np.mgrid[0:h, 0:w]

    valid = np.isfinite(z) & (z > float(min_depth)) & (z < float(max_depth))
    if valid.sum() == 0:
        return np.zeros((color_intrinsics.height, color_intrinsics.width), dtype=np.float32)

    z_d = z[valid]
    x_d = (xs[valid].astype(np.float32) - depth_intrinsics.cx) / depth_intrinsics.fx * z_d
    y_d = (ys[valid].astype(np.float32) - depth_intrinsics.cy) / depth_intrinsics.fy * z_d
    pts_d = np.stack([x_d, y_d, z_d], axis=1)

    transform = np.asarray(depth_to_color, dtype=np.float32)
    if transform.shape != (4, 4):
        raise ValueError("Expected depth_to_color to have shape 4x4")
    pts_c = (transform[:3, :3] @ pts_d.T).T + transform[:3, 3]
    z_c = pts_c[:, 2]

    valid_c = np.isfinite(z_c) & (z_c > float(min_depth)) & (z_c < float(max_depth))
    if valid_c.sum() == 0:
        return np.zeros((color_intrinsics.height, color_intrinsics.width), dtype=np.float32)

    pts_c = pts_c[valid_c]
    z_c = z_c[valid_c]
    u = np.rint(color_intrinsics.fx * pts_c[:, 0] / z_c + color_intrinsics.cx).astype(np.int32)
    v = np.rint(color_intrinsics.fy * pts_c[:, 1] / z_c + color_intrinsics.cy).astype(np.int32)

    inside = (
        (u >= 0)
        & (u < color_intrinsics.width)
        & (v >= 0)
        & (v < color_intrinsics.height)
    )
    aligned = np.zeros((color_intrinsics.height, color_intrinsics.width), dtype=np.float32)
    if inside.sum() == 0:
        return aligned

    u = u[inside]
    v = v[inside]
    z_c = z_c[inside]
    flat_idx = v * color_intrinsics.width + u
    order = np.argsort(z_c)
    flat = aligned.reshape(-1)
    for idx, depth_value in zip(flat_idx[order], z_c[order]):
        if flat[idx] == 0.0:
            flat[idx] = depth_value
    return aligned


def rgbd_pointcloud_from_records(
    root,
    records,
    intrinsics,
    depth_stride=4,
    init_frames=300,
    max_points=250000,
    min_depth=0.1,
    max_depth=8.0,
):
    points_all = []
    colors_all = []
    selected = records[:init_frames] if init_frames and init_frames > 0 else records

    for record in selected:
        c2w = np.asarray(record["c2w"], dtype=np.float32)
        depth_path = Path(root) / record["depth"]
        rgb_path = Path(root) / record["rgb"]
        if not depth_path.exists() or not rgb_path.exists():
            continue

        depth = np.array(Image.open(depth_path))
        rgb = np.array(Image.open(rgb_path).convert("RGB")).astype(np.float32) / 255.0
        z = depth_to_meters(depth, intrinsics.depth_scale)
        h, w = z.shape
        ys, xs = np.mgrid[0:h:depth_stride, 0:w:depth_stride]
        z_v = z[ys, xs]
        valid = np.isfinite(z_v) & (z_v > float(min_depth)) & (z_v < float(max_depth))
        if valid.sum() == 0:
            continue

        xs_v = xs[valid].astype(np.float32)
        ys_v = ys[valid].astype(np.float32)
        z_v = z_v[valid].astype(np.float32)
        x = (xs_v - intrinsics.cx) / intrinsics.fx * z_v
        y = (ys_v - intrinsics.cy) / intrinsics.fy * z_v
        pts_cam = np.stack([x, y, z_v], axis=1)
        pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]

        rgb_h, rgb_w = rgb.shape[:2]
        rgb_x = np.clip((xs_v / max(w - 1, 1) * (rgb_w - 1)).round().astype(np.int32), 0, rgb_w - 1)
        rgb_y = np.clip((ys_v / max(h - 1, 1) * (rgb_h - 1)).round().astype(np.int32), 0, rgb_h - 1)
        rgb_v = rgb[rgb_y, rgb_x]
        points_all.append(pts_world.astype(np.float32))
        colors_all.append(rgb_v.astype(np.float32))

    if not points_all:
        raise RuntimeError("Could not initialize RGB-D point cloud from sequence frames.")

    points = np.concatenate(points_all, axis=0)
    colors = np.concatenate(colors_all, axis=0)
    if max_points and points.shape[0] > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
    normals = np.zeros_like(points, dtype=np.float32)
    return points, colors, normals
