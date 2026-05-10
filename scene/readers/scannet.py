import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image

from scene.readers.common import BasicPointCloud, CameraInfo, SceneInfo, fetchPlyFlexible, getNerfppNorm, storePly
from utils.graphics_utils import getWorld2View2, focal2fov
from utils.sh_utils import SH2RGB

def _numeric_stem(path):
    stem = Path(path).stem
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)

def _read_scannet_intrinsic(path):
    K4 = np.loadtxt(path).astype(np.float32)
    if K4.shape != (4, 4):
        raise ValueError(f"Expected 4x4 ScanNet intrinsic matrix at {path}")
    fx = float(K4[0, 0])
    fy = float(K4[1, 1])
    cx = float(K4[0, 2])
    cy = float(K4[1, 2])
    return fx, fy, cx, cy

def _find_scannet_intrinsics(path):
    intrinsic_dir = os.path.join(path, "intrinsic")
    color_intr = os.path.join(intrinsic_dir, "intrinsic_color.txt")
    depth_intr = os.path.join(intrinsic_dir, "intrinsic_depth.txt")

    if not os.path.exists(color_intr):
        raise FileNotFoundError(f"Missing ScanNet color intrinsics: {color_intr}")

    fx, fy, cx, cy = _read_scannet_intrinsic(color_intr)

    if os.path.exists(depth_intr):
        dfx, dfy, dcx, dcy = _read_scannet_intrinsic(depth_intr)
    else:
        dfx, dfy, dcx, dcy = fx, fy, cx, cy

    return {
        "color": (fx, fy, cx, cy),
        "depth": (dfx, dfy, dcx, dcy),
    }

def _read_scannet_pose(path):
    pose = np.loadtxt(path).astype(np.float32)
    if pose.shape != (4, 4):
        return None
    if not np.isfinite(pose).all():
        return None
    if abs(np.linalg.det(pose[:3, :3])) < 1e-6:
        return None
    return pose

def _list_scannet_color_frames(path):
    color_dir = os.path.join(path, "color")
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(glob.glob(os.path.join(color_dir, ext)))
    return sorted(files, key=_numeric_stem)

def _scannet_frame_id_from_color(path):
    return Path(path).stem

def readScanNetCameras(path, frame_stride=10, max_frames=0, eval=False, eval_hold=20):
    intr = _find_scannet_intrinsics(path)
    fx, fy, cx, cy = intr["color"]

    color_files = _list_scannet_color_frames(path)
    if frame_stride > 1:
        color_files = color_files[::frame_stride]
    if max_frames and max_frames > 0:
        color_files = color_files[:max_frames]

    cam_infos = []
    skipped = 0

    for image_path in color_files:
        frame_id = _scannet_frame_id_from_color(image_path)
        pose_path = os.path.join(path, "pose", f"{frame_id}.txt")
        if not os.path.exists(pose_path):
            skipped += 1
            continue

        c2w = _read_scannet_pose(pose_path)
        if c2w is None:
            skipped += 1
            continue

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # ScanNet pose files are camera-to-world transforms.
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3].T
        T = w2c[:3, 3]

        FovX = focal2fov(fx, width)
        FovY = focal2fov(fy, height)

        cam_infos.append(
            CameraInfo(
                uid=len(cam_infos),
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_path=image_path,
                image_name=frame_id,
                width=width,
                height=height,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
            )
        )

    print(
        f"Loaded ScanNet cameras: {len(cam_infos)} "
        f"(skipped invalid/missing poses: {skipped})"
    )

    if eval and eval_hold > 0:
        train_cam_infos = [c for i, c in enumerate(cam_infos) if i % eval_hold != 0]
        test_cam_infos = [c for i, c in enumerate(cam_infos) if i % eval_hold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    return train_cam_infos, test_cam_infos

def _load_scannet_rgbd_pointcloud(
    path,
    train_cam_infos,
    depth_stride=8,
    init_frames=200,
    max_points=250000,
    depth_scale=1000.0,
):
    intr = _find_scannet_intrinsics(path)
    dfx, dfy, dcx, dcy = intr["depth"]

    points_all = []
    colors_all = []

    selected = train_cam_infos
    if init_frames and init_frames > 0:
        selected = selected[:init_frames]

    for cam in selected:
        frame_id = cam.image_name
        depth_path = os.path.join(path, "depth", f"{frame_id}.png")
        if not os.path.exists(depth_path):
            continue

        c2w = np.linalg.inv(getWorld2View2(cam.R, cam.T))

        depth = np.array(Image.open(depth_path)).astype(np.float32) / float(depth_scale)
        rgb = np.array(cam.image.convert("RGB")).astype(np.float32) / 255.0

        h, w = depth.shape
        ys, xs = np.mgrid[0:h:depth_stride, 0:w:depth_stride]
        z = depth[ys, xs]

        valid = np.isfinite(z) & (z > 0.1) & (z < 10.0)
        if valid.sum() == 0:
            continue

        xs_v = xs[valid].astype(np.float32)
        ys_v = ys[valid].astype(np.float32)
        z_v = z[valid].astype(np.float32)

        x = (xs_v - dcx) / dfx * z_v
        y = (ys_v - dcy) / dfy * z_v
        pts_cam = np.stack([x, y, z_v], axis=1)

        pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]

        rgb_h, rgb_w = rgb.shape[:2]
        rgb_x = np.clip((xs_v / max(w - 1, 1) * (rgb_w - 1)).round().astype(np.int32), 0, rgb_w - 1)
        rgb_y = np.clip((ys_v / max(h - 1, 1) * (rgb_h - 1)).round().astype(np.int32), 0, rgb_h - 1)
        rgb_v = rgb[rgb_y, rgb_x]

        points_all.append(pts_world.astype(np.float32))
        colors_all.append(rgb_v.astype(np.float32))

    if not points_all:
        raise RuntimeError("Could not initialize ScanNet point cloud from RGB-D frames.")

    points = np.concatenate(points_all, axis=0)
    colors = np.concatenate(colors_all, axis=0)

    if max_points and points.shape[0] > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    normals = np.zeros_like(points, dtype=np.float32)
    return BasicPointCloud(points=points, colors=colors, normals=normals)

def _find_scannet_mesh(path):
    candidates = []
    candidates.extend(glob.glob(os.path.join(path, "*_vh_clean_2.ply")))
    candidates.extend(glob.glob(os.path.join(path, "*_vh_clean.ply")))
    candidates.extend(
        p for p in glob.glob(os.path.join(path, "*.ply"))
        if not Path(p).name.startswith("scannet_")
    )
    return candidates[0] if candidates else None

def readScanNetSceneInfo(
    path,
    eval,
    frame_stride=10,
    max_frames=0,
    eval_hold=20,
    init_type="rgbd",
    depth_stride=8,
    init_frames=200,
    max_init_points=250000,
    depth_scale=1000.0,
    num_pts=100000,
):
    train_cam_infos, test_cam_infos = readScanNetCameras(
        path=path,
        frame_stride=frame_stride,
        max_frames=max_frames,
        eval=eval,
        eval_hold=eval_hold,
    )

    if len(train_cam_infos) == 0:
        raise RuntimeError(f"No valid ScanNet cameras found in {path}")

    nerf_normalization = getNerfppNorm(train_cam_infos)
    init_type = init_type.lower()

    if init_type == "mesh":
        mesh_path = _find_scannet_mesh(path)
        if mesh_path is None:
            raise RuntimeError("ScanNet mesh init requested, but no *_vh_clean*.ply found.")
        ply_path = os.path.join(path, "scannet_mesh_init.ply")
        pcd = fetchPlyFlexible(mesh_path)
        storePly(ply_path, pcd.points, np.clip(pcd.colors * 255.0, 0, 255))
    elif init_type == "rgbd":
        ply_path = os.path.join(path, "scannet_rgbd_init.ply")
        if not os.path.exists(ply_path):
            print(
                f"Generating ScanNet RGB-D init point cloud "
                f"(stride={depth_stride}, frames={init_frames}, max={max_init_points})..."
            )
            pcd = _load_scannet_rgbd_pointcloud(
                path=path,
                train_cam_infos=train_cam_infos,
                depth_stride=depth_stride,
                init_frames=init_frames,
                max_points=max_init_points,
                depth_scale=depth_scale,
            )
            storePly(ply_path, pcd.points, np.clip(pcd.colors * 255.0, 0, 255))
    elif init_type == "random":
        ply_path = os.path.join(path, "scannet_random.ply")
        print(f"Generating random ScanNet point cloud ({num_pts})...")
        xyz = (
            np.random.random((num_pts, 3))
            * nerf_normalization["radius"] * 3 * 2
            - (nerf_normalization["radius"] * 3)
        )
        shs = np.random.random((num_pts, 3)) / 255.0
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        raise ValueError("ScanNet init_type must be one of: rgbd, mesh, random")

    pcd = fetchPlyFlexible(ply_path)

    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )

