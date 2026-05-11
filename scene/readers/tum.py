import os
from pathlib import Path

import numpy as np
from PIL import Image

from scene.readers.common import BasicPointCloud, CameraInfo, SceneInfo, fetchPlyFlexible, getNerfppNorm, storePly
from utils.graphics_utils import getWorld2View2, focal2fov
from utils.pointcloud_preprocess import (
    pointcloud_cache_suffix,
    pointcloud_preprocess_config,
    preprocess_pointcloud,
)
from utils.sh_utils import SH2RGB

def _tum_default_intrinsics(path, sequence=""):
    name = (sequence or Path(path).name).lower()

    # Common TUM RGB-D intrinsics for the 640x480 RGB stream.
    if "freiburg1" in name or "fr1" in name:
        return 517.3, 516.5, 318.6, 255.3
    if "freiburg2" in name or "fr2" in name:
        return 520.9, 521.0, 325.1, 249.7
    if "freiburg3" in name or "fr3" in name:
        return 535.4, 539.2, 320.1, 247.6

    print(
        "[TUM] Could not infer freiburg1/2/3 from path. "
        "Falling back to Freiburg1 intrinsics. "
        "Pass --tum_sequence freiburg2 or freiburg3 if needed."
    )
    return 517.3, 516.5, 318.6, 255.3

def _read_tum_list(path):
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            entries.append((float(parts[0]), parts[1:]))
    return entries

def _read_tum_poses(path):
    poses = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                continue

            t = float(parts[0])
            tvec = np.array([float(v) for v in parts[1:4]], dtype=np.float32)
            qxyzw = np.array([float(v) for v in parts[4:8]], dtype=np.float32)
            poses.append((t, tvec, qxyzw))
    return poses

def _quat_xyzw_to_rotmat(q):
    qx, qy, qz, qw = q
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)

    s = 2.0 / n
    x, y, z, w = qx, qy, qz, qw

    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ], dtype=np.float32)

def _tum_pose_to_c2w(tvec, qxyzw):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = _quat_xyzw_to_rotmat(qxyzw)
    c2w[:3, 3] = tvec
    return c2w

def _associate_tum(rgb_entries, depth_entries, pose_entries, max_dt=0.03):
    depth_times = np.array([t for t, _ in depth_entries], dtype=np.float64)
    pose_times = np.array([t for t, _, _ in pose_entries], dtype=np.float64)

    associations = []

    for rgb_t, rgb_data in rgb_entries:
        if len(depth_times) == 0 or len(pose_times) == 0:
            break

        d_idx = int(np.argmin(np.abs(depth_times - rgb_t)))
        p_idx = int(np.argmin(np.abs(pose_times - rgb_t)))

        if abs(depth_times[d_idx] - rgb_t) > max_dt:
            continue
        if abs(pose_times[p_idx] - rgb_t) > max_dt:
            continue

        depth_t, depth_data = depth_entries[d_idx]
        pose_t, tvec, qxyzw = pose_entries[p_idx]

        associations.append({
            "rgb_t": rgb_t,
            "rgb_rel": rgb_data[0],
            "depth_t": depth_t,
            "depth_rel": depth_data[0],
            "pose_t": pose_t,
            "tvec": tvec,
            "qxyzw": qxyzw,
        })

    return associations

def readTUMCameras(
    path,
    frame_stride=1,
    max_frames=0,
    eval=False,
    eval_hold=8,
    association_max_dt=0.03,
    sequence="",
):
    rgb_path = os.path.join(path, "rgb.txt")
    depth_path = os.path.join(path, "depth.txt")
    gt_path = os.path.join(path, "groundtruth.txt")

    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"Missing TUM rgb.txt: {rgb_path}")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Missing TUM depth.txt: {depth_path}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Missing TUM groundtruth.txt: {gt_path}")

    rgb_entries = _read_tum_list(rgb_path)
    depth_entries = _read_tum_list(depth_path)
    pose_entries = _read_tum_poses(gt_path)

    assocs = _associate_tum(
        rgb_entries=rgb_entries,
        depth_entries=depth_entries,
        pose_entries=pose_entries,
        max_dt=association_max_dt,
    )

    if frame_stride > 1:
        assocs = assocs[::frame_stride]
    if max_frames and max_frames > 0:
        assocs = assocs[:max_frames]

    fx, fy, cx, cy = _tum_default_intrinsics(path, sequence=sequence)

    cam_infos = []
    skipped = 0

    for assoc in assocs:
        image_path = os.path.join(path, assoc["rgb_rel"])
        if not os.path.exists(image_path):
            skipped += 1
            continue

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        c2w = _tum_pose_to_c2w(assoc["tvec"], assoc["qxyzw"])
        if not np.isfinite(c2w).all():
            skipped += 1
            continue

        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3].T
        T = w2c[:3, 3]

        FovX = focal2fov(fx, width)
        FovY = focal2fov(fy, height)
        image_name = f"{assoc['rgb_t']:.6f}".replace(".", "_")

        cam_infos.append(
            CameraInfo(
                uid=len(cam_infos),
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_path=image_path,
                image_name=image_name,
                width=width,
                height=height,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
            )
        )

    print(f"Loaded TUM cameras: {len(cam_infos)} skipped={skipped}")

    if eval and eval_hold > 0:
        train_cam_infos = [c for i, c in enumerate(cam_infos) if i % eval_hold != 0]
        test_cam_infos = [c for i, c in enumerate(cam_infos) if i % eval_hold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    return train_cam_infos, test_cam_infos, assocs

def _load_tum_rgbd_pointcloud(
    path,
    train_cam_infos,
    associations,
    depth_stride=4,
    init_frames=300,
    max_points=250000,
    depth_scale=5000.0,
    sequence="",
):
    fx, fy, cx, cy = _tum_default_intrinsics(path, sequence=sequence)

    selected_cams = train_cam_infos
    if init_frames and init_frames > 0:
        selected_cams = selected_cams[:init_frames]

    assoc_by_name = {
        f"{a['rgb_t']:.6f}".replace(".", "_"): a
        for a in associations
    }

    points_all = []
    colors_all = []

    for cam in selected_cams:
        assoc = assoc_by_name.get(cam.image_name)
        if assoc is None:
            continue

        depth_file = os.path.join(path, assoc["depth_rel"])
        if not os.path.exists(depth_file):
            continue

        depth = np.array(Image.open(depth_file)).astype(np.float32) / float(depth_scale)
        rgb = np.array(cam.image.convert("RGB")).astype(np.float32) / 255.0

        h, w = depth.shape
        ys, xs = np.mgrid[0:h:depth_stride, 0:w:depth_stride]
        z = depth[ys, xs]

        valid = np.isfinite(z) & (z > 0.1) & (z < 8.0)
        if valid.sum() == 0:
            continue

        xs_v = xs[valid].astype(np.float32)
        ys_v = ys[valid].astype(np.float32)
        z_v = z[valid].astype(np.float32)

        x = (xs_v - cx) / fx * z_v
        y = (ys_v - cy) / fy * z_v
        pts_cam = np.stack([x, y, z_v], axis=1)

        w2c = getWorld2View2(cam.R, cam.T)
        c2w = np.linalg.inv(w2c)
        pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]

        rgb_h, rgb_w = rgb.shape[:2]
        rgb_x = np.clip(
            (xs_v / max(w - 1, 1) * (rgb_w - 1)).round().astype(np.int32),
            0,
            rgb_w - 1,
        )
        rgb_y = np.clip(
            (ys_v / max(h - 1, 1) * (rgb_h - 1)).round().astype(np.int32),
            0,
            rgb_h - 1,
        )
        rgb_v = rgb[rgb_y, rgb_x]

        points_all.append(pts_world.astype(np.float32))
        colors_all.append(rgb_v.astype(np.float32))

    if not points_all:
        raise RuntimeError("Could not initialize TUM point cloud from RGB-D frames.")

    points = np.concatenate(points_all, axis=0)
    colors = np.concatenate(colors_all, axis=0)

    if max_points and points.shape[0] > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    normals = np.zeros_like(points, dtype=np.float32)
    return BasicPointCloud(points=points, colors=colors, normals=normals)

def readTUMSceneInfo(
    path,
    eval,
    frame_stride=1,
    max_frames=0,
    eval_hold=8,
    init_type="rgbd",
    depth_stride=4,
    init_frames=300,
    max_init_points=250000,
    depth_scale=5000.0,
    association_max_dt=0.03,
    sequence="",
    num_pts=100000,
    pointcloud_preprocess="none",
    pcd_voxel_size=0.0,
    pcd_outlier_filter="none",
    pcd_stat_nb_neighbors=20,
    pcd_stat_std_ratio=2.0,
    pcd_radius=0.05,
    pcd_min_neighbors=4,
    pcd_estimate_normals=False,
    pcd_force_regenerate=False,
):
    train_cam_infos, test_cam_infos, associations = readTUMCameras(
        path=path,
        frame_stride=frame_stride,
        max_frames=max_frames,
        eval=eval,
        eval_hold=eval_hold,
        association_max_dt=association_max_dt,
        sequence=sequence,
    )

    if len(train_cam_infos) == 0:
        raise RuntimeError(f"No valid TUM cameras found in {path}")

    nerf_normalization = getNerfppNorm(train_cam_infos)
    init_type = init_type.lower()

    if init_type == "rgbd":
        preprocess_cfg = pointcloud_preprocess_config(
            pointcloud_preprocess=pointcloud_preprocess,
            pcd_voxel_size=pcd_voxel_size,
            pcd_outlier_filter=pcd_outlier_filter,
            pcd_stat_nb_neighbors=pcd_stat_nb_neighbors,
            pcd_stat_std_ratio=pcd_stat_std_ratio,
            pcd_radius=pcd_radius,
            pcd_min_neighbors=pcd_min_neighbors,
            pcd_estimate_normals=pcd_estimate_normals,
            pcd_force_regenerate=pcd_force_regenerate,
        )
        ply_path = os.path.join(path, f"tum_rgbd_init{pointcloud_cache_suffix(preprocess_cfg)}.ply")
        if preprocess_cfg.force_regenerate or not os.path.exists(ply_path):
            print(
                f"Generating TUM RGB-D init point cloud "
                f"(stride={depth_stride}, frames={init_frames}, max={max_init_points})..."
            )
            pcd = _load_tum_rgbd_pointcloud(
                path=path,
                train_cam_infos=train_cam_infos,
                associations=associations,
                depth_stride=depth_stride,
                init_frames=init_frames,
                max_points=max_init_points,
                depth_scale=depth_scale,
                sequence=sequence,
            )
            points, colors, normals, _ = preprocess_pointcloud(
                pcd.points,
                pcd.colors,
                pcd.normals,
                config=preprocess_cfg,
                label="tum_rgbd_init",
            )
            storePly(ply_path, points, np.clip(colors * 255.0, 0, 255), normals=normals)
    elif init_type == "random":
        ply_path = os.path.join(path, "tum_random.ply")
        print(f"Generating random TUM point cloud ({num_pts})...")
        xyz = (
            np.random.random((num_pts, 3))
            * nerf_normalization["radius"] * 3 * 2
            - (nerf_normalization["radius"] * 3)
        )
        shs = np.random.random((num_pts, 3)) / 255.0
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        raise ValueError("TUM init_type must be one of: rgbd, random")

    pcd = fetchPlyFlexible(ply_path)

    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
