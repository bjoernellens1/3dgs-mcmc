import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

from scene.readers.common import BasicPointCloud, CameraInfo, SceneInfo, fetchPlyFlexible, getNerfppNorm, storePly
from utils.graphics_utils import focal2fov
from utils.pointcloud_preprocess import (
    pointcloud_cache_suffix,
    pointcloud_preprocess_config,
    preprocess_pointcloud,
)
from utils.sh_utils import SH2RGB


def _sample_pointcloud(pcd, max_points, seed=42):
    if not max_points or max_points <= 0 or pcd.points.shape[0] <= max_points:
        return pcd

    rng = np.random.default_rng(seed)
    idx = rng.choice(pcd.points.shape[0], size=max_points, replace=False)
    return BasicPointCloud(
        points=pcd.points[idx].astype(np.float32),
        colors=pcd.colors[idx].astype(np.float32),
        normals=pcd.normals[idx].astype(np.float32),
    )


def _look_at_c2w(eye, target, up=np.array([0.0, 0.0, 1.0], dtype=np.float32)):
    eye = eye.astype(np.float32)
    target = target.astype(np.float32)
    forward = target - eye
    forward /= max(float(np.linalg.norm(forward)), 1e-8)

    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right /= max(float(np.linalg.norm(right)), 1e-8)

    down = np.cross(forward, right)
    down /= max(float(np.linalg.norm(down)), 1e-8)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w


def _replica_camera_path(points, num_views):
    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    center = (xyz_min + xyz_max) * 0.5
    extent = np.maximum(xyz_max - xyz_min, 1e-3)

    rx = max(float(extent[0]) * 0.28, 0.35)
    ry = max(float(extent[1]) * 0.28, 0.35)
    z = float(np.percentile(points[:, 2], 55.0))
    target_z = float(np.percentile(points[:, 2], 50.0))

    c2ws = []
    for i in range(num_views):
        angle = 2.0 * math.pi * i / max(num_views, 1)
        eye = np.array([
            center[0] + math.cos(angle) * rx,
            center[1] + math.sin(angle) * ry,
            z,
        ], dtype=np.float32)
        target = np.array([
            center[0] - math.cos(angle) * rx * 0.35,
            center[1] - math.sin(angle) * ry * 0.35,
            target_z,
        ], dtype=np.float32)
        c2ws.append(_look_at_c2w(eye, target))
    return c2ws


def _render_point_splat(points, colors, c2w, width, height, fx, fy, cx, cy, splat_radius=1):
    rot = c2w[:3, :3]
    eye = c2w[:3, 3]
    pts_cam = (points - eye) @ rot
    z = pts_cam[:, 2]
    valid = np.isfinite(z) & (z > 0.05)
    if not np.any(valid):
        return np.full((height, width, 3), 127, dtype=np.uint8)

    pts_cam = pts_cam[valid]
    rgb = colors[valid]
    z = pts_cam[:, 2]
    u = np.rint(fx * (pts_cam[:, 0] / z) + cx).astype(np.int32)
    v = np.rint(fy * (pts_cam[:, 1] / z) + cy).astype(np.int32)
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(in_bounds):
        return np.full((height, width, 3), 127, dtype=np.uint8)

    u = u[in_bounds]
    v = v[in_bounds]
    z = z[in_bounds]
    rgb = np.clip(rgb[in_bounds], 0.0, 1.0)

    image = np.full((height, width, 3), 127, dtype=np.uint8)
    flat_rgb = image.reshape(-1, 3)

    radius = max(int(splat_radius), 0)
    offsets = [(0, 0)]
    if radius > 0:
        offsets = [
            (du, dv)
            for dv in range(-radius, radius + 1)
            for du in range(-radius, radius + 1)
        ]

    for du, dv in offsets:
        uu = u + du
        vv = v + dv
        keep = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
        if not np.any(keep):
            continue

        flat_idx = vv[keep] * width + uu[keep]
        zz = z[keep]
        cc = rgb[keep]

        order = np.lexsort((zz, flat_idx))
        flat_sorted = flat_idx[order]
        first = np.unique(flat_sorted, return_index=True)[1]
        chosen = order[first]
        flat_rgb[flat_idx[chosen]] = np.rint(cc[chosen] * 255.0).astype(np.uint8)

    return image


def _raycast_replica_frame(scene, mesh_t, c2w, width, height, fx, fy, cx, cy):
    """Render one RGB + depth frame via open3d mesh raycasting.

    Returns (rgb_jpeg_bytes, depth_png_bytes).
    Depth is z-component in metres, stored as uint16 mm PNG.
    Miss pixels: rgb=(0,0,0), depth=0.
    """
    import io as _io

    import numpy as np
    import open3d as o3d
    import open3d.t.geometry as o3tg
    from PIL import Image

    w2c = np.linalg.inv(c2w)
    K = o3d.core.Tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=o3d.core.Dtype.Float64,
    )
    E = o3d.core.Tensor(w2c.astype(np.float64), dtype=o3d.core.Dtype.Float64)

    rays = o3tg.RaycastingScene.create_rays_pinhole(K, E, width, height)
    result = scene.cast_rays(rays)

    t_hit = result["t_hit"].numpy()          # [H, W] float32
    prim_id = result["primitive_ids"].numpy()  # [H, W] uint32
    prim_uv = result["primitive_uvs"].numpy()  # [H, W, 2] float32

    hit = np.isfinite(t_hit)

    # ray-length → z-component
    us, vs = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    z_factor = 1.0 / np.sqrt(((us - cx) / fx) ** 2 + ((vs - cy) / fy) ** 2 + 1.0)
    z_depth = t_hit * z_factor
    z_depth[~hit] = 0.0

    # barycentric vertex-color interpolation
    vc = mesh_t.vertex["colors"].numpy()       # [N, 3] float32 in [0, 1]
    tri = mesh_t.triangle["indices"].numpy()   # [M, 3] int32

    rgb = np.zeros((height * width, 3), dtype=np.float32)
    flat_hit = hit.ravel()
    if flat_hit.any():
        pids = prim_id.ravel()[flat_hit]
        uvs = prim_uv.reshape(-1, 2)[flat_hit]
        f = tri[pids]
        u_b = uvs[:, 0:1]
        v_b = uvs[:, 1:2]
        rgb[flat_hit] = (1 - u_b - v_b) * vc[f[:, 0]] + u_b * vc[f[:, 1]] + v_b * vc[f[:, 2]]
    rgb = rgb.reshape(height, width, 3)

    buf_rgb = _io.BytesIO()
    Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(
        buf_rgb, format="JPEG", quality=92
    )

    depth_mm = np.round(z_depth * 1000.0).clip(0, 65535).astype(np.uint16)
    buf_dep = _io.BytesIO()
    Image.fromarray(depth_mm, mode="I;16").save(buf_dep, format="PNG")

    return buf_rgb.getvalue(), buf_dep.getvalue()


def _ensure_replica_views(path, pcd, c2ws, width, height, fov_degrees, render_points, splat_radius):
    view_dir = Path(path) / (
        f"replica_views_w{width}_h{height}_n{len(c2ws)}_fov{int(round(fov_degrees))}_r{splat_radius}"
    )
    view_dir.mkdir(parents=True, exist_ok=True)

    fx = width / (2.0 * math.tan(math.radians(fov_degrees) * 0.5))
    fy = fx
    cx = width * 0.5
    cy = height * 0.5

    render_pcd = _sample_pointcloud(pcd, render_points, seed=7)
    paths = []
    for idx, c2w in enumerate(c2ws):
        image_path = view_dir / f"{idx:06d}.png"
        if not image_path.exists():
            image = _render_point_splat(
                render_pcd.points,
                render_pcd.colors,
                c2w,
                width,
                height,
                fx,
                fy,
                cx,
                cy,
                splat_radius=splat_radius,
            )
            Image.fromarray(image, mode="RGB").save(image_path)
        paths.append(str(image_path))
    return paths, fx, fy, cx, cy


def readReplicaSceneInfo(
    path,
    eval,
    init_type="mesh",
    num_views=120,
    width=640,
    height=480,
    eval_hold=8,
    fov_degrees=70.0,
    max_init_points=250000,
    render_points=300000,
    splat_radius=1,
    num_pts=250000,
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
    mesh_path = os.path.join(path, "mesh.ply")
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Missing Replica mesh: {mesh_path}")

    mesh_pcd = fetchPlyFlexible(mesh_path)
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

    if init_type == "mesh":
        pcd = _sample_pointcloud(mesh_pcd, max_init_points, seed=42)
        ply_path = os.path.join(
            path,
            f"replica_mesh_init_{pcd.points.shape[0]}{pointcloud_cache_suffix(preprocess_cfg)}.ply",
        )
        if preprocess_cfg.force_regenerate or not os.path.exists(ply_path):
            points, colors, normals, _ = preprocess_pointcloud(
                pcd.points,
                pcd.colors,
                pcd.normals,
                config=preprocess_cfg,
                label="replica_mesh_init",
            )
            storePly(ply_path, points, np.clip(colors * 255.0, 0, 255), normals=normals)
        pcd = fetchPlyFlexible(ply_path)
    elif init_type == "random":
        pcd_for_norm = _sample_pointcloud(mesh_pcd, max_init_points, seed=42)
        xyz_min = pcd_for_norm.points.min(axis=0)
        xyz_max = pcd_for_norm.points.max(axis=0)
        center = (xyz_min + xyz_max) * 0.5
        extent = max(float(np.max(xyz_max - xyz_min)), 1.0)
        rng = np.random.default_rng(42)
        xyz = center + (rng.random((num_pts, 3), dtype=np.float32) - 0.5) * extent
        shs = rng.random((num_pts, 3), dtype=np.float32) / 255.0
        ply_path = os.path.join(path, f"replica_random_{num_pts}.ply")
        if not os.path.exists(ply_path):
            storePly(ply_path, xyz, SH2RGB(shs) * 255.0)
        pcd = fetchPlyFlexible(ply_path)
    else:
        raise ValueError("Replica init_type must be one of: mesh, random")

    c2ws = _replica_camera_path(mesh_pcd.points, int(num_views))
    image_paths, fx, fy, cx, cy = _ensure_replica_views(
        path,
        mesh_pcd,
        c2ws,
        int(width),
        int(height),
        float(fov_degrees),
        int(render_points),
        int(splat_radius),
    )

    fov_x = focal2fov(fx, width)
    fov_y = focal2fov(fy, height)
    cam_infos = []
    for idx, (image_path, c2w) in enumerate(zip(image_paths, c2ws)):
        w2c = np.linalg.inv(c2w)
        cam_infos.append(
            CameraInfo(
                uid=idx,
                R=w2c[:3, :3].T,
                T=w2c[:3, 3],
                FovY=fov_y,
                FovX=fov_x,
                image=Image.open(image_path).convert("RGB"),
                image_path=image_path,
                image_name=f"replica_{idx:06d}",
                width=int(width),
                height=int(height),
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
            )
        )

    print(
        f"Loaded Replica mesh scene: cameras={len(cam_infos)} "
        f"init_points={pcd.points.shape[0]} mesh_vertices={mesh_pcd.points.shape[0]}"
    )

    if eval and eval_hold > 0:
        train_cam_infos = [c for i, c in enumerate(cam_infos) if i % eval_hold != 0]
        test_cam_infos = [c for i, c in enumerate(cam_infos) if i % eval_hold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
