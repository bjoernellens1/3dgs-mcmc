import os

import numpy as np
from PIL import Image

from scene.readers.common import CameraInfo, SceneInfo, fetchPlyFlexible, getNerfppNorm, storePly
from utils.graphics_utils import focal2fov
from utils.pointcloud_preprocess import (
    pointcloud_cache_suffix,
    pointcloud_preprocess_config,
    preprocess_pointcloud,
)
from utils.rgbd_frames import c2w_to_camera_rt, read_frame_records, read_intrinsics, read_metadata, rgbd_pointcloud_from_records
from utils.sh_utils import SH2RGB

def readRGBDSequenceSceneInfo(
    path,
    eval,
    eval_hold=8,
    init_type="rgbd",
    depth_stride=4,
    init_frames=300,
    max_init_points=250000,
    min_depth=0.1,
    max_depth=8.0,
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
    intr = read_intrinsics(os.path.join(path, "intrinsics.json"))
    records = read_frame_records(os.path.join(path, "frames.jsonl"))
    if not os.path.exists(os.path.join(path, "metadata.json")):
        print("[RGBDSequence] Warning: metadata.json missing; sequence may be incomplete.")
    if len(records) == 0:
        raise RuntimeError(f"No RGB-D frames found in {path}")

    cam_infos = []
    valid_records = []
    skipped = 0
    for record in records:
        image_path = os.path.join(path, record["rgb"])
        if not os.path.exists(image_path):
            skipped += 1
            continue
        try:
            R, T = c2w_to_camera_rt(record["c2w"])
        except ValueError:
            skipped += 1
            continue

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        FovX = focal2fov(intr.fx, width)
        FovY = focal2fov(intr.fy, height)
        image_name = f"{int(record.get('id', len(cam_infos))):06d}"

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
                fx=intr.fx,
                fy=intr.fy,
                cx=intr.cx,
                cy=intr.cy,
            )
        )
        valid_records.append(record)

    print(f"Loaded RGB-D sequence cameras: {len(cam_infos)} skipped={skipped}")
    if len(cam_infos) == 0:
        raise RuntimeError(f"No valid RGB-D sequence cameras found in {path}")

    if eval and eval_hold > 0:
        train_cam_infos = [c for i, c in enumerate(cam_infos) if i % eval_hold != 0]
        test_cam_infos = [c for i, c in enumerate(cam_infos) if i % eval_hold == 0]
        train_records = [r for i, r in enumerate(valid_records) if i % eval_hold != 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
        train_records = valid_records

    nerf_normalization = getNerfppNorm(train_cam_infos)
    init_type = init_type.lower()
    if init_type == "sfm":
        init_type = "rgbd"

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
        ply_path = os.path.join(path, f"init_rgbd{pointcloud_cache_suffix(preprocess_cfg)}.ply")
        if preprocess_cfg.force_regenerate or not os.path.exists(ply_path):
            print(
                f"Generating RGB-D init point cloud "
                f"(stride={depth_stride}, frames={init_frames}, max={max_init_points})..."
            )
            points, colors, normals = rgbd_pointcloud_from_records(
                root=path,
                records=train_records,
                intrinsics=intr,
                depth_stride=depth_stride,
                init_frames=init_frames,
                max_points=max_init_points,
                min_depth=min_depth,
                max_depth=max_depth,
            )
            points, colors, normals, _ = preprocess_pointcloud(
                points,
                colors,
                normals,
                config=preprocess_cfg,
                label="rgbd_sequence_init",
            )
            storePly(ply_path, points, np.clip(colors * 255.0, 0, 255), normals=normals)
    elif init_type == "random":
        ply_path = os.path.join(path, "rgbd_random.ply")
        print(f"Generating random RGB-D sequence point cloud ({num_pts})...")
        xyz = (
            np.random.random((num_pts, 3))
            * nerf_normalization["radius"] * 3 * 2
            - (nerf_normalization["radius"] * 3)
        )
        shs = np.random.random((num_pts, 3)) / 255.0
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        raise ValueError("RGBDSequence init_type must be one of: rgbd, random")

    pcd = fetchPlyFlexible(ply_path)
    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
