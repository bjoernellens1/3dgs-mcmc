#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import glob
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fx: float = None
    fy: float = None
    cx: float = None
    cy: float = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            cx = intr.params[1]
            cy = intr.params[2]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            fx = focal_length_x
            fy = focal_length_x
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            cx = intr.params[2]
            cy = intr.params[3]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            fx = focal_length_x
            fy = focal_length_y
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              fx=fx, fy=fy, cx=cx, cy=cy)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def fetchPlyFlexible(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    names = vertices.data.dtype.names

    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float32)

    if all(c in names for c in ("red", "green", "blue")):
        colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T.astype(np.float32) / 255.0
    else:
        colors = np.full_like(positions, 0.5, dtype=np.float32)

    if all(n in names for n in ("nx", "ny", "nz")):
        normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T.astype(np.float32)
    else:
        normals = np.zeros_like(positions, dtype=np.float32)

    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, init_type="sfm", num_pts=100000):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if init_type == "sfm":
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
    elif init_type == "random":
        ply_path = os.path.join(path, "random.ply")
        print(f"Generating random point cloud ({num_pts})...")
        
        xyz = np.random.random((num_pts, 3)) * nerf_normalization["radius"]* 3*2 -(nerf_normalization["radius"]*3)
        
        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        print("Please specify a correct init_type: random or sfm")
        exit(0)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

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
        ply_path = os.path.join(path, "tum_rgbd_init.ply")
        if not os.path.exists(ply_path):
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
            storePly(ply_path, pcd.points, np.clip(pcd.colors * 255.0, 0, 255))
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

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "ScanNet": readScanNetSceneInfo,
    "TUM": readTUMSceneInfo,
}
