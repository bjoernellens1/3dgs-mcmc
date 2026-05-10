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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.gsplat_model import GsplatGaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, init_type=args.init_type)
        elif (
            os.path.exists(os.path.join(args.source_path, "rgb.txt"))
            and os.path.exists(os.path.join(args.source_path, "depth.txt"))
            and os.path.exists(os.path.join(args.source_path, "groundtruth.txt"))
            and os.path.isdir(os.path.join(args.source_path, "rgb"))
            and os.path.isdir(os.path.join(args.source_path, "depth"))
        ):
            print("Found TUM RGB-D dataset layout.")
            scene_info = sceneLoadTypeCallbacks["TUM"](
                args.source_path,
                args.eval,
                frame_stride=args.tum_frame_stride,
                max_frames=args.tum_max_frames,
                eval_hold=args.tum_eval_hold,
                init_type=args.tum_init,
                depth_stride=args.tum_depth_stride,
                init_frames=args.tum_init_frames,
                max_init_points=args.tum_max_init_points,
                depth_scale=args.tum_depth_scale,
                association_max_dt=args.tum_association_max_dt,
                sequence=args.tum_sequence,
                num_pts=args.tum_random_num_pts,
            )
        elif (
            os.path.exists(os.path.join(args.source_path, "frames.jsonl"))
            and os.path.exists(os.path.join(args.source_path, "intrinsics.json"))
        ):
            print("Found generic RGB-D sequence layout.")
            scene_info = sceneLoadTypeCallbacks["RGBDSequence"](
                args.source_path,
                args.eval,
                eval_hold=args.rgbd_eval_hold,
                init_type=args.init_type,
                depth_stride=args.rgbd_depth_stride,
                init_frames=args.rgbd_init_frames,
                max_init_points=args.rgbd_max_init_points,
                min_depth=args.rgbd_min_depth,
                max_depth=args.rgbd_max_depth,
                num_pts=args.rgbd_random_num_pts,
            )
        elif (
            os.path.exists(os.path.join(args.source_path, "color"))
            and os.path.exists(os.path.join(args.source_path, "pose"))
            and os.path.exists(os.path.join(args.source_path, "intrinsic"))
        ):
            print("Found ScanNet RGB-D scene layout.")
            scene_info = sceneLoadTypeCallbacks["ScanNet"](
                args.source_path,
                args.eval,
                frame_stride=args.scannet_frame_stride,
                max_frames=args.scannet_max_frames,
                eval_hold=args.scannet_eval_hold,
                init_type=args.scannet_init,
                depth_stride=args.scannet_depth_stride,
                init_frames=args.scannet_init_frames,
                max_init_points=args.scannet_max_init_points,
                depth_scale=args.scannet_depth_scale,
                num_pts=args.scannet_random_num_pts,
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
