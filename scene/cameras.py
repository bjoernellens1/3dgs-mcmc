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

import math
import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


def prepare_camera_for_render(cam, device="cuda"):
    """
    Pre-compute and cache gsplat-compatible tensors on the camera object.

    This avoids per-render DeviceCopy / tensor-construction overhead that
    triggers torch._inductor warnings and costs host→device transfer time.
    Call once per camera after scene load (before training starts).
    """
    if hasattr(cam, "_gsplat_ready") and cam._gsplat_ready:
        return

    # View matrix: Inria stores transpose of W2C; gsplat expects actual W2C.
    cam._gsplat_viewmat = (
        cam.world_view_transform
        .transpose(0, 1)
        .to(device=device, dtype=torch.float32)
        .contiguous()
    )

    # Intrinsics matrix K
    W = int(cam.image_width)
    H = int(cam.image_height)
    fx = W / (2.0 * math.tan(cam.FoVx / 2.0))
    fy = H / (2.0 * math.tan(cam.FoVy / 2.0))
    cx = W / 2.0
    cy = H / 2.0

    cam._gsplat_K = torch.empty((3, 3), device=device, dtype=torch.float32)
    cam._gsplat_K.zero_()
    cam._gsplat_K[0, 0] = fx
    cam._gsplat_K[1, 1] = fy
    cam._gsplat_K[0, 2] = cx
    cam._gsplat_K[1, 2] = cy
    cam._gsplat_K[2, 2] = 1.0
    cam._gsplat_K = cam._gsplat_K.contiguous()

    # Camera center (float32, same dtype as means)
    cam._gsplat_camera_center = (
        cam.camera_center
        .to(device=device, dtype=torch.float32)
        .view(1, 3)
        .contiguous()
    )

    # Ensure original image is on GPU (noop if already there)
    if hasattr(cam, "original_image") and cam.original_image.device.type != device:
        cam.original_image = cam.original_image.to(device=device).contiguous()

    cam._gsplat_ready = True

