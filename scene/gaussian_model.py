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

import torch
import numpy as np
import math
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
# DEBUG: force Python fallback for ROCm stability
from utils.rocm_knn_fallback import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.reloc_utils import compute_relocation_cuda


def _dense_grad(grad):
    if grad is None:
        return None
    if getattr(grad, "layout", torch.strided) != torch.strided:
        return grad.to_dense()
    return grad


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.optimizer_type = "adam"
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        # Streaming state
        self.birth_frame = torch.empty(0, dtype=torch.int32, device="cuda")
        self.support_count = torch.empty(0, dtype=torch.int16, device="cuda")
        self.provisional = torch.empty(0, dtype=torch.bool, device="cuda")
        self.anchor_xyz = torch.empty((0, 3), dtype=torch.float32, device="cuda")
        self.anchor_iter = torch.empty(0, dtype=torch.int32, device="cuda")
        self.depth_conflict_count = torch.empty(0, dtype=torch.int16, device="cuda")
        self.depth_last_support_uid = torch.empty(0, dtype=torch.int32, device="cuda")
        self.depth_last_conflict_uid = torch.empty(0, dtype=torch.int32, device="cuda")
        self.setup_functions()

    _STREAMING_BUFFER_NAMES: tuple = (
        "birth_frame", "support_count", "provisional", "anchor_iter",
        "depth_conflict_count", "depth_last_support_uid", "depth_last_conflict_uid",
        "lifecycle_state", "utility_ema", "anchor_scale_log", "anchor_opacity_logit",
        "anchor_xyz",
    )

    def capture(self):
        streaming_state = {}
        for name in self._STREAMING_BUFFER_NAMES:
            buf = getattr(self, name, None)
            if buf is not None:
                streaming_state[name] = buf.detach().cpu()
        return {
            "layout": "legacy",
            "active_sh_degree": self.active_sh_degree,
            "xyz": self._xyz,
            "features_dc": self._features_dc,
            "features_rest": self._features_rest,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "opacity": self._opacity,
            "max_radii2D": self.max_radii2D,
            "xyz_gradient_accum": self.xyz_gradient_accum,
            "denom": self.denom,
            "optimizer": self.optimizer.state_dict(),
            "spatial_lr_scale": self.spatial_lr_scale,
            "streaming_state": streaming_state,
        }

    def restore(self, model_args, training_args):
        if isinstance(model_args, tuple):
            # Backward-compat: old tuple-format checkpoint
            (self.active_sh_degree,
             self._xyz, self._features_dc, self._features_rest,
             self._scaling, self._rotation, self._opacity,
             self.max_radii2D, xyz_gradient_accum, denom,
             opt_dict, self.spatial_lr_scale,
             *_legacy_extras) = model_args
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)
            if len(_legacy_extras) >= 3:
                self.birth_frame, self.support_count, self.provisional = _legacy_extras[:3]
            return
        self.active_sh_degree = model_args["active_sh_degree"]
        self._xyz = model_args["xyz"]
        self._features_dc = model_args["features_dc"]
        self._features_rest = model_args["features_rest"]
        self._scaling = model_args["scaling"]
        self._rotation = model_args["rotation"]
        self._opacity = model_args["opacity"]
        self.max_radii2D = model_args["max_radii2D"]
        self.spatial_lr_scale = model_args["spatial_lr_scale"]
        self.training_setup(training_args)
        self.xyz_gradient_accum = model_args["xyz_gradient_accum"]
        self.denom = model_args["denom"]
        self.optimizer.load_state_dict(model_args["optimizer"])
        streaming_state = model_args.get("streaming_state", {})
        if streaming_state:
            for name, buf in streaming_state.items():
                setattr(self, name, buf.to(device="cuda"))
        elif any(hasattr(self, n) for n in self._STREAMING_BUFFER_NAMES):
            print(
                "[checkpoint] Warning: checkpoint has no streaming_state; "
                "lifecycle/anchor/support buffers reset to defaults.",
                flush=True,
            )

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    # ------------------------------------------------------------------
    # Layout-agnostic adapter properties — mirrors GsplatGaussianModel so
    # streaming code can work without branching on model_layout.
    # ------------------------------------------------------------------

    @property
    def raw_opacities(self):
        """Raw (pre-sigmoid) opacity logits, shape (N,)."""
        return self._opacity.squeeze(-1)

    @property
    def raw_scales(self):
        """Raw (log) scale parameters, shape (N, 3)."""
        return self._scaling

    @property
    def means_param(self):
        """The means / xyz position parameter."""
        return self._xyz

    @property
    def geometry_params(self):
        """Geometric learnable parameters [xyz, scaling, rotation]."""
        return [self._xyz, self._scaling, self._rotation]

    @property
    def all_learnable_params(self):
        """All learnable parameters as a list."""
        return [self._xyz, self._features_dc, self._features_rest,
                self._opacity, self._scaling, self._rotation]

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def as_gsplat_params(self, activated=True):
        """
        Return an upstream-style splat parameter mapping.

        activated=True is intended for rasterization/viewer code. activated=False
        exposes raw trainable tensors for checkpoint/export/strategy adapters.
        """
        if activated:
            scales = self.get_scaling
            quats = self.get_rotation
            opacities = self.get_opacity.squeeze(-1)
        else:
            scales = self._scaling
            quats = self._rotation
            opacities = self._opacity.squeeze(-1)

        return {
            "means": self._xyz,
            "scales": scales,
            "quats": quats,
            "opacities": opacities,
            "sh0": self._features_dc,
            "shN": self._features_rest,
        }
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, init_scale_mode="fixed", init_scale=0.01, voxel_size=0.02):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        N = fused_point_cloud.shape[0]
        if init_scale_mode == "fixed":
            dist2 = torch.full((N,), init_scale ** 2, device="cuda")
        elif init_scale_mode == "knn":
            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        elif init_scale_mode == "voxel":
            dist2 = torch.full((N,), (voxel_size * 0.5) ** 2, device="cuda")
        else:
            raise ValueError(f"Unknown init_scale_mode: {init_scale_mode}")
        scales = torch.log(torch.sqrt(dist2)*0.1)[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.visibility_ema = torch.zeros((fused_point_cloud.shape[0], 1), device="cuda")
        # Initialize streaming buffers for the initial point cloud
        count = self.get_xyz.shape[0]
        self.birth_frame = torch.zeros(count, dtype=torch.int32, device="cuda")
        self.support_count = torch.zeros(count, dtype=torch.int16, device="cuda")
        self.provisional = torch.zeros(count, dtype=torch.bool, device="cuda")
        self.anchor_xyz = self._xyz.detach().clone()
        self.anchor_iter = torch.zeros(count, dtype=torch.int32, device="cuda")
        self.depth_conflict_count = torch.zeros(count, dtype=torch.int16, device="cuda")
        self.depth_last_support_uid = torch.full((count,), -1, dtype=torch.int32, device="cuda")
        self.depth_last_conflict_uid = torch.full((count,), -1, dtype=torch.int32, device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer_type = getattr(training_args, "optimizer_type", "adam").lower()
        if self.optimizer_type in {"adam", "default"}:
            self.optimizer_type = "adam"
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "selective_adam":
            try:
                from gsplat.optimizers import SelectiveAdam
            except Exception as exc:
                raise RuntimeError(
                    "--optimizer_type selective_adam requires gsplat.optimizers.SelectiveAdam "
                    "inside the project container."
                ) from exc
            self.optimizer = SelectiveAdam(l, eps=1e-15, betas=(0.9, 0.999))
        else:
            raise ValueError(
                f"Unsupported optimizer_type '{self.optimizer_type}'. "
                "Expected 'adam' or 'selective_adam'."
            )
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def normalize_rotation_params(self, mask=None):
        with torch.no_grad():
            if mask is None:
                self._rotation.copy_(torch.nn.functional.normalize(self._rotation, dim=1, eps=1e-8))
            else:
                self._rotation[mask] = torch.nn.functional.normalize(
                    self._rotation[mask], dim=1, eps=1e-8
                )

    def prepare_selective_adam_step(self, allow_dense_grads=False):
        """
        Prepare gradients and state tensors for SelectiveAdam step.

        IMPORTANT — sparse tensor semantics:
            gsplat's ``sparse_grad=True`` returns sparse COO gradients for
            visible Gaussians. However, SelectiveAdam's fused CUDA ``adam()``
            kernel requires strided (dense) gradient tensors — it accesses
            per-row gradients via direct pointer arithmetic. Therefore this
            method **always** converts sparse COO grads to dense strided
            before the optimizer step.

        This means the training profile is:
            **active-set / visible-row selective updates**
        NOT:
            fully end-to-end sparse tensor training

        The active-set benefit still comes from:
        1. Sparse rasterizer backward  → fewer grad entries computed
        2. Active-set regularizers     → O(visible) compute
        3. Active-set energy losses    → O(visible) compute (or periodic global)
        4. SelectiveAdam step          → only visible rows updated
        5. Post-backward masking       → invisible rows explicitly zeroed

        Args:
            allow_dense_grads: kept for API compatibility; dense conversion
                is always required by the CUDA kernel regardless of this flag.
        """
        for group in self.optimizer.param_groups:
            param = group["params"][0]
            if not param.is_contiguous():
                param.data = param.data.contiguous()
            if param.grad is not None:
                g_layout = getattr(param.grad, "layout", torch.strided)
                if g_layout != torch.strided:
                    # SelectiveAdam CUDA kernel requires strided (dense) grads.
                    # Always convert — the post-backward masking in train.py
                    # already zeros invisible rows in strided grads, and
                    # sparse→dense naturally produces zeros for invisible rows.
                    param.grad = param.grad.to_dense().contiguous()
                elif not param.grad.is_contiguous():
                    param.grad = param.grad.contiguous()
            stored_state = self.optimizer.state.get(param, None)
            if stored_state is not None:
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in stored_state and not stored_state[key].is_contiguous():
                        stored_state[key] = stored_state[key].contiguous()

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        self.visibility_ema = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.visibility_ema = self.visibility_ema[valid_points_mask]
        self.birth_frame = self.birth_frame[valid_points_mask]
        self.support_count = self.support_count[valid_points_mask]
        self.provisional = self.provisional[valid_points_mask]
        self.anchor_xyz = self.anchor_xyz[valid_points_mask]
        self.anchor_iter = self.anchor_iter[valid_points_mask]
        if self.depth_conflict_count.shape[0] == mask.shape[0]:
            self.depth_conflict_count = self.depth_conflict_count[valid_points_mask]
        if self.depth_last_support_uid.shape[0] == mask.shape[0]:
            self.depth_last_support_uid = self.depth_last_support_uid[valid_points_mask]
        if self.depth_last_conflict_uid.shape[0] == mask.shape[0]:
            self.depth_last_conflict_uid = self.depth_last_conflict_uid[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, reset_params=True, birth_frame=0, is_provisional=False):
        old_count = self.get_xyz.shape[0]
        old_visibility_ema = getattr(self, "visibility_ema", None)
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if reset_params:
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
            self.visibility_ema = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        else:
            new_count = self.get_xyz.shape[0] - old_count
            if (
                old_visibility_ema is not None
                and old_visibility_ema.shape[0] == old_count
            ):
                self.visibility_ema = torch.cat(
                    (
                        old_visibility_ema,
                        torch.zeros(
                            (new_count, 1),
                            device=old_visibility_ema.device,
                            dtype=old_visibility_ema.dtype,
                        ),
                    ),
                    dim=0,
                )
            else:
                self.visibility_ema = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # Handle streaming buffers
        N_new = new_xyz.shape[0]
        new_birth = torch.full((N_new,), int(birth_frame), dtype=torch.int32, device="cuda")
        new_support = torch.zeros(N_new, dtype=torch.int16, device="cuda")
        new_provisional = torch.full((N_new,), bool(is_provisional), dtype=torch.bool, device="cuda")
        
        # If the existing buffers are empty (e.g. first init), just set them
        if self.birth_frame.shape[0] == 0:
            self.birth_frame = new_birth
            self.support_count = new_support
            self.provisional = new_provisional
            self.anchor_xyz = new_xyz.detach().clone()
            self.anchor_iter = torch.zeros(N_new, dtype=torch.int32, device="cuda")
            self.depth_conflict_count = torch.zeros(N_new, dtype=torch.int16, device="cuda")
            self.depth_last_support_uid = torch.full((N_new,), -1, dtype=torch.int32, device="cuda")
            self.depth_last_conflict_uid = torch.full((N_new,), -1, dtype=torch.int32, device="cuda")
        else:
            self.birth_frame = torch.cat([self.birth_frame, new_birth], dim=0)
            self.support_count = torch.cat([self.support_count, new_support], dim=0)
            self.provisional = torch.cat([self.provisional, new_provisional], dim=0)
            self.anchor_xyz = torch.cat([self.anchor_xyz, new_xyz.detach().clone()], dim=0)
            self.anchor_iter = torch.cat([
                self.anchor_iter,
                torch.zeros(N_new, dtype=torch.int32, device="cuda"),
            ], dim=0)
            self.depth_conflict_count = torch.cat([
                self.depth_conflict_count,
                torch.zeros(N_new, dtype=torch.int16, device="cuda"),
            ], dim=0)
            self.depth_last_support_uid = torch.cat([
                self.depth_last_support_uid,
                torch.full((N_new,), -1, dtype=torch.int32, device="cuda"),
            ], dim=0)
            self.depth_last_conflict_uid = torch.cat([
                self.depth_last_conflict_uid,
                torch.full((N_new,), -1, dtype=torch.int32, device="cuda"),
            ], dim=0)

    def add_points_as_gaussians(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        init_scale: float = 0.01,
        init_opacity: float = 0.5,
        use_knn_scale: bool = False,
        normals: torch.Tensor = None,
        scales: torch.Tensor = None,
        rotations: torch.Tensor = None,
        is_provisional: bool = False,
        birth_frame: int = 0,
        opacities_raw: torch.Tensor = None,
        sh_rest: torch.Tensor = None,
    ) -> int:
        """
        Append new Gaussians initialised from 3-D world-space points and RGB
        colours (float32, range [0, 1]).  Extends the optimizer with
        zero-initialised momentum state for the new entries.

        opacities_raw: pre-logit opacity tensor shape (N,) or (N,1). When provided,
        bypasses init_opacity.
        sh_rest: higher-order SH coefficients shape (N, R-1, 3) in legacy transposed
        layout (N, num_sh-1, 3). When provided, bypasses zeros init.

        use_knn_scale: derive per-Gaussian scale from k-NN distance (matches
        bootstrap quality); falls back to fixed init_scale when False.

        Returns the number of Gaussians actually added.
        """
        N = points.shape[0]
        if N == 0:
            return 0

        from utils.sh_utils import RGB2SH
        fused_color = RGB2SH(colors.to(device="cuda", dtype=torch.float32))
        num_sh = (self.max_sh_degree + 1) ** 2
        features = torch.zeros(
            (N, 3, num_sh), device="cuda", dtype=torch.float32
        )
        features[:, :3, 0] = fused_color

        new_xyz = points.to(device="cuda", dtype=torch.float32)
        new_f_dc = features[:, :, 0:1].transpose(1, 2).contiguous()

        if sh_rest is not None:
            # sh_rest expected as (N, R-1, 3); convert to legacy (N, 3, R-1) then transpose
            sr = sh_rest.to(device="cuda", dtype=torch.float32)
            if sr.shape[1] != num_sh - 1:
                padded = torch.zeros((N, num_sh - 1, 3), device="cuda")
                copy_len = min(sr.shape[1], num_sh - 1)
                padded[:, :copy_len, :] = sr[:, :copy_len, :]
                sr = padded
            # Legacy layout: _features_rest is (N, R-1, 3) — match that directly
            new_f_rest = sr.contiguous()
        else:
            new_f_rest = features[:, :, 1:].transpose(1, 2).contiguous()

        if opacities_raw is not None:
            new_opacities = opacities_raw.to(device="cuda", dtype=torch.float32).reshape(N, 1)
        else:
            new_opacities = inverse_sigmoid(
                torch.full((N, 1), float(init_opacity), device="cuda", dtype=torch.float32)
            )

        if scales is not None:
            new_scaling = scales.to(device="cuda", dtype=torch.float32)
        elif use_knn_scale and N > 1:
            from utils.rocm_knn_fallback import distCUDA2
            dist_sq = distCUDA2(new_xyz)
            scales_1d = torch.clamp(dist_sq.sqrt() * 0.5, 1e-4, 0.1)
            new_scaling = torch.log(scales_1d).unsqueeze(-1).expand(-1, 3).contiguous()
        else:
            scale_val = math.log(math.sqrt(init_scale ** 2) * 0.1)
            new_scaling = torch.full((N, 3), scale_val, device="cuda", dtype=torch.float32)

        if rotations is not None:
            new_rotation = rotations.to(device="cuda", dtype=torch.float32)
        else:
            new_rotation = torch.zeros((N, 4), device="cuda", dtype=torch.float32)
            new_rotation[:, 0] = 1.0  # wxyz identity

        self.densification_postfix(
            new_xyz, new_f_dc, new_f_rest, new_opacities, new_scaling, new_rotation,
            reset_params=False, birth_frame=birth_frame, is_provisional=is_provisional,
        )
        return N

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = {"xyz": self._xyz,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "opacity": self._opacity,
            "scaling" : self._scaling,
            "rotation" : self._rotation}

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)

            if stored_state is None:
                stored_state = {}
                stored_state["step"] = torch.tensor(0, dtype=torch.float32)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
            elif inds is not None:
                stored_state["exp_avg"][inds] = 0
                stored_state["exp_avg_sq"][inds] = 0
            else:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

            if group['params'][0] in self.optimizer.state:
                del self.optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            self.optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"] 

        torch.cuda.empty_cache()
        
        return optimizable_tensors

    
    def _update_params(self, idxs, ratio):
        new_opacity, new_scaling = compute_relocation_cuda(
            opacity_old=self.get_opacity[idxs, 0],
            scale_old=self.get_scaling[idxs],
            N=ratio[idxs, 0] + 1
        )
        new_opacity = torch.clamp(new_opacity.unsqueeze(-1), max=1.0 - torch.finfo(torch.float32).eps, min=0.005)
        new_opacity = self.inverse_opacity_activation(new_opacity)
        new_scaling = self.scaling_inverse_activation(new_scaling.reshape(-1, 3))

        return self._xyz[idxs], self._features_dc[idxs], self._features_rest[idxs], new_opacity, new_scaling, self._rotation[idxs]


    def _sample_alives(self, probs=None, num=0, alive_indices=None, scores=None, temperature=1.0):
        """
        Sample alive Gaussians for relocation or growth.
        
        Backward compatibility:
            Old callers pass probs=... directly.
            New callers pass scores=... for utility-guided sampling.
        """
        if scores is not None:
            # Utility-guided: robustly normalize before softmax so a few
            # outliers do not collapse all relocation/growth sampling.
            scores = torch.nan_to_num(scores.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            scores = torch.log1p(torch.clamp(scores, min=0.0))
            median = scores.median()
            mad = (scores - median).abs().median().clamp_min(1e-6)
            scores = ((scores - median) / mad).clamp(-5.0, 5.0)
            probs = torch.softmax(scores / max(float(temperature), 1e-6), dim=0)
        else:
            # Legacy: normalize provided probabilities
            probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs).unsqueeze(-1)
        return sampled_idxs, ratio
    

    def relocate_gs(self, dead_mask=None):

        if dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask 
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if dead_indices.shape[0] <= 0 or alive_indices.shape[0] <= 0:
            return

        # sample from alive ones based on opacity
        probs = (self.get_opacity[alive_indices, 0]) 
        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])

        (
            self._xyz[dead_indices], 
            self._features_dc[dead_indices],
            self._features_rest[dead_indices],
            self._opacity[dead_indices],
            self._scaling[dead_indices],
            self._rotation[dead_indices] 
        ) = self._update_params(reinit_idx, ratio=ratio)
        
        self._opacity[reinit_idx] = self._opacity[dead_indices]
        self._scaling[reinit_idx] = self._scaling[dead_indices]

        self.replace_tensors_to_optimizer(inds=reinit_idx)
        self.visibility_ema[dead_indices] = 0.0
        self.visibility_ema[reinit_idx] = self.visibility_ema[reinit_idx].clamp_max(0.5)

    def relocate_gs_energy_guided(
        self,
        dead_mask=None,
        parent_scores=None,
        temperature=1.0,
        exclude_parent_mask=None,
    ):
        """
        Utility-guided relocation.
        
        Backward compatibility: this is called when --energy-mcmc is enabled.
        The old relocate_gs() remains available for --no-energy-mcmc.
        """
        if dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask
        if exclude_parent_mask is not None and exclude_parent_mask.shape[0] == alive_mask.shape[0]:
            dead_mask = dead_mask & ~exclude_parent_mask
            alive_mask &= ~exclude_parent_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if dead_indices.shape[0] <= 0 or alive_indices.shape[0] <= 0:
            return

        # Sample alive parents by utility scores
        if parent_scores is not None:
            scores = parent_scores[alive_indices]
        else:
            scores = None
        
        probs = (self.get_opacity[alive_indices, 0])
        reinit_idx, ratio = self._sample_alives(
            alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0],
            scores=scores, temperature=temperature,
        )

        (
            self._xyz[dead_indices], 
            self._features_dc[dead_indices],
            self._features_rest[dead_indices],
            self._opacity[dead_indices],
            self._scaling[dead_indices],
            self._rotation[dead_indices] 
        ) = self._update_params(reinit_idx, ratio=ratio)
        
        self._opacity[reinit_idx] = self._opacity[dead_indices]
        self._scaling[reinit_idx] = self._scaling[dead_indices]

        self.replace_tensors_to_optimizer(inds=reinit_idx)
        self.visibility_ema[dead_indices] = 0.0
        self.visibility_ema[reinit_idx] = self.visibility_ema[reinit_idx].clamp_max(0.5)

    def add_new_gs_energy_guided(
        self,
        cap_max,
        growth_factor=1.05,
        parent_scores=None,
        temperature=1.0,
        exclude_parent_mask=None,
    ):
        """
        Utility-guided growth.
        
        Backward compatibility: this is called when --energy-mcmc is enabled.
        The old add_new_gs() remains available for --no-energy-mcmc.
        """
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(growth_factor * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        # Sample parents by utility scores
        parent_indices = torch.arange(current_num_points, device=self.get_xyz.device)
        if exclude_parent_mask is not None and exclude_parent_mask.shape[0] == current_num_points:
            parent_indices = parent_indices[~exclude_parent_mask]
        if parent_indices.numel() == 0:
            return 0

        if parent_scores is not None:
            scores = parent_scores[parent_indices]
        else:
            scores = None
        
        probs = self.get_opacity.squeeze(-1)[parent_indices]
        add_idx, ratio = self._sample_alives(
            alive_indices=parent_indices, probs=probs, num=num_gs,
            scores=scores, temperature=temperature,
        )

        (
            new_xyz, 
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation 
        ) = self._update_params(add_idx, ratio=ratio)

        self._opacity[add_idx] = new_opacity
        self._scaling[add_idx] = new_scaling

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, reset_params=False)
        self.replace_tensors_to_optimizer(inds=add_idx)
        return num_gs

    def add_new_gs(self, cap_max, growth_factor=1.05):
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(growth_factor * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        probs = self.get_opacity.squeeze(-1) 
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)

        (
            new_xyz, 
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation 
        ) = self._update_params(add_idx, ratio=ratio)

        self._opacity[add_idx] = new_opacity
        self._scaling[add_idx] = new_scaling

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, reset_params=False)
        self.replace_tensors_to_optimizer(inds=add_idx)

        return num_gs

    def _sample_taming_indices(self, scores, candidate_mask, budget):
        budget = int(max(0, budget))
        if budget <= 0:
            return torch.empty(0, device=self.get_xyz.device, dtype=torch.long)

        candidate_indices = candidate_mask.nonzero(as_tuple=True)[0]
        if candidate_indices.numel() == 0:
            return candidate_indices

        budget = min(budget, candidate_indices.numel())
        candidate_scores = scores[candidate_indices].detach().float().clamp_min(0.0)
        if not torch.any(candidate_scores > 0):
            candidate_scores = torch.ones_like(candidate_scores)
        sampled = torch.multinomial(candidate_scores, budget, replacement=False)
        return candidate_indices[sampled]

    def _densify_clone_indices(self, selected_indices):
        if selected_indices.numel() == 0:
            return 0

        new_xyz = self._xyz[selected_indices]
        new_features_dc = self._features_dc[selected_indices]
        new_features_rest = self._features_rest[selected_indices]
        new_opacities = self._opacity[selected_indices]
        new_scaling = self._scaling[selected_indices]
        new_rotation = self._rotation[selected_indices]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            reset_params=True,
        )
        self.visibility_ema = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        return int(selected_indices.numel())

    def _densify_split_indices(self, selected_indices, N=2):
        if selected_indices.numel() == 0:
            return 0

        old_count = self.get_xyz.shape[0]
        stds = self.get_scaling[selected_indices].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_indices]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_indices].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_indices].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_indices].repeat(N, 1)
        new_features_dc = self._features_dc[selected_indices].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_indices].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_indices].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            reset_params=True,
        )

        prune_filter = torch.zeros(self.get_xyz.shape[0], device="cuda", dtype=bool)
        prune_filter[selected_indices] = True
        # Only original parents are pruned; appended children are kept.
        prune_filter[old_count:] = False
        self.prune_points(prune_filter)
        self.visibility_ema = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        return int(selected_indices.numel())

    def densify_with_taming_scores(
        self,
        scores,
        target_count,
        extent,
        min_opacity=0.005,
        max_screen_size=None,
        iteration=None,
        prune_stop_iter=3200,
        grad_threshold=0.0002,
        split_children=2,
    ):
        current_count = self.get_xyz.shape[0]
        target_count = int(target_count)
        if target_count <= current_count:
            return {"cloned": 0, "split": 0, "pruned": 0, "target": target_count}

        scores = scores.to(device=self.get_xyz.device, dtype=torch.float32)
        if scores.shape[0] != current_count:
            padded_scores = torch.zeros(current_count, device=self.get_xyz.device, dtype=torch.float32)
            n = min(current_count, scores.shape[0])
            padded_scores[:n] = scores[:n]
            scores = padded_scores

        grad_vars = self.xyz_gradient_accum / self.denom
        grad_vars[grad_vars.isnan()] = 0.0
        grad = _dense_grad(self._xyz.grad)
        if not torch.any(torch.norm(grad_vars, dim=-1) > 0) and grad is not None:
            grad_vars = grad.detach()
        score_qualifiers = scores > 0
        grad_qualifiers = torch.norm(grad_vars, dim=-1) >= grad_threshold
        growth_qualifiers = score_qualifiers | grad_qualifiers
        clone_qualifiers = self.get_scaling.max(dim=1).values <= self.percent_dense * extent
        split_qualifiers = self.get_scaling.max(dim=1).values > self.percent_dense * extent

        clone_candidates = clone_qualifiers & growth_qualifiers
        split_candidates = split_qualifiers & growth_qualifiers
        total_clones = int(clone_candidates.sum().item())
        total_splits = int(split_candidates.sum().item())
        total_candidates = total_clones + total_splits
        if total_candidates <= 0:
            return {"cloned": 0, "split": 0, "pruned": 0, "target": target_count}

        growth_budget = min(target_count - current_count, total_candidates)
        clone_budget = int(math.floor(growth_budget * total_clones / total_candidates))
        split_budget = growth_budget - clone_budget
        if split_budget > total_splits:
            clone_budget += split_budget - total_splits
            split_budget = total_splits
        if clone_budget > total_clones:
            split_budget += clone_budget - total_clones
            clone_budget = total_clones

        clone_indices = self._sample_taming_indices(scores, clone_candidates, clone_budget)
        split_indices = self._sample_taming_indices(scores, split_candidates, split_budget)

        cloned = self._densify_clone_indices(clone_indices)
        split = self._densify_split_indices(split_indices, N=split_children)

        prune_mask = (self.get_opacity < min_opacity).squeeze(-1)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        pruned = 0
        if prune_mask.any() and (iteration is None or iteration < prune_stop_iter):
            self.prune_points(prune_mask)
            self.visibility_ema = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            pruned = int(prune_mask.sum().item())

        torch.cuda.empty_cache()
        return {"cloned": cloned, "split": split, "pruned": pruned, "target": target_count}
