import math
import os

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from torch import nn

from utils.general_utils import get_expon_lr_func, inverse_sigmoid
from utils.rocm_knn_fallback import distCUDA2
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p


class OptimizerDictProxy:
    def __init__(self, optimizers):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        groups = []
        for optimizer in self.optimizers.values():
            groups.extend(optimizer.param_groups)
        return groups

    @property
    def state(self):
        merged = {}
        for optimizer in self.optimizers.values():
            merged.update(optimizer.state)
        return merged

    def step(self, visibility=None):
        for optimizer in self.optimizers.values():
            if visibility is not None:
                optimizer.step(visibility)
            else:
                optimizer.step()

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()}

    def load_state_dict(self, state_dict):
        for name, state in state_dict.items():
            if name in self.optimizers:
                self.optimizers[name].load_state_dict(state)


class GsplatGaussianModel:
    uses_gsplat_layout = True

    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.params = nn.ParameterDict()
        self.optimizers = {}
        self.optimizer = OptimizerDictProxy(self.optimizers)
        self.optimizer_type = "adam"
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.xyz_scheduler_args = None
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.visibility_ema = torch.empty(0)
        # Streaming state
        self.birth_frame = torch.empty(0, dtype=torch.int32, device="cuda")
        self.support_count = torch.empty(0, dtype=torch.int16, device="cuda")
        self.provisional = torch.empty(0, dtype=torch.bool, device="cuda")

    @property
    def _xyz(self):
        return self.params["means"]

    @property
    def _scaling(self):
        return self.params["scales"]

    @property
    def _rotation(self):
        return self.params["quats"]

    @property
    def _opacity(self):
        return self.params["opacities"].unsqueeze(-1)

    @property
    def _features_dc(self):
        return self.params["sh0"]

    @property
    def _features_rest(self):
        return self.params["shN"]

    @property
    def get_xyz(self):
        return self.params["means"]

    @property
    def get_scaling(self):
        return torch.exp(self.params["scales"])

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self.params["quats"], dim=1, eps=1e-8)

    @property
    def get_opacity(self):
        return torch.sigmoid(self.params["opacities"]).unsqueeze(-1)

    @property
    def get_features(self):
        return torch.cat((self.params["sh0"], self.params["shN"]), dim=1)

    def as_gsplat_params(self, activated=True):
        if activated:
            return {
                "means": self.params["means"],
                "scales": self.get_scaling,
                "quats": self.get_rotation,
                "opacities": torch.sigmoid(self.params["opacities"]),
                "sh0": self.params["sh0"],
                "shN": self.params["shN"],
            }
        return self.params

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd, spatial_lr_scale: float, init_scale_mode="fixed", init_scale=0.01, voxel_size=0.02):
        self.spatial_lr_scale = spatial_lr_scale
        points_np = np.asarray(pcd.points)
        colors_np = np.asarray(pcd.colors)
        fused_point_cloud = torch.tensor(points_np, dtype=torch.float32, device="cuda")
        fused_color = RGB2SH(torch.tensor(colors_np, dtype=torch.float32, device="cuda"))
        num_sh = (self.max_sh_degree + 1) ** 2
        features = torch.zeros((fused_color.shape[0], num_sh, 3), dtype=torch.float32, device="cuda")
        features[:, 0, :] = fused_color

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        N = fused_point_cloud.shape[0]
        if init_scale_mode == "fixed":
            dist2 = torch.full((N,), init_scale ** 2, device="cuda")
        elif init_scale_mode == "knn":
            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(points_np).float().cuda()), 0.0000001)
        elif init_scale_mode == "voxel":
            dist2 = torch.full((N,), (voxel_size * 0.5) ** 2, device="cuda")
        else:
            raise ValueError(f"Unknown init_scale_mode: {init_scale_mode}")
        scales = torch.log(torch.sqrt(dist2) * 0.1)[..., None].repeat(1, 3)
        quats = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        quats[:, 0] = 1
        opacities = inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0],), dtype=torch.float32, device="cuda"))

        self.params = nn.ParameterDict({
            "means": nn.Parameter(fused_point_cloud.requires_grad_(True)),
            "scales": nn.Parameter(scales.requires_grad_(True)),
            "quats": nn.Parameter(quats.requires_grad_(True)),
            "opacities": nn.Parameter(opacities.requires_grad_(True)),
            "sh0": nn.Parameter(features[:, :1, :].contiguous().requires_grad_(True)),
            "shN": nn.Parameter(features[:, 1:, :].contiguous().requires_grad_(True)),
        })
        self._reset_running_state()

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.optimizer_type = getattr(training_args, "optimizer_type", "adam").lower()
        if self.optimizer_type == "default":
            self.optimizer_type = "adam"

        lr_by_name = {
            "means": training_args.position_lr_init * self.spatial_lr_scale,
            "scales": training_args.scaling_lr,
            "quats": training_args.rotation_lr,
            "opacities": training_args.opacity_lr,
            "sh0": training_args.feature_lr,
            "shN": training_args.feature_lr / 20.0,
        }

        if self.optimizer_type == "selective_adam":
            try:
                from gsplat.optimizers import SelectiveAdam
            except Exception as exc:
                raise RuntimeError(
                    "--optimizer_type selective_adam requires gsplat.optimizers.SelectiveAdam "
                    "inside the project container."
                ) from exc
            optimizer_cls = SelectiveAdam
            kwargs = {"eps": 1e-15, "betas": (0.9, 0.999)}
        elif self.optimizer_type == "adam":
            optimizer_cls = torch.optim.Adam
            kwargs = {"lr": 0.0, "eps": 1e-15}
        else:
            raise ValueError(
                f"Unsupported optimizer_type '{self.optimizer_type}'. "
                "Expected 'adam' or 'selective_adam'."
            )

        self.optimizers = {}
        for name, lr in lr_by_name.items():
            param_group = {"params": [self.params[name]], "lr": lr, "name": name}
            self.optimizers[name] = optimizer_cls([param_group], **kwargs)
        self.optimizer = OptimizerDictProxy(self.optimizers)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )
        self._reset_running_state()

    def update_learning_rate(self, iteration):
        lr = self.xyz_scheduler_args(iteration)
        self.optimizers["means"].param_groups[0]["lr"] = lr
        return lr

    def normalize_rotation_params(self, mask=None):
        with torch.no_grad():
            quats = self.params["quats"]
            if mask is None:
                quats.copy_(torch.nn.functional.normalize(quats, dim=1, eps=1e-8))
            else:
                quats[mask] = torch.nn.functional.normalize(quats[mask], dim=1, eps=1e-8)

    def prepare_selective_adam_step(self, allow_dense_grads=False):
        for optimizer in self.optimizers.values():
            for group in optimizer.param_groups:
                param = group["params"][0]
                if not param.is_contiguous():
                    param.data = param.data.contiguous()
                if param.grad is not None:
                    if getattr(param.grad, "layout", torch.strided) != torch.strided:
                        param.grad = param.grad.to_dense().contiguous()
                    elif not param.grad.is_contiguous():
                        param.grad = param.grad.contiguous()
                stored_state = optimizer.state.get(param, None)
                if stored_state is not None:
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in stored_state and not stored_state[key].is_contiguous():
                            stored_state[key] = stored_state[key].contiguous()

    def capture(self):
        return {
            "layout": "gsplat",
            "active_sh_degree": self.active_sh_degree,
            "params": {name: param.detach() for name, param in self.params.items()},
            "optimizers": self.optimizer.state_dict(),
            "spatial_lr_scale": self.spatial_lr_scale,
            "visibility_ema": self.visibility_ema,
        }

    def restore(self, model_args, training_args):
        if not isinstance(model_args, dict) or model_args.get("layout") != "gsplat":
            raise ValueError("GsplatGaussianModel can only restore gsplat-layout checkpoints.")
        self.active_sh_degree = model_args["active_sh_degree"]
        self.spatial_lr_scale = model_args["spatial_lr_scale"]
        self.params = nn.ParameterDict({
            name: nn.Parameter(tensor.detach().cuda().requires_grad_(True))
            for name, tensor in model_args["params"].items()
        })
        self.training_setup(training_args)
        self.optimizer.load_state_dict(model_args.get("optimizers", {}))
        self.visibility_ema = model_args.get(
            "visibility_ema",
            torch.zeros((self.get_xyz.shape[0], 1), device="cuda"),
        ).to(device="cuda")

    def construct_list_of_attributes(self):
        names = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(self.params["sh0"].shape[1] * self.params["sh0"].shape[2]):
            names.append(f"f_dc_{i}")
        for i in range(self.params["shN"].shape[1] * self.params["shN"].shape[2]):
            names.append(f"f_rest_{i}")
        names.append("opacity")
        for i in range(self.params["scales"].shape[1]):
            names.append(f"scale_{i}")
        for i in range(self.params["quats"].shape[1]):
            names.append(f"rot_{i}")
        return names

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self.params["means"].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.params["sh0"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.params["shN"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.params["opacities"].detach().unsqueeze(-1).cpu().numpy()
        scale = self.params["scales"].detach().cpu().numpy()
        rotation = self.params["quats"].detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = sorted(
            [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = sorted(
            [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")],
            key=lambda x: int(x.split("_")[-1]),
        )
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self.params = nn.ParameterDict({
            "means": nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device="cuda").requires_grad_(True)),
            "scales": nn.Parameter(torch.tensor(scales, dtype=torch.float32, device="cuda").requires_grad_(True)),
            "quats": nn.Parameter(torch.tensor(rots, dtype=torch.float32, device="cuda").requires_grad_(True)),
            "opacities": nn.Parameter(torch.tensor(opacities, dtype=torch.float32, device="cuda").requires_grad_(True)),
            "sh0": nn.Parameter(torch.tensor(features_dc, dtype=torch.float32, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)),
            "shN": nn.Parameter(torch.tensor(features_extra, dtype=torch.float32, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)),
        })
        self.active_sh_degree = self.max_sh_degree
        self._reset_running_state()

    def _reset_running_state(self):
        count = self.get_xyz.shape[0] if len(self.params) else 0
        self.xyz_gradient_accum = torch.zeros((count, 1), device="cuda")
        self.denom = torch.zeros((count, 1), device="cuda")
        self.max_radii2D = torch.zeros((count,), device="cuda")
        self.visibility_ema = torch.zeros((count, 1), device="cuda")
        self.birth_frame = torch.zeros(count, dtype=torch.int32, device="cuda")
        self.support_count = torch.zeros(count, dtype=torch.int16, device="cuda")
        self.provisional = torch.zeros(count, dtype=torch.bool, device="cuda")

    def prune_points(self, mask):
        valid_points_mask = ~mask
        
        # Prune optimizable parameters
        for name, optimizer in self.optimizers.items():
            group = optimizer.param_groups[0]
            old_param = group["params"][0]
            stored_state = optimizer.state.get(old_param, None)
            
            new_param = nn.Parameter(old_param[valid_points_mask].detach().requires_grad_(True))
            if stored_state is not None:
                new_state = {}
                for k, v in stored_state.items():
                    if isinstance(v, torch.Tensor):
                        new_state[k] = v[valid_points_mask]
                    else:
                        new_state[k] = v
                del optimizer.state[old_param]
                optimizer.state[new_param] = new_state
            group["params"][0] = new_param
            self.params[name] = new_param

        # Prune running state buffers
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.visibility_ema = self.visibility_ema[valid_points_mask]
        self.birth_frame = self.birth_frame[valid_points_mask]
        self.support_count = self.support_count[valid_points_mask]
        self.provisional = self.provisional[valid_points_mask]

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
    ) -> int:
        """
        Append new Gaussians initialised from 3-D world-space points and RGB
        colours (float32, range [0, 1]).  Extends each per-parameter optimizer
        with zero-initialised momentum state for the new entries.

        use_knn_scale: derive per-Gaussian scale from k-NN distance (matches
        bootstrap quality); falls back to fixed init_scale when False.

        Returns the number of Gaussians actually added.
        """
        N = points.shape[0]
        if N == 0:
            return 0

        from utils.sh_utils import RGB2SH
        from utils.general_utils import inverse_sigmoid

        num_sh = (self.max_sh_degree + 1) ** 2
        fused_color = RGB2SH(colors.to(device="cuda", dtype=torch.float32))  # (N, 3)

        # Scale initialisation
        if scales is not None:
            log_scales = scales.to(device="cuda", dtype=torch.float32)
        elif use_knn_scale and N > 1:
            from utils.rocm_knn_fallback import distCUDA2
            pts_cuda = points.to(device="cuda", dtype=torch.float32)
            dist_sq = distCUDA2(pts_cuda)  # (N,) squared dist to nearest neighbour
            scales_1d = torch.clamp(dist_sq.sqrt() * 0.5, 1e-4, 0.1)
            log_scales = torch.log(scales_1d).unsqueeze(-1).expand(-1, 3).contiguous()
        else:
            log_scales = torch.full((N, 3), math.log(init_scale * 0.1), device="cuda")

        if rotations is not None:
            new_quats = rotations.to(device="cuda", dtype=torch.float32)
        else:
            new_quats = torch.cat([                                                    # wxyz identity
                torch.ones(N, 1, device="cuda"),
                torch.zeros(N, 3, device="cuda"),
            ], dim=1)

        new_tensors = {
            "means": points.to(device="cuda", dtype=torch.float32).contiguous(),
            "sh0": fused_color[:, None, :].contiguous(),                          # (N, 1, 3)
            "shN": torch.zeros((N, num_sh - 1, 3), device="cuda"),                # (N, R-1, 3)
            "scales": log_scales,
            "quats": new_quats,
            "opacities": inverse_sigmoid(torch.full((N,), float(init_opacity), device="cuda")),
        }

        for name, ext in new_tensors.items():
            optimizer = self.optimizers[name]
            group = optimizer.param_groups[0]
            old_param = group["params"][0]
            stored_state = optimizer.state.get(old_param, None)

            new_param = nn.Parameter(
                torch.cat([old_param.detach(), ext.detach()], dim=0).requires_grad_(True)
            )
            if stored_state is not None:
                new_state = {}
                for k, v in stored_state.items():
                    if isinstance(v, torch.Tensor) and v.dim() == ext.dim():
                        # Per-element momentum state (exp_avg, exp_avg_sq) — extend with zeros
                        zeros = torch.zeros(N, *ext.shape[1:], dtype=v.dtype, device=v.device)
                        new_state[k] = torch.cat([v, zeros], dim=0)
                    else:
                        # Scalar step counter or other non-extensible state — keep as-is
                        new_state[k] = v
                del optimizer.state[old_param]
                optimizer.state[new_param] = new_state
            group["params"][0] = new_param
            self.params[name] = new_param

        # Extend running state buffers to match the new count
        self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros(N, device="cuda")])
        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, torch.zeros(N, 1, device="cuda")])
        self.denom = torch.cat([self.denom, torch.zeros(N, 1, device="cuda")])
        self.visibility_ema = torch.cat([self.visibility_ema, torch.zeros(N, 1, device="cuda")])
        
        # Extend streaming buffers
        new_birth = torch.full((N,), int(birth_frame), dtype=torch.int32, device="cuda")
        new_support = torch.zeros(N, dtype=torch.int16, device="cuda")
        new_provisional = torch.full((N,), bool(is_provisional), dtype=torch.bool, device="cuda")
        
        self.birth_frame = torch.cat([self.birth_frame, new_birth], dim=0)
        self.support_count = torch.cat([self.support_count, new_support], dim=0)
        self.provisional = torch.cat([self.provisional, new_provisional], dim=0)

        return N
