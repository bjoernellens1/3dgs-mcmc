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

from argparse import ArgumentParser, Namespace, BooleanOptionalAction
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action=BooleanOptionalAction)
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action=BooleanOptionalAction)
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.cap_max = 500000
        self.init_type = "sfm"
        self.model_layout = "gsplat"
        self.init_scale_mode = "fixed"
        self.init_scale = 0.01
        self.scannet_frame_stride = 10
        self.scannet_max_frames = 0
        self.scannet_eval_hold = 20
        self.scannet_init = "rgbd"
        self.scannet_depth_stride = 8
        self.scannet_init_frames = 200
        self.scannet_max_init_points = 250000
        self.scannet_depth_scale = 1000.0
        self.scannet_random_num_pts = 250000
        self.tum_frame_stride = 1
        self.tum_max_frames = 0
        self.tum_eval_hold = 8
        self.tum_init = "rgbd"
        self.tum_depth_stride = 4
        self.tum_init_frames = 300
        self.tum_max_init_points = 250000
        self.tum_depth_scale = 5000.0
        self.tum_association_max_dt = 0.03
        self.tum_sequence = ""
        self.tum_random_num_pts = 250000
        self.rgbd_eval_hold = 8
        self.rgbd_depth_stride = 4
        self.rgbd_init_frames = 300
        self.rgbd_max_init_points = 250000
        self.rgbd_min_depth = 0.1
        self.rgbd_max_depth = 8.0
        self.rgbd_random_num_pts = 250000
        self.pointcloud_preprocess = "open3d"
        self.pcd_voxel_size = 0.02
        self.pcd_outlier_filter = "none"
        self.pcd_stat_nb_neighbors = 20
        self.pcd_stat_std_ratio = 2.0
        self.pcd_radius = 0.05
        self.pcd_min_neighbors = 4
        self.pcd_estimate_normals = False
        self.pcd_force_regenerate = False
        self.replica_init = "mesh"
        self.replica_num_views = 120
        self.replica_width = 640
        self.replica_height = 480
        self.replica_eval_hold = 8
        self.replica_fov = 70.0
        self.replica_max_init_points = 250000
        self.replica_render_points = 300000
        self.replica_splat_radius = 1
        self.replica_random_num_pts = 250000
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.tile_size = 16
        self.sh_backend = "python"
        self.near_plane = 0.01
        self.far_plane = 1e10
        self.radius_clip = 0.0
        self.eps2d = 0.3
        self.render_mode = "RGB"
        self.absgrad = False
        self.rasterize_mode = "classic"
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 25_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        self.optimizer_type = "selective_adam"
        self.gsplat_sparse_grad = True
        self.sparse_policy = "active_set"
        self.sh_update_interval = 16
        self.parallelism_profile = "safe"
        self.selective_adam_allow_dense_grads = False
        self.sparse_mode_detach_sh_dir = True
        self.selective_adam_zero_invisible_grads = True
        self.sparse_energy_global_interval = 500
        self.benchmark_dir = ""
        self.noise_lr = 5e5
        self.mcmc_noise_stop_iter = 30_000
        self.scale_reg = 0.01
        self.opacity_reg = 0.01
        self.densification_strategy = "gsplat_energy_mcmc"
        self.mcmc_stop_growth_iter = 12000
        self.mcmc_growth_factor_start = 1.05
        self.mcmc_growth_factor_min = 1.002
        self.mcmc_growth_factor_tau = 0.35
        self.mcmc_grow_interval_min = 100
        self.mcmc_grow_interval_max = 2000
        self.mcmc_grow_tau = 0.35
        self.energy_mcmc = True
        self.lambda_eff_count = 0.1
        self.lambda_opacity_entropy = 0.01
        self.energy_w_alpha = 1.0
        self.energy_w_vis = 2.0
        self.energy_w_grad = 3.0
        self.energy_w_scale = 0.5
        self.energy_w_dead = 1.0
        self.energy_temp_start = 2.0
        self.energy_temp_min = 0.3
        self.energy_temp_tau = 0.4
        self.energy_beta_opacity = 1.0
        self.energy_beta_scale = 0.5
        self.energy_alpha_dead = 0.005
        self.energy_w_support = 2.0
        # MCMC schedule parameters (exposed for ablation studies)
        self.mcmc_relocate_interval_min = 50
        self.mcmc_relocate_interval_max = 500
        self.mcmc_relocate_tau = 0.65
        self.mcmc_cap_growth_power = 2.0
        self.mcmc_cap_interval_strength = 4.0
        self.mcmc_cap_interval_power = 2.0
        self.mcmc_cap_stop_ratio = 0.98
        self.mcmc_dead_opacity_start = 0.003
        self.mcmc_dead_opacity_end = 0.010
        self.mcmc_dead_opacity_power = 1.5
        self.mcmc_use_target_deficit = False
        self.mcmc_target_splat_end = 150000
        self.mcmc_target_q_start = 0.05
        self.mcmc_target_q_end = 0.85
        self.mcmc_target_tau = 0.45
        # Taming-3DGS strategy parameters. These are inert unless
        # --densification_strategy is set to "taming" or "hybrid".
        self.taming_budget = -1.0
        self.taming_budget_mode = "final_count"
        self.taming_cams = 3
        self.taming_score_interval = 100
        self.taming_min_opacity = 0.005
        self.taming_prune_stop_iter = 3200
        self.taming_view_importance = 50.0
        self.taming_edge_importance = 50.0
        self.taming_mse_importance = 50.0
        self.taming_grad_importance = 25.0
        self.taming_dist_importance = 50.0
        self.taming_opacity_importance = 100.0
        self.taming_depth_importance = 5.0
        self.taming_loss_importance = 10.0
        self.taming_radii_importance = 10.0
        self.taming_scale_importance = 25.0
        self.taming_count_importance = 0.1
        self.taming_blend_importance = 50.0
        # torch.compile configuration (off by default — only opt-in for ablation)
        self.compile_mode = "off"
        self.compile_dynamic = False
        self.compile_after_iter = 12000
        self.compile_growth_margin = 500
        self.compile_sh = False
        self.compile_energy = False
        self.compile_utility = False
        # Persistent web viewer. Disabled by default; use --web-viewer to enable.
        self.web_viewer_enabled = False
        self.web_viewer_port = 6010
        self.web_viewer_host = "0.0.0.0"
        self.web_viewer_backend = "process"
        self.web_viewer_cache_dir = ""
        self.web_viewer_image_interval = 100
        self.web_viewer_fixed_camera = False
        self.web_viewer_keep_alive = True
        self.web_viewer_scene_cache_interval = 0
        self.web_viewer_scene_cache_keep = 3
        self.record_video = False
        self.record_video_cameras = ""
        self.record_video_interval = 100
        self.record_video_fps = 30
        self.record_video_crf = 23
        self.record_video_preset = "veryfast"
        self.scalar_log_interval = 100
        self.geometry_log_interval = 500
        self.sfm_anchor_interval = 2000
        self.strategy_log_interval = 500
        self.mcmc_control_log_interval = 500
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
