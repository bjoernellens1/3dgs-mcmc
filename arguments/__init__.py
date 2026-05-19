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
        # Eval is on by default — pass --no-eval to disable the test holdout.
        self.eval = True
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
        self.hypersim_cam_id = "cam_00"
        self.hypersim_frame_stride = 1
        self.hypersim_eval_hold = 8
        self.orbbec_color_topic = "/camera/color/image_raw/compressed"
        self.orbbec_depth_topic = "/camera/depth/image_raw/compressed"
        self.orbbec_pose_topic = "/camera_pose"
        self.orbbec_pose_source = "camera_pose"
        self.orbbec_camera_info_topic = "/camera/color/camera_info"
        self.orbbec_sync_threshold_ms = 33.0
        self.orbbec_open3d_odom_max_failure_ratio = 0.25
        self.orbbec_open3d_odom_cache = True
        self.orbbec_open3d_odom_cache_dir = ""
        self.orbbec_open3d_odom_stride = 1
        self.orbbec_open3d_odom_downscale = 1
        self.orbbec_open3d_odom_max_trans_per_edge = 0.15
        self.orbbec_open3d_odom_max_rot_deg_per_edge = 8.0
        self.orbbec_open3d_odom_method = "hybrid"
        self.orbbec_open3d_odom_depth_diff_max = 0.07
        self.orbbec_open3d_icp_max_distance = 0.07
        self.orbbec_open3d_icp_robust_kernel = "huber"
        self.orbbec_open3d_icp_sigma = 0.05
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
        self.lambda_lpips = 0.0
        self.lpips_interval = 10
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
        # Training-progress video: render a fixed camera every N iters,
        # write training_progress.mp4 at end of training.
        self.progress_video_interval = 0     # 0 = disabled (default off; enable with --progress_video_interval 200)
        self.progress_video_fps = 10         # output FPS
        self.scalar_log_interval = 100
        self.geometry_log_interval = 500
        self.sfm_anchor_interval = 2000
        self.strategy_log_interval = 500
        self.mcmc_control_log_interval = 500
        super().__init__(parser, "Optimization Parameters")


class StreamingParams(ParamGroup):
    def __init__(self, parser):
        # Enable streaming replay mode
        self.streaming_replay = False
        self.streaming_input_fps = 30.0
        self.streaming_wallclock = False          # False = deterministic step-based simulation (legacy)
        self.streaming_steps_per_frame = 50       # release one frame every N training iterations
        # Ingestion pacing mode: iter_based | dataset_fps | wallclock_strict
        # iter_based      — deterministic, release every streaming_steps_per_frame iters (default)
        # dataset_fps     — simulated clock; train as many iters as possible per real-time second;
        #                   cap arrivals at streaming_input_fps_cap (never drops frames)
        # wallclock_strict — true wall-clock pacing; drops frames when training is slow
        self.streaming_ingestion_mode = "iter_based"
        self.streaming_input_fps_cap = 30.0       # max frame rate for dataset_fps / wallclock_strict
        self.streaming_max_frames = 0             # 0 = all frames in dataset
        self.streaming_frame_stride = 1           # Release every Nth frame from the source
        self.streaming_initial_frames = 5         # frames used for bootstrap point cloud + init
        self.streaming_keyframe_window = 8        # recent cameras used for local training
        self.streaming_replay_buffer = 32         # size of older-frame replay ring buffer
        self.streaming_global_replay_ratio = 0.1  # fraction of steps drawn from replay buffer
        # Depth-based incremental Gaussian insertion (Phase 2)
        self.streaming_insert_from_depth = True
        self.streaming_depth_stride = 8
        self.streaming_max_new_gaussians_per_frame = 1000
        self.streaming_insert_voxel_size = 0.02
        self.streaming_min_depth = 0.1
        self.streaming_max_depth = 8.0
        # Compatibility mode: make depth insertion obey cap_max. Default False
        # because SLAM map size is unknown and depth insertion is sensor-driven.
        self.streaming_depth_respects_cap = False
        # Local MCMC: restrict noise/reloc to visible/active Gaussians only
        self.streaming_mcmc_local_only = True
        # Global maintenance: run full MCMC pass every N iterations (0 = never)
        self.streaming_global_maintenance_interval = 200
        # Save PLY every N frames ingested (0 = iteration-based only)
        self.streaming_save_frame_interval = 50
        # Initial opacity for depth-inserted Gaussians — high enough for gradient signal
        self.streaming_insert_opacity = 0.05
        # Batch N frames of depth-insertion candidates into a single add_points_as_gaussians
        # call, reducing O(N²) optimizer-state rebuild to O(N/batch) calls. 0 = disable batching.
        self.streaming_insertion_batch_frames = 1
        # Flush the insertion batch early if it exceeds this many pending points (0 = no early flush).
        self.streaming_insertion_batch_max_points = 8000
        # Coverage voxel multiplier (deprecated — occupancy hash enforces a single voxel size)
        self.streaming_cover_voxel_size = 0.0
        self.streaming_cover_voxel_multiplier = 1.0
        # Rebuild occupancy hash from current Gaussian positions every N iters (0 = only on stale prune)
        self.streaming_occupancy_rebuild_interval = 200
        # Depth discontinuity threshold: reject pixels where |dz/dx|+|dz/dy| > this (metres)
        self.streaming_depth_edge_threshold = 0.02
        # Grazing-angle rejection: reject surface normals > this angle from view direction (degrees)
        self.streaming_max_view_angle = 70.0
        # Use KNN-based initial scale for inserted Gaussians (matches bootstrap quality)
        self.streaming_insert_knn_scale = True
        # Two-frame depth consistency threshold (metres; 0 = disabled)
        self.streaming_depth_consistency_thresh = 0.05
        # Hold-out every Nth arriving frame for test evaluation (0 = disabled).
        # Default 8 ≈ 12.5% test split, evenly spread along the trajectory.
        # Test PSNR + comparison renders are produced post-training whenever
        # this is > 0 (mandatory unless explicitly turned off).
        self.streaming_eval_hold = 8
        # SLAM lifecycle: provisional -> persistent (Step 8)
        self.streaming_min_support_views = 2      # required multi-view confirmations
        self.streaming_support_window = 8         # check support against recent frames
        self.streaming_provisional_max_age = 20   # frames before pruning low-support points
        self.streaming_promote_opacity = 0.3      # opacity boost on promotion
        # Save renders at frame-PLY save milestones (opt-in to avoid overhead)
        self.streaming_render_at_saves = False
        # Pre-training bootstrap snapshot: render bootstrap views before training starts
        # and write to iter_0_bootstrap_views/ (opt-in; adds overhead on startup)
        self.streaming_report_pre_training = False
        # Freeze bootstrap Gaussians (birth_frame==0) from MCMC noise displacement.
        # After each step_post_backward, their positions are restored to the pre-noise
        # values. Tests whether position displacement of early Gaussians causes forgetting.
        self.streaming_anchor_bootstrap = False
        # Diagnostic training-mode gate — controls how much optimisation happens:
        #   "normal"         — standard streaming (default)
        #   "placement_only" — skip backward, optimizer step, and all MCMC;
        #                       inserted points stay exactly where placed (H1 ablation)
        #   "colors_only"    — zero LR on means/scales/quats/opacities, SH trains;
        #                       geometry frozen, colors converge to GT (H2 ablation)
        self.streaming_training_mode = "normal"
        # H3 ablation: replace anisotropic surfel scales (tx, ty, 0.2·min)
        # with isotropic scales (geometric mean of tx·ty for all three axes).
        # Prevents edge-on streaking from flat-disc insertions.
        self.streaming_insert_isotropic_scale = False
        # Scale clamping for depth-inserted Gaussians:
        #   scale_mult:         multiplier on the raw depth-derived footprint
        #   scale_max:          hard upper bound per axis (metres)
        #   normal_scale_ratio: z-axis fraction of in-plane scale (anisotropic mode)
        self.streaming_insert_scale_mult = 0.5
        self.streaming_insert_scale_max = 0.05
        self.streaming_insert_normal_scale_ratio = 0.15
        # Free-space / floater loss weight (penalises opacity rendered in front of surface)
        self.streaming_free_space_loss_weight = 0.0
        # Depth supervision loss weight (0 = disabled)
        self.streaming_depth_loss_weight = 0.05
        # Depth-filtering profile. "default" preserves the generic settings.
        self.streaming_camera_profile = "default"
        self.streaming_frame_admission = "all"
        self.streaming_keyframe_min_translation = 0.05
        self.streaming_keyframe_min_rotation_deg = 5.0
        self.streaming_keyframe_min_overlap = 0.25
        self.streaming_keyframe_max_overlap = 0.90
        self.streaming_keyframe_max_gap = 10
        self.streaming_keyframe_coverage_alpha = 0.3
        self.streaming_keyframe_admit_eval_holdouts = False
        # Depth loss type: "l1" or "huber"
        self.streaming_depth_loss_type = "huber"
        # H7: Freeze confirmed geometry gradients during streaming training.
        # When enabled, xyz/scales/quats gradients are zeroed for old confirmed
        # Gaussians so the optimizer cannot drag existing good splats to explain
        # new views — depth insertion must handle new geometry instead.
        self.streaming_freeze_old_geometry = False
        # A Gaussian is "young" (gradients allowed) for this many frames after birth.
        self.streaming_young_age_frames = 5
        # Extra-strict freeze for this many steps immediately after a new frame arrives.
        # During this window, only provisional splats get geometry gradients.
        self.streaming_freeze_new_frame_steps = 50
        # H8: New-frame warmup — train exclusively on the just-arrived frame for this
        # many steps before mixing in the local window / replay.
        self.streaming_new_frame_warmup_steps = 0
        # H9: Anchor loss for young provisional splats.
        # Penalises drift from the depth-insertion position while the splat is young.
        self.streaming_anchor_loss_weight = 0.0
        # Anchor penalty decays linearly to zero over this many training iterations.
        self.streaming_anchor_decay_steps = 500
        # H10: Global keyframe reservoir.
        # Every Nth ingested train frame is kept in a permanent reservoir for
        # trajectory-wide replay. 0 = disabled.
        self.streaming_global_reservoir_stride = 0
        # H11: Submap-stitching mode parameters.
        # streaming_training_mode = "submap_stitch" enables the submap path.
        self.streaming_submap_frames = 20       # frames per independent submap
        self.streaming_submap_iters = 3000      # optimisation iterations per submap
        self.streaming_global_refine_iters = 5000  # final global refinement iters
        # -----------------------------------------------------------------------
        # Four-state Gaussian lifecycle (PROVISIONAL=0, YOUNG=1, MATURE=2, FROZEN=3)
        # Requires --streaming_lifecycle_enabled to activate.
        # Without it, all behaviour is identical to prior H4d config.
        # -----------------------------------------------------------------------
        self.streaming_lifecycle_enabled = False
        # Frame age (since birth) at which a YOUNG Gaussian becomes MATURE
        self.streaming_mature_age_frames = 15
        # Minimum utility EMA for YOUNG→MATURE promotion (0 = age-only)
        self.streaming_mature_min_utility = 0.1
        # Frame age at which a MATURE Gaussian becomes FROZEN (-1 = never)
        self.streaming_freeze_age_frames = -1
        # EMA decay for per-Gaussian utility tracking
        self.streaming_utility_ema_beta = 0.95
        # -----------------------------------------------------------------------
        # Soft-anchor anti-fade losses for MATURE Gaussians (Component C)
        # All three weights default to 0.0 (opt-in). Recommended: ~0.05 each.
        # -----------------------------------------------------------------------
        self.streaming_mature_anchor_xyz_weight = 0.0
        self.streaming_mature_anchor_scale_weight = 0.0
        self.streaming_mature_anchor_opacity_weight = 0.0
        # -----------------------------------------------------------------------
        # Stratified replay sampling (Component D)
        # -----------------------------------------------------------------------
        # sampling_mode: "legacy" (existing ring-buffer logic) or "stratified"
        self.streaming_sampling_mode = "legacy"
        # Comma-separated ratios for [recent, covisible, global_reservoir, hard_frames]
        # Must sum to ~1.0. Only used when streaming_sampling_mode = "stratified".
        self.streaming_sampling_ratios = "0.70,0.15,0.10,0.05"
        # Depth comparison export: save rendered_depth vs sensor_depth side-by-side PNGs
        # for every ingested training frame. Output goes to <model_path>/depth_comparison/.
        # Set to True to enable; requires streaming_depth_loss_weight > 0 (depth must be rendered).
        self.streaming_export_depth_comparison = False
        # Minimum shared Gaussians to consider two frames covisible
        self.streaming_covisible_min_shared = 200
        # Number of recent high-loss frames to keep as "hard frames"
        self.streaming_hard_frame_history = 8
        # -----------------------------------------------------------------------
        # KNN insertion dedup (Component E)
        # -----------------------------------------------------------------------
        self.streaming_insert_knn_dedup = False
        # Radius factor: candidate rejected if nearest existing Gaussian is within
        # depth * streaming_insert_knn_radius_factor metres
        self.streaming_insert_knn_radius_factor = 0.005
        # Max existing Gaussians to query (beyond this, subsample via voxel grid)
        self.streaming_insert_knn_max_existing = 200000
        # -----------------------------------------------------------------------
        # Insertion telemetry (Component F)
        # -----------------------------------------------------------------------
        self.streaming_insertion_debug = False
        self.streaming_insertion_debug_ply = False
        # Save estimated trajectory as TUM .txt + PNG plot + ATE/RPE metrics
        # to <model_path>/trajectory_eval/. Default on.
        self.streaming_trajectory_eval = True
        super().__init__(parser, "Streaming Parameters")


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
