import sys

with open("train_streaming.py", "r") as f:
    lines = f.readlines()

insert_idx = 0
for i, line in enumerate(lines):
    if line.startswith("def _run_submap_stitch("):
        insert_idx = i
        break

rolling_seed_code = """
def _run_rolling_seed(
    gaussians,
    streaming_scene,
    opt,
    pipe,
    args,
    background,
    dataset,
    tb_writer,
    save_worker,
    testing_iterations,
    saving_iterations,
    sh_degree_schedule,
):
    \"\"\"
    Diagnostic mode: Aggregates geometry by streaming through windows of frames,
    building depth point clouds, and inserting them into the global map via the
    occupancy grid. No training occurs during the streaming phase.
    Finally, performs global optimization for the specified iterations.
    \"\"\"
    import torch
    import os
    import numpy as np
    from tqdm import tqdm
    from gaussian_renderer import render

    submap_frames = max(2, int(getattr(args, "streaming_submap_frames", 20)))
    refine_iters  = max(0, int(getattr(args, "streaming_global_refine_iters", 5000)))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print(f"[rolling_seed] mode: {submap_frames} frames/window, {refine_iters} global refine iters", flush=True)

    # Collect all frames
    all_frames = list(streaming_scene._all_frames)
    all_train_cameras = list(streaming_scene.train_cameras)
    while streaming_scene.has_next_frame():
        result = streaming_scene.ingest_next_frame()
        if result is not None:
            cam, frame, is_train = result
            if is_train:
                all_train_cameras.append(cam)

    eval_hold = getattr(args, "streaming_eval_hold", 0)
    train_frames = [f for f in all_frames if eval_hold <= 0 or f.index % eval_hold != 0]
    n_paired = min(len(train_frames), len(all_train_cameras))
    frame_cam_pairs = list(zip(train_frames[:n_paired], all_train_cameras[:n_paired]))

    windows = [
        frame_cam_pairs[i: i + submap_frames]
        for i in range(0, len(frame_cam_pairs), submap_frames)
    ]

    print(f"[rolling_seed] Aggregating geometry across {len(windows)} windows...", flush=True)
    
    for sm_idx, sm_pairs in enumerate(windows):
        if not sm_pairs:
            continue
            
        sm_frames, sm_cams = zip(*sm_pairs)
        
        # Build PCD from this window
        from scene.streaming_scene import StreamingScene
        _tmp_scene = StreamingScene.__new__(StreamingScene)
        _tmp_scene.args = args
        _tmp_scene.cameras_extent = streaming_scene.cameras_extent
        pcd = _tmp_scene._build_pcd_from_frames(list(sm_frames))
        
        if pcd is None or pcd.points.shape[0] == 0:
            continue
            
        # Filter points through global occupancy grid
        voxel_size = getattr(args, "streaming_insert_voxel_size", 0.02)
        occupied = streaming_scene.check_occupancy(pcd.points, voxel_size, check_neighbors=True)
        unoccupied = ~occupied
        
        if not unoccupied.any():
            continue
            
        # Extract new points
        new_pts = torch.tensor(pcd.points[unoccupied], dtype=torch.float32, device="cuda")
        new_cols = torch.tensor(pcd.colors[unoccupied], dtype=torch.float32, device="cuda")
        
        # Add to global gaussians
        added = gaussians.add_points_as_gaussians(
            new_pts, new_cols,
            init_scale=0.01,
            init_opacity=0.3,
            use_knn_scale=True,
            is_provisional=False,
            birth_frame=0,
        )
        
        # Update occupancy
        streaming_scene.add_to_occupancy_hash(new_pts)
        print(f"[rolling_seed] Window {sm_idx+1}/{len(windows)}: added {added} points. Total: {gaussians.get_xyz.shape[0]}", flush=True)

    print(f"[rolling_seed] Geometry aggregation complete. Total points: {gaussians.get_xyz.shape[0]}", flush=True)
    
    if refine_iters <= 0:
        print("[rolling_seed] No global refinement requested. Done.", flush=True)
        return
        
    print(f"[rolling_seed] Running {refine_iters} global refinement iters over {len(all_train_cameras)} cameras...", flush=True)
    
    # Re-setup training to recreate optimizers for all points
    gaussians.training_setup(opt)
    _global_is_selective = getattr(gaussians, "optimizer_type", "adam") == "selective_adam"
    
    from utils.mcmc_schedule import MCMCScheduleConfig, get_mcmc_schedule
    from utils.strategies import make_mcmc_strategy
    from utils.compiled_kernels import configure_torch_compile, set_compile_iteration
    
    densification_strategy = getattr(opt, "densification_strategy", "gsplat_energy_mcmc").lower()
    mcmc_cfg = MCMCScheduleConfig(
        start_iter=opt.densify_from_iter,
        stop_growth_iter=getattr(opt, "mcmc_stop_growth_iter", 12_000),
        stop_reloc_iter=opt.densify_until_iter,
        growth_factor_start=getattr(opt, "mcmc_growth_factor_start", 1.05),
    )
    mcmc_strategy = make_mcmc_strategy(densification_strategy, gaussians=gaussians, args=args)
    mcmc_strategy.initialize_state(gaussians=gaussians, args=args)
    configure_torch_compile(args)
    
    from utils.loss_utils import l1_loss, ssim as _ssim_fn
    import random as _rnd
    
    for _it in tqdm(range(1, refine_iters + 1), desc="Global refine"):
        set_compile_iteration(_it)
        cam = _rnd.choice(all_train_cameras)
        
        xyz_lr = gaussians.update_learning_rate(_it)
        if _it in sh_degree_schedule:
            gaussians.oneupSHdegree()
            
        bg = torch.rand((3,), device="cuda") if opt.random_background else background
        
        pkg = render(cam, gaussians, pipe, bg)
        img = pkg["render"]
        gt = cam.original_image
        loss = (1.0 - opt.lambda_dssim) * l1_loss(img, gt) + opt.lambda_dssim * (1.0 - _ssim_fn(img, gt))
        
        mcmc_strategy.step_pre_backward(gaussians=gaussians, args=args, iteration=_it, render_pkg=pkg, loss=loss)
        loss.backward()
        
        if _it < refine_iters:
            if _global_is_selective:
                gaussians.prepare_selective_adam_step()
                _vis_all = pkg["visibility_filter"].detach()
                gaussians.optimizer.step(visibility=_vis_all)
                gaussians.normalize_rotation_params(mask=_vis_all)
            else:
                gaussians.optimizer.step()
                if pipe.gsplat_sparse_grad:
                    gaussians.normalize_rotation_params()
                    
            gaussians.optimizer.zero_grad(set_to_none=True)
            
            # MCMC
            mcmc_strategy.inject_noise(gaussians=gaussians, args=args, xyz_lr=xyz_lr, visible=pkg["visibility_filter"].detach() if _global_is_selective else None, sparse_active_set=_global_is_selective, iteration=_it)
            
            sched = get_mcmc_schedule(_it, gaussians.get_xyz.shape[0], getattr(args, "cap_max", -1), mcmc_cfg)
            mcmc_strategy.step_post_backward(gaussians=gaussians, args=args, sched=sched, iteration=_it, utility=None, temperature=1.0, use_energy_mcmc=False, tb_writer=tb_writer, should_log_strategy=lambda i: False, render_pkg=pkg, lr=xyz_lr)

    # Save final refined PLY
    refined_ply_dir = os.path.join(args.model_path, "point_cloud", "iteration_final")
    os.makedirs(refined_ply_dir, exist_ok=True)
    gaussians.save_ply(os.path.join(refined_ply_dir, "point_cloud.ply"))

    # Eval
    if streaming_scene.getTestCameras():
        try:
            from utils.comparison_report import write_post_training_report
            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            _eval_bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            write_post_training_report(
                args.model_path, refine_iters, gaussians, all_train_cameras, streaming_scene.getTestCameras(),
                render, pipe, _eval_bg, tb_writer=tb_writer, subdir="rolling_seed_final"
            )
        except Exception as _e:
            print(f"[rolling_seed] Evaluation failed: {_e}", flush=True)

    print("[rolling_seed] Training complete.", flush=True)

"""

lines.insert(insert_idx, rolling_seed_code)

with open("train_streaming.py", "w") as f:
    f.writelines(lines)

