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
import json
import math
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
try:
    from gaussian_renderer import network_gui
except Exception:
    network_gui = None
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.gaussian_model import build_scaling_rotation
from utils.mcmc_schedule import MCMCScheduleConfig, get_mcmc_schedule
from utils.energy_mcmc import (
    compute_effective_count_loss,
    compute_opacity_entropy_loss,
    compute_gaussian_utility,
    compute_dead_mask,
)
from utils.geometry_metrics import (
    update_visibility_ema,
    compute_geometry_dashboard,
)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, sh_degree_schedule):
    if dataset.cap_max == -1:
        print("Please specify the maximum number of Gaussians using --cap_max.")
        exit()
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    mcmc_cfg = MCMCScheduleConfig(
        start_iter=opt.densify_from_iter,
        stop_growth_iter=getattr(opt, "mcmc_stop_growth_iter", 12_000),
        stop_reloc_iter=opt.densify_until_iter,
        growth_factor_start=getattr(opt, "mcmc_growth_factor_start", 1.05),
        growth_factor_min=getattr(opt, "mcmc_growth_factor_min", 1.002),
        growth_factor_tau=getattr(opt, "mcmc_growth_factor_tau", 0.35),
        grow_interval_min=getattr(opt, "mcmc_grow_interval_min", 100),
        grow_interval_max=getattr(opt, "mcmc_grow_interval_max", 2000),
        grow_tau=getattr(opt, "mcmc_grow_tau", 0.35),
    )
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        xyz_lr = gaussians.update_learning_rate(iteration)

        # Increase SH degree at configured schedule (default: 1000, 2000, 3000)
        if iteration in sh_degree_schedule:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image = render_pkg["render"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss = loss + args.opacity_reg * torch.abs(gaussians.get_opacity).mean()
        loss = loss + args.scale_reg * torch.abs(gaussians.get_scaling).mean()

        # Energy-guided MCMC losses (Stage A: soft loss steering)
        # Backward compatibility: only active when --energy_mcmc is enabled (default)
        if args.energy_mcmc:
            # Effective count loss: steer toward target population curve
            L_eff = compute_effective_count_loss(
                gaussians.get_opacity,
                iteration=iteration,
                cap_max=args.cap_max,
            )
            loss = loss + args.lambda_eff_count * L_eff

            # Opacity entropy loss: encourage decisive alive/dead opacities
            L_entropy = compute_opacity_entropy_loss(gaussians.get_opacity)
            loss = loss + args.lambda_opacity_entropy * L_entropy

        loss.backward()

        # Update visibility EMA for geometry dashboard
        with torch.no_grad():
            update_visibility_ema(gaussians, render_pkg["is_used"])

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations) or (args.save_interval > 0 and iteration % args.save_interval == 0):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Geometry failure dashboard
            if iteration % 100 == 0:
                with torch.no_grad():
                    # Load SfM points once for anchor distance
                    if not hasattr(scene, "_sfm_points"):
                        try:
                            import numpy as np
                            from scene.dataset_readers import fetchPly
                            sfm_pcd = fetchPly(os.path.join(dataset.source_path, "sparse", "0", "points3D.ply"))
                            scene._sfm_points = torch.tensor(np.asarray(sfm_pcd.points), dtype=torch.float32, device="cuda")
                        except Exception:
                            scene._sfm_points = None

                    geo = compute_geometry_dashboard(gaussians, sfm_points=getattr(scene, "_sfm_points", None))
                    gaussians._last_geo = geo

                    for k, v in geo.items():
                        if tb_writer:
                            tb_writer.add_scalar(f"geometry/{k}", v, iteration)

                    # Print concise dashboard every 500 iters
                    if iteration % 500 == 0:
                        print(
                            f"[geo] iter={iteration} "
                            f"N={int(geo['num_gaussians'])} "
                            f"LSOM={geo.get('low_support_opacity_mass', 0.0):.4f} "
                            f"OSF={geo.get('opacity_scale_floater_score', 0.0):.4f} "
                            f"mean_sup={geo.get('mean_support', 0.0):.4f}",
                            flush=True,
                        )

            sched = get_mcmc_schedule(
                iteration=iteration,
                current_n=gaussians.get_xyz.shape[0],
                cap_max=args.cap_max,
                cfg=mcmc_cfg,
            )

            # Optional closed-loop: suppress growth if LSOM is rising
            if args.energy_mcmc and iteration > opt.densify_from_iter:
                geo = getattr(gaussians, "_last_geo", {})
                lsom = geo.get("low_support_opacity_mass", 0.0)
                if lsom > 0.15:
                    # Too many unsupported splats: halve growth factor
                    sched["growth_factor"] = max(1.0, sched["growth_factor"] * 0.5)
                    print(f"[mcmc-control] LSOM={lsom:.3f} > 0.15, suppressing growth", flush=True)

            # -----------------------------------------------------------------
            # MCMC relocation and growth
            # Two paths: energy-guided (new default) vs schedule-only (legacy)
            # Backward compatibility: --no-energy_mcmc runs the old path
            # -----------------------------------------------------------------
            if args.energy_mcmc:
                with torch.no_grad():
                    utility = compute_gaussian_utility(
                        gaussians=gaussians,
                        render_pkg=render_pkg,
                        iteration=iteration,
                    )

                # Temperature annealing for birth/death sampling
                u_temp = min(iteration / 30000.0, 1.0)
                tau_t = max(getattr(args, "energy_temp_tau", 0.4), 1e-6)
                denom_t = 1.0 - math.exp(-1.0 / tau_t)
                alpha_t = (1.0 - math.exp(-u_temp / tau_t)) / denom_t
                temperature = (
                    args.energy_temp_min
                    + (args.energy_temp_start - args.energy_temp_min)
                    * (1.0 - alpha_t)
                )

                if sched["allow_relocation"] and iteration % sched["relocate_interval"] == 0:
                    dead_mask = compute_dead_mask(
                        gaussians=gaussians,
                        utility=utility,
                        opacity_threshold=sched["dead_opacity_threshold"],
                        utility_quantile=0.05,
                    )

                    dead_count = int(dead_mask.sum().item())
                    gaussians.relocate_gs_energy_guided(
                        dead_mask=dead_mask,
                        parent_scores=utility,
                        temperature=temperature,
                    )

                    print(
                        f"[mcmc-reloc] iter={iteration} "
                        f"dead={dead_count} "
                        f"thr={sched['dead_opacity_threshold']:.5f} "
                        f"reloc_int={sched['relocate_interval']} "
                        f"rho={sched['rho']:.3f} "
                        f"N={gaussians.get_xyz.shape[0]}",
                        flush=True,
                    )

                if sched["allow_growth"] and iteration % sched["grow_interval"] == 0:
                    before = gaussians.get_xyz.shape[0]
                    added = gaussians.add_new_gs_energy_guided(
                        cap_max=args.cap_max,
                        growth_factor=sched["growth_factor"],
                        parent_scores=utility,
                        temperature=temperature,
                    )
                    after = gaussians.get_xyz.shape[0]

                    print(
                        f"[mcmc-grow] iter={iteration} "
                        f"added={added} "
                        f"N={before}->{after} "
                        f"factor={sched['growth_factor']:.4f} "
                        f"grow_int={sched['grow_interval']} "
                        f"rho={sched['rho']:.3f}",
                        flush=True,
                    )
            else:
                # LEGACY PATH: schedule-only MCMC (no energy guidance)
                # Kept for reproducibility and backward compatibility.
                if sched["allow_relocation"] and iteration % sched["relocate_interval"] == 0:
                    dead_mask = (
                        gaussians.get_opacity <= sched["dead_opacity_threshold"]
                    ).squeeze(-1)

                    dead_count = int(dead_mask.sum().item())
                    gaussians.relocate_gs(dead_mask=dead_mask)

                    print(
                        f"[mcmc-reloc] iter={iteration} "
                        f"dead={dead_count} "
                        f"thr={sched['dead_opacity_threshold']:.5f} "
                        f"reloc_int={sched['relocate_interval']} "
                        f"rho={sched['rho']:.3f} "
                        f"N={gaussians.get_xyz.shape[0]}",
                        flush=True,
                    )

                if sched["allow_growth"] and iteration % sched["grow_interval"] == 0:
                    before = gaussians.get_xyz.shape[0]
                    added = gaussians.add_new_gs(
                        cap_max=args.cap_max,
                        growth_factor=sched["growth_factor"],
                    )
                    after = gaussians.get_xyz.shape[0]

                    print(
                        f"[mcmc-grow] iter={iteration} "
                        f"added={added} "
                        f"N={before}->{after} "
                        f"factor={sched['growth_factor']:.4f} "
                        f"grow_int={sched['grow_interval']} "
                        f"rho={sched['rho']:.3f}",
                        flush=True,
                    )

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                actual_covariance = L @ L.transpose(1, 2)

                def op_sigmoid(x, k=100, x0=0.995):
                    return 1 / (1 + torch.exp(-k * (x - x0)))
                
                noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1- gaussians.get_opacity))*args.noise_lr*xyz_lr
                noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                gaussians._xyz.add_(noise)

            if (iteration in checkpoint_iterations) or (args.checkpoint_interval > 0 and iteration % args.checkpoint_interval == 0):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_interval", type=int, default=2000,
                        help="Save a checkpoint every N iterations (independent of --checkpoint_iterations).")
    parser.add_argument("--save_interval", type=int, default=2000,
                        help="Save a PLY point cloud every N iterations (independent of --save_iterations).")
    parser.add_argument("--sh_degree_schedule", nargs="+", type=int, default=[1000, 2000, 3000],
                        help="Iterations at which to increase SH degree (default: 1000 2000 3000).")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    
    if args.config is not None:
        # Load the configuration file
        config = load_config(args.config)
        # Set the configuration parameters on args, if they are not already set by command line arguments
        for key, value in config.items():
            setattr(args, key, value)

    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.sh_degree_schedule)

    # All done
    print("\nTraining complete.")
