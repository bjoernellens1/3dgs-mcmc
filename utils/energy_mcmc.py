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

"""
Energy-Guided Adaptive MCMC for 3DGS.

This module provides differentiable energy losses and per-Gaussian utility
scores to guide MCMC relocation and growth decisions.

Backwards compatibility: all functions here are called conditionally from
train.py when --energy_mcmc is enabled (default). The old fixed-interval
path remains available via --no-energy-mcmc.
"""
import torch
import math


def _dense_grad(grad):
    if grad is None:
        return None
    if getattr(grad, "layout", torch.strided) != torch.strided:
        return grad.to_dense()
    return grad


def _named_grad(gaussians, legacy_attr, gsplat_key=None):
    params = getattr(gaussians, "params", None)
    if params is not None and gsplat_key is not None and gsplat_key in params:
        grad = _dense_grad(params[gsplat_key].grad)
        if grad is not None:
            return grad
    return _dense_grad(getattr(gaussians, legacy_attr).grad)


def compute_effective_count(opacities, dead_threshold=0.005, softness=0.002):
    """
    Compute soft effective Gaussian count.

    N_eff = sum_j sigmoid((alpha_j - alpha_dead) / s)

    Args:
        opacities: [N, 1] raw opacity parameters (gaussians._opacity, before sigmoid)
                   because this function participates in the loss graph and
                   must be differentiable w.r.t. raw parameters.
        dead_threshold: opacity threshold
        softness: softness of the sigmoid
    Returns:
        N_eff scalar tensor
    """
    from utils.compiled_kernels import effective_count_core
    return effective_count_core(opacities, dead_threshold, softness)


def compute_effective_count_loss(
    opacities, iteration, cap_max,
    dead_threshold=0.005, softness=0.002,
    q_start=0.05, q_end=0.85, tau_N=0.45,
    max_iterations=30000, target_splat_end=None,
):
    """
    Loss that steers the effective Gaussian count toward a target curve.

    L_eff = ((N_eff - N_target) / cap_max)^2

    Args:
        opacities: [N, 1] raw opacity parameters (gaussians._opacity, before sigmoid)
                   because this function participates in the loss graph.
        iteration: current iteration
        cap_max: maximum allowed Gaussians and fallback target scale
        q_start: initial target cap fraction
        q_end: final target cap fraction
        tau_N: exponential interpolation tau for target curve
        max_iterations: training horizon used to normalize the target curve
        target_splat_end: optional soft target count independent of cap_max
    Returns:
        scalar loss tensor
    """
    from utils.compiled_kernels import effective_count_loss_core

    if cap_max <= 0:
        zero = torch.tensor(0.0, device=opacities.device)
        return zero, zero.detach(), 0.0

    target_scale = target_splat_end if target_splat_end is not None and target_splat_end > 0 else cap_max
    if target_scale <= 0:
        zero = torch.tensor(0.0, device=opacities.device)
        return zero, zero.detach(), 0.0

    u = min(iteration / float(max(1, max_iterations)), 1.0)

    # Exponential interpolation for q
    tau = max(tau_N, 1e-6)
    denom = 1.0 - math.exp(-1.0 / tau)
    alpha_q = (1.0 - math.exp(-u / tau)) / denom
    q = q_start + (q_end - q_start) * alpha_q

    N_target = target_scale * q
    loss, N_eff = effective_count_loss_core(opacities, N_target, target_scale, dead_threshold, softness)
    return loss, N_eff.detach(), N_target


def compute_opacity_entropy_loss(opacities):
    """
    Opacity entropy loss: pushes opacities toward 0 or 1 (decisive).

    L_entropy = mean(-a log(a) - (1-a) log(1-a))

    Minimizing entropy makes MCMC dead/alive decisions cleaner.

    Args:
        opacities: [N, 1] raw opacity parameters (gaussians._opacity, before sigmoid)
                   because this function participates in the loss graph.
    """
    from utils.compiled_kernels import opacity_entropy_core
    return opacity_entropy_core(opacities)


def compute_gaussian_utility(
    gaussians, render_pkg, iteration,
    w_alpha=1.0, w_vis=2.0, w_grad=3.0,
    w_scale=0.5, w_dead=1.0,
    w_support=2.0,
    beta_opacity=1.0, beta_scale=0.5,
    alpha_dead=0.005,
):
    """
    Compute per-Gaussian utility score for MCMC birth/death decisions.

    U_j = w_alpha * alpha_j
        + w_vis * v_j (single-frame visibility)
        + w_support * support_j (multi-view EMA visibility)
        + w_grad * norm_grad_j
        - w_scale * scale_penalty_j
        - w_dead * dead_penalty_j

    Args:
        gaussians: GaussianModel instance
        render_pkg: dict from render() containing visibility info
        iteration: current iteration
        various weight coefficients
    Returns:
        [N] utility scores tensor
    """
    from utils.compiled_kernels import utility_core

    device = gaussians.get_xyz.device
    N = gaussians.get_xyz.shape[0]

    # Opacity term: get_opacity is already activated (sigmoid applied)
    alpha = gaussians.get_opacity.squeeze(-1)

    # Visibility term (current frame)
    visibility = render_pkg.get("visibility_filter", torch.ones(N, device=device, dtype=torch.bool))
    v = visibility.float()

    # Support EMA term (multi-view visibility, not just current frame)
    support = gaussians.visibility_ema.squeeze(-1) if hasattr(gaussians, "visibility_ema") else v

    # Gradient term: norm of xyz gradient
    # NOTE: requires that loss.backward() has been called
    xyz_grad = torch.zeros(N, device=device)
    grad = _named_grad(gaussians, "_xyz", "means")
    if grad is not None:
        xyz_grad = grad.norm(dim=1)

    opacity_grad = torch.zeros(N, device=device)
    grad = _named_grad(gaussians, "_opacity", "opacities")
    if grad is not None:
        opacity_grad = grad.abs().flatten()

    scale_grad = torch.zeros(N, device=device)
    grad = _named_grad(gaussians, "_scaling", "scales")
    if grad is not None:
        scale_grad = grad.norm(dim=1)

    # Scale penalty: penalize oversized Gaussians
    max_scale = gaussians.get_scaling.max(dim=1).values

    return utility_core(
        alpha, v, support, xyz_grad, opacity_grad, scale_grad, max_scale,
        w_alpha=w_alpha, w_vis=w_vis, w_support=w_support, w_grad=w_grad,
        w_scale=w_scale, w_dead=w_dead, beta_opacity=beta_opacity,
        beta_scale=beta_scale, alpha_dead=alpha_dead,
    )


def compute_dead_mask(
    gaussians, utility=None,
    opacity_threshold=0.005,
    support_threshold=0.01,
    utility_quantile=0.05,
    use_utility_quantile=True,
    min_visibility_count=3,
    scale_kill_threshold=0.20,
    scale_moderate_threshold=0.05,
):
    """
    Compute dead mask combining opacity, support, scale, and optionally utility.

    Composite rule:
        1. Unconditional kill: alpha < opacity_threshold * 0.1  (near-zero, no recovery)
        2. Conservative kill:  alpha < opacity_threshold AND support < support_threshold
        3. Scale kill:         max_scale > scale_kill_threshold AND support < support_threshold
           (large Gaussians must be well-supported or they are runaway floaters)
        4. Medium-scale kill:  max_scale > scale_moderate_threshold
                               AND alpha < opacity_threshold * 4
                               AND support < support_threshold
           (semi-transparent medium-size splats that are unsupported = blue-dot floaters)
        5. Optional utility bottom-quantile kill (opacity-gated)

    Args:
        gaussians: GaussianModel instance
        utility: [N] optional utility scores
        opacity_threshold: opacity dead threshold
        support_threshold: support EMA dead threshold
        utility_quantile: bottom quantile for utility-based death
        use_utility_quantile: whether to append utility quantile death
        min_visibility_count: unused (kept for API compatibility)
        scale_kill_threshold: max_scale above which low support = dead (default 0.2 m)
        scale_moderate_threshold: max_scale above which relaxed opacity gate applies (default 0.05 m)
    Returns:
        [N] bool mask
    """
    # get_opacity is already activated (sigmoid applied)
    alpha = gaussians.get_opacity.squeeze(-1)
    support = gaussians.visibility_ema.squeeze(-1) if hasattr(gaussians, "visibility_ema") else torch.ones_like(alpha)

    very_low_opacity = opacity_threshold * 0.1  # ~0.0005 — unambiguously dead

    # Clause 1: unconditionally kill near-zero opacity
    dead = alpha < very_low_opacity

    # Clause 2: both opacity AND support below threshold
    dead = dead | ((alpha < opacity_threshold) & (support < support_threshold))

    # Clauses 3+4: scale-aware kills — large/medium unsupported Gaussians are floaters
    if hasattr(gaussians, "get_scaling"):
        max_scale = gaussians.get_scaling.max(dim=-1).values
        # Clause 3: large Gaussian with any low support → floater (white-streak case)
        if scale_kill_threshold > 0:
            dead = dead | ((max_scale > scale_kill_threshold) & (support < support_threshold))
        # Clause 4: medium-scale semi-transparent with low support → blue-dot floater
        if scale_moderate_threshold > 0:
            dead = dead | (
                (max_scale > scale_moderate_threshold)
                & (alpha < opacity_threshold * 4)
                & (support < support_threshold)
            )

    if use_utility_quantile and utility is not None and utility.numel() > 0:
        q_val = torch.quantile(utility, utility_quantile)
        dead = dead | ((alpha < opacity_threshold) & (utility < q_val))

    return dead
