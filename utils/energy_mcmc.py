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
    alpha = torch.sigmoid(opacities).squeeze(-1)
    return torch.sigmoid((alpha - dead_threshold) / softness).sum()


def compute_effective_count_loss(
    opacities, iteration, cap_max,
    dead_threshold=0.005, softness=0.002,
    q_start=0.05, q_end=0.85, tau_N=0.45
):
    """
    Loss that steers the effective Gaussian count toward a target curve.
    
    L_eff = ((N_eff - N_target) / cap_max)^2
    
    Args:
        opacities: [N, 1] raw opacity parameters (gaussians._opacity, before sigmoid)
                   because this function participates in the loss graph.
        iteration: current iteration
        cap_max: maximum allowed Gaussians
        q_start: initial target cap fraction
        q_end: final target cap fraction
        tau_N: exponential interpolation tau for target curve
    Returns:
        scalar loss tensor
    """
    if cap_max <= 0:
        return torch.tensor(0.0, device=opacities.device)
    
    # Normalize progress
    # NOTE: using hardcoded 30k as max for q curve; could be parameterized
    u = min(iteration / 30000.0, 1.0)
    
    # Exponential interpolation for q
    tau = max(tau_N, 1e-6)
    denom = 1.0 - math.exp(-1.0 / tau)
    alpha_q = (1.0 - math.exp(-u / tau)) / denom
    q = q_start + (q_end - q_start) * alpha_q
    
    N_eff = compute_effective_count(opacities, dead_threshold, softness)
    N_target = cap_max * q
    
    loss = ((N_eff - N_target) / cap_max) ** 2
    return loss


def compute_opacity_entropy_loss(opacities):
    """
    Opacity entropy loss: pushes opacities toward 0 or 1 (decisive).
    
    L_entropy = mean(-a log(a) - (1-a) log(1-a))
    
    Minimizing entropy makes MCMC dead/alive decisions cleaner.
    
    Args:
        opacities: [N, 1] raw opacity parameters (gaussians._opacity, before sigmoid)
                   because this function participates in the loss graph.
    """
    alpha = torch.sigmoid(opacities).squeeze(-1)
    eps = 1e-6
    entropy = -alpha * torch.log(alpha + eps) - (1.0 - alpha) * torch.log(1.0 - alpha + eps)
    return entropy.mean()


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
    if gaussians._xyz.grad is not None:
        xyz_grad = gaussians._xyz.grad.norm(dim=1)
    
    opacity_grad = torch.zeros(N, device=device)
    if gaussians._opacity.grad is not None:
        opacity_grad = gaussians._opacity.grad.abs().squeeze(-1)
    
    scale_grad = torch.zeros(N, device=device)
    if gaussians._scaling.grad is not None:
        scale_grad = gaussians._scaling.grad.norm(dim=1)
    
    norm_grad = xyz_grad + beta_opacity * opacity_grad + beta_scale * scale_grad
    
    # Scale penalty: penalize oversized Gaussians
    max_scale = gaussians.get_scaling.max(dim=1).values
    scale_penalty = torch.clamp(max_scale - 0.1, min=0.0) ** 2
    
    # Dead penalty
    dead_penalty = (alpha < alpha_dead).float()
    
    utility = (
        w_alpha * alpha
        + w_vis * v
        + w_support * support
        + w_grad * norm_grad
        - w_scale * scale_penalty
        - w_dead * dead_penalty
    )
    
    return utility


def compute_dead_mask(
    gaussians, utility=None,
    opacity_threshold=0.005,
    support_threshold=0.01,
    utility_quantile=0.05,
    use_utility_quantile=True,
    min_visibility_count=3,
):
    """
    Compute dead mask combining opacity, support, and optionally utility.
    
    dead_j = alpha_j < opacity_threshold AND support_j < support_threshold
    optionally OR utility_j < quantile(utility, q)
    
    Args:
        gaussians: GaussianModel instance
        utility: [N] optional utility scores
        opacity_threshold: opacity dead threshold
        support_threshold: support EMA dead threshold
        utility_quantile: bottom quantile for utility-based death
        use_utility_quantile: whether to append utility quantile death
        min_visibility_count: minimum times visible to survive
    Returns:
        [N] bool mask
    """
    # get_opacity is already activated (sigmoid applied)
    alpha = gaussians.get_opacity.squeeze(-1)
    support = gaussians.visibility_ema.squeeze(-1) if hasattr(gaussians, "visibility_ema") else torch.ones_like(alpha)
    
    # Core death: low opacity AND low support
    dead = (alpha < opacity_threshold) & (support < support_threshold)
    
    if use_utility_quantile and utility is not None and utility.numel() > 0:
        q_val = torch.quantile(utility, utility_quantile)
        dead = dead | (utility < q_val)
    
    # NOTE: visibility tracking across iterations would require
    # accumulating visibility counts. For now we use current frame only.
    # Future enhancement: track xyz_gradient_accum as proxy for activity.
    
    return dead
