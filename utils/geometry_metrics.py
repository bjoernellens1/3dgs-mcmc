"""
Geometry failure dashboard metrics for 3DGS-MCMC.

Backwards compatibility: all functions are called conditionally from train.py.
They do not affect the rendering or training pipeline unless explicitly used.
"""
import torch


def update_visibility_ema(gaussians, is_used, beta=0.98):
    """
    Update exponential moving average of per-Gaussian visibility.

    Args:
        gaussians: GaussianModel with visibility_ema attribute
        is_used: [N] bool tensor from render_pkg["is_used"]
        beta: EMA decay factor
    """
    used = is_used.float().view(-1, 1)
    N = used.shape[0]

    if not hasattr(gaussians, "visibility_ema") or gaussians.visibility_ema.numel() != N:
        gaussians.visibility_ema = torch.zeros(N, 1, device=used.device, dtype=used.dtype)

    gaussians.visibility_ema.mul_(beta).add_(used, alpha=(1.0 - beta))


def compute_low_support_opacity_mass(gaussians, tau_support=0.03):
    """
    LSOM: fraction of total opacity belonging to poorly-supported Gaussians.

    LSOM = sum(alpha_j * 1[support_j < tau]) / sum(alpha_j)

    Returns scalar tensor.
    """
    alpha = torch.sigmoid(gaussians.get_opacity).squeeze(-1)
    support = gaussians.visibility_ema.squeeze(-1)
    low_support = support < tau_support

    numerator = alpha[low_support].sum()
    denominator = alpha.sum() + 1e-8
    return numerator / denominator


def compute_sfm_anchor_outlier_mass(gaussians, sfm_points, tau_dist=0.5):
    """
    SfM anchor outlier mass: fraction of opacity far from nearest SfM point.

    Args:
        gaussians: GaussianModel
        sfm_points: [M, 3] tensor of SfM point positions (on same device)
        tau_dist: distance threshold in world units
    Returns scalar tensor.
    """
    if sfm_points is None or sfm_points.numel() == 0:
        return torch.tensor(0.0, device=gaussians.get_xyz.device)

    xyz = gaussians.get_xyz
    alpha = torch.sigmoid(gaussians.get_opacity).squeeze(-1)

    # Nearest SfM distance per Gaussian
    dists = torch.cdist(xyz, sfm_points)  # [N, M]
    min_dists = dists.min(dim=1).values

    far_mask = min_dists > tau_dist
    numerator = alpha[far_mask].sum()
    denominator = alpha.sum() + 1e-8
    return numerator / denominator


def compute_opacity_scale_floater_score(gaussians, s_ref=0.15, tau_support=0.03):
    """
    OSF: fraction of opacity in large, low-support splats.

    OSF = sum(alpha_j * 1[s_max_j > s_ref and support_j < tau]) / sum(alpha_j)

    Returns scalar tensor.
    """
    alpha = torch.sigmoid(gaussians.get_opacity).squeeze(-1)
    max_scale = gaussians.get_scaling.max(dim=1).values
    support = gaussians.visibility_ema.squeeze(-1)

    floater_mask = (max_scale > s_ref) & (support < tau_support)
    numerator = alpha[floater_mask].sum()
    denominator = alpha.sum() + 1e-8
    return numerator / denominator


def compute_geometry_dashboard(gaussians, sfm_points=None):
    """
    Compute all geometry metrics and return as a dict.
    """
    N = gaussians.get_xyz.shape[0]
    alpha = torch.sigmoid(gaussians.get_opacity)

    metrics = {
        "num_gaussians": float(N),
        "mean_opacity": float(alpha.mean().item()),
        "mean_support": float(gaussians.visibility_ema.mean().item()) if hasattr(gaussians, "visibility_ema") else 0.0,
        "visible_fraction": float((gaussians.visibility_ema.squeeze(-1) > 0.01).float().mean().item()) if hasattr(gaussians, "visibility_ema") else 0.0,
    }

    if hasattr(gaussians, "visibility_ema") and gaussians.visibility_ema.numel() == N:
        metrics["low_support_opacity_mass"] = float(compute_low_support_opacity_mass(gaussians).item())
        metrics["opacity_scale_floater_score"] = float(compute_opacity_scale_floater_score(gaussians).item())

    if sfm_points is not None and sfm_points.numel() > 0:
        metrics["sfm_anchor_outlier_mass"] = float(compute_sfm_anchor_outlier_mass(gaussians, sfm_points).item())

    return metrics
