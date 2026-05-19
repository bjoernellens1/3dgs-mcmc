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
    # get_opacity is already activated (sigmoid applied)
    alpha = gaussians.get_opacity.squeeze(-1)
    support = gaussians.visibility_ema.squeeze(-1)
    low_support = support < tau_support

    numerator = alpha[low_support].sum()
    denominator = alpha.sum() + 1e-8
    return numerator / denominator


def _deterministic_subsample(values, max_items):
    if values.shape[0] <= max_items:
        return values
    indices = torch.linspace(
        0,
        values.shape[0] - 1,
        steps=max_items,
        device=values.device,
    ).long()
    return values[indices]


def compute_sfm_anchor_outlier_mass(
    gaussians,
    sfm_points,
    tau_dist=0.5,
    max_gaussians=8192,
    max_sfm_points=4096,
    chunk_size=2048,
):
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
    # get_opacity is already activated (sigmoid applied)
    alpha = gaussians.get_opacity.squeeze(-1)
    if xyz.shape[0] > max_gaussians:
        indices = torch.linspace(
            0,
            xyz.shape[0] - 1,
            steps=max_gaussians,
            device=xyz.device,
        ).long()
        xyz = xyz[indices]
        alpha = alpha[indices]
    sfm_points = _deterministic_subsample(sfm_points, max_sfm_points)

    min_dists = []
    for start in range(0, xyz.shape[0], chunk_size):
        chunk = xyz[start:start + chunk_size]
        dists = torch.cdist(chunk, sfm_points)
        min_dists.append(dists.min(dim=1).values)
    min_dists = torch.cat(min_dists, dim=0)

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
    # get_opacity is already activated (sigmoid applied)
    alpha = gaussians.get_opacity.squeeze(-1)
    max_scale = gaussians.get_scaling.max(dim=1).values
    support = gaussians.visibility_ema.squeeze(-1)

    floater_mask = (max_scale > s_ref) & (support < tau_support)
    numerator = alpha[floater_mask].sum()
    denominator = alpha.sum() + 1e-8
    return numerator / denominator


def compute_geometry_dashboard(gaussians, sfm_points=None):
    """
    Compute all geometry metrics and return as a dict.

    Batches CUDA syncs: all GPU work happens first, then a single
    synchronize point extracts all scalar values.
    """
    N = gaussians.get_xyz.shape[0]
    # get_opacity is already activated (sigmoid applied)
    alpha = gaussians.get_opacity

    # --- Stage 1: compute all GPU tensors (no sync) ---
    mean_opacity = alpha.mean()
    
    has_vis = hasattr(gaussians, "visibility_ema") and gaussians.visibility_ema.numel() == N
    if has_vis:
        support = gaussians.visibility_ema.squeeze(-1)
        mean_support = support.mean()
        visible_fraction = (support > 0.01).float().mean()
        lsom = compute_low_support_opacity_mass(gaussians)
        osf = compute_opacity_scale_floater_score(gaussians)
    else:
        mean_support = None
        visible_fraction = None
        lsom = None
        osf = None

    has_sfm = sfm_points is not None and sfm_points.numel() > 0
    if has_sfm:
        outlier_mass = compute_sfm_anchor_outlier_mass(gaussians, sfm_points)
    else:
        outlier_mass = None

    # --- Stage 2: single CUDA sync point, then extract floats ---
    # Move all GPU scalars to CPU in one shot, then .item() is cheap (CPU→CPU)
    cpu_vals = {
        "mean_opacity": mean_opacity.cpu(),
    }
    if has_vis:
        cpu_vals["mean_support"] = mean_support.cpu()
        cpu_vals["visible_fraction"] = visible_fraction.cpu()
        cpu_vals["lsom"] = lsom.cpu()
        cpu_vals["osf"] = osf.cpu()
    if has_sfm:
        cpu_vals["outlier_mass"] = outlier_mass.cpu()

    # .item() on CPU tensors: zero CUDA sync
    metrics = {
        "num_gaussians": float(N),
        "mean_opacity": float(cpu_vals["mean_opacity"].item()),
    }
    if has_vis:
        metrics["mean_support"] = float(cpu_vals["mean_support"].item())
        metrics["visible_fraction"] = float(cpu_vals["visible_fraction"].item())
        metrics["low_support_opacity_mass"] = float(cpu_vals["lsom"].item())
        metrics["opacity_scale_floater_score"] = float(cpu_vals["osf"].item())
    if has_sfm:
        metrics["sfm_anchor_outlier_mass"] = float(cpu_vals["outlier_mass"].item())

    return metrics
