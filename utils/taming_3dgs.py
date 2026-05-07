import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim


def _dense_grad(grad):
    if grad is None:
        return None
    if getattr(grad, "layout", torch.strided) != torch.strided:
        return grad.to_dense()
    return grad


@dataclass
class TamingScoreWeights:
    view_importance: float = 50.0
    edge_importance: float = 50.0
    mse_importance: float = 50.0
    grad_importance: float = 25.0
    dist_importance: float = 50.0
    opacity_importance: float = 100.0
    depth_importance: float = 5.0
    loss_importance: float = 10.0
    radii_importance: float = 10.0
    scale_importance: float = 25.0
    count_importance: float = 0.1
    blend_importance: float = 50.0


def get_taming_budget(args):
    budget = getattr(args, "taming_budget", -1.0)
    if budget is not None and budget > 0:
        return budget
    return getattr(args, "cap_max", -1)


def get_taming_count_array(start_count, budget, opt, mode="final_count"):
    if mode == "multiplier":
        final_budget = int(start_count * float(budget))
    elif mode == "final_count":
        final_budget = int(budget)
    else:
        raise ValueError(f"Unsupported Taming budget mode: {mode}")

    interval = max(1, getattr(opt, "taming_score_interval", 0) or opt.densification_interval)
    num_steps = max(1, (opt.densify_until_iter - opt.densify_from_iter) // interval)
    if final_budget <= start_count:
        return [start_count for _ in range(num_steps + 1)]

    # Corrected Eq. 2 from the Taming-3DGS project page.
    # A(x) = ((B - S - kN) / N^2) x^2 + kx + S; k = 2(B - S) / N
    n_steps = float(num_steps)
    k = 2.0 * (final_budget - start_count) / n_steps
    a = (final_budget - start_count - k * n_steps) / (n_steps * n_steps)
    counts = []
    for x in range(num_steps + 1):
        count = int(round(a * (x ** 2) + k * x + start_count))
        counts.append(min(final_budget, max(start_count, count)))
    counts[-1] = final_budget
    return counts


def get_taming_score_weights(args):
    return TamingScoreWeights(
        view_importance=getattr(args, "taming_view_importance", 50.0),
        edge_importance=getattr(args, "taming_edge_importance", 50.0),
        mse_importance=getattr(args, "taming_mse_importance", 50.0),
        grad_importance=getattr(args, "taming_grad_importance", 25.0),
        dist_importance=getattr(args, "taming_dist_importance", 50.0),
        opacity_importance=getattr(args, "taming_opacity_importance", 100.0),
        depth_importance=getattr(args, "taming_depth_importance", 5.0),
        loss_importance=getattr(args, "taming_loss_importance", 10.0),
        radii_importance=getattr(args, "taming_radii_importance", 10.0),
        scale_importance=getattr(args, "taming_scale_importance", 25.0),
        count_importance=getattr(args, "taming_count_importance", 0.1),
        blend_importance=getattr(args, "taming_blend_importance", 50.0),
    )


def normalize_score(weight, values):
    values = torch.nan_to_num(values.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    valid = values > 0
    out = torch.zeros_like(values, dtype=torch.float32)
    if not torch.any(valid) or weight == 0:
        return out
    median = torch.median(values[valid]).clamp_min(1e-6)
    out[valid] = float(weight) * (values[valid] / median)
    return out


def compute_edge_map(image):
    if image.dim() != 3:
        raise ValueError("Expected image tensor with shape [C, H, W]")
    image = image.detach().float()
    if image.shape[0] >= 3:
        gray = 0.299 * image[0:1] + 0.587 * image[1:2] + 0.114 * image[2:3]
    else:
        gray = image[:1]

    sobel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=image.device,
        dtype=image.dtype,
    )
    sobel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        device=image.device,
        dtype=image.dtype,
    )
    gx = F.conv2d(gray.unsqueeze(0), sobel_x.unsqueeze(0), padding=1)
    gy = F.conv2d(gray.unsqueeze(0), sobel_y.unsqueeze(0), padding=1)
    edges = torch.sqrt(gx.square() + gy.square()).squeeze(0).squeeze(0)
    denom = edges.max() - edges.min()
    if denom <= 1e-8:
        return torch.zeros_like(edges)
    return (edges - edges.min()) / denom


def compute_loss_map(rendered_image, gt_image, edge_map, weights: TamingScoreWeights):
    l1_map = torch.mean(torch.abs(rendered_image.detach() - gt_image.detach()), dim=0)
    denom = l1_map.max() - l1_map.min()
    if denom > 1e-8:
        l1_map = (l1_map - l1_map.min()) / denom
    else:
        l1_map = torch.zeros_like(l1_map)
    return weights.mse_importance * l1_map + weights.edge_importance * edge_map.to(l1_map.device)


def compute_photometric_loss(viewpoint_cam, image, lambda_dssim=0.2):
    gt_image = viewpoint_cam.original_image.to(image.device)
    ll1 = l1_loss(image, gt_image)
    return (1.0 - lambda_dssim) * ll1 + lambda_dssim * (1.0 - ssim(image, gt_image))


def sample_taming_cameras(cameras, num_cams):
    if num_cams == -1 or num_cams >= len(cameras):
        return list(cameras)
    return random.sample(list(cameras), max(1, num_cams))


def _require_stat(render_pkg, key):
    value = render_pkg.get(key, None)
    if value is None:
        raise RuntimeError(f"Renderer did not return required exact Taming stat '{key}'.")
    return value.detach()


def compute_taming_scores(scene, camlist, edge_maps, gaussians, pipe, bg, weights, opt):
    num_points = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device
    scores_by_view = torch.zeros((len(camlist), num_points), device=device, dtype=torch.float32)

    opacity = gaussians.get_opacity.detach().squeeze(-1)
    scales = torch.prod(gaussians.get_scaling.detach(), dim=1)
    support = gaussians.visibility_ema.detach().squeeze(-1) if hasattr(gaussians, "visibility_ema") else torch.zeros_like(opacity)

    grad = _dense_grad(gaussians._xyz.grad)
    if grad is not None:
        xyz_grad = grad.detach().norm(dim=1)
    else:
        xyz_grad = torch.zeros(num_points, device=device)

    for view_idx, viewpoint_cam in enumerate(camlist):
        base_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        render_image = base_pkg["render"]
        photometric_loss = compute_photometric_loss(
            viewpoint_cam,
            render_image,
            lambda_dssim=getattr(opt, "lambda_dssim", 0.2),
        ).detach()
        gt_image = viewpoint_cam.original_image.to(device)
        pixel_weights = compute_loss_map(render_image, gt_image, edge_maps[view_idx], weights)

        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            pixel_weights=pixel_weights,
            return_taming_stats=True,
        )
        if not render_pkg.get("taming_stats_exact", False):
            raise RuntimeError(
                "Taming scoring requires exact renderer stats. "
                f"Renderer reported exact={render_pkg.get('taming_stats_exact')} "
                f"backend={render_pkg.get('taming_stats_backend', 'unknown')} "
                f"approximation={render_pkg.get('taming_stats_approximation', 'unknown')}"
            )

        visibility = render_pkg.get("visibility_filter", torch.ones(num_points, device=device, dtype=torch.bool))
        if visibility.dtype != torch.bool:
            visible_mask = torch.zeros(num_points, device=device, dtype=torch.bool)
            visible_mask[visibility.squeeze()] = True
        else:
            visible_mask = visibility

        radii = _require_stat(render_pkg, "gaussian_radii").float()
        depths = _require_stat(render_pkg, "gaussian_depths").abs()
        loss_accum = _require_stat(render_pkg, "accum_weights").float()
        dist_accum = _require_stat(render_pkg, "accum_dist").float()
        blend_accum = _require_stat(render_pkg, "accum_blend").float()
        count_accum = _require_stat(render_pkg, "accum_count").float()

        gaussian_importance = (
            normalize_score(weights.grad_importance, xyz_grad)
            + normalize_score(weights.opacity_importance, opacity)
            + normalize_score(weights.depth_importance, depths)
            + normalize_score(weights.radii_importance, radii)
            + normalize_score(weights.scale_importance, scales)
            + normalize_score(weights.count_importance, count_accum + support)
        )
        pixel_importance = (
            normalize_score(weights.dist_importance, dist_accum)
            + normalize_score(weights.loss_importance, loss_accum)
            + normalize_score(weights.blend_importance, blend_accum)
        )
        scores_by_view[view_idx][visible_mask] = (
            weights.view_importance
            * photometric_loss
            * (gaussian_importance + pixel_importance)
        )[visible_mask]

    scores = scores_by_view.sum(dim=0)
    if not torch.any(scores > 0):
        scores = opacity + support + 1e-6
    return torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
