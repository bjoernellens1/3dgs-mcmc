import math
import torch
from gsplat.rendering import rasterization
from gsplat.cuda._wrapper import rasterize_to_indices_in_range


def _fov2focal(fov, pixels):
    return pixels / (2.0 * math.tan(fov / 2.0))


def _prepare_pixel_weights(pixel_weights, width, height, device):
    weights = pixel_weights.detach()
    if weights.dim() == 3:
        weights = weights.mean(dim=0)
    elif weights.dim() == 4:
        weights = weights.mean(dim=(0, 1))
    if weights.shape != (height, width):
        raise ValueError(
            f"Expected pixel_weights with spatial shape {(height, width)}, "
            f"got {tuple(weights.shape)}"
        )
    return weights.to(device=device, dtype=torch.float32).contiguous()


def _expand_packed_stat(meta, key, ids, num_points, device, dtype=torch.float32, reduce_radii=False):
    values = meta.get(key, None)
    out = torch.zeros(num_points, device=device, dtype=dtype)
    if values is None or ids is None:
        return out
    values = values.detach()
    if reduce_radii and values.dim() > 1:
        values = values.max(dim=-1).values
    out[ids] = values.to(device=device, dtype=dtype)
    return out


def _compute_camera_depths(means, viewmat):
    ones = torch.ones((means.shape[0], 1), device=means.device, dtype=means.dtype)
    homog = torch.cat((means, ones), dim=1)
    camera_space = homog @ viewmat.transpose(0, 1)
    return camera_space[:, 2]


def _compute_exact_taming_stats(meta, pixel_weights, num_points, width, height, device):
    if pixel_weights is None:
        return {
            "accum_weights": torch.zeros(num_points, device=device, dtype=torch.float32),
            "accum_count": torch.zeros(num_points, device=device, dtype=torch.int32),
            "accum_blend": torch.zeros(num_points, device=device, dtype=torch.float32),
            "accum_dist": torch.zeros(num_points, device=device, dtype=torch.float32),
        }

    required = ("means2d", "conics", "opacities", "isect_offsets", "flatten_ids")
    missing = [key for key in required if key not in meta or meta[key] is None]
    if missing:
        raise RuntimeError(
            "Exact Taming stats require gsplat intersection metadata; "
            f"missing keys: {missing}"
        )

    packed_ids = meta.get("gaussian_ids", None)
    if packed_ids is None:
        raise RuntimeError("Exact Taming stats currently require gsplat packed metadata.")

    weights = _prepare_pixel_weights(pixel_weights, width, height, device)
    means2d = meta["means2d"].detach().to(device=device, dtype=torch.float32).contiguous()
    conics = meta["conics"].detach().to(device=device, dtype=torch.float32).contiguous()
    opacities = meta["opacities"].detach().to(device=device, dtype=torch.float32).contiguous()
    isect_offsets = meta["isect_offsets"].detach().contiguous()
    flatten_ids = meta["flatten_ids"].detach().contiguous()

    if means2d.numel() == 0 or flatten_ids.numel() == 0:
        return {
            "accum_weights": torch.zeros(num_points, device=device, dtype=torch.float32),
            "accum_count": torch.zeros(num_points, device=device, dtype=torch.int32),
            "accum_blend": torch.zeros(num_points, device=device, dtype=torch.float32),
            "accum_dist": torch.zeros(num_points, device=device, dtype=torch.float32),
        }

    transmittances = torch.ones((1, height, width), device=device, dtype=torch.float32)
    local_ids, pixel_ids, image_ids = rasterize_to_indices_in_range(
        0,
        2**31 - 1,
        transmittances,
        means2d[None],
        conics[None],
        opacities[None],
        width,
        height,
        int(meta["tile_size"]),
        isect_offsets,
        flatten_ids,
    )
    if local_ids.numel() == 0:
        return {
            "accum_weights": torch.zeros(num_points, device=device, dtype=torch.float32),
            "accum_count": torch.zeros(num_points, device=device, dtype=torch.int32),
            "accum_blend": torch.zeros(num_points, device=device, dtype=torch.float32),
            "accum_dist": torch.zeros(num_points, device=device, dtype=torch.float32),
        }
    if torch.any(image_ids != 0):
        raise RuntimeError("Exact Taming stats expected a single rendered camera.")

    pixel_ids = pixel_ids.long()
    local_ids = local_ids.long()

    pix_x = (pixel_ids % width).to(dtype=torch.float32)
    pix_y = torch.div(pixel_ids, width, rounding_mode="floor").to(dtype=torch.float32)
    xy = means2d[local_ids]
    d_x = xy[:, 0] - pix_x
    d_y = xy[:, 1] - pix_y
    con = conics[local_ids]
    power = -0.5 * (con[:, 0] * d_x.square() + con[:, 2] * d_y.square()) - con[:, 1] * d_x * d_y
    alpha = torch.clamp(opacities[local_ids] * torch.exp(power), max=0.99)

    # rasterize_to_indices_in_range returns the same accepted Gaussian/pixel
    # intersections used by gsplat rasterization. Reconstruct the pre-hit
    # transmittance per pixel in raster order to match Taming's T * alpha blend.
    order = torch.argsort(pixel_ids, stable=True)
    pixel_sorted = pixel_ids[order]
    alpha_sorted = alpha[order]
    log_survival = torch.log1p(-alpha_sorted.clamp(max=0.999999))
    cumulative = torch.cumsum(log_survival, dim=0)
    group_start = torch.ones_like(pixel_sorted, dtype=torch.bool)
    group_start[1:] = pixel_sorted[1:] != pixel_sorted[:-1]
    group_ids = torch.cumsum(group_start.to(torch.int64), dim=0) - 1
    start_positions = torch.nonzero(group_start, as_tuple=False).squeeze(-1)
    start_cum_before = cumulative[start_positions] - log_survival[start_positions]
    exclusive = cumulative - start_cum_before[group_ids] - log_survival
    trans_sorted = torch.exp(exclusive)
    trans = torch.empty_like(trans_sorted)
    trans[order] = trans_sorted
    blend = trans * alpha

    global_ids = packed_ids[local_ids].long()
    accum_weights = torch.zeros(num_points, device=device, dtype=torch.float32)
    accum_weights.scatter_add_(0, global_ids, weights.flatten()[pixel_ids])
    accum_count = torch.zeros(num_points, device=device, dtype=torch.int32)
    accum_count.scatter_add_(0, global_ids, torch.ones_like(global_ids, dtype=torch.int32))
    accum_blend = torch.zeros(num_points, device=device, dtype=torch.float32)
    accum_blend.scatter_add_(0, global_ids, blend.to(dtype=torch.float32))
    accum_dist = torch.zeros(num_points, device=device, dtype=torch.float32)
    accum_dist.scatter_add_(0, global_ids, torch.sqrt(d_x.square() + d_y.square()).to(dtype=torch.float32))

    return {
        "accum_weights": accum_weights,
        "accum_count": accum_count,
        "accum_blend": accum_blend,
        "accum_dist": accum_dist,
    }


def render(viewpoint_camera, pc, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, override_color=None, pixel_weights=None,
           return_taming_stats=False, update_sh_rest=True):
    """
    Render the scene using gsplat (ROCm-compatible backend).

    Background tensor (bg_color) must be on GPU!
    """
    device = pc.get_xyz.device
    W = int(viewpoint_camera.image_width)
    H = int(viewpoint_camera.image_height)

    means = pc.get_xyz.contiguous()
    scales = pc.get_scaling.contiguous()
    sparse_grad = bool(getattr(pipe, "gsplat_sparse_grad", False))
    # gsplat sparse gradients cannot backpropagate through PyTorch's dense
    # quaternion normalization. The accelerated path keeps raw rotations
    # normalized after optimizer steps and passes them directly.
    quats = (pc._rotation if sparse_grad else pc.get_rotation).contiguous()
    opacities = pc.get_opacity.squeeze(-1).contiguous()

    if scaling_modifier != 1.0:
        scales = scales * scaling_modifier

    # Camera convention mapping
    # -------------------------------------------------------------------------
    # Inria's codebase stores world_view_transform as the *transpose* of the
    # actual world-to-camera matrix. gsplat expects the actual W2C matrix.
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).to(device, non_blocking=True).contiguous()

    # Build pinhole intrinsics from FoV
    fx = _fov2focal(viewpoint_camera.FoVx, W)
    fy = _fov2focal(viewpoint_camera.FoVy, H)
    cx = W / 2.0
    cy = H / 2.0
    K = torch.tensor(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        device=device, dtype=torch.float32,
    ).contiguous()

    viewmats = viewmat[None].contiguous()
    Ks = K[None].contiguous()

    # Background: gsplat expects flat [C] for packed=True
    bg = bg_color.contiguous() if bg_color is not None else None

    # Colors / SH handling
    # -------------------------------------------------------------------------
    # Python-side SH evaluation to bypass gsplat ROCm SH backward kernel.
    # Optimized: avoid pc.get_features round-trip, slice only active coeffs,
    # and special-case degree 0 to skip eval_sh entirely.
    from utils.sh_utils import eval_sh

    if pc.active_sh_degree == 0:
        colors = torch.clamp(pc._features_dc[:, 0, :] + 0.5, 0.0, 1.0).contiguous()
    else:
        active_coeffs = (pc.active_sh_degree + 1) ** 2
        features_dc = pc._features_dc.transpose(1, 2)  # [N, 3, 1]
        features_rest = pc._features_rest[:, :active_coeffs - 1, :].transpose(1, 2)
        if not update_sh_rest:
            features_rest = features_rest.detach()
        shs_view = torch.cat((features_dc, features_rest), dim=2).contiguous()

        cam_center = viewpoint_camera.camera_center.to(device=device, dtype=means.dtype).view(1, 3)
        dir_pp = means - cam_center
        dir_pp_normalized = torch.nn.functional.normalize(dir_pp, dim=1, eps=1e-8)

        colors = torch.clamp(
            eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized) + 0.5,
            0.0,
            1.0,
        ).contiguous()

    sh_degree = None

    # tile_size=8 causes NaN gradients on ROCm/gfx1151 with wave32-patched gsplat.
    # Default is 16. See docs/ROCM_PODMAN.md.
    tile_size = getattr(pipe, "tile_size", 16)
    render_colors, render_alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=W,
        height=H,
        sh_degree=None,
        packed=True,
        tile_size=tile_size,
        backgrounds=bg,
        render_mode="RGB",
        sparse_grad=sparse_grad,
        absgrad=False,
        rasterize_mode="classic",
    )

    # gsplat returns [C, H, W, 3]; Inria backend returns [3, H, W]
    image = render_colors[0].permute(2, 0, 1).contiguous()

    # Expand packed metadata back to full [N] arrays for compatibility
    N = means.shape[0]
    radii = torch.zeros(N, device=device, dtype=torch.int32)
    means2d = torch.zeros(N, 2, device=device, dtype=torch.float32)
    is_used = torch.zeros(N, device=device, dtype=torch.bool)

    if "gaussian_ids" in meta and meta["gaussian_ids"] is not None:
        ids = meta["gaussian_ids"]
        tile_radii = meta["radii"]  # [M, 2] — (x_radius, y_radius) per gaussian
        # Inria backend returns a single scalar radius = max of x/y extent
        radii_scalar = tile_radii.max(dim=-1).values.to(torch.int32)  # [M]
        radii[ids] = radii_scalar
        means2d[ids] = meta["means2d"]
        is_used[ids] = True
    else:
        radii = meta.get("radii", torch.zeros(N, device=device, dtype=torch.int32))
        means2d = meta.get("means2d", torch.zeros(N, 2, device=device, dtype=torch.float32))
        is_used = radii > 0

    # Old code creates a dummy screenspace_points tensor for gradient hooks.
    # The MCMC training loop does not use it, but we keep it for API parity.
    screenspace_points = torch.zeros_like(means, dtype=means.dtype, requires_grad=True, device=device)
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    visibility_filter = radii > 0

    result = {
        "render": image,
        "viewspace_points": screenspace_points,
        "visibility_filter": visibility_filter,
        "radii": radii,
        "is_used": is_used,
        "alpha": render_alphas[0],
        "meta": meta,
        "sparse_grad": sparse_grad,
    }

    if return_taming_stats or pixel_weights is not None:
        radii_float = radii.to(dtype=torch.float32)
        depths = meta.get("depths", None)
        if depths is None:
            depths = meta.get("depth", None)
        if depths is not None and depths.shape[0] != N:
            full_depths = torch.zeros(N, device=device, dtype=torch.float32)
            if "gaussian_ids" in meta and meta["gaussian_ids"] is not None:
                full_depths[meta["gaussian_ids"]] = depths.detach().to(dtype=torch.float32)
            depths = full_depths
        elif depths is None:
            depths = _compute_camera_depths(means.detach(), viewmat.detach()).to(dtype=torch.float32)
        else:
            depths = depths.detach().to(dtype=torch.float32)

        exact_stats = _compute_exact_taming_stats(meta, pixel_weights, N, W, H, device)
        result.update({
            "gaussian_depths": depths,
            "gaussian_radii": radii_float,
            **exact_stats,
            "taming_stats_exact": True,
            "taming_stats_backend": "gsplat_intersections",
            "taming_stats_approximation": None,
        })

    return result
