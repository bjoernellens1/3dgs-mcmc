import torch

try:
    from gsplat import rasterize_to_indices_in_range
    HAS_EXACT_TAMING_STATS = True
    TAMING_STATS_BACKEND = "gsplat_public"
except Exception:
    try:
        from gsplat.cuda._wrapper import rasterize_to_indices_in_range
        HAS_EXACT_TAMING_STATS = True
        TAMING_STATS_BACKEND = "gsplat_cuda_wrapper"
    except Exception:
        rasterize_to_indices_in_range = None
        HAS_EXACT_TAMING_STATS = False
        TAMING_STATS_BACKEND = "unavailable"


def prepare_pixel_weights(pixel_weights, width, height, device):
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


def compute_camera_depths(means, viewmat):
    ones = torch.ones((means.shape[0], 1), device=means.device, dtype=means.dtype)
    homog = torch.cat((means, ones), dim=1)
    camera_space = homog @ viewmat.transpose(0, 1)
    return camera_space[:, 2]


def empty_taming_stats(num_points, device):
    return {
        "accum_weights": torch.zeros(num_points, device=device, dtype=torch.float32),
        "accum_count": torch.zeros(num_points, device=device, dtype=torch.int32),
        "accum_blend": torch.zeros(num_points, device=device, dtype=torch.float32),
        "accum_dist": torch.zeros(num_points, device=device, dtype=torch.float32),
    }


def compute_exact_taming_stats(meta, pixel_weights, num_points, width, height, device):
    if not HAS_EXACT_TAMING_STATS:
        raise RuntimeError(
            "Exact Taming stats require gsplat.rasterize_to_indices_in_range or "
            "gsplat.cuda._wrapper.rasterize_to_indices_in_range. Disable Taming "
            "scoring or install a gsplat build that exposes this utility."
        )

    if pixel_weights is None:
        return empty_taming_stats(num_points, device)

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

    weights = prepare_pixel_weights(pixel_weights, width, height, device)
    means2d = meta["means2d"].detach().to(device=device, dtype=torch.float32).contiguous()
    conics = meta["conics"].detach().to(device=device, dtype=torch.float32).contiguous()
    opacities = meta["opacities"].detach().to(device=device, dtype=torch.float32).contiguous()
    isect_offsets = meta["isect_offsets"].detach().contiguous()
    flatten_ids = meta["flatten_ids"].detach().contiguous()

    if means2d.numel() == 0 or flatten_ids.numel() == 0:
        return empty_taming_stats(num_points, device)

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
        return empty_taming_stats(num_points, device)
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
