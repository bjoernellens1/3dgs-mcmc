import math
import torch
from gsplat.rendering import rasterization
from utils.taming_stats import (
    TAMING_STATS_BACKEND,
    compute_camera_depths,
    compute_exact_taming_stats,
)


def _fov2focal(fov, pixels):
    return pixels / (2.0 * math.tan(fov / 2.0))


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

def _camera_tensors(viewpoint_camera, device):
    gsplat = getattr(viewpoint_camera, "gsplat", None)
    if gsplat is not None:
        target = torch.device(device)
        if (
            gsplat.viewmat.device == target
            and gsplat.K.device == target
            and gsplat.camera_center.device == target
            and gsplat.viewmat.dtype == torch.float32
            and gsplat.K.dtype == torch.float32
            and gsplat.camera_center.dtype == torch.float32
            and gsplat.viewmat.is_contiguous()
            and gsplat.K.is_contiguous()
            and gsplat.camera_center.is_contiguous()
        ):
            return gsplat.viewmat, gsplat.K, gsplat.camera_center
        viewmat = gsplat.viewmat.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
        K = gsplat.K.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
        cam_center = gsplat.camera_center.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
        return viewmat, K, cam_center

    if hasattr(viewpoint_camera, "_gsplat_viewmat"):
        viewmat = viewpoint_camera._gsplat_viewmat.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
        K = viewpoint_camera._gsplat_K.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
        cam_center = viewpoint_camera._gsplat_camera_center.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
        return viewmat, K, cam_center

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).to(device, non_blocking=True).contiguous()
    W = int(viewpoint_camera.image_width)
    H = int(viewpoint_camera.image_height)
    fx = getattr(viewpoint_camera, "fx", None)
    fy = getattr(viewpoint_camera, "fy", None)
    cx = getattr(viewpoint_camera, "cx", None)
    cy = getattr(viewpoint_camera, "cy", None)
    fx = float(fx) if fx is not None else _fov2focal(viewpoint_camera.FoVx, W)
    fy = float(fy) if fy is not None else _fov2focal(viewpoint_camera.FoVy, H)
    cx = float(cx) if cx is not None else W / 2.0
    cy = float(cy) if cy is not None else H / 2.0
    K = torch.tensor(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        device=device, dtype=torch.float32,
    ).contiguous()
    cam_center = viewpoint_camera.camera_center.to(device=device, dtype=torch.float32).view(1, 3).contiguous()
    return viewmat, K, cam_center


def _python_sh_colors(pc, means, cam_center, sparse_grad, pipe, update_sh_rest, compiled):
    if pc.active_sh_degree == 0:
        return torch.clamp(pc._features_dc[:, 0, :] + 0.5, 0.0, 1.0).contiguous()

    features_rest = pc._features_rest
    if not update_sh_rest:
        features_rest = features_rest.detach()

    _cc = cam_center.to(dtype=means.dtype) if cam_center.dtype != means.dtype else cam_center
    dir_source = means.detach() if (sparse_grad and getattr(pipe, "sparse_mode_detach_sh_dir", True)) else means
    dir_pp = dir_source - _cc
    dir_pp_normalized = torch.nn.functional.normalize(dir_pp, dim=1, eps=1e-8)

    from utils.compiled_kernels import sh_to_rgb, sh_to_rgb_eager
    if compiled:
        return sh_to_rgb(pc.active_sh_degree, pc._features_dc, features_rest, dir_pp_normalized)
    return sh_to_rgb_eager(pc.active_sh_degree, pc._features_dc, features_rest, dir_pp_normalized)


def _gsplat_sh_coeffs(pc, update_sh_rest):
    features_rest = pc._features_rest if update_sh_rest else pc._features_rest.detach()
    return torch.cat((pc._features_dc, features_rest), dim=1).contiguous()


def render(viewpoint_camera, pc, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, override_color=None, pixel_weights=None,
           return_taming_stats=False, update_sh_rest=True, render_depth=False):
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
    #
    # Use cached camera tensors prepared by scene.cameras.prepare_camera_for_render
    # and keep a fallback for MiniCam / interactive paths.
    viewmat, K, cam_center = _camera_tensors(viewpoint_camera, device)

    viewmats = viewmat[None].contiguous()
    Ks = K[None].contiguous()

    # Background handling.
    # RGB mode: pass flat [3] tensor — gsplat packed mode accepts this.
    # RGB+D mode: gsplat's Python-level code does cat([backgrounds, zeros([C,1])],
    # dim=-1) which requires 2D input, but the CUDA kernel then asserts 1D. These
    # two requirements are contradictory in packed+C=1 mode (gsplat bug). Workaround:
    # pass bg=None for the rasterization call and manually composite afterward.
    if render_depth:
        effective_render_mode = "RGB+D"
        bg = None  # composited manually after rasterization
    else:
        effective_render_mode = getattr(pipe, "render_mode", "RGB")
        bg = bg_color.contiguous() if bg_color is not None else None

    # Colors / SH handling
    # -------------------------------------------------------------------------
    # Python-side SH evaluation to bypass gsplat ROCm SH backward kernel.
    # Optimized: avoid pc.get_features round-trip, slice only active coeffs,
    # and special-case degree 0 to skip eval_sh entirely.
    # For deg >= 1, the full SH pipeline is dispatched through the compile
    # registry (utils.compiled_kernels.sh_to_rgb) for potential torch.compile
    # acceleration. Features rest is passed as full [N, 15, 3] so that function
    # input shapes remain stable across SH degree changes.
    sh_backend = str(getattr(pipe, "sh_backend", "python")).lower()
    if override_color is not None:
        colors = override_color.contiguous()
        sh_degree = None
    elif sh_backend == "gsplat":
        colors = _gsplat_sh_coeffs(pc, update_sh_rest=update_sh_rest)
        sh_degree = pc.active_sh_degree
    else:
        if sh_backend not in {"python", "compiled_python"}:
            raise ValueError(
                f"Unsupported SH backend '{sh_backend}'. "
                "Expected 'python', 'compiled_python', or 'gsplat'."
            )
        colors = _python_sh_colors(
            pc, means, cam_center, sparse_grad, pipe, update_sh_rest,
            compiled=sh_backend == "compiled_python",
        )
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
        near_plane=getattr(pipe, "near_plane", 0.01),
        far_plane=getattr(pipe, "far_plane", 1e10),
        radius_clip=getattr(pipe, "radius_clip", 0.0),
        eps2d=getattr(pipe, "eps2d", 0.3),
        sh_degree=sh_degree,
        packed=True,
        tile_size=tile_size,
        backgrounds=bg,
        render_mode=effective_render_mode,
        sparse_grad=sparse_grad,
        absgrad=getattr(pipe, "absgrad", False),
        rasterize_mode=getattr(pipe, "rasterize_mode", "classic"),
    )

    # gsplat returns [C, H, W, channels]; Inria backend returns [channels, H, W]
    rendered = render_colors[0].permute(2, 0, 1).contiguous()
    image = rendered[:3] if rendered.shape[0] >= 3 else rendered

    # When render_depth=True we pass bg=None to avoid a gsplat packed+RGB+D bug.
    # Manually composite the background onto the RGB channels using the alpha map.
    if render_depth and bg_color is not None:
        alpha = render_alphas[0].permute(2, 0, 1)  # [1, H, W]
        image = image + bg_color[:, None, None] * (1.0 - alpha)
        rendered = torch.cat([image, rendered[3:]], dim=0)  # rebuild [4, H, W]

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
        "render_full": rendered,
        "viewspace_points": screenspace_points,
        "visibility_filter": visibility_filter,
        "radii": radii,
        "is_used": is_used,
        "alpha": render_alphas[0],
        "meta": meta,
        "sparse_grad": sparse_grad,
    }
    if render_depth:
        # rendered[3:4] is the alpha-composited depth in camera space (metres)
        result["rendered_depth"] = rendered[3:4]  # [1, H, W]

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
            depths = compute_camera_depths(means.detach(), viewmat.detach()).to(dtype=torch.float32)
        else:
            depths = depths.detach().to(dtype=torch.float32)

        exact_stats = compute_exact_taming_stats(meta, pixel_weights, N, W, H, device)
        result.update({
            "gaussian_depths": depths,
            "gaussian_radii": radii_float,
            **exact_stats,
            "taming_stats_exact": True,
            "taming_stats_backend": TAMING_STATS_BACKEND,
            "taming_stats_approximation": None,
        })

    return result
