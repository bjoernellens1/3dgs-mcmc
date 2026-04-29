import math
import torch
from gsplat.rendering import rasterization
def _fov2focal(fov, pixels):
    return pixels / (2.0 * math.tan(fov / 2.0))


def render(viewpoint_camera, pc, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, override_color=None):
    """
    Render the scene using gsplat (ROCm-compatible backend).

    Background tensor (bg_color) must be on GPU!
    """
    device = pc.get_xyz.device
    W = int(viewpoint_camera.image_width)
    H = int(viewpoint_camera.image_height)

    means = pc.get_xyz.contiguous()
    scales = pc.get_scaling.contiguous()
    quats = pc.get_rotation.contiguous()
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

    # Background: gsplat expects flat [C] (broadcast over all pixels)
    bg = bg_color.contiguous() if bg_color is not None else None

    # Colors / SH handling
    # -------------------------------------------------------------------------
    # pc.get_features returns [N, K, 3] where K = (max_sh_degree+1)**2.
    # gsplat expects the same layout when sh_degree is provided.
    colors = pc.get_features.contiguous()
    sh_degree = pc.active_sh_degree

    if override_color is not None:
        if override_color.dim() == 1:
            override_color = override_color.unsqueeze(0).expand(means.shape[0], -1)
        colors = override_color.contiguous()
        sh_degree = None
    elif pipe.convert_SHs_python:
        from utils.sh_utils import eval_sh
        shs_view = colors.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_xyz.shape[0], 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        colors_precomp = torch.clamp_min(
            eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized) + 0.5, 0.0
        )
        colors = colors_precomp.contiguous()
        sh_degree = None

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
        sh_degree=sh_degree,
        packed=True,
        tile_size=8,  # 8 performs better on AMD GPUs (ROCm/gsplat default)
        backgrounds=bg,
        render_mode="RGB",
        sparse_grad=False,
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

    return {
        "render": image,
        "viewspace_points": screenspace_points,
        "visibility_filter": visibility_filter,
        "radii": radii,
        "is_used": is_used,
        "alpha": render_alphas[0],
        "meta": meta,
    }
