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
