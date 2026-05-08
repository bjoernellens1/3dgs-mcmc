"""
Compiled kernel wrappers for tensor-only helpers.

Provides eager + torch.compile variants with graceful ROCm fallback.
Compile only after ``compile_after_iter`` to avoid growth-phase churn.

Defaults to **off**. Enable per-kernel via CLI flags (e.g. ``--compile_sh``).
"""
import torch

# ---------------------------------------------------------------------------
# Compilation availability & ROCm detection
# ---------------------------------------------------------------------------
_COMPILE_AVAILABLE = hasattr(torch, "compile")
_IS_ROCM = False
try:
    _IS_ROCM = torch.version.hip is not None
except Exception:
    pass


def _compile_or_eager(fn, enabled=True, mode="reduce-overhead", dynamic=True):
    """
    Try ``torch.compile(fn)``; on any failure return the eager function.
    """
    if not _COMPILE_AVAILABLE or not enabled:
        return fn
    try:
        compiled = torch.compile(fn, mode=mode, dynamic=dynamic, fullgraph=False)
        return compiled
    except Exception as exc:
        print(
            f"[compiled-kernels] torch.compile failed for {fn.__name__}: {exc}\n"
            f"[compiled-kernels] Falling back to eager."
        )
        return fn


# ---------------------------------------------------------------------------
# Registry that switches eager -> compiled after a configurable iteration.
# ---------------------------------------------------------------------------
class _CompiledKernelRegistry:
    def __init__(self):
        self._eager = {}
        self._compiled = {}
        self._active = {}
        self._enabled_globally = True
        self._mode = "reduce-overhead"
        self._dynamic = True
        self._compile_after_iter = 0
        self._current_iter = 0
        self._compiled_once = set()

    def configure(self, enabled=True, mode="reduce-overhead", dynamic=True, compile_after_iter=0):
        self._enabled_globally = enabled
        self._mode = mode
        self._dynamic = dynamic
        self._compile_after_iter = compile_after_iter

    def set_iteration(self, iteration):
        self._current_iter = iteration

    def register(self, name, eager_fn, enabled=True):
        self._eager[name] = eager_fn
        self._active[name] = enabled
        # compiled version is created lazily on first use after compile_after_iter
        self._compiled.pop(name, None)
        self._compiled_once.discard(name)
        return self

    def _get_compiled(self, name):
        if name not in self._compiled:
            eager_fn = self._eager[name]
            compiled = _compile_or_eager(
                eager_fn,
                enabled=self._active.get(name, True) and self._enabled_globally,
                mode=self._mode,
                dynamic=self._dynamic,
            )
            self._compiled[name] = compiled
            if compiled is not eager_fn and name not in self._compiled_once:
                self._compiled_once.add(name)
                print(
                    f"[compiled-kernels] Compiled '{name}' "
                    f"(mode={self._mode}, dynamic={self._dynamic})"
                )
        return self._compiled[name]

    def __call__(self, name, *args, **kwargs):
        if self._current_iter < self._compile_after_iter:
            return self._eager[name](*args, **kwargs)
        try:
            return self._get_compiled(name)(*args, **kwargs)
        except Exception as exc:
            print(
                f"[compiled-kernels] runtime failure in '{name}': {exc}\n"
                f"[compiled-kernels] Disabling '{name}', falling back to eager.",
                flush=True,
            )
            self._compiled[name] = self._eager[name]
            self._active[name] = False
            return self._eager[name](*args, **kwargs)

    def eager(self, name, *args, **kwargs):
        """Force eager execution (useful for debugging / graph breaks)."""
        return self._eager[name](*args, **kwargs)


REGISTRY = _CompiledKernelRegistry()


# ---------------------------------------------------------------------------
# Eager kernel implementations
# ---------------------------------------------------------------------------

def _sh_to_rgb_deg1(features_dc, features_rest, dir_pp_normalized):
    """
    Full SH-to-RGB for degree 1.
    Takes full [N, 15, 3] features_rest, slices internally to degree-1 coeffs.
    """
    from utils.sh_utils import eval_sh
    dc = features_dc.transpose(1, 2)                         # [N, 3, 1]
    rest = features_rest[:, :3, :].transpose(1, 2)           # [N, 3, 3]
    shs = torch.cat((dc, rest), dim=2).contiguous()          # [N, 3, 4]
    colors = eval_sh(1, shs, dir_pp_normalized)
    return torch.clamp(colors + 0.5, 0.0, 1.0).contiguous()


def _sh_to_rgb_deg2(features_dc, features_rest, dir_pp_normalized):
    """
    Full SH-to-RGB for degree 2.
    Takes full [N, 15, 3] features_rest, slices internally to degree-2 coeffs.
    """
    from utils.sh_utils import eval_sh
    dc = features_dc.transpose(1, 2)
    rest = features_rest[:, :8, :].transpose(1, 2)
    shs = torch.cat((dc, rest), dim=2).contiguous()
    colors = eval_sh(2, shs, dir_pp_normalized)
    return torch.clamp(colors + 0.5, 0.0, 1.0).contiguous()


def _sh_to_rgb_deg3(features_dc, features_rest, dir_pp_normalized):
    """
    Full SH-to-RGB for degree 3.
    Takes full [N, 15, 3] features_rest, uses all 15 rest coeffs.
    """
    from utils.sh_utils import eval_sh
    dc = features_dc.transpose(1, 2)
    rest = features_rest[:, :15, :].transpose(1, 2)
    shs = torch.cat((dc, rest), dim=2).contiguous()
    colors = eval_sh(3, shs, dir_pp_normalized)
    return torch.clamp(colors + 0.5, 0.0, 1.0).contiguous()


def _effective_count_core(opacities, dead_threshold, softness):
    """
    Core of compute_effective_count.
    Args:
        opacities: [N, 1] raw opacity params
    Returns:
        scalar N_eff
    """
    alpha = torch.sigmoid(opacities).squeeze(-1)
    return torch.sigmoid((alpha - dead_threshold) / softness).sum()


def _effective_count_loss_core(opacities, target, target_scale, dead_threshold, softness):
    """
    Core of compute_effective_count_loss.
    Returns (loss, n_eff) tuple — avoids recomputing N_eff for logging.
    """
    alpha = torch.sigmoid(opacities).squeeze(-1)
    n_eff = torch.sigmoid((alpha - dead_threshold) / softness).sum()
    loss = ((n_eff - target) / target_scale) ** 2
    return loss, n_eff


def _opacity_entropy_core(opacities):
    """
    Core of compute_opacity_entropy_loss.
    Args:
        opacities: [N, 1] raw
    Returns:
        scalar mean entropy
    """
    alpha = torch.sigmoid(opacities).squeeze(-1)
    eps = 1e-6
    entropy = -alpha * torch.log(alpha + eps) - (1.0 - alpha) * torch.log(1.0 - alpha + eps)
    return entropy.mean()


def _utility_core(
    alpha, v, support, xyz_grad, opacity_grad, scale_grad,
    max_scale, w_alpha, w_vis, w_support, w_grad, w_scale,
    w_dead, beta_opacity, beta_scale, alpha_dead,
):
    """
    Pure-tensor core of compute_gaussian_utility.
    All inputs are 1-D tensors of length N (or scalars for weights).
    Returns:
        [N] utility scores
    """
    norm_grad = xyz_grad + beta_opacity * opacity_grad + beta_scale * scale_grad
    scale_penalty = torch.clamp(max_scale - 0.1, min=0.0) ** 2
    dead_penalty = (alpha < alpha_dead).float()
    return (
        w_alpha * alpha + w_vis * v + w_support * support + w_grad * norm_grad
        - w_scale * scale_penalty - w_dead * dead_penalty
    )


def _active_reg_core(opacity, scaling, w_opacity, w_scale):
    """
    Active-set L1 regularizer core.
    NOTE: Always runs eager — active-set shapes change every iteration.
    """
    return w_opacity * torch.abs(opacity).mean() + w_scale * torch.abs(scaling).mean()


# ---------------------------------------------------------------------------
# Public API — functions that dispatch through the registry
# ---------------------------------------------------------------------------

def sh_to_rgb(deg, features_dc, features_rest, dir_pp_normalized):
    """
    Full SH-to-RGB pipeline (transpose → slice → cat → eval_sh → clamp → contiguous).

    Features rest is always passed as full [N, 15, 3] regardless of active degree;
    the degree-specific wrapper slices internally. This keeps input shapes stable
    across SH degree changes, avoiding recompilation.

    Args:
        deg: SH degree (1-3). Degree 0 callers should use the inline path.
        features_dc: [N, 1, 3] raw DC features
        features_rest: [N, 15, 3] full rest features (caller may have detached)
        dir_pp_normalized: [N, 3] unit view directions
    Returns:
        [N, 3] clamped RGB colors
    """
    if deg == 1:
        return REGISTRY("sh_to_rgb_deg1", features_dc, features_rest, dir_pp_normalized)
    elif deg == 2:
        return REGISTRY("sh_to_rgb_deg2", features_dc, features_rest, dir_pp_normalized)
    elif deg == 3:
        return REGISTRY("sh_to_rgb_deg3", features_dc, features_rest, dir_pp_normalized)
    else:
        raise ValueError(f"Unsupported SH degree for compiled path: {deg}")


def effective_count_core(opacities, dead_threshold=0.005, softness=0.002):
    return REGISTRY("effective_count_core", opacities, dead_threshold, softness)


def effective_count_loss_core(opacities, target, target_scale, dead_threshold=0.005, softness=0.002):
    """Returns (loss, n_eff) tuple."""
    return REGISTRY(
        "effective_count_loss_core", opacities, target, target_scale,
        dead_threshold, softness,
    )


def opacity_entropy_core(opacities):
    return REGISTRY("opacity_entropy_core", opacities)


def utility_core(
    alpha, v, support, xyz_grad, opacity_grad, scale_grad,
    max_scale, w_alpha=1.0, w_vis=2.0, w_support=2.0, w_grad=3.0,
    w_scale=0.5, w_dead=1.0, beta_opacity=1.0, beta_scale=0.5, alpha_dead=0.005,
):
    return REGISTRY(
        "utility_core", alpha, v, support, xyz_grad, opacity_grad, scale_grad,
        max_scale, w_alpha, w_vis, w_support, w_grad, w_scale, w_dead,
        beta_opacity, beta_scale, alpha_dead,
    )


def active_reg_core(opacity, scaling, w_opacity=0.01, w_scale=0.01):
    return REGISTRY("active_reg_core", opacity, scaling, w_opacity, w_scale)


# ---------------------------------------------------------------------------
# Setup / configuration helpers
# ---------------------------------------------------------------------------

def configure_torch_compile(args):
    """
    Call once at training start with the parsed arg namespace.

    Reads ``compile_mode``, ``compile_dynamic``, ``compile_after_iter``,
    and per-kernel flags.

    Kernels are **always** registered so that eager fallback works even
    when ``compile_mode=off``.
    """
    # --- Register all kernels (eager fallback always needed) ---
    REGISTRY.register(
        "sh_to_rgb_deg1", _sh_to_rgb_deg1,
        enabled=bool(getattr(args, "compile_sh", False)),
    )
    REGISTRY.register(
        "sh_to_rgb_deg2", _sh_to_rgb_deg2,
        enabled=bool(getattr(args, "compile_sh", False)),
    )
    REGISTRY.register(
        "sh_to_rgb_deg3", _sh_to_rgb_deg3,
        enabled=bool(getattr(args, "compile_sh", False)),
    )
    # Tiny helpers — default-off, opt-in for ablation
    REGISTRY.register(
        "effective_count_core", _effective_count_core,
        enabled=bool(getattr(args, "compile_energy", False)),
    )
    REGISTRY.register(
        "effective_count_loss_core", _effective_count_loss_core,
        enabled=bool(getattr(args, "compile_energy", False)),
    )
    REGISTRY.register(
        "opacity_entropy_core", _opacity_entropy_core,
        enabled=bool(getattr(args, "compile_energy", False)),
    )
    REGISTRY.register(
        "utility_core", _utility_core,
        enabled=bool(getattr(args, "compile_utility", False)),
    )
    # active_reg always eager — shape changes every iteration
    REGISTRY.register("active_reg_core", _active_reg_core, enabled=False)

    mode = getattr(args, "compile_mode", "off")
    if mode == "off":
        REGISTRY.configure(enabled=False)
        print("[compiled-kernels] torch.compile disabled (--compile_mode=off)")
        return

    # ROCm warning
    if _IS_ROCM:
        print(
            "[compiled-kernels] WARNING: torch.compile on ROCm is experimental "
            "for this workload and may be slower. Use for ablation only.",
            flush=True,
        )

    # Dynamo/Inductor cache limits to prevent unbounded graph growth
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.cache_size_limit = 16
        torch._dynamo.config.accumulated_cache_size_limit = 64
    except Exception:
        pass

    dynamic = bool(getattr(args, "compile_dynamic", True))
    after_iter = int(getattr(args, "compile_after_iter", 12000))
    REGISTRY.configure(enabled=True, mode=mode, dynamic=dynamic, compile_after_iter=after_iter)
    print(
        f"[compiled-kernels] mode={mode} dynamic={dynamic} after_iter={after_iter} "
        f"rocm={_IS_ROCM}"
    )


def set_compile_iteration(iteration):
    """Call every iteration so the registry knows when to switch eager -> compiled."""
    REGISTRY.set_iteration(iteration)
