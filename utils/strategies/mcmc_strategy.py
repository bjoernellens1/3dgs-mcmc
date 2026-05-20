import inspect

import torch

from utils.energy_mcmc import compute_dead_mask
from utils.general_utils import build_scaling_rotation
from utils.streaming_depth_lock import depth_lock_enabled, depth_locked_mask


def _dead_clause_breakdown(gaussians, sched, args):
    """Return per-clause dead counts for TB logging (non-exclusive — Gaussian may match several)."""
    alpha = gaussians.get_opacity.squeeze(-1)
    support = (
        gaussians.visibility_ema.squeeze(-1)
        if hasattr(gaussians, "visibility_ema")
        else torch.ones_like(alpha)
    )
    thresh = sched["dead_opacity_threshold"]
    sup_thresh = 0.01
    scale_kill = float(getattr(args, "scale_kill_threshold", 0.20))
    scale_mod = float(getattr(args, "scale_moderate_threshold", 0.05))
    aniso_ratio = float(getattr(args, "anisotropy_kill_ratio", 20.0))

    counts = {
        "very_low_opacity": int((alpha < thresh * 0.1).sum()),
        "low_opacity_low_support": int(((alpha < thresh) & (support < sup_thresh)).sum()),
    }
    if hasattr(gaussians, "get_scaling"):
        s = gaussians.get_scaling
        max_s = s.max(dim=-1).values
        min_s = s.min(dim=-1).values.clamp_min(1e-6)
        aniso = max_s / min_s
        counts["large_low_support"] = int(((max_s > scale_kill) & (support < sup_thresh)).sum())
        counts["moderate_transparent"] = int(
            ((max_s > scale_mod) & (alpha < thresh * 4) & (support < sup_thresh)).sum()
        )
        counts["anisotropic"] = int(
            ((aniso > aniso_ratio) & (alpha < thresh * 4) & (support < sup_thresh)).sum()
        ) if aniso_ratio > 0 else 0
    return counts


def _locked_depth_mask(gaussians, args):
    if not depth_lock_enabled(args):
        return None
    mask = depth_locked_mask(gaussians, args)
    if mask.shape[0] != gaussians.get_xyz.shape[0] or not mask.any():
        return None
    return mask


def _exclude_locked_from_mask(mask, gaussians, args):
    locked = _locked_depth_mask(gaussians, args)
    if locked is None or mask is None or mask.shape[0] != locked.shape[0]:
        return mask
    return mask & ~locked


class ScheduledMCMCStrategy:
    """Upstream-shaped adapter around this repo's scheduled Energy-MCMC logic."""

    def __init__(self, name):
        self.name = name
        self._next_reloc_iter = None
        self._next_grow_iter = None

    def initialize_state(self, **_kwargs):
        return {}

    def step_pre_backward(self, **_kwargs):
        return None

    def _due(self, name, iteration, interval, start_iter):
        attr = f"_next_{name}_iter"
        next_iter = getattr(self, attr, None)
        if next_iter is None:
            next_iter = int(start_iter) + max(1, int(interval))
            setattr(self, attr, next_iter)
        if int(iteration) < next_iter:
            return False
        setattr(self, attr, int(iteration) + max(1, int(interval)))
        return True

    def inject_noise(self, gaussians, args, xyz_lr, visible=None, sparse_active_set=False, iteration=0):
        if iteration > int(getattr(args, "mcmc_noise_stop_iter", getattr(args, "iterations", 30_000))):
            return
        with torch.no_grad():
            if sparse_active_set and visible is not None:
                noise_idx = visible.nonzero(as_tuple=True)[0]
            else:
                noise_idx = torch.arange(gaussians.get_xyz.shape[0], device=gaussians.get_xyz.device)
            locked = _locked_depth_mask(gaussians, args)
            if locked is not None:
                noise_idx = noise_idx[~locked[noise_idx]]
            if noise_idx.numel() == 0:
                return

            # Cap per-axis scale before building noise covariance.
            # Without this, needles (e.g. scale=(0.3, 0.02, 0.02)) have a 225×
            # eigenvalue ratio, so the noise direction is overwhelmingly biased
            # along the major axis — causing persistent axis-aligned drift.
            noise_max_scale = float(getattr(args, "noise_max_scale", 0.05))
            scales_for_noise = gaussians.get_scaling[noise_idx]
            if noise_max_scale > 0:
                scales_for_noise = scales_for_noise.clamp(max=noise_max_scale)
            L = build_scaling_rotation(
                scales_for_noise,
                gaussians.get_rotation[noise_idx],
            )
            actual_covariance = L @ L.transpose(1, 2)

            def op_sigmoid(x, k=100, x0=0.995):
                return 1 / (1 + torch.exp(-k * (x - x0)))

            noise = torch.randn_like(gaussians._xyz[noise_idx]) * (
                op_sigmoid(1 - gaussians.get_opacity[noise_idx])
            ) * args.noise_lr * xyz_lr
            noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
            max_step = float(getattr(args, "noise_max_step", 0.01))
            if max_step > 0:
                # Clamp per-Gaussian L2 displacement to max_step metres
                step_norm = noise.norm(dim=-1, keepdim=True).clamp_min(max_step)
                noise = noise * (max_step / step_norm)
            gaussians._xyz[noise_idx].add_(noise)

    def step_post_backward(
        self,
        gaussians,
        args,
        sched,
        iteration,
        utility=None,
        temperature=1.0,
        use_energy_mcmc=True,
        tb_writer=None,
        should_log_strategy=None,
        render_pkg=None,
        lr=None,
    ):
        if self.name not in {"mcmc", "hybrid"}:
            return

        should_log = should_log_strategy or (lambda _iteration: False)
        if use_energy_mcmc:
            with torch.no_grad():
                if sched["allow_relocation"] and self._due(
                    "reloc", iteration, sched["relocate_interval"], getattr(args, "densify_from_iter", 500)
                ):
                    dead_mask = compute_dead_mask(
                        gaussians=gaussians,
                        utility=utility,
                        opacity_threshold=sched["dead_opacity_threshold"],
                        utility_quantile=0.05,
                        scale_kill_threshold=float(getattr(args, "scale_kill_threshold", 0.20)),
                        scale_moderate_threshold=float(getattr(args, "scale_moderate_threshold", 0.05)),
                        anisotropy_kill_ratio=float(getattr(args, "anisotropy_kill_ratio", 20.0)),
                    )
                    if tb_writer:
                        for clause, cnt in _dead_clause_breakdown(gaussians, sched, args).items():
                            tb_writer.add_scalar(f"mcmc/dead_{clause}", cnt, iteration)
                    dead_mask = _exclude_locked_from_mask(dead_mask, gaussians, args)
                    dead_count = int(dead_mask.sum().item())
                    gaussians.relocate_gs_energy_guided(
                        dead_mask=dead_mask,
                        parent_scores=utility,
                        temperature=temperature,
                        exclude_parent_mask=_locked_depth_mask(gaussians, args),
                    )
                    self._log_reloc(
                        tb_writer, should_log, iteration, dead_count, sched,
                        gaussians.get_xyz.shape[0],
                    )

                if self.name == "mcmc" and sched["allow_growth"] and self._due(
                    "grow", iteration, sched["grow_interval"], getattr(args, "densify_from_iter", 500)
                ):
                    before = gaussians.get_xyz.shape[0]
                    added = gaussians.add_new_gs_energy_guided(
                        cap_max=args.cap_max,
                        growth_factor=sched["growth_factor"],
                        parent_scores=utility,
                        temperature=temperature,
                        exclude_parent_mask=_locked_depth_mask(gaussians, args),
                    )
                    self._log_growth(tb_writer, should_log, iteration, before, gaussians.get_xyz.shape[0], added, sched)
            return

        with torch.no_grad():
            if sched["allow_relocation"] and self._due(
                "reloc", iteration, sched["relocate_interval"], getattr(args, "densify_from_iter", 500)
            ):
                dead_mask = (gaussians.get_opacity <= sched["dead_opacity_threshold"]).squeeze(-1)
                dead_mask = _exclude_locked_from_mask(dead_mask, gaussians, args)
                dead_count = int(dead_mask.sum().item())
                gaussians.relocate_gs(dead_mask=dead_mask)
                self._log_reloc(tb_writer, should_log, iteration, dead_count, sched, gaussians.get_xyz.shape[0])

            if self.name == "mcmc" and sched["allow_growth"] and self._due(
                "grow", iteration, sched["grow_interval"], getattr(args, "densify_from_iter", 500)
            ):
                before = gaussians.get_xyz.shape[0]
                added = gaussians.add_new_gs(
                    cap_max=args.cap_max,
                    growth_factor=sched["growth_factor"],
                )
                self._log_growth(tb_writer, should_log, iteration, before, gaussians.get_xyz.shape[0], added, sched)

    def _log_reloc(self, tb_writer, should_log, iteration, dead_count, sched, count):
        if tb_writer:
            tb_writer.add_scalar("mcmc/dead_count", dead_count, iteration)
        if should_log(iteration):
            print(
                f"[mcmc-reloc] iter={iteration} "
                f"dead={dead_count} "
                f"thr={sched['dead_opacity_threshold']:.5f} "
                f"reloc_int={sched['relocate_interval']} "
                f"rho={sched['rho']:.3f} "
                f"N={count}",
                flush=True,
            )

    def _log_growth(self, tb_writer, should_log, iteration, before, after, added, sched):
        if tb_writer:
            tb_writer.add_scalar("mcmc/added_count", added, iteration)
            tb_writer.add_scalar("mcmc/growth_delta_N", after - before, iteration)
        if should_log(iteration):
            print(
                f"[mcmc-grow] iter={iteration} "
                f"added={added} "
                f"N={before}->{after} "
                f"factor={sched['growth_factor']:.4f} "
                f"grow_int={sched['grow_interval']} "
                f"rho={sched['rho']:.3f}",
                flush=True,
            )


class GsplatMCMCBaselineStrategy(ScheduledMCMCStrategy):
    """Official-like MCMC baseline using opacity relocation, fixed growth, and noise."""

    def __init__(self):
        super().__init__("gsplat_mcmc")

    def step_post_backward(
        self,
        gaussians,
        args,
        sched,
        iteration,
        utility=None,
        temperature=1.0,
        use_energy_mcmc=False,
        tb_writer=None,
        should_log_strategy=None,
    ):
        should_log = should_log_strategy or (lambda _iteration: False)
        refine_start = int(getattr(args, "densify_from_iter", 500))
        refine_stop = int(getattr(args, "mcmc_stop_growth_iter", 12_000))
        refine_every = max(1, int(getattr(args, "densification_interval", 100)))
        if not (refine_start <= iteration < refine_stop) or iteration % refine_every != 0:
            return

        with torch.no_grad():
            min_opacity = float(getattr(args, "mcmc_dead_opacity_end", 0.01))
            dead_mask = (gaussians.get_opacity <= min_opacity).squeeze(-1)
            dead_mask = _exclude_locked_from_mask(dead_mask, gaussians, args)
            dead_count = int(dead_mask.sum().item())
            gaussians.relocate_gs(dead_mask=dead_mask)
            self._log_reloc(
                tb_writer,
                should_log,
                iteration,
                dead_count,
                {
                    "dead_opacity_threshold": min_opacity,
                    "relocate_interval": refine_every,
                    "rho": sched.get("rho", 0.0),
                },
                gaussians.get_xyz.shape[0],
            )

            before = gaussians.get_xyz.shape[0]
            added = gaussians.add_new_gs(
                cap_max=args.cap_max,
                growth_factor=1.05,
            )
            self._log_growth(
                tb_writer,
                should_log,
                iteration,
                before,
                gaussians.get_xyz.shape[0],
                added,
                {
                    "growth_factor": 1.05,
                    "grow_interval": refine_every,
                    "rho": sched.get("rho", 0.0),
                },
            )


class UpstreamGsplatMCMCStrategy:
    name = "gsplat_mcmc"

    def __init__(self):
        self.strategy = None
        self.state = None

    def initialize_state(self, gaussians, args, **_kwargs):
        from gsplat.strategy import MCMCStrategy

        requested = {
            "cap_max": int(getattr(args, "cap_max", 1_000_000)),
            "noise_lr": float(getattr(args, "noise_lr", 5e5)),
            "refine_start_iter": int(getattr(args, "densify_from_iter", 500)),
            "refine_stop_iter": int(getattr(args, "mcmc_stop_growth_iter", 25_000)),
            "refine_every": int(getattr(args, "densification_interval", 100)),
            "min_opacity": float(getattr(args, "mcmc_dead_opacity_end", 0.005)),
            "verbose": bool(getattr(args, "mcmc_strategy_verbose", False)),
            "noise_injection_stop_iter": int(
                getattr(args, "mcmc_noise_stop_iter", getattr(args, "iterations", 30_000))
            ),
        }
        supported = inspect.signature(MCMCStrategy).parameters
        kwargs = {key: value for key, value in requested.items() if key in supported}
        self.strategy = MCMCStrategy(**kwargs)
        self.strategy.check_sanity(gaussians.params, gaussians.optimizers)
        self.state = self.strategy.initialize_state()
        return self.state

    def step_pre_backward(self, **kwargs):
        if self.strategy is not None:
            self.strategy.step_pre_backward(**kwargs)

    def inject_noise(self, **_kwargs):
        # Upstream MCMCStrategy injects noise in step_post_backward.
        return None

    def step_post_backward(
        self,
        gaussians,
        args,
        sched,
        iteration,
        utility=None,
        temperature=1.0,
        use_energy_mcmc=False,
        tb_writer=None,
        should_log_strategy=None,
        render_pkg=None,
        lr=None,
    ):
        if self.strategy is None or self.state is None:
            self.initialize_state(gaussians=gaussians, args=args)
        before = gaussians.get_xyz.shape[0]
        info = render_pkg.get("meta", {}) if render_pkg is not None else {}
        self.strategy.step_post_backward(
            params=gaussians.params,
            optimizers=gaussians.optimizers,
            state=self.state,
            step=iteration,
            info=info,
            lr=float(lr if lr is not None else 0.0),
        )
        after = gaussians.get_xyz.shape[0]
        if tb_writer:
            tb_writer.add_scalar("mcmc/growth_delta_N", after - before, iteration)
        should_log = should_log_strategy or (lambda _iteration: False)
        if should_log(iteration):
            print(
                f"[gsplat-mcmc] iter={iteration} N={before}->{after} "
                f"lr={float(lr if lr is not None else 0.0):.6g}",
                flush=True,
            )


class GsplatEnergyMCMCStrategy:
    """Energy-guided MCMC using gsplat's ParameterDict and optimizer-dict contract."""

    name = "gsplat_energy_mcmc"

    def __init__(self):
        self.strategy = None
        self.state = None
        self.noise_stop_iter = None
        self._next_reloc_iter = None
        self._next_grow_iter = None

    def initialize_state(self, gaussians, args, **_kwargs):
        from gsplat.strategy import MCMCStrategy

        requested = {
            "cap_max": int(getattr(args, "cap_max", 1_000_000)),
            "noise_lr": float(getattr(args, "noise_lr", 5e5)),
            "refine_start_iter": int(getattr(args, "densify_from_iter", 500)),
            "refine_stop_iter": int(getattr(args, "mcmc_stop_growth_iter", 25_000)),
            "refine_every": int(getattr(args, "densification_interval", 100)),
            "min_opacity": float(getattr(args, "mcmc_dead_opacity_end", 0.005)),
            "verbose": bool(getattr(args, "mcmc_strategy_verbose", False)),
            "noise_injection_stop_iter": int(
                getattr(args, "mcmc_noise_stop_iter", getattr(args, "iterations", 30_000))
            ),
        }
        supported = inspect.signature(MCMCStrategy).parameters
        kwargs = {key: value for key, value in requested.items() if key in supported}
        self.strategy = MCMCStrategy(**kwargs)
        self.noise_stop_iter = requested["noise_injection_stop_iter"]
        self.strategy.check_sanity(gaussians.params, gaussians.optimizers)
        self.state = self.strategy.initialize_state()
        return self.state

    def step_pre_backward(self, **kwargs):
        if self.strategy is not None and hasattr(self.strategy, "step_pre_backward"):
            self.strategy.step_pre_backward(**kwargs)

    def inject_noise(self, **_kwargs):
        # Like upstream MCMCStrategy, this strategy injects noise in step_post_backward.
        return None

    def _due(self, name, iteration, interval, start_iter):
        attr = f"_next_{name}_iter"
        next_iter = getattr(self, attr, None)
        if next_iter is None:
            next_iter = int(start_iter) + max(1, int(interval))
            setattr(self, attr, next_iter)
        if int(iteration) < next_iter:
            return False
        setattr(self, attr, int(iteration) + max(1, int(interval)))
        return True

    def step_post_backward(
        self,
        gaussians,
        args,
        sched,
        iteration,
        utility=None,
        temperature=1.0,
        use_energy_mcmc=True,
        tb_writer=None,
        should_log_strategy=None,
        render_pkg=None,
        lr=None,
    ):
        if self.strategy is None or self.state is None:
            self.initialize_state(gaussians=gaussians, args=args)

        params = gaussians.params
        optimizers = gaussians.optimizers
        self.state["binoms"] = self.state["binoms"].to(params["means"].device)
        binoms = self.state["binoms"]
        before = params["means"].shape[0]
        n_relocated = 0
        n_added = 0

        with torch.no_grad():
            if sched["allow_relocation"] and self._due(
                "reloc", iteration, sched["relocate_interval"], getattr(args, "densify_from_iter", 500)
            ):
                if use_energy_mcmc and utility is not None:
                    dead_mask = compute_dead_mask(
                        gaussians=gaussians,
                        utility=utility,
                        opacity_threshold=sched["dead_opacity_threshold"],
                        utility_quantile=0.05,
                        scale_kill_threshold=float(getattr(args, "scale_kill_threshold", 0.20)),
                        scale_moderate_threshold=float(getattr(args, "scale_moderate_threshold", 0.05)),
                        anisotropy_kill_ratio=float(getattr(args, "anisotropy_kill_ratio", 20.0)),
                    )
                    if tb_writer:
                        for clause, cnt in _dead_clause_breakdown(gaussians, sched, args).items():
                            tb_writer.add_scalar(f"mcmc/dead_{clause}", cnt, iteration)
                else:
                    dead_mask = (
                        torch.sigmoid(params["opacities"].flatten())
                        <= sched["dead_opacity_threshold"]
                    )
                dead_mask = _exclude_locked_from_mask(dead_mask, gaussians, args)

                n_relocated = self._relocate_energy_guided(
                    gaussians=gaussians,
                    args=args,
                    dead_mask=dead_mask,
                    utility=utility if use_energy_mcmc else None,
                    temperature=temperature,
                    binoms=binoms,
                    min_opacity=max(float(sched["dead_opacity_threshold"]), self.strategy.min_opacity),
                )

            if sched["allow_growth"] and self._due(
                "grow", iteration, sched["grow_interval"], getattr(args, "densify_from_iter", 500)
            ):
                n_added = self._add_energy_guided(
                    gaussians=gaussians,
                    args=args,
                    utility=utility if use_energy_mcmc else None,
                    temperature=temperature,
                    binoms=binoms,
                    min_opacity=self.strategy.min_opacity,
                    cap_max=self.strategy.cap_max,
                    growth_factor=sched["growth_factor"],
                )

        if n_relocated > 0 or n_added > 0:
            torch.cuda.empty_cache()

        if self.noise_stop_iter is None or iteration <= self.noise_stop_iter:
            from gsplat.strategy.ops import inject_noise_to_position

            inject_noise_to_position(
                params=params,
                optimizers=optimizers,
                state={},
                scaler=float(lr if lr is not None else 0.0) * self.strategy.noise_lr,
            )

        after = params["means"].shape[0]
        if tb_writer:
            tb_writer.add_scalar("mcmc/dead_count", n_relocated, iteration)
            tb_writer.add_scalar("mcmc/added_count", n_added, iteration)
            tb_writer.add_scalar("mcmc/growth_delta_N", after - before, iteration)
            tb_writer.add_scalar("mcmc/energy_guided", int(use_energy_mcmc and utility is not None), iteration)

        should_log = should_log_strategy or (lambda _iteration: False)
        if should_log(iteration):
            mode = "energy" if use_energy_mcmc and utility is not None else "opacity"
            print(
                f"[gsplat-energy-mcmc] iter={iteration} mode={mode} "
                f"relocated={n_relocated} added={n_added} "
                f"factor={sched['growth_factor']:.4f} grow_int={sched['grow_interval']} "
                f"N={before}->{after} lr={float(lr if lr is not None else 0.0):.6g}",
                flush=True,
            )

    def _energy_weights(self, fallback_weights, utility, indices, temperature):
        weights = fallback_weights.detach().flatten().clamp_min(0.0)
        if utility is None:
            return weights
        if utility.shape[0] < int(indices.max().item()) + 1:
            return weights

        scores = utility.detach().float()[indices]
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        scores = torch.log1p(torch.clamp(scores, min=0.0))
        if scores.numel() == 0:
            return weights

        median = scores.median()
        mad = (scores - median).abs().median().clamp_min(1e-6)
        scores = ((scores - median) / mad).clamp(-5.0, 5.0)
        probs = torch.softmax(scores / max(float(temperature), 1e-6), dim=0)
        if not torch.isfinite(probs).all() or probs.sum() <= 0:
            return weights
        return probs

    def _relocate_energy_guided(self, gaussians, args, dead_mask, utility, temperature, binoms, min_opacity):
        if dead_mask is None or dead_mask.sum() == 0:
            return 0

        from gsplat.strategy.ops import _multinomial_sample, _update_param_with_optimizer, compute_relocation

        params = gaussians.params
        optimizers = gaussians.optimizers
        opacities = torch.sigmoid(params["opacities"].flatten())
        locked = _locked_depth_mask(gaussians, args)
        if locked is not None and locked.shape[0] == dead_mask.shape[0]:
            dead_mask = dead_mask & ~locked
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = (~dead_mask).nonzero(as_tuple=True)[0]
        if locked is not None and locked.shape[0] == opacities.shape[0]:
            alive_indices = alive_indices[~locked[alive_indices]]
        n = int(dead_indices.numel())
        if n == 0 or alive_indices.numel() == 0:
            return 0

        weights = self._energy_weights(opacities[alive_indices], utility, alive_indices, temperature)
        sampled_local = _multinomial_sample(weights, n, replacement=True)
        sampled_idxs = alive_indices[sampled_local]
        ratios = torch.bincount(sampled_idxs, minlength=opacities.shape[0])[sampled_idxs] + 1

        eps = torch.finfo(torch.float32).eps
        new_opacities, new_scales = compute_relocation(
            opacities=opacities[sampled_idxs],
            scales=torch.exp(params["scales"])[sampled_idxs],
            ratios=ratios,
            binoms=binoms,
        )
        new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)

        def param_fn(name, p):
            if name == "opacities":
                p[sampled_idxs] = torch.logit(new_opacities)
            elif name == "scales":
                p[sampled_idxs] = torch.log(new_scales)
            p[dead_indices] = p[sampled_idxs]
            return torch.nn.Parameter(p, requires_grad=p.requires_grad)

        def optimizer_fn(_key, v):
            v[sampled_idxs] = 0
            return v

        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
        self._reset_relocated_state(gaussians, dead_indices, sampled_idxs)
        return n

    def _add_energy_guided(self, gaussians, args, utility, temperature, binoms, min_opacity, cap_max, growth_factor):
        from gsplat.strategy.ops import _multinomial_sample, _update_param_with_optimizer, compute_relocation

        params = gaussians.params
        optimizers = gaussians.optimizers
        current_n = params["means"].shape[0]
        target_n = min(int(cap_max), int(float(growth_factor) * current_n))
        n = max(0, target_n - current_n)
        if n <= 0:
            return 0

        opacities = torch.sigmoid(params["opacities"].flatten())
        all_indices = torch.arange(current_n, device=opacities.device)
        locked = _locked_depth_mask(gaussians, args)
        if locked is not None and locked.shape[0] == current_n:
            all_indices = all_indices[~locked]
        if all_indices.numel() == 0:
            return 0
        weights = self._energy_weights(opacities[all_indices], utility, all_indices, temperature)
        sampled_local = _multinomial_sample(weights, n, replacement=True)
        sampled_idxs = all_indices[sampled_local]
        ratios = torch.bincount(sampled_idxs, minlength=current_n)[sampled_idxs] + 1

        eps = torch.finfo(torch.float32).eps
        new_opacities, new_scales = compute_relocation(
            opacities=opacities[sampled_idxs],
            scales=torch.exp(params["scales"])[sampled_idxs],
            ratios=ratios,
            binoms=binoms,
        )
        new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)

        def param_fn(name, p):
            if name == "opacities":
                p[sampled_idxs] = torch.logit(new_opacities)
            elif name == "scales":
                p[sampled_idxs] = torch.log(new_scales)
            p_new = torch.cat([p, p[sampled_idxs]])
            return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)

        def optimizer_fn(_key, v):
            v_new = torch.zeros((len(sampled_idxs), *v.shape[1:]), device=v.device)
            return torch.cat([v, v_new])

        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
        self._append_running_state(gaussians, current_n, n)
        return n

    def _reset_relocated_state(self, gaussians, dead_indices, sampled_idxs):
        visibility_ema = getattr(gaussians, "visibility_ema", None)
        if visibility_ema is not None and visibility_ema.numel() > 0:
            if visibility_ema.shape[0] == gaussians.get_xyz.shape[0]:
                visibility_ema[dead_indices] = 0.0
                visibility_ema[sampled_idxs] = visibility_ema[sampled_idxs].clamp_max(0.5)
        
        provisional = getattr(gaussians, "provisional", None)
        if provisional is not None and provisional.shape[0] == gaussians.get_xyz.shape[0]:
            gaussians.provisional[dead_indices] = gaussians.provisional[sampled_idxs]
            gaussians.birth_frame[dead_indices] = gaussians.birth_frame[sampled_idxs]
            gaussians.support_count[dead_indices] = gaussians.support_count[sampled_idxs]
        for name in ("depth_conflict_count", "depth_last_support_uid", "depth_last_conflict_uid"):
            tensor = getattr(gaussians, name, None)
            if tensor is not None and tensor.shape[0] == gaussians.get_xyz.shape[0]:
                tensor[dead_indices] = tensor[sampled_idxs]

    def _append_running_state(self, gaussians, old_count, added_count):
        for name in ("visibility_ema", "xyz_gradient_accum", "denom", "birth_frame", "support_count", "provisional",
                     "anchor_iter", "depth_conflict_count", "lifecycle_state", "utility_ema"):
            tensor = getattr(gaussians, name, None)
            if tensor is not None and tensor.shape[0] == old_count:
                pad = torch.zeros((added_count, *tensor.shape[1:]), device=tensor.device, dtype=tensor.dtype)
                setattr(gaussians, name, torch.cat([tensor, pad], dim=0))
        for name in ("depth_last_support_uid", "depth_last_conflict_uid"):
            tensor = getattr(gaussians, name, None)
            if tensor is not None and tensor.shape[0] == old_count:
                pad = torch.full(
                    (added_count,),
                    -1,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                setattr(gaussians, name, torch.cat([tensor, pad], dim=0))
        # anchor_xyz needs its own path: pad with the new Gaussian positions so anchor=current pos at birth
        anchor_xyz = getattr(gaussians, "anchor_xyz", None)
        if anchor_xyz is not None and anchor_xyz.shape[0] == old_count:
            new_means = gaussians.get_xyz[-added_count:].detach().clone()
            gaussians.anchor_xyz = torch.cat([anchor_xyz, new_means], dim=0)
        # anchor_scale_log / anchor_opacity_logit: pad with current values
        anchor_scale_log = getattr(gaussians, "anchor_scale_log", None)
        if anchor_scale_log is not None and anchor_scale_log.shape[0] == old_count:
            new_scales = gaussians.params["scales"][-added_count:].detach().clone() if hasattr(gaussians, "params") else torch.zeros((added_count, 3), device=anchor_scale_log.device)
            gaussians.anchor_scale_log = torch.cat([anchor_scale_log, new_scales], dim=0)
        anchor_opacity_logit = getattr(gaussians, "anchor_opacity_logit", None)
        if anchor_opacity_logit is not None and anchor_opacity_logit.shape[0] == old_count:
            new_ops = gaussians.params["opacities"][-added_count:].detach().clone() if hasattr(gaussians, "params") else torch.zeros(added_count, device=anchor_opacity_logit.device)
            gaussians.anchor_opacity_logit = torch.cat([anchor_opacity_logit, new_ops], dim=0)
        radii = getattr(gaussians, "max_radii2D", None)
        if radii is not None and radii.numel() > 0 and radii.shape[0] == old_count:
            pad = torch.zeros((added_count,), device=radii.device, dtype=radii.dtype)
            gaussians.max_radii2D = torch.cat([radii, pad], dim=0)


class GsplatDefaultStrategy:
    """Gradient-based densification using gsplat's DefaultStrategy (clone/split/prune)."""

    name = "gsplat_default"

    def __init__(self):
        self.strategy = None
        self.state = None

    def initialize_state(self, gaussians, args, scene_scale=1.0, **_kwargs):
        from gsplat.strategy import DefaultStrategy

        requested = {
            "prune_opa": float(getattr(args, "mcmc_dead_opacity_end", 0.005)),
            "grow_grad2d": float(getattr(args, "default_grow_grad2d", 0.0002)),
            "grow_scale3d": float(getattr(args, "default_grow_scale3d", 0.01)),
            "refine_start_iter": int(getattr(args, "densify_from_iter", 500)),
            "refine_stop_iter": int(getattr(args, "mcmc_stop_growth_iter", 15_000)),
            "refine_every": int(getattr(args, "densification_interval", 100)),
            "reset_every": 999_999_999,
            "verbose": bool(getattr(args, "mcmc_strategy_verbose", False)),
        }
        supported = inspect.signature(DefaultStrategy).parameters
        kwargs = {k: v for k, v in requested.items() if k in supported}
        self.strategy = DefaultStrategy(**kwargs)
        self.state = self.strategy.initialize_state(scene_scale=scene_scale)
        return self.state

    def _reset_accum_if_n_changed(self, gaussians):
        """Reset accumulated grad stats when Gaussian count changes (e.g. after depth insertion)."""
        if self.state is None:
            return
        n = gaussians.get_xyz.shape[0]
        for key in ("grad2d", "count", "radii"):
            v = self.state.get(key)
            if v is not None and v.shape[0] != n:
                self.state[key] = None

    def step_pre_backward(self, gaussians=None, args=None, iteration=None, render_pkg=None, **_kwargs):
        if self.strategy is None or self.state is None:
            return
        self._reset_accum_if_n_changed(gaussians)
        info = render_pkg.get("meta", {}) if render_pkg is not None else {}
        self.strategy.step_pre_backward(
            params=gaussians.params,
            optimizers=gaussians.optimizers,
            state=self.state,
            step=iteration,
            info=info,
        )

    def inject_noise(self, **_kwargs):
        return None

    def step_post_backward(
        self,
        gaussians,
        args,
        sched,
        iteration,
        utility=None,
        temperature=1.0,
        use_energy_mcmc=False,
        tb_writer=None,
        should_log_strategy=None,
        render_pkg=None,
        lr=None,
    ):
        if self.strategy is None or self.state is None:
            self.initialize_state(gaussians=gaussians, args=args)
        self._reset_accum_if_n_changed(gaussians)
        before = gaussians.get_xyz.shape[0]
        info = render_pkg.get("meta", {}) if render_pkg is not None else {}
        self.strategy.step_post_backward(
            params=gaussians.params,
            optimizers=gaussians.optimizers,
            state=self.state,
            step=iteration,
            info=info,
            packed=True,
        )
        after = gaussians.get_xyz.shape[0]
        if tb_writer:
            tb_writer.add_scalar("mcmc/growth_delta_N", after - before, iteration)
        should_log = should_log_strategy or (lambda _iteration: False)
        if should_log(iteration):
            print(f"[gsplat-default] iter={iteration} N={before}->{after}", flush=True)


def make_mcmc_strategy(name, gaussians=None, args=None):
    if name == "gsplat_energy_mcmc" and getattr(gaussians, "uses_gsplat_layout", False):
        return GsplatEnergyMCMCStrategy()
    if name == "gsplat_energy_mcmc":
        raise ValueError("--densification_strategy gsplat_energy_mcmc requires --model_layout gsplat.")
    if name == "gsplat_mcmc" and getattr(gaussians, "uses_gsplat_layout", False):
        return UpstreamGsplatMCMCStrategy()
    if name == "gsplat_mcmc":
        return GsplatMCMCBaselineStrategy()
    if name == "gsplat_default":
        if not getattr(gaussians, "uses_gsplat_layout", False):
            raise ValueError("--densification_strategy gsplat_default requires --model_layout gsplat.")
        return GsplatDefaultStrategy()
    return ScheduledMCMCStrategy(name)
