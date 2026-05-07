import math
from dataclasses import dataclass


@dataclass
class MCMCScheduleConfig:
    start_iter: int = 500
    stop_growth_iter: int = 12_000
    stop_reloc_iter: int = 25_000

    relocate_interval_min: int = 50
    relocate_interval_max: int = 500
    relocate_tau: float = 0.65

    grow_interval_min: int = 100
    grow_interval_max: int = 2000
    grow_tau: float = 0.35

    growth_factor_start: float = 1.05
    growth_factor_min: float = 1.002
    growth_factor_tau: float = 0.35

    cap_growth_power: float = 2.0
    cap_interval_strength: float = 4.0
    cap_interval_power: float = 2.0
    cap_stop_ratio: float = 0.98

    dead_opacity_start: float = 0.003
    dead_opacity_end: float = 0.010
    dead_opacity_power: float = 1.5

    # Target-deficit growth controller (optional, selectable)
    use_target_deficit: bool = False
    target_splat_end: int = 150_000
    target_q_start: float = 0.05
    target_q_end: float = 0.85
    target_tau: float = 0.45
    growth_factor_eps: float = 1e-4


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def progress(iteration, start, stop):
    return clamp((iteration - start) / float(max(1, stop - start)), 0.0, 1.0)


def exp_interp(u, start, end, tau):
    tau = max(tau, 1e-6)
    denom = 1.0 - math.exp(-1.0 / tau)
    alpha = (1.0 - math.exp(-u / tau)) / denom
    return start + (end - start) * alpha


def get_mcmc_schedule(iteration, current_n, cap_max, cfg: MCMCScheduleConfig):
    u_growth = progress(iteration, cfg.start_iter, cfg.stop_growth_iter)
    u_reloc = progress(iteration, cfg.start_iter, cfg.stop_reloc_iter)

    rho = 1.0
    if cap_max is not None and cap_max > 0:
        rho = clamp(current_n / float(cap_max), 0.0, 1.0)

    relocate_interval = exp_interp(
        u_reloc,
        cfg.relocate_interval_min,
        cfg.relocate_interval_max,
        cfg.relocate_tau,
    )

    base_grow_interval = exp_interp(
        u_growth,
        cfg.grow_interval_min,
        cfg.grow_interval_max,
        cfg.grow_tau,
    )

    grow_interval = base_grow_interval * (
        1.0 + cfg.cap_interval_strength * (rho ** cfg.cap_interval_power)
    )

    time_decay = math.exp(-u_growth / max(cfg.growth_factor_tau, 1e-6))
    cap_decay = max(0.0, 1.0 - rho) ** cfg.cap_growth_power

    if cfg.use_target_deficit:
        # Target-deficit growth controller. Reuse the growth progress instead
        # of assuming a fixed 30k-iteration run.
        q = exp_interp(u_growth, cfg.target_q_start, cfg.target_q_end, cfg.target_tau)
        N_target = cfg.target_splat_end * q
        deficit = max(0.0, (N_target - current_n) / max(N_target, 1.0))
        growth_factor = 1.0 + (
            (cfg.growth_factor_start - 1.0) * time_decay * (deficit ** cfg.cap_growth_power)
        )
    else:
        growth_factor = 1.0 + (
            (cfg.growth_factor_start - 1.0) * time_decay * cap_decay
        )

    if growth_factor < 1.0 + cfg.growth_factor_eps:
        growth_factor = 1.0
    else:
        growth_factor = max(cfg.growth_factor_min, growth_factor)

    dead_threshold = cfg.dead_opacity_start + (
        cfg.dead_opacity_end - cfg.dead_opacity_start
    ) * (u_reloc ** cfg.dead_opacity_power)

    allow_relocation = cfg.start_iter < iteration < cfg.stop_reloc_iter
    allow_growth = (
        cfg.start_iter < iteration < cfg.stop_growth_iter
        and rho < cfg.cap_stop_ratio
        and growth_factor > 1.0
    )

    return {
        "allow_relocation": allow_relocation,
        "allow_growth": allow_growth,
        "relocate_interval": max(1, int(round(relocate_interval))),
        "grow_interval": max(1, int(round(grow_interval))),
        "growth_factor": growth_factor,
        "dead_opacity_threshold": dead_threshold,
        "rho": rho,
        "u_growth": u_growth,
        "u_reloc": u_reloc,
    }
