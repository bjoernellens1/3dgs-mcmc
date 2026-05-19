"""Optimizer helpers for the gsplat parameter-dict + per-param optimizer convention.

The gsplat strategy API expects:
    params:     ParameterDict | dict[str, Parameter]
    optimizers: dict[str, Optimizer]

These helpers let streaming code operate on that dict directly without going
through OptimizerDictProxy, which exists only for backward compatibility with
the legacy training path.
"""
from __future__ import annotations

from typing import Iterator


def zero_grad(optimizers: dict, set_to_none: bool = True) -> None:
    for opt in optimizers.values():
        opt.zero_grad(set_to_none=set_to_none)


def step_all(optimizers: dict, visibility=None) -> None:
    for opt in optimizers.values():
        if visibility is not None:
            opt.step(visibility)
        else:
            opt.step()


def iter_param_groups(optimizers: dict) -> Iterator[dict]:
    for opt in optimizers.values():
        yield from opt.param_groups
