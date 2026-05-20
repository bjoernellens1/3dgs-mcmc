"""Unit tests for depth-first streaming geometry lock helpers."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.streaming_depth_lock import (  # noqa: E402
    depth_conflict_prune_mask,
    depth_lock_candidate_mask,
    depth_locked_mask,
    record_depth_conflicts,
    restore_depth_locked_centers,
    zero_depth_locked_mean_grads,
)


class _DummyGaussians:
    def __init__(self, xyz: torch.Tensor, birth_frame: torch.Tensor):
        self.means_param = nn.Parameter(xyz.clone())
        self.birth_frame = birth_frame.to(dtype=torch.int32)
        self.anchor_xyz = xyz.clone()

    @property
    def get_xyz(self):
        return self.means_param


def _args(**overrides) -> Namespace:
    data = {
        "streaming_depth_lock_centers": False,
        "streaming_depth_lock_bootstrap": False,
        "streaming_depth_lock_insertions": False,
        "streaming_depth_multiview_conflict_enabled": False,
        "streaming_depth_lock_prune_conflicts": False,
        "streaming_depth_unlock_conflict_views": 2,
        "streaming_depth_lock_conflict_thresh": 0.05,
        "streaming_depth_consistency_thresh": 0.05,
    }
    data.update(overrides)
    return Namespace(**data)


def test_depth_lock_defaults_do_not_mask_or_restore_centers():
    g = _DummyGaussians(
        torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
        torch.tensor([0, 1]),
    )
    args = _args()
    assert not depth_lock_candidate_mask(g, args).any()

    g.means_param.grad = torch.ones_like(g.means_param)
    assert zero_depth_locked_mean_grads(g, args) == 0
    assert torch.all(g.means_param.grad == 1.0)

    with torch.no_grad():
        g.means_param.add_(10.0)
    before = g.get_xyz.detach().clone()
    stats = restore_depth_locked_centers(g, args)
    assert stats["locked"] == 0
    assert torch.equal(g.get_xyz, before)


def test_depth_lock_zeroes_and_restores_selected_centers():
    xyz = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    g = _DummyGaussians(xyz, torch.tensor([0, 4, 0]))
    args = _args(
        streaming_depth_lock_centers=True,
        streaming_depth_lock_bootstrap=True,
    )

    mask = depth_locked_mask(g, args)
    assert mask.tolist() == [True, False, True]

    g.means_param.grad = torch.ones_like(g.means_param)
    assert zero_depth_locked_mean_grads(g, args, mask=mask) == 2
    assert torch.equal(g.means_param.grad[mask], torch.zeros(2, 3))
    assert torch.equal(g.means_param.grad[~mask], torch.ones(1, 3))

    with torch.no_grad():
        g.means_param.add_(1.0)
    stats = restore_depth_locked_centers(g, args, mask=mask)
    assert stats["locked"] == 2
    assert torch.equal(g.get_xyz[mask], g.anchor_xyz[mask])
    assert torch.equal(g.get_xyz[~mask], xyz[~mask] + 1.0)


def test_depth_conflicts_unlock_or_prune_after_distinct_views():
    g = _DummyGaussians(
        torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]]),
        torch.tensor([0, 2, 0]),
    )
    args = _args(
        streaming_depth_lock_centers=True,
        streaming_depth_lock_bootstrap=True,
        streaming_depth_lock_insertions=True,
        streaming_depth_multiview_conflict_enabled=True,
        streaming_depth_unlock_conflict_views=2,
    )

    idx = torch.tensor([0, 2])
    assert record_depth_conflicts(g, idx, view_uid=10) == 2
    assert record_depth_conflicts(g, idx, view_uid=10) == 0
    assert record_depth_conflicts(g, idx, view_uid=11) == 2
    assert g.depth_conflict_count.tolist() == [2, 0, 2]

    assert depth_locked_mask(g, args).tolist() == [False, True, False]
    assert not depth_conflict_prune_mask(g, args).any()

    prune_data = vars(args).copy()
    prune_data["streaming_depth_lock_prune_conflicts"] = True
    prune_args = _args(**prune_data)
    assert depth_locked_mask(g, prune_args).tolist() == [True, True, True]
    assert depth_conflict_prune_mask(g, prune_args).tolist() == [True, False, True]
