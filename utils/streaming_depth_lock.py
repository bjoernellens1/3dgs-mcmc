from __future__ import annotations

from typing import Any

import torch


def depth_lock_enabled(args: Any) -> bool:
    return bool(getattr(args, "streaming_depth_lock_centers", False))


def depth_conflict_enabled(args: Any) -> bool:
    return bool(getattr(args, "streaming_depth_multiview_conflict_enabled", False))


def depth_lock_conflict_threshold(args: Any) -> float:
    value = float(getattr(args, "streaming_depth_lock_conflict_thresh", 0.0))
    if value > 0:
        return value
    return float(getattr(args, "streaming_depth_consistency_thresh", 0.05))


def _point_count(gaussians) -> int:
    return int(gaussians.get_xyz.shape[0])


def _resize_1d_buffer(gaussians, name: str, *, fill_value: int, dtype: torch.dtype) -> None:
    count = _point_count(gaussians)
    xyz = gaussians.get_xyz
    tensor = getattr(gaussians, name, None)
    if tensor is None:
        setattr(
            gaussians,
            name,
            torch.full((count,), fill_value, dtype=dtype, device=xyz.device),
        )
        return
    if tensor.shape[0] == count:
        return
    if tensor.shape[0] > count:
        setattr(gaussians, name, tensor[:count].contiguous())
        return
    pad = torch.full(
        (count - tensor.shape[0],),
        fill_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    setattr(gaussians, name, torch.cat([tensor, pad], dim=0))


def ensure_depth_lock_buffers(gaussians) -> None:
    _resize_1d_buffer(gaussians, "depth_conflict_count", fill_value=0, dtype=torch.int16)
    _resize_1d_buffer(gaussians, "depth_last_support_uid", fill_value=-1, dtype=torch.int32)
    _resize_1d_buffer(gaussians, "depth_last_conflict_uid", fill_value=-1, dtype=torch.int32)


def depth_lock_candidate_mask(gaussians, args: Any) -> torch.Tensor:
    xyz = gaussians.get_xyz
    mask = torch.zeros((xyz.shape[0],), dtype=torch.bool, device=xyz.device)
    if not depth_lock_enabled(args):
        return mask
    birth_frame = getattr(gaussians, "birth_frame", None)
    anchor_xyz = getattr(gaussians, "anchor_xyz", None)
    if birth_frame is None or anchor_xyz is None:
        return mask
    if birth_frame.shape[0] != xyz.shape[0] or anchor_xyz.shape[0] != xyz.shape[0]:
        return mask
    if bool(getattr(args, "streaming_depth_lock_bootstrap", False)):
        mask |= birth_frame == 0
    if bool(getattr(args, "streaming_depth_lock_insertions", False)):
        mask |= birth_frame > 0
    return mask


def depth_locked_mask(gaussians, args: Any) -> torch.Tensor:
    mask = depth_lock_candidate_mask(gaussians, args)
    if not mask.any():
        return mask
    if (
        depth_conflict_enabled(args)
        and not bool(getattr(args, "streaming_depth_lock_prune_conflicts", False))
    ):
        ensure_depth_lock_buffers(gaussians)
        min_views = max(1, int(getattr(args, "streaming_depth_unlock_conflict_views", 2)))
        conflicts = getattr(gaussians, "depth_conflict_count")
        if conflicts.shape[0] == mask.shape[0]:
            mask &= conflicts < min_views
    return mask


def zero_depth_locked_mean_grads(gaussians, args: Any, mask: torch.Tensor | None = None) -> int:
    if mask is None:
        mask = depth_locked_mask(gaussians, args)
    elif mask.shape[0] != gaussians.get_xyz.shape[0]:
        mask = depth_locked_mask(gaussians, args)
    if not mask.any():
        return 0
    means = gaussians.means_param
    if means.grad is None:
        return int(mask.sum().item())
    if getattr(means.grad, "layout", torch.strided) != torch.strided:
        means.grad = means.grad.to_dense().contiguous()
    means.grad[mask] = 0.0
    return int(mask.sum().item())


def restore_depth_locked_centers(gaussians, args: Any, mask: torch.Tensor | None = None) -> dict:
    if mask is None:
        mask = depth_locked_mask(gaussians, args)
    elif mask.shape[0] != gaussians.get_xyz.shape[0]:
        mask = depth_locked_mask(gaussians, args)
    if not mask.any() or not hasattr(gaussians, "anchor_xyz"):
        return {"locked": 0, "max_drift": 0.0, "mean_drift": 0.0}
    means = gaussians.means_param
    anchors = gaussians.anchor_xyz
    if anchors.shape[0] != means.shape[0]:
        return {"locked": 0, "max_drift": 0.0, "mean_drift": 0.0}
    with torch.no_grad():
        drift = (means.detach()[mask] - anchors[mask]).norm(dim=1)
        stats = {
            "locked": int(mask.sum().item()),
            "max_drift": float(drift.max().item()) if drift.numel() else 0.0,
            "mean_drift": float(drift.mean().item()) if drift.numel() else 0.0,
        }
        means.data[mask] = anchors[mask].to(device=means.device, dtype=means.dtype)
    return stats


def record_depth_conflicts(gaussians, indices: torch.Tensor, view_uid: int) -> int:
    if indices.numel() == 0:
        return 0
    ensure_depth_lock_buffers(gaussians)
    last = gaussians.depth_last_conflict_uid
    counts = gaussians.depth_conflict_count
    fresh = last[indices] != int(view_uid)
    if not fresh.any():
        return 0
    fresh_indices = indices[fresh]
    max_count = torch.iinfo(counts.dtype).max
    counts[fresh_indices] = torch.clamp(
        counts[fresh_indices].to(torch.int32) + 1,
        max=max_count,
    ).to(dtype=counts.dtype)
    last[fresh_indices] = int(view_uid)
    return int(fresh_indices.numel())


def record_depth_support_views(gaussians, indices: torch.Tensor, view_uid: int) -> int:
    if indices.numel() == 0:
        return 0
    ensure_depth_lock_buffers(gaussians)
    last = gaussians.depth_last_support_uid
    fresh = last[indices] != int(view_uid)
    if not fresh.any():
        return 0
    fresh_indices = indices[fresh]
    last[fresh_indices] = int(view_uid)
    return int(fresh_indices.numel())


def depth_conflict_prune_mask(gaussians, args: Any) -> torch.Tensor:
    mask = depth_lock_candidate_mask(gaussians, args)
    if (
        not mask.any()
        or not depth_conflict_enabled(args)
        or not bool(getattr(args, "streaming_depth_lock_prune_conflicts", False))
    ):
        return torch.zeros_like(mask)
    ensure_depth_lock_buffers(gaussians)
    min_views = max(1, int(getattr(args, "streaming_depth_unlock_conflict_views", 2)))
    conflicts = gaussians.depth_conflict_count
    if conflicts.shape[0] != mask.shape[0]:
        return torch.zeros_like(mask)
    return mask & (conflicts >= min_views)


def depth_lock_geometry_stats(gaussians, args: Any) -> dict:
    xyz = gaussians.get_xyz.detach()
    if xyz.numel() == 0:
        return {}
    anchor_xyz = getattr(gaussians, "anchor_xyz", None)
    center_source = anchor_xyz.detach() if anchor_xyz is not None and anchor_xyz.shape == xyz.shape else xyz
    center = center_source.median(dim=0).values
    radius = (xyz - center).norm(dim=1)
    candidate = depth_lock_candidate_mask(gaussians, args)
    locked = depth_locked_mask(gaussians, args)
    stats = {
        "radius_p95_m": float(torch.quantile(radius.float(), 0.95).item()),
        "radius_p99_m": float(torch.quantile(radius.float(), 0.99).item()),
        "radius_max_m": float(radius.max().item()),
        "count_gt_5m": int((radius > 5.0).sum().item()),
        "count_gt_10m": int((radius > 10.0).sum().item()),
        "count_gt_50m": int((radius > 50.0).sum().item()),
        "count_gt_100m": int((radius > 100.0).sum().item()),
        "candidate_count": int(candidate.sum().item()),
        "locked_count": int(locked.sum().item()),
    }
    if anchor_xyz is not None and anchor_xyz.shape == xyz.shape and candidate.any():
        drift = (xyz[candidate] - anchor_xyz.detach()[candidate]).norm(dim=1)
        stats["candidate_drift_max_m"] = float(drift.max().item())
        stats["candidate_drift_mean_m"] = float(drift.mean().item())
        stats["candidate_drift_gt_5cm"] = int((drift > 0.05).sum().item())
        stats["candidate_drift_gt_1m"] = int((drift > 1.0).sum().item())
    if hasattr(gaussians, "depth_conflict_count") and gaussians.depth_conflict_count.shape[0] == xyz.shape[0]:
        conflicts = gaussians.depth_conflict_count
        stats["conflict_count_nonzero"] = int((conflicts > 0).sum().item())
        stats["conflict_count_max"] = int(conflicts.max().item()) if conflicts.numel() else 0
    return stats
