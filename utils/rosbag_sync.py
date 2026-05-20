"""Strict header-stamp RGB-D-pose synchronization for rosbag streaming.

Used by the Orbbec ROS2 bag loader (and any future ROS1/live loader) to turn
three independent streams of timestamped messages into a list of one-to-one
RGB+D(+pose) tuples with auditable per-frame delta-times. The matcher is a
monotonic two-pointer scan: depth indices never go backwards and are never
reused, so jitter or drops cannot cause the same depth frame to be paired
with two consecutive color frames.

All timestamps are integer nanoseconds. The synchronizer is pure Python +
NumPy and has no dependency on rosbags or PyTorch, so it is unit-testable
without ROS or GPU.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class StampedMsg:
    """A single deserialized message with both header and bag timestamps.

    payload is opaque to the synchronizer (bytes for images, 4x4 np.ndarray
    for poses, etc.). frame_id/encoding/format are populated where available
    so alignment checks can verify color/depth come from the same optical
    frame.
    """
    topic: str
    header_ns: int
    bag_ns: int
    payload: Any = field(repr=False, compare=False, default=None)
    frame_id: str = ""
    encoding: str = ""
    format: str = ""


@dataclass
class SyncedRGBD:
    color: StampedMsg
    depth: StampedMsg
    pose: Optional[StampedMsg]
    rgb_depth_dt_ns: int
    rgb_pose_dt_ns: Optional[int]
    index_color: int
    index_depth: int
    index_pose: Optional[int]


def _dt_stats(dts_ns: list[int]) -> dict:
    if not dts_ns:
        return {"count": 0}
    a = np.asarray(dts_ns, dtype=np.int64)
    absa = np.abs(a)
    return {
        "count": int(a.size),
        "min_ns": int(a.min()),
        "max_ns": int(a.max()),
        "mean_ns": float(a.mean()),
        "median_ns": float(np.median(a)),
        "abs_median_ns": float(np.median(absa)),
        "abs_p95_ns": float(np.percentile(absa, 95)),
        "abs_p99_ns": float(np.percentile(absa, 99)),
        "abs_max_ns": int(absa.max()),
    }


@dataclass
class SyncStats:
    accepted: int = 0
    rejected_no_depth: int = 0
    rejected_no_pose: int = 0
    reused_depth: int = 0
    rgb_depth_dt_ns: dict = field(default_factory=dict)
    rgb_pose_dt_ns: Optional[dict] = None
    header_minus_bag_ns: dict[str, dict] = field(default_factory=dict)
    estimated_offset_ns: int = 0
    compensated_dt_ns: Optional[dict] = None
    p95_violation: bool = False


def estimate_stream_offset_ns(color_ts: list[int], depth_ts: list[int]) -> int:
    """Median nearest-neighbor (depth - color) over all colors.

    Robust to outliers and to streams that are not the same length. Returns
    0 if either side is empty.
    """
    if not color_ts or not depth_ts:
        return 0
    c = np.asarray(color_ts, dtype=np.int64)
    d = np.sort(np.asarray(depth_ts, dtype=np.int64))
    j = np.searchsorted(d, c)
    j = np.clip(j, 0, len(d) - 1)
    j_lo = np.clip(j - 1, 0, len(d) - 1)
    cand_hi = d[j] - c
    cand_lo = d[j_lo] - c
    pick_lo = np.abs(cand_lo) < np.abs(cand_hi)
    nearest = np.where(pick_lo, cand_lo, cand_hi)
    return int(np.median(nearest))


_CLOCK_SKEW_WARN_NS = 60 * 1_000_000_000  # 60 seconds


def _header_minus_bag(msgs: list[StampedMsg]) -> dict:
    """(header.stamp - bag receive time) summary.

    The diagnostic is only meaningful when both clocks share an epoch. Many
    sensor drivers publish header.stamp as time-since-boot or a separate
    NTP-disciplined clock, producing huge constant offsets that swamp the
    real per-message latency. We flag that case via `clock_skew_warning` so
    callers can ignore the field instead of treating the gap as latency.
    """
    if not msgs:
        return {"count": 0}
    diffs = [m.header_ns - m.bag_ns for m in msgs if m.header_ns and m.bag_ns]
    stats = _dt_stats(diffs)
    if stats.get("count"):
        stats["clock_skew_warning"] = bool(stats.get("abs_median_ns", 0) > _CLOCK_SKEW_WARN_NS)
    return stats


def sync_color_depth_unique(
    color: list[StampedMsg],
    depth: list[StampedMsg],
    max_dt_ns: int,
    offset_ns: int = 0,
) -> tuple[list[tuple[int, int, int]], SyncStats]:
    """Monotonic one-to-one RGB↔depth matcher.

    For each color message (in arrival order), advance the depth pointer as
    long as the next depth is closer in time to (color + offset). Accept the
    pair when |dt| <= max_dt_ns and the depth index has not been consumed
    yet. The depth pointer never moves backwards, so a depth message is at
    most matched once.

    Args:
        color, depth: header-stamp-sorted message lists.
        max_dt_ns: maximum |depth.header_ns - (color.header_ns + offset_ns)|.
        offset_ns: add this constant to each color stamp before comparing —
            used to compensate a systematic depth lag (positive = depth
            arrives later than color).

    Returns:
        pairs: list of (color_idx, depth_idx, signed_raw_dt_ns) where
            signed_raw_dt = depth.header_ns - color.header_ns (no offset
            applied, so callers can audit the raw distribution).
        stats: partial SyncStats (rgb_depth_dt_ns / accepted / rejected_no_depth
            / reused_depth; pose fields left default).
    """
    pairs: list[tuple[int, int, int]] = []
    raw_dts: list[int] = []
    stats = SyncStats()

    if not color or not depth:
        stats.rejected_no_depth = len(color)
        stats.rgb_depth_dt_ns = _dt_stats([])
        return pairs, stats

    j = 0
    n_depth = len(depth)
    for ci, cm in enumerate(color):
        target = cm.header_ns + offset_ns
        # Advance j while the next depth is strictly closer to target. Strict
        # `<` keeps the matcher deterministic on exact ties (picks earlier).
        while j + 1 < n_depth and abs(depth[j + 1].header_ns - target) < abs(depth[j].header_ns - target):
            j += 1
        if j >= n_depth:
            stats.rejected_no_depth += 1
            continue
        dt_signed_raw = depth[j].header_ns - cm.header_ns
        dt_for_threshold = depth[j].header_ns - target
        if abs(dt_for_threshold) > max_dt_ns:
            stats.rejected_no_depth += 1
            continue
        pairs.append((ci, j, dt_signed_raw))
        raw_dts.append(dt_signed_raw)
        stats.accepted += 1
        j += 1  # one-to-one: never reuse this depth

    stats.rgb_depth_dt_ns = _dt_stats(raw_dts)
    # Stash raw dts for later compensation re-stat; non-public field.
    stats.rgb_depth_dt_ns["_raw"] = raw_dts
    return pairs, stats


def attach_pose(
    pairs: list[tuple[int, int, int]],
    color: list[StampedMsg],
    depth: list[StampedMsg],
    pose: list[StampedMsg],
    max_pose_dt_ns: int,
    pose_required: bool,
) -> tuple[list[SyncedRGBD], SyncStats]:
    """Attach the nearest pose (by header timestamp) to each RGB-D pair.

    Pose stream is searched via binary search over its header timestamps.
    Frames whose |pose_dt| > max_pose_dt_ns are dropped iff pose_required.
    """
    out: list[SyncedRGBD] = []
    pose_dts: list[int] = []
    stats = SyncStats()

    pose_ts = np.array([p.header_ns for p in pose], dtype=np.int64) if pose else np.empty(0, np.int64)

    for ci, di, dt_depth in pairs:
        cm = color[ci]
        dm = depth[di]
        pose_match: Optional[StampedMsg] = None
        pose_idx: Optional[int] = None
        pose_dt: Optional[int] = None

        if pose_ts.size:
            k = int(np.searchsorted(pose_ts, cm.header_ns))
            k = min(k, pose_ts.size - 1)
            if k > 0 and abs(pose_ts[k - 1] - cm.header_ns) < abs(pose_ts[k] - cm.header_ns):
                k -= 1
            cand_dt = int(pose_ts[k] - cm.header_ns)
            if abs(cand_dt) <= max_pose_dt_ns:
                pose_match = pose[k]
                pose_idx = k
                pose_dt = cand_dt
                pose_dts.append(cand_dt)
            elif pose_required:
                stats.rejected_no_pose += 1
                continue
        elif pose_required:
            stats.rejected_no_pose += 1
            continue

        out.append(SyncedRGBD(
            color=cm,
            depth=dm,
            pose=pose_match,
            rgb_depth_dt_ns=dt_depth,
            rgb_pose_dt_ns=pose_dt,
            index_color=ci,
            index_depth=di,
            index_pose=pose_idx,
        ))
        stats.accepted += 1

    stats.rgb_pose_dt_ns = _dt_stats(pose_dts) if pose_dts else None
    return out, stats


def check_p95_within_threshold(stats: SyncStats, max_dt_ns: int) -> bool:
    """True iff abs_p95 of the RGB-depth distribution is within threshold."""
    p95 = stats.rgb_depth_dt_ns.get("abs_p95_ns")
    if p95 is None:
        return True
    return float(p95) <= float(max_dt_ns)


def merge_stats(
    pair_stats: SyncStats,
    pose_stats: SyncStats,
    color: list[StampedMsg],
    depth: list[StampedMsg],
    pose: list[StampedMsg],
    estimated_offset_ns: int,
    offset_applied_ns: int,
) -> SyncStats:
    """Combine matcher + pose-attach stats and add header-vs-bag diagnostics."""
    merged = SyncStats()
    merged.accepted = pose_stats.accepted
    merged.rejected_no_depth = pair_stats.rejected_no_depth
    merged.rejected_no_pose = pose_stats.rejected_no_pose
    merged.reused_depth = 0  # matcher is one-to-one by construction
    # Strip the private _raw field before exposing.
    rd = dict(pair_stats.rgb_depth_dt_ns)
    raw = rd.pop("_raw", None)
    merged.rgb_depth_dt_ns = rd
    merged.rgb_pose_dt_ns = pose_stats.rgb_pose_dt_ns
    merged.header_minus_bag_ns = {
        "color": _header_minus_bag(color),
        "depth": _header_minus_bag(depth),
        "pose": _header_minus_bag(pose),
    }
    merged.estimated_offset_ns = int(estimated_offset_ns)
    if offset_applied_ns and raw:
        merged.compensated_dt_ns = _dt_stats([d - offset_applied_ns for d in raw])
    return merged


def write_sync_report(path: str | os.PathLike, stats: SyncStats, extras: dict | None = None) -> None:
    """Write a JSON sync report. Parent directory is created if missing."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"stats": asdict(stats)}
    if extras:
        payload["extras"] = extras
    p.write_text(json.dumps(payload, indent=2, default=_json_default))


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def format_stats_oneline(stats: SyncStats) -> str:
    """Compact human-readable summary for stdout."""
    rd = stats.rgb_depth_dt_ns
    parts = [
        f"accepted={stats.accepted}",
        f"reject_depth={stats.rejected_no_depth}",
        f"reject_pose={stats.rejected_no_pose}",
        f"reused_depth={stats.reused_depth}",
    ]
    if rd.get("count"):
        parts.append(
            f"rgb-depth dt: med={rd['abs_median_ns']/1e6:.2f}ms "
            f"p95={rd['abs_p95_ns']/1e6:.2f}ms "
            f"max={rd['abs_max_ns']/1e6:.2f}ms"
        )
    if stats.rgb_pose_dt_ns and stats.rgb_pose_dt_ns.get("count"):
        rp = stats.rgb_pose_dt_ns
        parts.append(
            f"rgb-pose dt: med={rp['abs_median_ns']/1e6:.2f}ms "
            f"p95={rp['abs_p95_ns']/1e6:.2f}ms"
        )
    if stats.estimated_offset_ns:
        parts.append(f"est_offset={stats.estimated_offset_ns/1e6:+.2f}ms")
    return " | ".join(parts)
