"""Unit tests for utils/rosbag_sync.

Pure-Python tests — no rosbags, no PyTorch, no GPU. Run with:

    python -m pytest tests/test_rosbag_sync.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.rosbag_sync import (  # noqa: E402
    StampedMsg,
    SyncedRGBD,
    attach_pose,
    check_p95_within_threshold,
    estimate_stream_offset_ns,
    merge_stats,
    sync_color_depth_unique,
    write_sync_report,
)
from utils.streaming_frames import streaming_frame_from_synced_rgbd  # noqa: E402


MS = 1_000_000  # ns per ms


def _ms_msgs(topic: str, times_ms: list[float], bag_offset_ms: float = 0.0) -> list[StampedMsg]:
    return [
        StampedMsg(
            topic=topic,
            header_ns=int(t * MS),
            bag_ns=int((t + bag_offset_ms) * MS),
            payload=i,
            frame_id="camera_color_optical_frame",
        )
        for i, t in enumerate(times_ms)
    ]


# ---------- estimate_stream_offset_ns ----------

def test_offset_zero_when_streams_aligned():
    color = [i * 33.0 for i in range(20)]
    depth = list(color)
    assert estimate_stream_offset_ns([int(c * MS) for c in color],
                                     [int(d * MS) for d in depth]) == 0


def test_offset_detects_constant_lag():
    color_ms = [i * 33.0 for i in range(50)]
    depth_ms = [c + 14.0 for c in color_ms]  # depth lags color by 14 ms
    est = estimate_stream_offset_ns([int(c * MS) for c in color_ms],
                                    [int(d * MS) for d in depth_ms])
    assert abs(est - 14 * MS) <= 1, f"expected ~14ms offset, got {est/MS:.3f}ms"


def test_offset_empty():
    assert estimate_stream_offset_ns([], [1, 2, 3]) == 0
    assert estimate_stream_offset_ns([1, 2, 3], []) == 0


# ---------- sync_color_depth_unique ----------

def test_perfectly_synced_all_accepted():
    color = _ms_msgs("c", [i * 33.0 for i in range(10)])
    depth = _ms_msgs("d", [i * 33.0 for i in range(10)])
    pairs, stats = sync_color_depth_unique(color, depth, max_dt_ns=5 * MS)
    assert stats.accepted == 10
    assert stats.rejected_no_depth == 0
    color_idxs = [p[0] for p in pairs]
    depth_idxs = [p[1] for p in pairs]
    assert color_idxs == list(range(10))
    assert depth_idxs == list(range(10))


def test_no_depth_reuse_on_drop():
    # Depth #3 is missing.
    color = _ms_msgs("c", [i * 33.0 for i in range(10)])
    depth_times = [i * 33.0 for i in range(10) if i != 3]
    depth = _ms_msgs("d", depth_times)
    pairs, stats = sync_color_depth_unique(color, depth, max_dt_ns=5 * MS)
    accepted_color = {p[0] for p in pairs}
    assert 3 not in accepted_color
    depth_idxs = [p[1] for p in pairs]
    assert len(depth_idxs) == len(set(depth_idxs)), "depth was reused"
    assert stats.rejected_no_depth >= 1


def test_constant_offset_rejected_without_compensation():
    color = _ms_msgs("c", [i * 33.0 for i in range(20)])
    depth = _ms_msgs("d", [i * 33.0 + 14.0 for i in range(20)])
    pairs, stats = sync_color_depth_unique(color, depth, max_dt_ns=5 * MS, offset_ns=0)
    assert stats.accepted == 0
    assert stats.rejected_no_depth == 20


def test_constant_offset_passes_with_compensation():
    color = _ms_msgs("c", [i * 33.0 for i in range(20)])
    depth = _ms_msgs("d", [i * 33.0 + 14.0 for i in range(20)])
    pairs, stats = sync_color_depth_unique(color, depth, max_dt_ns=2 * MS, offset_ns=14 * MS)
    assert stats.accepted == 20
    assert stats.rejected_no_depth == 0


def test_ties_pick_earlier_and_no_backward_motion():
    # Two depths equidistant from color at t=10ms: depths at 8 and 12.
    color = _ms_msgs("c", [10.0, 33.0])
    depth = _ms_msgs("d", [8.0, 12.0, 33.0])
    pairs, stats = sync_color_depth_unique(color, depth, max_dt_ns=5 * MS)
    assert pairs[0] == (0, 0, -2 * MS)
    depth_idxs = [p[1] for p in pairs]
    assert depth_idxs == sorted(depth_idxs)
    assert len(depth_idxs) == len(set(depth_idxs))


def test_jitter_p95_reflects_distribution():
    rng = np.random.default_rng(0)
    n = 200
    color_ms = np.arange(n) * 33.0
    jitter_ms = rng.normal(0.0, 1.0, size=n).clip(-3, 3)
    depth_ms = color_ms + jitter_ms
    color = _ms_msgs("c", color_ms.tolist())
    depth = _ms_msgs("d", depth_ms.tolist())
    pairs, stats = sync_color_depth_unique(color, depth, max_dt_ns=5 * MS)
    assert stats.accepted == n
    rd = stats.rgb_depth_dt_ns
    assert rd["abs_p95_ns"] > rd["abs_median_ns"]
    assert rd["abs_p95_ns"] < 5 * MS


def test_empty_streams():
    pairs, stats = sync_color_depth_unique([], [], max_dt_ns=5 * MS)
    assert pairs == []
    assert stats.accepted == 0


# ---------- attach_pose ----------

def test_attach_pose_drops_when_required_and_far():
    color = _ms_msgs("c", [0.0, 33.0, 66.0])
    depth = _ms_msgs("d", [0.0, 33.0, 66.0])
    pose = _ms_msgs("p", [0.0, 200.0, 400.0])
    pairs, _ = sync_color_depth_unique(color, depth, max_dt_ns=5 * MS)
    out, stats = attach_pose(pairs, color, depth, pose,
                             max_pose_dt_ns=10 * MS, pose_required=True)
    assert len(out) == 1
    assert stats.accepted == 1
    assert stats.rejected_no_pose == 2


def test_attach_pose_optional_keeps_frames():
    color = _ms_msgs("c", [0.0, 33.0])
    depth = _ms_msgs("d", [0.0, 33.0])
    pose: list[StampedMsg] = []
    pairs, _ = sync_color_depth_unique(color, depth, max_dt_ns=5 * MS)
    out, stats = attach_pose(pairs, color, depth, pose,
                             max_pose_dt_ns=10 * MS, pose_required=False)
    assert len(out) == 2
    assert all(s.pose is None for s in out)


# ---------- p95 guard ----------

def test_p95_violation_flag():
    color = _ms_msgs("c", [i * 33.0 for i in range(100)])
    depth = _ms_msgs("d", [i * 33.0 + (3.0 if i % 4 == 0 else 0.2)
                           for i in range(100)])
    _, stats = sync_color_depth_unique(color, depth, max_dt_ns=5 * MS)
    assert not check_p95_within_threshold(stats, max_dt_ns=1 * MS)
    assert check_p95_within_threshold(stats, max_dt_ns=5 * MS)


# ---------- end-to-end report ----------

def test_write_sync_report_roundtrip(tmp_path):
    color = _ms_msgs("c", [i * 33.0 for i in range(5)], bag_offset_ms=1.0)
    depth = _ms_msgs("d", [i * 33.0 + 0.2 for i in range(5)], bag_offset_ms=1.5)
    pose = _ms_msgs("p", [i * 33.0 for i in range(5)], bag_offset_ms=0.5)
    pairs, ps = sync_color_depth_unique(color, depth, max_dt_ns=5 * MS, offset_ns=0)
    out, qs = attach_pose(pairs, color, depth, pose, max_pose_dt_ns=10 * MS, pose_required=True)
    merged = merge_stats(ps, qs, color, depth, pose,
                         estimated_offset_ns=200_000, offset_applied_ns=0)
    report = tmp_path / "sync_report.json"
    write_sync_report(report, merged, extras={"bag": "synthetic"})
    import json
    payload = json.loads(report.read_text())
    assert payload["stats"]["accepted"] == 5
    assert payload["stats"]["reused_depth"] == 0
    assert "rgb_depth_dt_ns" in payload["stats"]
    assert "compensated_dt_ns" in payload["stats"]
    assert payload["extras"]["bag"] == "synthetic"


def test_merge_stats_compensation_recomputed():
    color = _ms_msgs("c", [i * 33.0 for i in range(50)])
    depth = _ms_msgs("d", [i * 33.0 + 14.0 for i in range(50)])
    pairs, ps = sync_color_depth_unique(color, depth, max_dt_ns=20 * MS, offset_ns=0)
    out, qs = attach_pose(pairs, color, depth, [], max_pose_dt_ns=10 * MS, pose_required=False)
    merged = merge_stats(ps, qs, color, depth, [],
                         estimated_offset_ns=14 * MS, offset_applied_ns=14 * MS)
    assert merged.compensated_dt_ns is not None
    assert merged.compensated_dt_ns["abs_p95_ns"] <= 1, (
        f"compensated p95 should be ~0, got {merged.compensated_dt_ns['abs_p95_ns']}"
    )
    assert abs(merged.rgb_depth_dt_ns["median_ns"] - 14 * MS) < 1


def test_streaming_frame_from_synced_rgbd_audit_fields():
    c2w = np.eye(4, dtype=np.float32)
    rec = SyncedRGBD(
        color=StampedMsg("c", header_ns=1_000_000_000, bag_ns=1_000_001_000, payload=b"rgb"),
        depth=StampedMsg("d", header_ns=1_003_000_000, bag_ns=1_003_001_000, payload=b"depth"),
        pose=StampedMsg("p", header_ns=999_000_000, bag_ns=999_001_000, payload=c2w),
        rgb_depth_dt_ns=3 * MS,
        rgb_pose_dt_ns=-1 * MS,
        index_color=7,
        index_depth=8,
        index_pose=9,
    )
    frame = streaming_frame_from_synced_rgbd(
        rec,
        index=0,
        fx=500.0, fy=501.0, cx=320.0, cy=240.0,
        width=640, height=480,
    )
    assert frame.timestamp == 1.0
    assert frame.color_ts_ns == rec.color.header_ns
    assert frame.depth_ts_ns == rec.depth.header_ns
    assert frame.pose_ts_ns == rec.pose.header_ns
    assert frame.rgb_depth_dt_ns == 3 * MS
    assert frame.rgb_pose_dt_ns == -1 * MS
    assert frame.color_bag_ns == rec.color.bag_ns
    assert frame.depth_bag_ns == rec.depth.bag_ns
    assert frame._rgb_bytes == b"rgb"
    assert frame._depth_bytes == b"depth"
    np.testing.assert_allclose(frame.c2w, c2w)
