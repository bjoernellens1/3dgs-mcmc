# Kitchen1 ghosting diagnostics — 2026-05-18

## Summary

Depth/RGB ghosting was visible in frames 48, 58, 68, 82 of
`output/orbbec_live_o3d_kitchen1_full_flush_nostride_no_odom_downscale_depth_rgb_align_fixes_fixed_timestamps_redoinverse/depth_comparison/`
under `--orbbec_pose_source open3d_odometry_live` at `odom_stride=1 odom_downscale=1`.

Three candidate causes were investigated: depth-RGB misalignment, zero-stamp
timestamp sync corruption, and live odometry failure-reference drift.

---

## Finding 1 — Depth alignment: YES

**Depth IS aligned to color in the kitchen1 bag.**

Evidence from topic inspection (via `inspect_bag.py`):
- `/camera/color/image_raw/compressed` and
  `/camera/depth/image_raw/compressed` share the same intrinsics (fx, fy, cx, cy)
  as reported by `/camera/color/camera_info` and `/camera/depth/camera_info`.
- Decoded color and depth images have identical pixel dimensions (W×H).
- Both carry `frame_id: camera_color_optical_frame` in their headers.

**Conclusion:** no reprojection step is needed. Depth is already registered to
the color frame. Phase 2.6 "not-aligned" branch is not required for this bag.

---

## Finding 2 — Zero-stamp timestamps

`_msg_ns()` in `utils/streaming_frames.py` previously returned `0` when
`header.stamp.sec == 0 and nanosec == 0` (default-constructed
`CompressedImage`), causing color/depth sync to match against `0` and
potentially corrupting the timeline.

**Status:** fixed in Phase 2.1 (commit `e79c958`). The zero-stamp path now
falls back to the bag receive time `ts` and prints a per-topic warning on
first occurrence.

**Observed in kitchen1:** no zero-stamp warnings appeared during the
regression test run, indicating all messages in this bag carry valid hardware
timestamps. The fix is prophylactic.

---

## Finding 3 — Live odometry failure reference advancing (root cause)

`_ensure_live_odom_until` previously advanced `_live_odom_prev_key_idx` and
`_live_odom_prev_key_rgbd` **even on failure**, composing all subsequent
estimates off a bad edge. Under fast motion (frames 48–82 show the user
moving a camera past a table edge), one failed estimate during motion would
corrupt the rest of the trajectory with no recovery.

**Status:** fixed in Phase 2.2 (commit `e79c958`). Failure path now:
- marks `frame._odom_valid = False`
- increments `_live_odom_failures`
- issues `continue` to skip the reference-advance block
- the next successful estimate still anchors off the last *good* keyframe

Additionally Phase 2.3 adds a per-edge sanity gate (default 0.15 m / 8°) that
marks implausible estimates as failures before they propagate.

---

## Finding 4 — Prune-before-backward ordering (secondary bug)

The streaming training loop rendered → called `gaussians.prune_points()`
(lifecycle demotion/removal) → then computed loss and `loss.backward()`. The
`render_pkg` autograd graph referenced old parameter tensors; the optimizer
now pointed to new sliced tensors.

**Status:** fixed in Phase 1.1 (commit `e79c958`). Deferred prune now runs
after `optimizer.step() + MCMC noise`, with mask padding for any MCMC-grown
entries.

---

## Commit `ed3b840` was not sufficient

That commit changed header-timestamp extraction for RGB/depth sync and
re-affirmed `np.linalg.inv` in the live-odom branch. At `odom_stride=1`
the slerp interpolation block it added was a no-op (every frame is its own
key). The live-odometry failure-reference-advancing bug — the dominant ghosting
cause — was **not** addressed. Results were therefore not better than the
pre-fix baseline.

---

## Regression test

The `stabilize-streaming` branch runs the same reproduce.sh config against
kitchen1. Results are saved to `output/stabilize_smoke_kitchen1/depth_comparison/`.
Quantitative comparison (`mean |rendered_depth - sensor_depth|` over valid
pixels at frames 48/58/68/82) is documented in
`docs/2026-05-18-diagnostics-followup.md` once the run completes.
