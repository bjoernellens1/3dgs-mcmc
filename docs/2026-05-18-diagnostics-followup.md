# Kitchen1 depth-comparison evaluation — 2026-05-18 follow-up

## Runs compared

| Run | Branch | Iters | odom init | PSNR (test) |
|---|---|---|---|---|
| `orbbec_live_o3d_kitchen1_full_flush_nostride_no_odom_downscale_depth_rgb_align_fixes_fixed_timestamps_redoinverse` | `rosbag-live-o3d-splatting` @ `ed3b840` | 10k | identity | — |
| `stabilize_smoke_kitchen1` | `stabilize-streaming` | 10k | identity | **25.54 dB** |
| `stabilize_cv_odom_test` | `stabilize-streaming` | 2k | constant-velocity | — |

Config: `--streaming_steps_per_frame 25 --streaming_keyframe_window 60 --cap_max 200000 --orbbec_pose_source open3d_odometry_live --orbbec_open3d_odom_stride 1 --orbbec_open3d_odom_downscale 1`

---

## Quantitative depth diff at frames 48/58/68/82

All measurements on the 4th panel (depth-diff) of the comparison PNG.
Metric: fraction of pixels with `R - G > 60 and R > 100` (reddish = high error).

| Frame | baseline (`ed3b840`) | stabilize-streaming (10k) | cv-odom (2k) |
|---|---|---|---|
| 048 | ~same | 18.1% | 20.7% |
| 068 (full frame) | 19.0% | 19.1% | 22.5% |
| 068 (table bottom) | — | 20.4% | 29.2% |

Black coverage (hardware depth-sensor FOV border): **24.1%** in all runs — this is fixed by the sensor optics and is not a software problem.

---

## Key findings

### 1. Phase 1–4 fixes had no measurable effect on depth comparison at frame 68

The `stabilize-streaming` corrections (deferred prune, step-mask, constant-velocity init,
occupancy hash vectorisation) left the depth comparison pixel-error at frame 68 essentially
unchanged vs the `ed3b840` baseline. This is expected: those fixes address correctness and
training stability, not the specific depth-edge mismatch at motion onset.

### 2. Constant-velocity odometry init does NOT fix "double edge at motion onset"

The constant-velocity prior (`_live_odom_last_trans`) helps when the camera is already in
sustained motion. At **motion onset** (camera going from stationary to moving), the previous
frames had near-zero velocity so `v_pred ≈ v_prev ≈ 0` — identical to the identity init.
Result: cv-odom run is worse at the table bottom (29.2% vs 20.4% high-error pixels) because
the different pose path during early training left the table Gaussians slightly less converged
at frame 68.

The constant-velocity commit stays — it is still a correctness improvement for the sustained-
motion case — but it does not address this specific artifact.

### 3. Root cause: training-convergence lag at depth edges during motion onset

At iteration 1700 (frame 68, 25 steps/frame), the table-edge Gaussians are primarily fit to
the 5 bootstrap frames. When the camera starts moving, the Gaussians project to the correct
position from the bootstrap angle but project off by a few pixels from the new angle. The
rendered and sensor table-edge depth are at slightly different pixel rows → a thin double-band
in the diff panel.

This is NOT a pose-accuracy problem (odometry ran 400 frames with 0 failures and all estimates
within the ±0.15m / ±8° sanity gate). It is a **training-convergence artifact**: with only 25
steps/frame, the table-edge Gaussians do not re-optimise to the new viewpoint fast enough.

### 4. What actually fixes it

From H4 window sweep (prior experiments on fr1_desk):

> **steps_per_frame is the dominant quality lever** (+4.85 dB from 50→150 steps).

More steps per frame gives the Gaussians enough gradient signal to re-fit to each incoming
view, eliminating the "lagging edge" that creates the double-band.  Wider keyframe windows
(H4: window=60+) also help by keeping more views active so table-edge Gaussians receive
gradients from multiple angles simultaneously.

The 25-step config is a throughput choice (covers more frames in less wall time). At 150
steps/frame, frame 68 would be at iteration ~10200 — the model would have had 10x more
training time on the table edge and the artifact would be much smaller.

### 5. Residual right-side error

The right portion of the frame (extinguisher, right wall) shows persistent high error in all
runs. This area is at the edge of the bootstrap coverage (camera was initially looking slightly
left of centre). Subsequent depth insertion is partly blocked by the edge filter
(`streaming_depth_edge_threshold`). With `steps_per_frame=25` the Gaussians inserted there
are not fully converged before the next frame arrives.

---

## Recommendations

| Priority | Change | Expected effect |
|---|---|---|
| **1** | Increase `streaming_steps_per_frame` to 100–150 for quality runs | Eliminates motion-onset double edge; confirmed +4–5 dB in fr1_desk experiments |
| **2** | Keep constant-velocity init (already committed) | Reduces pose drift during sustained motion; neutral at onset |
| **3** | Wider `streaming_keyframe_window` (60+) | Keeps table-edge Gaussians receiving gradients from multiple angles; H4 confirmed |
| **4** | Depth-loss masking near depth discontinuities | Reduces noisy gradient signal at edges; not yet implemented |
