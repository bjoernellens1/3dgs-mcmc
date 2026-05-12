# Streaming Replay Implementation Plan

**Branch:** `feature/streaming-replay` (new)  
**Base:** `perf/acceleration`  
**Goal:** Turn offline full-dataset optimizer into stream-simulated local mapper

---

## Architecture Summary

### Current Offline Path
```
Dataset → Scene (loads ALL cameras) → full point cloud init →
random sampling from complete set → global MCMC
```

### New Streaming Path
```
Dataset → StreamingFrameSource (ordered frames) → FrameScheduler →
StreamingScene (recent window + replay buffer) →
initial bootstrap from K frames → incremental depth insertion →
local training window → selective Adam on active set
```

### Key Design Decisions
1. **TUM is pilot dataset** — cleanest timestamp-ordered frame structure
2. **Deterministic stepping by default** (`steps_per_frame`) — reproducible experiments
3. **Wall-clock mode opt-in** (`streaming_wallclock=True`) — real-time simulation
4. **Separate module** (`utils/streaming_training.py`) — offline path untouched
5. **Active-set selective Adam** — already default on `perf/acceleration`, fits perfectly
6. **Incremental depth insertion** explicit — not through MCMC growth only

---

## Phase 1: Core Streaming Infrastructure

### Step 1.1: Add streaming config parameters
**File:** `arguments/__init__.py`

Add new `StreamingParams` group (or extend `OptimizationParams`):
```python
self.streaming_replay = False
self.streaming_input_fps = 30.0
self.streaming_wallclock = False
self.streaming_steps_per_frame = 1
self.streaming_max_frames = 0
self.streaming_initial_frames = 1
self.streaming_keyframe_window = 8
self.streaming_replay_buffer = 32
self.streaming_global_replay_ratio = 0.1
self.streaming_local_only = True
self.streaming_active_frustum_margin = 0.10
self.streaming_active_recent_iters = 500
self.streaming_active_support_threshold = 0.02
self.streaming_insert_from_depth = True
self.streaming_depth_stride = 8
self.streaming_max_new_gaussians_per_frame = 2000
self.streaming_insert_voxel_size = 0.02
self.streaming_min_depth = 0.1
self.streaming_max_depth = 8.0
self.streaming_global_maintenance_interval = 100
self.streaming_mcmc_local_only = True
```

### Step 1.2: Add StreamingRGBDFrame dataclass
**File:** `utils/streaming_frames.py` (NEW)

```python
@dataclass
class StreamingRGBDFrame:
    index: int
    timestamp: float
    rgb_path: str
    depth_path: Optional[str]
    c2w: np.ndarray          # [4,4] world-to-camera or camera-to-world?
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    depth_scale: float = 1000.0
```

Note: Need to check CameraInfo convention in codebase for c2w vs w2c.

### Step 1.3: Add FrameScheduler
**File:** `utils/stream_scheduler.py` (NEW)

```python
class FrameScheduler:
    def __init__(self, fps, steps_per_frame=1, wallclock=False):
        self.fps = float(fps)
        self.steps_per_frame = max(1, int(steps_per_frame))
        self.wallclock = bool(wallclock)
        self.start_time = None
        self.next_frame_idx = 0

    def should_release(self, iteration):
        if not self.wallclock:
            return iteration == 1 or iteration % self.steps_per_frame == 0
        # ... wall-clock branch

    def mark_released(self):
        self.next_frame_idx += 1
```

### Step 1.4: Add TUM StreamingFrameSource
**File:** `scene/stream_sources/tum_stream.py` (NEW)

Leverage existing `readTUMCameras()` in `scene/readers/tum.py` to get ordered CameraInfo list, then yield `StreamingRGBDFrame` records one by one.

```python
class TUMStreamingFrameSource:
    def __init__(self, path, frame_stride=1, max_frames=0, ...):
        # Reuse readTUMCameras() to get associations
        self.cam_infos = readTUMCameras(path=path, ...)
        self.idx = 0

    def next(self):
        if self.idx >= len(self.cam_infos):
            return None
        cam = self.cam_infos[self.idx]
        self.idx += 1
        return _cam_info_to_streaming_frame(cam)
```

Also create stub files for:
- `scene/stream_sources/rgbd_sequence_stream.py`
- `scene/stream_sources/scannet_stream.py`
- `scene/stream_sources/replica_stream.py`
- `scene/stream_sources/__init__.py` (registry)

### Step 1.5: Add StreamingScene
**File:** `scene/streaming_scene.py` (NEW)

```python
class StreamingScene:
    def __init__(self, args, gaussians, frame_source, resolution_scales=[1.0]):
        self.args = args
        self.gaussians = gaussians
        self.frame_source = frame_source
        self.train_cameras = []
        self.replay_cameras = []
        self.current_frame = None
        self.frame_idx = 0
        self.cameras_extent = 1.0

    def ingest_next_frame(self):
        frame = self.frame_source.next()
        if frame is None:
            return None
        cam = self._frame_to_camera(frame)
        self.train_cameras.append(cam)
        self.replay_cameras.append(cam)
        if len(self.replay_cameras) > self.args.streaming_replay_buffer:
            self.replay_cameras.pop(0)
        self.current_frame = cam
        self.frame_idx += 1
        return cam

    def get_local_cameras(self):
        k = self.args.streaming_keyframe_window
        return self.train_cameras[-k:]

    def sample_training_camera(self, iteration):
        if (self.replay_cameras
            and self.args.streaming_global_replay_ratio > 0
            and random.random() < self.args.streaming_global_replay_ratio):
            return random.choice(self.replay_cameras)
        local = self.get_local_cameras()
        return random.choice(local) if local else self.current_frame
```

`_frame_to_camera` must create a `Camera` object compatible with `render()`. Study `cameraList_from_camInfos()` in `utils/camera_utils.py`.

### Step 1.6: Add streaming_training() loop
**File:** `utils/streaming_training.py` (NEW)

```python
def streaming_training(dataset, opt, pipe, testing_iterations, saving_iterations,
                       checkpoint_iterations, checkpoint, debug_from,
                       sh_degree_schedule, run_args=None):
    args = run_args
    # ... (mirror offline training() setup)

    # Frame source selection
    frame_source = _create_frame_source(args)
    streaming_scene = StreamingScene(args, gaussians, frame_source)
    scheduler = FrameScheduler(
        fps=args.streaming_input_fps,
        steps_per_frame=args.streaming_steps_per_frame,
        wallclock=args.streaming_wallclock,
    )

    # Bootstrap: ingest initial frames, build point cloud, init gaussians
    _bootstrap_from_initial_frames(streaming_scene, gaussians, args)

    for iteration in range(first_iter, opt.iterations + 1):
        # Release new frame?
        if scheduler.should_release(iteration):
            new_frame = streaming_scene.ingest_next_frame()
            scheduler.mark_released()
            if new_frame is not None:
                _log_streaming_metrics(streaming_scene, iteration)

        # Sample training camera from local window
        viewpoint_cam = streaming_scene.sample_training_camera(iteration)

        # Render + loss + backward (same as offline)
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        # ... loss computation ...
        loss.backward()

        # Optimizer step
        if optimizer_type == "selective_adam":
            # In Phase 1: pass full visibility (all gaussians visible)
            # In Phase 2: pass active mask
            visible = torch.ones(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")
            gaussians.optimizer.step(visibility=visible)
        else:
            gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        # MCMC strategy (same as offline for Phase 1)
        mcmc_strategy.step_post_backward(...)

        # ... logging, saving, etc.
```

### Step 1.7: Add streaming branch to train.py
**File:** `train.py`

In `training()`, add early branch:
```python
if getattr(args, "streaming_replay", False):
    from utils.streaming_training import streaming_training
    return streaming_training(
        dataset=dataset, opt=opt, pipe=pipe,
        testing_iterations=testing_iterations,
        saving_iterations=saving_iterations,
        checkpoint_iterations=checkpoint_iterations,
        checkpoint=checkpoint, debug_from=debug_from,
        sh_degree_schedule=sh_degree_schedule,
        run_args=args,
    )
```

### Step 1.8: Bootstrap from initial frames
**File:** `utils/streaming_training.py` (function inside)

```python
def _bootstrap_from_initial_frames(streaming_scene, gaussians, args):
    initial_frames = []
    for _ in range(args.streaming_initial_frames):
        frame = streaming_scene.ingest_next_frame()
        if frame is None:
            break
        initial_frames.append(frame)

    # Build RGB-D point cloud from initial frames
    # Reuse _load_tum_rgbd_pointcloud logic or write generic version
    pcd = _build_pointcloud_from_frames(initial_frames, args)

    # Initialize gaussians (same as offline create_from_pcd)
    gaussians.create_from_pcd(
        pcd, cameras_extent,
        init_scale_mode=args.init_scale_mode,
        init_scale=args.init_scale,
        voxel_size=args.pcd_voxel_size,
    )
```

**Risk:** `_load_tum_rgbd_pointcloud` in TUM reader takes `train_cam_infos` + `associations` as inputs. For streaming, we have individual frames. Need to either:
- (a) Backproject a single frame's depth to get points
- (b) Collect multiple frames' points and merge

Option (b) is closer to current TUM reader behavior. But for streaming, option (a) per-frame + merge is cleaner.

### Phase 1 Deliverable
Command:
```bash
python train.py -s /data/tum/freiburg1_desk \
    --model_path /workspace/output/tum_streaming \
    --streaming_replay \
    --streaming_steps_per_frame 1 \
    --streaming_initial_frames 1 \
    --streaming_keyframe_window 8
```

Expected behavior: Frames consumed in timestamp order, training samples from recent window, no incremental insertion yet.

---

## Phase 2: Incremental Insertion + Active Set

### Step 2.1: Add `add_points_as_gaussians()` to models
**Files:** `scene/gaussian_model.py`, `scene/gsplat_model.py`

```python
def add_points_as_gaussians(self, points, colors, init_scale=0.01):
    """Append new Gaussians from RGB-D backprojected points."""
    # points: [N, 3] numpy or tensor
    # colors: [N, 3] numpy or tensor (0-1 range)
    # Returns: number of added points

    # Convert to torch
    new_xyz = torch.tensor(points, dtype=torch.float32, device="cuda")
    new_colors = torch.tensor(colors, dtype=torch.float32, device="cuda")

    # SH features
    fused_color = RGB2SH(new_colors)
    features = torch.zeros((new_xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
    features[:, :3, 0] = fused_color

    # Scales
    dist2 = torch.full((new_xyz.shape[0],), init_scale ** 2, device="cuda")
    scales = torch.log(torch.sqrt(dist2) * 0.1)[..., None].repeat(1, 3)

    # Rotations
    rots = torch.zeros((new_xyz.shape[0], 4), device="cuda")
    rots[:, 0] = 1

    # Opacities
    opacities = inverse_sigmoid(0.5 * torch.ones((new_xyz.shape[0], 1), dtype=torch.float, device="cuda"))

    # Append via cat_tensors_to_optimizer
    self.cat_tensors_to_optimizer({
        "xyz": new_xyz,
        "f_dc": features[:, :, 0:1].transpose(1, 2).contiguous(),
        "f_rest": features[:, :, 1:].transpose(1, 2).contiguous(),
        "opacity": opacities,
        "scaling": scales,
        "rotation": rots,
    })
```

For `GsplatGaussianModel`, similar but using `self.params` and the per-param optimizer structure.

### Step 2.2: Add depth backprojection utility
**File:** `utils/streaming_insert.py` (NEW)

```python
def backproject_rgbd_frame(frame: StreamingRGBDFrame,
                           stride: int = 8,
                           min_depth: float = 0.1,
                           max_depth: float = 8.0):
    """Backproject a single RGB-D frame to 3D points in world space."""
    # Load RGB and depth images
    rgb = cv2.imread(frame.rgb_path)  # or PIL
    depth = cv2.imread(frame.depth_path, cv2.IMREAD_ANYDEPTH)

    # Build pixel grid with stride
    h, w = depth.shape
    u = np.arange(0, w, stride)
    v = np.arange(0, h, stride)
    uu, vv = np.meshgrid(u, v)

    # Sample depth
    d = depth[vv, uu].astype(np.float32) / frame.depth_scale
    valid = (d > min_depth) & (d < max_depth)

    # Backproject to camera space
    z = d[valid]
    x = (uu[valid] - frame.cx) * z / frame.fx
    y = (vv[valid] - frame.cy) * z / frame.fy
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)  # [N, 4]

    # Transform to world
    pts_world = (frame.c2w @ pts_cam.T).T[:, :3]

    # Sample colors
    colors = rgb[vv, uu][valid] / 255.0

    return pts_world, colors
```

### Step 2.3: Add voxel filter + coverage check
**File:** `utils/streaming_insert.py`

```python
def voxel_downsample(points, colors, voxel_size=0.02):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pcd.points), np.asarray(pcd.colors)

def filter_existing_coverage(points, colors, existing_xyz, radius=0.02):
    """Remove points already covered by existing Gaussians."""
    # KD-tree or brute-force distance check
    # For small N, simple torch cdist is fine
    existing = torch.tensor(existing_xyz, dtype=torch.float32, device="cuda")
    pts = torch.tensor(points, dtype=torch.float32, device="cuda")
    d = torch.cdist(pts, existing)
    min_d = d.min(dim=1)[0]
    keep = min_d > radius
    return points[keep.cpu().numpy()], colors[keep.cpu().numpy()]
```

### Step 2.4: Integrate insertion into streaming loop
**File:** `utils/streaming_training.py`

In the frame ingestion block:
```python
if scheduler.should_release(iteration):
    new_frame = streaming_scene.ingest_next_frame()
    scheduler.mark_released()
    if new_frame is not None and args.streaming_insert_from_depth:
        _insert_gaussians_from_frame(gaussians, new_frame, args)
```

### Step 2.5: Add frustum active mask
**File:** `utils/frustum_active_set.py` (NEW)

```python
def compute_frustum_active_mask(means, camera, margin=0.10, near=0.05, far=8.0):
    """
    means: [N, 3] tensor
    camera: Camera object with world_view_transform, fx, fy, cx, cy, image_width, image_height
    Returns: [N] bool tensor
    """
    # World to camera transform
    w2c = camera.world_view_transform.T[:4, :4]  # Check convention!
    xyz_h = torch.cat([means, torch.ones_like(means[:, :1])], dim=1)
    cam_xyz = xyz_h @ w2c.to(means.device)

    z = cam_xyz[:, 2]
    x = cam_xyz[:, 0] / z
    y = cam_xyz[:, 1] / z

    u = camera.fx * x + camera.cx
    v = camera.fy * y + camera.cy

    W, H = camera.image_width, camera.image_height

    return (
        (z > near)
        & (z < far)
        & (u >= -margin * W)
        & (u < (1.0 + margin) * W)
        & (v >= -margin * H)
        & (v < (1.0 + margin) * H)
    )
```

### Step 2.6: Wire active mask into optimizer step
**File:** `utils/streaming_training.py`

```python
# After backward()
if args.streaming_replay and args.streaming_local_only:
    active_mask = compute_frustum_active_mask(
        gaussians.get_xyz,
        viewpoint_cam,
        margin=args.streaming_active_frustum_margin,
    )
    # Also add recent-gaussian mask and low-support mask
    recent_mask = ...  # Gaussians inserted within last N iterations
    support_mask = gaussians.visibility_ema.squeeze() > args.streaming_active_support_threshold
    active_mask = active_mask | recent_mask | support_mask
else:
    active_mask = torch.ones(gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda")

if optimizer_type == "selective_adam":
    gaussians.optimizer.step(visibility=active_mask)
else:
    gaussians.optimizer.step()
```

### Step 2.7: Local MCMC only
**File:** `utils/streaming_training.py`

After optimizer step, modify MCMC strategy call:
```python
if args.streaming_replay and args.streaming_mcmc_local_only:
    # Apply MCMC only to active Gaussians
    # Modify mcmc_strategy to accept active_mask
    mcmc_strategy.step_post_backward(
        gaussians=gaussians,
        args=args,
        sched=sched,
        iteration=iteration,
        utility=utility,
        temperature=temperature,
        use_energy_mcmc=use_energy_mcmc,
        active_mask=active_mask,  # NEW parameter
        ...
    )
else:
    # Global MCMC (existing path)
    mcmc_strategy.step_post_backward(...)
```

Need to modify `utils/strategies/mcmc_strategy.py` to accept optional `active_mask`.

### Phase 2 Deliverable
Command:
```bash
python train.py -s /data/tum/freiburg1_desk \
    --model_path /workspace/output/tum_streaming \
    --streaming_replay \
    --streaming_steps_per_frame 1 \
    --streaming_initial_frames 1 \
    --streaming_insert_from_depth \
    --streaming_depth_stride 8 \
    --streaming_max_new_gaussians_per_frame 2000 \
    --streaming_mcmc_local_only
```

Expected: Each new frame inserts ~0-2000 new Gaussians from depth. Training only updates visible/active Gaussians. MCMC only relocates/prunes active set.

---

## Phase 3: Residual Training + Metrics

### Step 3.1: Patch-based residual training (optional, advanced)
Instead of full-frame render + loss, train on high-error patches only.

```python
# After render
error_map = torch.abs(image - gt_image).mean(dim=0)  # [H, W]
# Sample patches from high-error regions
patch_coords = sample_patches_from_error_map(error_map, num_patches=4, patch_size=64)
# Render only patches (requires gsplat patch rendering support)
```

**Risk:** gsplat rasterizer may not support arbitrary patch rendering efficiently. Alternative: full-frame render but weight loss by error map.

### Step 3.2: Frustum Gaussian culling before rasterizer
Pre-filter Gaussians outside view frustum before calling `render()`.

```python
if args.streaming_replay:
    visible_mask = compute_frustum_active_mask(gaussians.get_xyz, viewpoint_cam, margin=0.0)
    # Create temporary view with only visible Gaussians
    # OR: pass visible_mask to render() if supported
```

**Risk:** gsplat render() may not accept a mask parameter. Need to check `gaussian_renderer/gsplat_backend.py`.

### Step 3.3: Streaming metrics logging
**File:** `utils/streaming_training.py`

Log to TensorBoard:
- `streaming/frame_id` — current frame index
- `streaming/total_frames` — total arrived
- `streaming/gaussian_count` — current N
- `streaming/inserted_this_frame` — new Gaussians from depth
- `streaming/active_count` — Gaussians in active mask
- `streaming/replay_buffer_size` — replay buffer length
- `streaming/keyframe_window_size` — local window size
- `streaming/fps` — effective input fps

### Step 3.4: Global maintenance pass
Periodic full-dataset maintenance (every `streaming_global_maintenance_interval` iterations):
```python
if iteration % args.streaming_global_maintenance_interval == 0:
    # Run full visibility check across all cameras
    # Prune globally dead Gaussians
    # Optional: global bundle adjustment-style refinement
```

### Step 3.5: Wall-clock mode
Enable `streaming_wallclock=True` for real-time simulation:
```python
scheduler = FrameScheduler(fps=30, wallclock=True)
# Training loop sleeps if ahead of real-time
# Frames drop if training is too slow
```

### Phase 3 Deliverable
Full streaming pipeline with:
- Incremental depth insertion
- Active-set training
- Local MCMC
- Residual/patch training (if feasible)
- Streaming metrics
- Optional wall-clock mode

---

## Implementation Order (Recommended)

1. **Step 1.1** — Config params (arguments/__init__.py)
2. **Step 1.2** — StreamingRGBDFrame (utils/streaming_frames.py)
3. **Step 1.3** — FrameScheduler (utils/stream_scheduler.py)
4. **Step 1.4** — TUM frame source (scene/stream_sources/tum_stream.py)
5. **Step 1.5** — StreamingScene (scene/streaming_scene.py)
6. **Step 1.6 + 1.8** — streaming_training() + bootstrap (utils/streaming_training.py)
7. **Step 1.7** — Branch in train.py
8. **Verify Phase 1** — run TUM streaming experiment
9. **Step 2.1** — add_points_as_gaussians() in models
10. **Step 2.2 + 2.3** — backprojection + voxel filter (utils/streaming_insert.py)
11. **Step 2.4** — Integrate insertion into loop
12. **Step 2.5** — Frustum active mask (utils/frustum_active_set.py)
13. **Step 2.6** — Wire active mask into optimizer
14. **Step 2.7** — Local MCMC (modify strategy)
15. **Verify Phase 2** — run with insertion + active set
16. **Step 3.1-3.5** — Residual training, metrics, wall-clock
17. **Verify Phase 3** — full pipeline

---

## Risk Register

| Risk | Mitigation |
|---|---|
| Camera object convention mismatch (c2w vs w2c) | Study existing `Camera` class carefully, test with known poses |
| gsplat render() doesn't accept mask | Fallback: render all, zero out inactive in loss; or modify gsplat backend |
| SelectiveAdam step() signature mismatch | Check `gsplat.optimizers.SelectiveAdam.step()` API |
| KD-tree for coverage check too slow on GPU | Use torch.cdist for small N, batch for large N |
| Open3D not available in container | Already added to Dockerfile, install at runtime for now |
| Memory leak from growing Gaussian count | Cap at streaming_max_new_gaussians_per_frame, prune aggressively |
| Frame ordering wrong (TUM timestamps) | Test with known sequence, compare to groundtruth trajectory |

---

## Files Created (summary)

New files: 11
- `utils/streaming_frames.py`
- `utils/stream_scheduler.py`
- `utils/streaming_training.py`
- `utils/streaming_insert.py`
- `utils/frustum_active_set.py`
- `scene/streaming_scene.py`
- `scene/stream_sources/__init__.py`
- `scene/stream_sources/tum_stream.py`
- `scene/stream_sources/rgbd_sequence_stream.py`
- `scene/stream_sources/scannet_stream.py`
- `scene/stream_sources/replica_stream.py`

Modified files: 5
- `arguments/__init__.py` — add streaming params
- `train.py` — add streaming branch
- `scene/gaussian_model.py` — add add_points_as_gaussians()
- `scene/gsplat_model.py` — add add_points_as_gaussians()
- `utils/strategies/mcmc_strategy.py` — accept active_mask

---

## Notes

- Keep offline path completely untouched — all streaming logic is additive
- Test TUM first, generalize to RGBDSequence/ScanNet/Replica after
- `perf/acceleration` branch defaults (gsplat, fixed init, open3d) are good baseline
- The `cap_max=500000` default is important for streaming to prevent unbounded growth
- AsyncSaveWorker from `perf/acceleration` works with streaming path as-is
