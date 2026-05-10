# ROS/Rosbag RGB-D Loader

ROS inputs are normalized into a static RGB-D sequence folder before training.
This keeps training reproducible and lets rosbag extraction, ROS2 replay, and
live cameras share one dataset path.

Expected normalized layout:

```text
data/rosbags/run1_rgbd/
  rgb/000000.png
  depth/000000.png
  frames.jsonl
  intrinsics.json
  init_rgbd.ply
  metadata.json
```

`frames.jsonl` stores color-camera optical-frame poses as `c2w` matrices. The
training loader converts them with the same convention used by TUM and ScanNet:

```python
w2c = np.linalg.inv(c2w)
R = w2c[:3, :3].T
T = w2c[:3, 3]
```

## Offline Rosbag Extraction

The offline extractor uses the optional Python `rosbags` package. It is not
installed in the ROCm training image by default.

```bash
pip install rosbags
python scripts/extract_rosbag_rgbd.py \
  --bag /path/to/run1 \
  --out data/rosbags/run1_rgbd \
  --rgb-topic /camera/color/image_raw \
  --depth-topic /camera/aligned_depth_to_color/image_raw \
  --camera-info-topic /camera/color/camera_info \
  --pose-source tf \
  --world-frame map \
  --camera-frame camera_color_optical_frame \
  --frame-stride 3 \
  --max-frames 2000 \
  --max-association-dt 0.03 \
  --depth-scale 1000.0 \
  --init-frames 300 \
  --max-init-points 250000
```

Train the extracted sequence normally:

```bash
python train.py \
  -s data/rosbags/run1_rgbd \
  -m output/run1_gs \
  --init_type rgbd \
  --cap_max 300000 \
  --tile_size 16
```

## Unaligned Depth

Prefer aligned depth when available, for example Realsense:

```text
/camera/aligned_depth_to_color/image_raw
```

If the bag only has a depth stream in a separate depth optical frame, ask the
extractor to reproject depth into the color camera:

```bash
python scripts/extract_rosbag_rgbd.py \
  --bag /path/to/run1 \
  --out data/rosbags/run1_rgbd \
  --rgb-topic /camera/color/image_raw \
  --depth-topic /camera/depth/image_raw \
  --camera-info-topic /camera/color/camera_info \
  --depth-camera-info-topic /camera/depth/camera_info \
  --pose-source tf \
  --world-frame map \
  --camera-frame camera_color_optical_frame \
  --color-frame camera_color_optical_frame \
  --depth-frame camera_depth_optical_frame \
  --no-depth-is-aligned-to-color \
  --depth-scale 1000.0
```

The output folder still stores only aligned-to-color depth images. The original
topic names, frame names, and reprojection setting are recorded in
`metadata.json`.

## ROS2 Live or Replay Capture

Run a replay or live camera in one terminal:

```bash
ros2 bag play /path/to/bag --clock
```

Then capture from ROS2 topics in another ROS2-enabled environment:

```bash
python scripts/capture_ros2_rgbd_sequence.py \
  --out data/rosbags/replay_capture \
  --rgb-topic /camera/color/image_raw \
  --depth-topic /camera/aligned_depth_to_color/image_raw \
  --camera-info-topic /camera/color/camera_info \
  --world-frame map \
  --camera-frame camera_color_optical_frame \
  --duration 120 \
  --frame-stride 3
```

This script requires optional ROS2 Python packages (`rclpy`, `cv_bridge`, and
the standard message packages). They are intentionally not installed into the
ROCm training image.

## Container Smoke

The normalized sequence loader can be checked in the project container:

```bash
docker compose run --rm -T \
  -v "$PWD":/workspace/3dgs-mcmc \
  train python train.py \
  -s output/rgbd_sequence_fixture \
  -m output/rgbd_sequence_smoke \
  --init_type rgbd \
  --cap_max 1000 \
  --iterations 2 \
  --resolution 4 \
  --no-web-viewer
```
