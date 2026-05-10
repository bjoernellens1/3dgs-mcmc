# Native TUM RGB-D Loader

This repo can load TUM RGB-D sequence folders directly. Depth is used only to
initialize the point cloud; training still uses the existing RGB photometric
gsplat path.

Expected layout:

```text
rgbd_dataset_freiburg1_xyz/
  rgb.txt
  depth.txt
  groundtruth.txt
  rgb/
  depth/
```

The loader associates RGB, depth, and ground-truth pose timestamps, uses the
usual Freiburg 1/2/3 RGB intrinsics, and writes cached initialization PLYs into
the sequence folder:

```text
tum_rgbd_init.ply
tum_random.ply
```

The sequence folder must be writable when creating one of these cached files.

Full run:

```bash
python train.py \
  -s /path/to/rgbd_dataset_freiburg1_xyz \
  -m output/tum_fr1_xyz \
  --tum_sequence freiburg1 \
  --tum_init rgbd \
  --tum_frame_stride 1 \
  --tum_depth_stride 4 \
  --tum_init_frames 300 \
  --tum_max_init_points 250000 \
  --cap_max 300000 \
  --iterations 30000 \
  --tile_size 16
```

Fast debug:

```bash
python train.py \
  -s /path/to/rgbd_dataset_freiburg1_xyz \
  -m output/tum_debug \
  --tum_sequence freiburg1 \
  --tum_init rgbd \
  --tum_frame_stride 5 \
  --tum_max_frames 300 \
  --tum_init_frames 50 \
  --tum_max_init_points 75000 \
  --cap_max 100000 \
  --resolution 2 \
  --iterations 3000
```

Container example:

```bash
docker compose run --rm -T \
  -v "$PWD":/workspace/3dgs-mcmc \
  -v /path/to/tum:/data/tum \
  train python train.py \
  -s /data/tum/rgbd_dataset_freiburg1_xyz \
  -m output/tum_debug \
  --tum_sequence freiburg1 \
  --tum_init rgbd \
  --tum_frame_stride 5 \
  --tum_max_frames 300 \
  --tum_init_frames 50 \
  --tum_max_init_points 75000 \
  --cap_max 100000 \
  --resolution 2 \
  --iterations 3000
```

Use `--tum_sequence freiburg2` or `--tum_sequence freiburg3` for those camera
families when the sequence name does not make it obvious.
