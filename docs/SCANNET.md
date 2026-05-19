# Native ScanNet Loader

This repo can load ScanNet RGB-D scene folders directly. Depth is used only to
initialize the point cloud; training still uses the existing RGB photometric
gsplat path.

Expected layout:

```text
scans/scene0000_00/
  color/
  depth/
  pose/
  intrinsic/
  scene0000_00_vh_clean_2.ply
```

The loader reads `intrinsic/intrinsic_color.txt`, optional
`intrinsic/intrinsic_depth.txt`, `color/*`, `depth/*.png`, and `pose/*.txt`.
It writes cached initialization PLYs into the scene folder:

```text
scannet_rgbd_init.ply
scannet_mesh_init.ply
scannet_random.ply
```

The scene folder must be writable when creating one of these cached files.

Full run:

```bash
python train.py \
  -s /path/to/scans/scene0000_00 \
  -m output/scannet_scene0000_00 \
  --scannet_frame_stride 10 \
  --scannet_init rgbd \
  --scannet_depth_stride 8 \
  --scannet_init_frames 200 \
  --scannet_max_init_points 250000 \
  --iterations 30000 \
  --tile_size 16
```

Fast debug:

```bash
python train.py \
  -s /path/to/scans/scene0000_00 \
  -m output/scannet_debug \
  --scannet_frame_stride 20 \
  --scannet_max_frames 300 \
  --scannet_init rgbd \
  --scannet_init_frames 50 \
  --scannet_max_init_points 75000 \
  --resolution 4 \
  --iterations 3000
```

Container example:

```bash
docker compose run --rm -T \
  -v "$PWD":/workspace/3dgs-mcmc \
  -v /path/to/scans:/data/scans \
  train python train.py \
  -s /data/scans/scene0000_00 \
  -m output/scannet_debug \
  --scannet_frame_stride 20 \
  --scannet_max_frames 300 \
  --scannet_init rgbd \
  --scannet_init_frames 50 \
  --scannet_max_init_points 75000 \
  --resolution 4 \
  --iterations 3000
```
