# RGB-D Evaluation Notes

## Dense RGB-D Initialization

For ScanNet and TUM evaluations, keep the RGB-D initialization density explicit in
the run table. A rich RGB-D point-cloud init materially improves scene structure
and early reconstruction quality compared with stale or thin cached init PLYs.

Observed during loader validation:

- Thin TUM runs reused an old `tum_rgbd_init.ply` with about 5k points and ended
  near 6.5k splats after 6k iterations. This was not a hard growth cap; it was a
  weak initialization plus conservative growth schedule.
- Full TUM rerun regenerated `tum_rgbd_init.ply` from all available associated
  RGB-D frames with 100k init points. It started at 100k splats, grew past 104k,
  and produced much better scene structure.
- ScanNet `.sens` runs improved strongly after regenerating a 50k RGB-D init
  from native poses/intrinsics instead of reusing the earlier smoke-test 5k init.

When comparing dataset loaders, SH settings, or growth policies, report:

- init source: `rgbd`, `mesh`, `random`, or cached PLY
- init PLY path and whether it was regenerated
- initial point count
- frame count used for RGB-D init
- depth stride and max init points
- final splat count
- train/eval PSNR and L1

This avoids conflating model-policy changes with the much larger effect of
physically plausible RGB-D point initialization.

## Default Results Table

Use this schema for RGB-D and mesh-scene evaluation tables by default:

| Run | Dataset | Scene | Loader | Iter | SH | Init | Init regenerated | PCD preprocess | Voxel size | Outlier filter | Preprocess points in/out | Init points | Init frames | Depth stride | Max init points | Train frames | Eval frames | Final splats | Train PSNR | Train L1 | Eval PSNR | Eval L1 | Notes |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| example | ScanNet | scene0011_00 | sens | 6000 | 3 | rgbd | yes | none | 0.0 | none | 50000/50000 | 50000 | 80 | 8 | 50000 | 119 | 0 | 52038 | 18.58 | 0.0689 | n/a | n/a | native poses/intrinsics |

The required default comparison columns are `Init regenerated`, `PCD preprocess`,
`Voxel size`, `Outlier filter`, `Preprocess points in/out`, `Init points`,
`Init frames`, `Depth stride`, and `Max init points`. Do not collapse them into a
single note field, because stale cached PLYs and point-cloud preprocessing can
dominate the result.

## Default Eval Baselines

Use full-quality inputs unless explicitly testing a fast debug setting:

- Use the stable sparse runtime defaults for eval tables:
  `--parallelism_profile safe`, `--optimizer_type selective_adam`,
  `--gsplat_sparse_grad`, and `--sh_update_interval 16`. These are now code
  defaults and should be recorded when comparing against older dense-Adam runs.

- ScanNet `.sens`: `--scannet_init rgbd --scannet_frame_stride 20
  --scannet_max_frames 120 --scannet_init_frames 80 --scannet_depth_stride 8
  --scannet_max_init_points 50000 --iterations 6000 --cap_max 100000`
- TUM full sequence: `--tum_init rgbd --tum_frame_stride 1 --tum_max_frames 0
  --tum_init_frames 0 --tum_depth_stride 4 --tum_max_init_points 100000
  --iterations 6000 --cap_max 200000`
- Replica mesh scenes: `--replica_init mesh --replica_num_views 120
  --replica_width 640 --replica_height 480 --replica_max_init_points 100000
  --replica_render_points 250000 --iterations 6000 --cap_max 200000`

For SH ablations, rerun the same baseline command with only `--sh_degree 1`
changed. For init ablations, keep the camera/frame settings fixed and change
only the init mode and init point budget.

For Open3D preprocessing ablations, keep all camera/frame/init settings fixed and
change only:

- `--pointcloud_preprocess open3d`
- `--pcd_voxel_size`
- `--pcd_outlier_filter`
- optional outlier parameters
- `--pcd_force_regenerate` when the selected cache file must be rebuilt

## 2026-05-11 Open3D 6k Runs

Branch: `feature/open3d-pointcloud-preprocess`.

These runs were launched before changing the runtime defaults to the stable
sparse path, so their logs show `parallelism_profile=off`,
`optimizer_type=adam`, `gsplat_sparse_grad=False`, and `sh_update_interval=1`.
For future eval rows, use the new defaults from the section above.

| Run | Dataset | Scene | Loader | Iter | SH | Init | Init regenerated | PCD preprocess | Voxel size | Outlier filter | Preprocess points in/out | Init points | Init frames | Depth stride | Max init points | Train frames | Eval frames | Final splats | Train PSNR | Train L1 | Tail it/s | Notes |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `output/open3d_eval_tum_freiburg1_desk_6k` | TUM | `rgbd_dataset_freiburg1_desk` | native TUM | 6000 | 3 | rgbd | yes | open3d | 0.02 | statistical | 100000/38289 | 38289 | all | 4 | 100000 | 596 | 0 | 45281 | 21.4997 | 0.04769 | 66.0 | Full associated RGB-D sequence; strong structure from dense RGB-D init. |
| `output/open3d_eval_scannet_scene0000_01_6k` | ScanNet | `scene0000_01` | `.sens` | 6000 | 3 | rgbd | yes | open3d | 0.02 | statistical | 50000/42582 | 42582 | 80 | 8 | 50000 | 120 | 0 | 45078 | 19.7702 | 0.05063 | 23.0 | Native poses/intrinsics; full input resolution kept. |
| `output/open3d_eval_bicycle_6k` | Mip-NeRF 360 | `bicycle` | COLMAP | 6000 | 3 | sfm | n/a | none | 0.0 | none | n/a | 54275 | n/a | n/a | n/a | 194 | 0 | 61243 | 20.5300 | 0.05971 | 16.1 | Open3D does not affect COLMAP init yet; run omitted `configs/bicycle.json`. |

Replica was not run in this pass. The expected public Replica mount was not
available, and the only path found was an empty trash directory:
`/home/bjoern/.local/share/Trash/files/rgbd-tracking-lab/data/raw/replica`.

### Throughput Findings

Per-iteration speed is dominated by rendered image resolution and runtime mode,
not by the number of available camera poses. Training samples one camera per
iteration, so camera count mostly affects loading, shuffling, and evaluation.

Observed camera/image sizes from the completed outputs:

- TUM: 596 cameras at `640x480`, about 0.31M pixels, tail throughput about
  66 it/s.
- ScanNet: 120 cameras at `1296x968`, about 1.25M pixels, tail throughput about
  23 it/s.
- Bicycle: 194 cameras, large source images auto-rescaled to about 1600 px wide
  during training, tail throughput about 16 it/s.

The current bicycle run was slower than expected for two reasons:

- It was launched without `configs/bicycle.json`, so it used `resolution=-1`
  and auto-rescaled only to the default 1600 px width. The established bicycle
  eval config uses `resolution=4`, which is the intended lower-resolution
  training setting for Mip-NeRF 360 scenes.
- It ran the old dense runtime defaults: dense Adam, no gsplat sparse gradients,
  and `sh_update_interval=1`. The proven stable sparse path is SelectiveAdam,
  `gsplat_sparse_grad=True`, and `sh_update_interval=16`; these are now code
  defaults.
