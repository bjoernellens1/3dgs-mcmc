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

| Run | Dataset | Scene | Loader | Iter | SH | Init | Init regenerated | Init points | Init frames | Depth stride | Max init points | Train frames | Eval frames | Final splats | Train PSNR | Train L1 | Eval PSNR | Eval L1 | Notes |
| --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| example | ScanNet | scene0011_00 | sens | 6000 | 3 | rgbd | yes | 50000 | 80 | 8 | 50000 | 119 | 0 | 52038 | 18.58 | 0.0689 | n/a | n/a | native poses/intrinsics |

The required default comparison columns are `Init regenerated`, `Init points`,
`Init frames`, `Depth stride`, and `Max init points`. Do not collapse them into a
single note field, because stale cached PLYs can dominate the result.

## Default Eval Baselines

Use full-quality inputs unless explicitly testing a fast debug setting:

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
