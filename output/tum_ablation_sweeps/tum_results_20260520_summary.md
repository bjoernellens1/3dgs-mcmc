# TUM sweep results 2026-05-20
## Odometry takeaway
- Best supported setting: `orbbec_open3d_odom_method=hybrid`, `rosbag_sync_threshold_ms=33`, no motion prior.
- ICP variants were slower and worse on the checked Freiburg1 scenes.

| run | frames | ATE RMSE | RPE trans | RPE rot deg | metrics |
| --- | ---: | ---: | ---: | ---: | --- |
| odom_freiburg1_desk_gt_raw | 120 | 0.000000 | 0.000000 | 0.0180 | `output/tum_ablation_sweeps/tum_odom_20260520_c/runs/odom_freiburg1_desk_gt_raw/metrics.json` |
| odom_freiburg2_desk_hybrid_sync33 | 120 | 0.011359 | 0.002607 | 0.2685 | `output/tum_ablation_sweeps/tum_odom_focused_20260520/runs/odom_freiburg2_desk_hybrid_sync33/metrics.json` |
| odom_freiburg1_xyz_hybrid_sync33 | 120 | 0.014370 | 0.004555 | 0.3435 | `output/tum_ablation_sweeps/tum_odom_20260520_c/runs/odom_freiburg1_xyz_hybrid_sync33/metrics.json` |
| odom_freiburg1_xyz_hybrid_prior | 120 | 0.014738 | 0.004584 | 0.3435 | `output/tum_ablation_sweeps/tum_odom_20260520_c/runs/odom_freiburg1_xyz_hybrid_prior/metrics.json` |
| odom_freiburg1_desk_hybrid_sync33 | 120 | 0.020659 | 0.007301 | 0.5686 | `output/tum_ablation_sweeps/tum_odom_20260520_c/runs/odom_freiburg1_desk_hybrid_sync33/metrics.json` |
| odom_freiburg1_desk_hybrid_prior | 120 | 0.024627 | 0.007341 | 0.5704 | `output/tum_ablation_sweeps/tum_odom_20260520_c/runs/odom_freiburg1_desk_hybrid_prior/metrics.json` |
| odom_freiburg1_xyz_hybrid_sync5 | 120 | 0.027377 | 0.025859 | 1.1578 | `output/tum_ablation_sweeps/tum_odom_20260520_c/runs/odom_freiburg1_xyz_hybrid_sync5/metrics.json` |
| odom_freiburg1_xyz_icp_d003 | 120 | 0.042602 | 0.024881 | 1.3909 | `output/tum_ablation_sweeps/tum_odom_20260520_c/runs/odom_freiburg1_xyz_icp_d003/metrics.json` |
| odom_freiburg3_long_office_household_hybrid_sync33 | 118 | 0.044531 | 0.005423 | 0.2792 | `output/tum_ablation_sweeps/tum_odom_focused_20260520/runs/odom_freiburg3_long_office_household_hybrid_sync33/metrics.json` |
| odom_freiburg3_long_office_household_hybrid_sync5 | 118 | 0.044756 | 0.005076 | 0.2798 | `output/tum_ablation_sweeps/tum_odom_focused_20260520/runs/odom_freiburg3_long_office_household_hybrid_sync5/metrics.json` |
| odom_freiburg1_desk_hybrid_sync5 | 90 | 0.053809 | 0.022534 | 1.1276 | `output/tum_ablation_sweeps/tum_odom_20260520_c/runs/odom_freiburg1_desk_hybrid_sync5/metrics.json` |
| odom_freiburg1_desk_icp_d003 | 120 | 0.121454 | 0.029357 | 2.7392 | `output/tum_ablation_sweeps/tum_odom_20260520_c/runs/odom_freiburg1_desk_icp_d003/metrics.json` |
| odom_freiburg2_desk_hybrid_sync5 | 96 | 0.121772 | 0.040500 | 1.6509 | `output/tum_ablation_sweeps/tum_odom_focused_20260520/runs/odom_freiburg2_desk_hybrid_sync5/metrics.json` |
| odom_freiburg1_desk_icp_d007 | 120 | 0.122304 | 0.029707 | 2.6709 | `output/tum_ablation_sweeps/tum_odom_20260520_c/runs/odom_freiburg1_desk_icp_d007/metrics.json` |
| odom_freiburg1_desk_icp_prior_gate | 120 | 0.122317 | 0.029774 | 2.6777 | `output/tum_ablation_sweeps/tum_odom_20260520_c/runs/odom_freiburg1_desk_icp_prior_gate/metrics.json` |

## Training takeaway
- Same-budget 8k winner: `streaming_provisional_max_age=40`.
- Overall winner: `streaming_steps_per_frame=150`, 20k iterations, default promotion age.
- Combining `spf150 + age40` did not beat `spf150` alone.

| run | iter | test PSNR | test LPIPS | report |
| --- | ---: | ---: | ---: | --- |
| short_freiburg1_desk_baseline_spf150 | 20000 | 19.7642 | 0.35594 | `output/tum_ablation_sweeps/strict_sync_short_v1/runs/short_freiburg1_desk_baseline_spf150/comparison/iter_20000/report.json` |
| confirm_freiburg1_desk_spf150_age40 | 20000 | 19.6188 | 0.35687 | `output/tum_ablation_sweeps/tum_confirm_20260520/confirm_freiburg1_desk_spf150_age40/comparison/iter_20000/report.json` |
| short_freiburg1_desk_promote_age40 | 8000 | 17.8935 | 0.41573 | `output/tum_ablation_sweeps/tum_short_focus_20260520/runs/short_freiburg1_desk_promote_age40/comparison/iter_8000/report.json` |
| short_freiburg1_desk_promote_opacity0.1 | 8000 | 17.8486 | 0.42144 | `output/tum_ablation_sweeps/tum_short_focus_20260520/runs/short_freiburg1_desk_promote_opacity0.1/comparison/iter_8000/report.json` |
| short_freiburg1_desk_baseline_spf50 | 8000 | 17.8229 | 0.41970 | `output/tum_ablation_sweeps/strict_sync_short_v1/runs/short_freiburg1_desk_baseline_spf50/comparison/iter_8000/report.json` |
| short_freiburg1_desk_promote_support3 | 8000 | 17.7565 | 0.42536 | `output/tum_ablation_sweeps/tum_short_focus_20260520/runs/short_freiburg1_desk_promote_support3/comparison/iter_8000/report.json` |
| short_freiburg1_desk_cap100000 | 8000 | 17.7486 | 0.42572 | `output/tum_ablation_sweeps/strict_sync_short_v1/runs/short_freiburg1_desk_cap100000/comparison/iter_8000/report.json` |
| short_freiburg1_desk_score_weighted_topk | 8000 | 17.7388 | 0.42034 | `output/tum_ablation_sweeps/tum_short_focus_20260520/runs/short_freiburg1_desk_score_weighted_topk/comparison/iter_8000/report.json` |
| short_freiburg1_desk_cap50000 | 8000 | 17.5940 | 0.43158 | `output/tum_ablation_sweeps/strict_sync_short_v1/runs/short_freiburg1_desk_cap50000/comparison/iter_8000/report.json` |
| short_freiburg1_desk_promote_support1 | 8000 | 17.5631 | 0.43070 | `output/tum_ablation_sweeps/tum_short_focus_20260520/runs/short_freiburg1_desk_promote_support1/comparison/iter_8000/report.json` |
| short_freiburg1_desk_score_depth_gap_only | 8000 | 17.1732 | 0.44278 | `output/tum_ablation_sweeps/tum_short_focus_20260520/runs/short_freiburg1_desk_score_depth_gap_only/comparison/iter_8000/report.json` |
