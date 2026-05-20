#!/usr/bin/env python3
"""Create and optionally execute TUM RGB-D odometry/training ablation suites."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import os
import re
import shlex
import subprocess
import tarfile
from pathlib import Path


DEFAULT_TUM_ROOT = "/mnt/cps_persistent1_shared/datasets/public/TUM/tum_rgbd"
SHORT_SCENES = [
    "freiburg1_desk",
    "freiburg1_xyz",
    "freiburg2_desk",
    "freiburg3_long_office_household",
]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def _family(scene: str) -> str:
    m = re.match(r"(freiburg[123])", scene)
    return m.group(1) if m else ""


def _find_first(paths):
    for path in paths:
        if path and Path(path).exists():
            return Path(path)
    return None


def _discover_scene(root: Path, scene: str, *, extract_raw: bool = False) -> dict:
    scene_dir = root / scene
    raw = None
    for child in sorted(scene_dir.glob("rgbd_dataset_*")):
        if (child / "rgb.txt").exists() and (child / "depth.txt").exists():
            raw = child
            break
    if raw is None and extract_raw:
        tgz = _find_first(sorted((scene_dir / "raw_tgz").glob("rgbd_dataset_*.tgz")))
        if tgz is not None:
            with tarfile.open(tgz) as tf:
                tf.extractall(scene_dir)
            for child in sorted(scene_dir.glob("rgbd_dataset_*")):
                if (child / "rgb.txt").exists() and (child / "depth.txt").exists():
                    raw = child
                    break
    bag = _find_first(sorted((scene_dir / "rosbag").glob("*.bag")))
    gt = None
    if raw is not None and (raw / "groundtruth.txt").exists():
        gt = raw / "groundtruth.txt"
    if gt is None:
        gt = _find_first(sorted((scene_dir / "groundtruth").glob("*-groundtruth.txt")))
    return {
        "scene": scene,
        "family": _family(scene),
        "scene_dir": str(scene_dir),
        "raw_path": str(raw) if raw else "",
        "bag_path": str(bag) if bag else "",
        "gt_path": str(gt) if gt else "",
    }


def _bool_flag(name: str, value: bool) -> list[str]:
    return [f"--{name}" if value else f"--no-{name}"]


def _append_flags(cmd: list[str], params: dict) -> list[str]:
    out = list(cmd)
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            out.extend(_bool_flag(key, value))
        elif isinstance(value, (list, tuple)):
            out.append(f"--{key}")
            out.extend(str(v) for v in value)
        else:
            out.extend([f"--{key}", str(value)])
    return out


def _inner_train_cmd(source: str, run_dir: Path, scene: dict, params: dict) -> list[str]:
    iterations = int(params.pop("iterations"))
    source_kind = params.pop("source_kind")
    cmd = [
        "python", "train.py",
        "-s", source,
        "-m", str(run_dir),
        "--streaming_replay",
        "--iterations", str(iterations),
        "--streaming_initial_frames", "5",
        "--streaming_eval_hold", "8",
        "--test_iterations", str(iterations),
        "--save_iterations", str(iterations),
        "--save_interval", "0",
        "--checkpoint_interval", "0",
        "--streaming_report_train_metrics_max", "32",
        "--streaming_report_test_metrics_max", "64",
        "--streaming_report_trajectory_max_frames", "80",
        "--tum_gt_path", scene["gt_path"],
        "--tum_association_max_dt", "0.03",
        "--quiet",
        "--disable_progress_bar",
    ]
    if source_kind == "raw":
        cmd.extend(["--tum_sequence", scene["family"], "--tum_frame_stride", "1"])
    else:
        cmd.extend([
            "--rosbag_profile", "tum",
            "--rosbag_sync_report_json", str(run_dir / "rosbag_sync_report.json"),
            "--orbbec_open3d_odom_cache_dir", str(run_dir / "open3d_odometry_cache"),
        ])
    return _append_flags(cmd, params)


def _inner_odom_cmd(source: str, run_dir: Path, scene: dict, params: dict) -> list[str]:
    source_kind = params.pop("source_kind")
    cmd = [
        "python", "scripts/evaluate_streaming_odometry.py",
        "-s", source,
        "-m", str(run_dir),
        "--tum_gt_path", scene["gt_path"],
        "--tum_association_max_dt", "0.03",
        "--streaming_max_frames", str(params.pop("streaming_max_frames", 120)),
        "--label", params.pop("label"),
    ]
    if source_kind == "raw":
        cmd.extend(["--tum_sequence", scene["family"], "--tum_frame_stride", "1"])
    else:
        cmd.extend([
            "--rosbag_profile", "tum",
            "--rosbag_sync_report_json", str(run_dir / "rosbag_sync_report.json"),
            "--orbbec_open3d_odom_cache_dir", str(run_dir / "open3d_odometry_cache"),
        ])
    return _append_flags(cmd, params)


def _full_cmd(inner: list[str], runner: str) -> list[str]:
    if runner == "container":
        return inner
    return ["docker", "compose", "run", "--rm", "-T", "train", *inner]


def _add_run(runs: list[dict], suite_dir: Path, scene: dict, phase: str, axis: str, params: dict, *, kind: str, source: str):
    run_id = _slug(f"{phase}_{scene['scene']}_{axis}")
    run_dir = suite_dir / "runs" / run_id
    params = dict(params)
    params["source_kind"] = "bag" if source.endswith(".bag") else "raw"
    inner = _inner_odom_cmd(source, run_dir, scene, params.copy()) if kind == "odometry" else _inner_train_cmd(source, run_dir, scene, params.copy())
    runs.append({
        "run_id": run_id,
        "phase": phase,
        "kind": kind,
        "axis": axis,
        "scene": scene["scene"],
        "source": source,
        "gt_path": scene["gt_path"],
        "run_dir": str(run_dir),
        "inner_command": inner,
    })


def build_runs(suite_dir: Path, scenes: list[dict], phase: str) -> list[dict]:
    runs: list[dict] = []
    include_odom = phase in {"all", "contract", "odom"}
    include_short = phase in {"all", "contract", "short"}
    include_confirm = phase in {"all", "contract", "confirm"}

    for scene in scenes:
        if include_odom:
            if scene["raw_path"]:
                _add_run(runs, suite_dir, scene, "odom", "gt_raw", {
                    "label": "gt_raw",
                    "streaming_max_frames": 120,
                }, kind="odometry", source=scene["raw_path"])
            if scene["bag_path"]:
                odom_variants = [
                    ("hybrid_sync5", {"label": "hybrid_sync5", "rosbag_sync_threshold_ms": 5, "orbbec_open3d_odom_method": "hybrid"}),
                    ("hybrid_sync33", {"label": "hybrid_sync33", "rosbag_sync_threshold_ms": 33, "orbbec_open3d_odom_method": "hybrid"}),
                    ("hybrid_prior", {"label": "hybrid_prior", "orbbec_open3d_odom_method": "hybrid", "orbbec_open3d_odom_motion_prior": True}),
                    ("icp_d003", {"label": "icp_d003", "orbbec_open3d_odom_method": "icp", "orbbec_open3d_icp_max_distance": 0.03}),
                    ("icp_d007", {"label": "icp_d007", "orbbec_open3d_odom_method": "icp", "orbbec_open3d_icp_max_distance": 0.07}),
                    ("icp_prior_gate", {
                        "label": "icp_prior_gate",
                        "orbbec_open3d_odom_method": "icp",
                        "orbbec_open3d_odom_motion_prior": True,
                        "orbbec_open3d_odom_motion_gate": True,
                    }),
                ]
                for axis, params in odom_variants:
                    params["streaming_max_frames"] = 120
                    _add_run(runs, suite_dir, scene, "odom", axis, params, kind="odometry", source=scene["bag_path"])

        if include_short and scene["raw_path"]:
            common = {
                "streaming_max_frames": 120,
                "streaming_steps_per_frame": 50,
                "iterations": 8000,
                "streaming_keyframe_window": 60,
                "cap_max": 200000,
                "streaming_free_space_loss_weight": 0.01,
            }
            short_variants = [("baseline_spf50", common)]
            for spf, iters in [(150, 20000)]:
                p = dict(common, streaming_steps_per_frame=spf, iterations=iters)
                short_variants.append((f"baseline_spf{spf}", p))
            for cap in [50000, 100000, 200000, 400000]:
                short_variants.append((f"cap{cap}", dict(common, cap_max=cap, streaming_depth_respects_cap=True)))
            for mode in ["alpha_only", "depth_gap_only", "weighted_topk"]:
                short_variants.append((f"score_{mode}", dict(common, streaming_insert_score_mode=mode)))
            for alpha in [0.2, 0.5]:
                short_variants.append((f"score_alpha{alpha}", dict(common, streaming_insert_alpha_threshold=alpha)))
            for support in [1, 3, 4]:
                short_variants.append((f"promote_support{support}", dict(common, streaming_min_support_views=support)))
            for age in [10, 40]:
                short_variants.append((f"promote_age{age}", dict(common, streaming_provisional_max_age=age)))
            for opacity in [0.1, 0.6]:
                short_variants.append((f"promote_opacity{opacity}", dict(common, streaming_promote_opacity=opacity)))
            for axis, params in short_variants:
                _add_run(runs, suite_dir, scene, "short", axis, params, kind="training", source=scene["raw_path"])

        if include_confirm and scene["raw_path"]:
            confirm = {
                "streaming_max_frames": 300,
                "streaming_steps_per_frame": 150,
                "iterations": 47000,
                "streaming_keyframe_window": 120,
                "streaming_global_replay_ratio": 0.2,
                "cap_max": 200000,
                "streaming_insert_score_mode": "weighted_topk",
                "streaming_min_support_views": 3,
                "streaming_free_space_loss_weight": 0.01,
            }
            _add_run(runs, suite_dir, scene, "confirm", "tuned_weighted_support3", confirm, kind="training", source=scene["raw_path"])
    return runs


def _load_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def collect_summary(run: dict) -> dict:
    run_dir = Path(run["run_dir"])
    traj = _load_json(run_dir / "trajectory_eval" / "metrics.json")
    if run["kind"] == "odometry":
        traj = _load_json(run_dir / "metrics.json")
    reports = sorted(glob.glob(str(run_dir / "comparison" / "iter_*" / "report.json")))
    report = _load_json(Path(reports[-1])) if reports else {}
    sync = _load_json(run_dir / "rosbag_sync_report.json")
    return {
        "run_id": run["run_id"],
        "phase": run["phase"],
        "kind": run["kind"],
        "axis": run["axis"],
        "scene": run["scene"],
        "ate_rmse": traj.get("ate_rmse"),
        "rpe_trans_rmse": traj.get("rpe_trans_rmse"),
        "rpe_rot_deg_rmse": traj.get("rpe_rot_deg_rmse"),
        "mean_test_psnr": report.get("mean_test_psnr"),
        "mean_test_lpips": report.get("mean_test_lpips"),
        "n_frames": traj.get("n_frames"),
        "sync_accepted": ((sync.get("stats") or {}).get("accepted") if sync else None),
        "run_dir": str(run_dir),
    }


def write_suite_files(suite_dir: Path, manifest: dict, summaries: list[dict]) -> None:
    suite_dir.mkdir(parents=True, exist_ok=True)
    with open(suite_dir / "sweep_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    with open(suite_dir / "summary.jsonl", "w") as f:
        for row in summaries:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")
    if summaries:
        with open(suite_dir / "summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
    ranked = sorted(
        summaries,
        key=lambda r: (
            float("inf") if r.get("ate_rmse") is None else float(r["ate_rmse"]),
            -(float("-inf") if r.get("mean_test_psnr") is None else float(r["mean_test_psnr"])),
        ),
    )
    with open(suite_dir / "leaderboard.md", "w") as f:
        f.write("| run | scene | axis | ATE RMSE | test PSNR | test LPIPS | dir |\n")
        f.write("| --- | --- | --- | ---: | ---: | ---: | --- |\n")
        for row in ranked:
            f.write(
                f"| {row['run_id']} | {row['scene']} | {row['axis']} | "
                f"{row.get('ate_rmse')} | {row.get('mean_test_psnr')} | "
                f"{row.get('mean_test_lpips')} | `{row['run_dir']}` |\n"
            )
    with open(suite_dir / "qualitative_index.html", "w") as f:
        f.write("<!doctype html><meta charset='utf-8'><title>TUM Ablation Qualitative Index</title>\n")
        f.write("<style>body{font-family:sans-serif} img{max-width:960px;width:100%;border:1px solid #ccc}</style>\n")
        f.write("<h1>TUM Ablation Qualitative Index</h1>\n")
        for row in ranked:
            run_dir = Path(row["run_dir"])
            sheets = sorted(glob.glob(str(run_dir / "comparison" / "iter_*" / "*contact_sheet.png")))
            if not sheets:
                continue
            rel = os.path.relpath(sheets[-1], suite_dir)
            f.write(f"<h2>{row['run_id']}</h2><p>{row['scene']} / {row['axis']}</p><img src='{rel}'>\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tum-root", default=DEFAULT_TUM_ROOT)
    ap.add_argument("--output-root", default="output/tum_ablation_sweeps")
    ap.add_argument("--suite-id", default="")
    ap.add_argument("--phase", choices=["contract", "odom", "short", "confirm", "all"], default="contract")
    ap.add_argument("--scenes", nargs="*", default=SHORT_SCENES)
    ap.add_argument("--extract-missing-raw", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--runner", choices=["host", "container"], default="host")
    args = ap.parse_args()

    suite_id = args.suite_id or dt.datetime.now().strftime("tum_%Y%m%d_%H%M%S")
    suite_dir = Path(args.output_root) / suite_id
    scenes = [_discover_scene(Path(args.tum_root), s, extract_raw=args.extract_missing_raw) for s in args.scenes]
    runs = build_runs(suite_dir, scenes, args.phase)
    summaries = []

    for run in runs:
        run_dir = Path(run["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        full = _full_cmd(run["inner_command"], args.runner)
        run["command"] = full
        with open(run_dir / "run.json", "w") as f:
            json.dump(run, f, indent=2)
            f.write("\n")
        with open(run_dir / "command.sh", "w") as f:
            f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
            f.write(shlex.join(full))
            f.write("\n")
        if args.execute:
            with open(run_dir / "stdout.log", "w") as log:
                proc = subprocess.run(full, cwd=Path(__file__).resolve().parents[1], stdout=log, stderr=subprocess.STDOUT)
            run["returncode"] = proc.returncode
            if proc.returncode != 0:
                print(f"[sweep] FAILED {run['run_id']} rc={proc.returncode}; see {run_dir / 'stdout.log'}")
        summaries.append(collect_summary(run))

    manifest = {
        "suite_id": suite_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "tum_root": args.tum_root,
        "phase": args.phase,
        "execute": args.execute,
        "runner": args.runner,
        "scenes": scenes,
        "runs": runs,
    }
    write_suite_files(suite_dir, manifest, summaries)
    print(f"[sweep] wrote {len(runs)} run contracts to {suite_dir}")
    if not args.execute:
        print("[sweep] dry-run only; pass --execute to launch commands")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
