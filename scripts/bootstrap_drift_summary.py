#!/usr/bin/env python3
"""
Compute and print per-iteration bootstrap-Gaussian drift from PLY snapshots.

Usage:
    python scripts/bootstrap_drift_summary.py output/MY_RUN [output/RUN2 ...]

Each run's `bootstrap_motion/` directory is scanned for PLY files written by
`--streaming_debug_bootstrap_motion`. The reference frame is always
`iter_000000_post_init_bootstrap.ply` (the seed before any optimizer step).
"""

import argparse
import glob
import os
import sys

import numpy as np

try:
    from plyfile import PlyData
except ImportError:
    print("plyfile not installed. Run:  pip install plyfile", file=sys.stderr)
    sys.exit(1)


def load_xyz(path: str) -> np.ndarray:
    d = PlyData.read(path)["vertex"]
    return np.stack([d["x"], d["y"], d["z"]], axis=1).astype(np.float32)


def summarise(run_dir: str, csv_out: str | None = None) -> None:
    motion_dir = os.path.join(run_dir, "bootstrap_motion")
    if not os.path.isdir(motion_dir):
        print(f"[SKIP] no bootstrap_motion/ in {run_dir}")
        return

    plys = sorted(glob.glob(os.path.join(motion_dir, "*.ply")))
    if not plys:
        print(f"[SKIP] no PLY files in {motion_dir}")
        return

    # Find reference — must be the post-init snapshot
    refs = [p for p in plys if "post_init" in os.path.basename(p)]
    if not refs:
        print(f"[WARN] no post_init PLY in {motion_dir}; using first file as ref")
        refs = [plys[0]]
    ref_path = refs[0]
    ref_xyz = load_xyz(ref_path)
    N = len(ref_xyz)

    rows = []
    print(f"\n=== {run_dir}  (N_bootstrap={N:,}) ===")
    print(f"{'iter':>8}  {'label':<30}  {'mean':>8}  {'p50':>8}  {'p95':>8}  {'max':>8}  {'>2cm':>6}  {'>5cm':>6}  {'>10cm':>7}")
    print("-" * 100)

    for p in plys:
        if p == ref_path:
            continue
        name = os.path.basename(p).replace(".ply", "")
        parts = name.split("_")
        try:
            itr = int(parts[1])
        except (IndexError, ValueError):
            itr = -1
        label = "_".join(parts[2:]) if len(parts) > 2 else name

        cur = load_xyz(p)
        if cur.shape != ref_xyz.shape:
            print(f"  {itr:>8}  {label:<30}  [shape mismatch: {cur.shape} vs {ref_xyz.shape}]")
            continue

        delta = np.linalg.norm(cur - ref_xyz, axis=1)
        mean_cm = delta.mean() * 100
        p50_cm  = np.quantile(delta, 0.50) * 100
        p95_cm  = np.quantile(delta, 0.95) * 100
        max_cm  = delta.max() * 100
        gt2  = int((delta > 0.02).sum())
        gt5  = int((delta > 0.05).sum())
        gt10 = int((delta > 0.10).sum())

        print(
            f"  {itr:>8}  {label:<30}  {mean_cm:>7.2f}cm  {p50_cm:>7.2f}cm  "
            f"{p95_cm:>7.2f}cm  {max_cm:>7.2f}cm  {gt2:>6}  {gt5:>6}  {gt10:>7}"
        )
        rows.append(
            dict(
                iter=itr,
                label=label,
                mean_cm=mean_cm,
                p50_cm=p50_cm,
                p95_cm=p95_cm,
                max_cm=max_cm,
                gt_2cm=gt2,
                gt_5cm=gt5,
                gt_10cm=gt10,
                N=N,
                run=run_dir,
            )
        )

    if csv_out and rows:
        import csv
        write_header = not os.path.exists(csv_out)
        with open(csv_out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                w.writeheader()
            w.writerows(rows)
        print(f"  → appended {len(rows)} rows to {csv_out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="Output directories to analyse")
    ap.add_argument("--csv", metavar="FILE", help="Append results to CSV file")
    args = ap.parse_args()

    for run in args.runs:
        summarise(run, csv_out=args.csv)


if __name__ == "__main__":
    main()
