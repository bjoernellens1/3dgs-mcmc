#!/usr/bin/env python3
"""
Benchmark comparison utility for sparse-mode ablations.

Usage:
    python utils/benchmark_compare.py <baseline.jsonl> <ablation.jsonl>

Expected input is the --benchmark_dir timings.jsonl output from train.py.

The script prints a side-by-side comparison of timing and memory metrics
across matching iterations.
"""

import json
import sys
import math
from collections import defaultdict


def load_benchmark(path):
    runs = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            runs[row["iteration"]] = row
    return runs


def fmt(v, unit=""):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    if isinstance(v, float):
        return f"{v:.3f}{unit}"
    return f"{v}{unit}"


def main():
    if len(sys.argv) != 3:
        print("Usage: python utils/benchmark_compare.py <baseline.jsonl> <ablation.jsonl>")
        sys.exit(1)

    base = load_benchmark(sys.argv[1])
    abl = load_benchmark(sys.argv[2])

    common_iters = sorted(set(base.keys()) & set(abl.keys()))
    if not common_iters:
        print("ERROR: No matching iterations found between the two runs.")
        sys.exit(1)

    # Use last 1/3 of iterations for steady-state comparison
    cutoff = common_iters[len(common_iters) // 3 * 2]
    steady = [it for it in common_iters if it >= cutoff]

    print(f"Common iterations: {len(common_iters)}")
    print(f"Steady-state window: {len(steady)} iterations (iter >= {cutoff})")
    print()

    # Collect timing keys from the first row
    timing_keys = set()
    for it in common_iters:
        if it in base:
            timing_keys.update(base[it].get("timing_ms", {}).keys())
    timing_keys = sorted(timing_keys)

    print(f"{'Metric':<35} {'Baseline':>15} {'Ablation':>15} {'Δ%':>10}")
    print("-" * 75)

    # Timing averages (steady state)
    print("\n--- Timing (ms/iter, steady-state average) ---")
    for key in timing_keys:
        base_vals = [base[it]["timing_ms"].get(key, 0.0) for it in steady if key in base[it].get("timing_ms", {})]
        abl_vals = [abl[it]["timing_ms"].get(key, 0.0) for it in steady if key in abl[it].get("timing_ms", {})]
        if not base_vals and not abl_vals:
            continue
        base_mean = sum(base_vals) / len(base_vals) if base_vals else 0.0
        abl_mean = sum(abl_vals) / len(abl_vals) if abl_vals else 0.0
        pct = ((abl_mean - base_mean) / max(base_mean, 1e-9)) * 100
        print(f"{'  ' + key:<35} {fmt(base_mean, 'ms'):>15} {fmt(abl_mean, 'ms'):>15} {fmt(pct, '%'):>10}")

    # GPU count stats
    print("\n--- Gaussian counts ---")
    base_n = [base[it].get("num_gaussians", 0) for it in steady]
    abl_n = [abl[it].get("num_gaussians", 0) for it in steady]
    base_n_mean = sum(base_n) / len(base_n) if base_n else 0
    abl_n_mean = sum(abl_n) / len(abl_n) if abl_n else 0
    print(f"{'  num_gaussians (mean)':<35} {fmt(base_n_mean):>15} {fmt(abl_n_mean):>15}")

    # Active-set stats (ablation only)
    abl_vis = [abl[it].get("num_visible", 0) for it in steady]
    abl_frac = [abl[it].get("active_fraction", 0.0) for it in steady]
    if abl_vis and any(v > 0 for v in abl_vis):
        vis_mean = sum(abl_vis) / len(abl_vis)
        frac_mean = sum(abl_frac) / len(abl_frac)
        print(f"{'  active_fraction (mean)':<35} {'':>15} {fmt(frac_mean):>15}")
        print(f"{'  num_visible (mean)':<35} {'':>15} {fmt(vis_mean):>15}")

    # Speed (it/s)
    print("\n--- Throughput ---")
    base_it_s = 1000.0 / sum(base[it]["timing_ms"].get("total_wall", 1.0) for it in steady) * len(steady) if steady else 0
    abl_it_s = 1000.0 / sum(abl[it]["timing_ms"].get("total_wall", 1.0) for it in steady) * len(steady) if steady else 0
    print(f"{'  iterations/sec (steady)':<35} {fmt(base_it_s, 'it/s'):>15} {fmt(abl_it_s, 'it/s'):>15} {fmt((abl_it_s - base_it_s) / max(base_it_s, 1e-9) * 100, '%'):>10}")

    print()
    print("--- Configuration ---")
    ref = base[common_iters[0]]
    print(f"  Baseline: optimizer_type={ref.get('optimizer_type')} "
          f"sparse_grad={ref.get('gsplat_sparse_grad')} "
          f"sh_update_interval={ref.get('sh_update_interval')} "
          f"profile={ref.get('parallelism_profile')}")
    ref2 = abl[common_iters[0]]
    print(f"  Ablation: optimizer_type={ref2.get('optimizer_type')} "
          f"sparse_grad={ref2.get('gsplat_sparse_grad')} "
          f"sh_update_interval={ref2.get('sh_update_interval')} "
          f"profile={ref2.get('parallelism_profile')}")


if __name__ == "__main__":
    main()
