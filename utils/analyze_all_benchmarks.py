#!/usr/bin/env python3
"""Analyze all benchmark runs in output/ and report patterns."""
import json, os, glob
from collections import defaultdict

def load_benchmarks(pattern):
    results = []
    for path in sorted(glob.glob(pattern)):
        try:
            rows = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            if not rows:
                continue
            cfg = rows[0]
            cutoff = len(rows) * 2 // 3
            steady = rows[cutoff:]
            if not steady:
                steady = rows

            timing_keys = [k for k in steady[0]["timing_ms"] if k != "total_wall"]
            avg_timing = {}
            for k in timing_keys:
                vals = [r["timing_ms"].get(k, 0) for r in steady]
                avg_timing[k] = sum(vals) / len(vals)
            total_wall = sum(r["timing_ms"]["total_wall"] for r in steady) / len(steady)

            num_gs = rows[-1].get("num_gaussians", 0)
            num_visible = rows[-1].get("num_visible", 0)
            active_frac = rows[-1].get("active_fraction", 0)

            results.append({
                "path": path,
                "iters": len(rows),
                "optimizer": cfg.get("optimizer_type", "?"),
                "sparse_grad": cfg.get("gsplat_sparse_grad", False),
                "sh_interval": cfg.get("sh_update_interval", 1),
                "profile": cfg.get("parallelism_profile", "off"),
                "total_wall_ms": total_wall,
                "num_gs": num_gs,
                "num_visible": num_visible,
                "active_frac": active_frac,
                "timing": avg_timing,
            })
        except Exception as e:
            print(f"  SKIP {path}: {e}")
    return results

results = []
for pat in ["output/**/timings.jsonl", "output/sparse_active_set*/**/timings.jsonl"]:
    results.extend(load_benchmarks(pat))

print(f"Found {len(results)} benchmark runs:\n")

results.sort(key=lambda r: r["total_wall_ms"])

hdr = f"{'Run':<50} {'Optimizer':<18} {'Sparse':<8} {'SH_int':<8} {'it/s':<8} {'Gauss':<8} {'Active%':<8} {'Backwd':<8} {'Optim':<8}"
print(hdr)
print("-" * len(hdr))
for r in results:
    short = r["path"].replace("output/", "").replace("/benchmark/timings.jsonl", "")[:48]
    its = 1000.0 / max(r["total_wall_ms"], 1)
    bw = r["timing"].get("backward", 0)
    opt = r["timing"].get("optimizer", 0)
    af = f"{r['active_frac']*100:.1f}" if r['active_frac'] else "-"
    print(f"{short:<50} {r['optimizer']:<18} {str(r['sparse_grad']):<8} {r['sh_interval']:<8} {its:<8.1f} {r['num_gs']:<8} {af:<8} {bw:<8.2f} {opt:<8.2f}")

print("\n=== PATTERNS ===\n")

by_opt = defaultdict(list)
for r in results:
    by_opt[r["optimizer"]].append(r)

print("1. By optimizer type (avg metrics):")
hdr2 = f"{'Optimizer':<18} {'Count':<6} {'Avg it/s':<10} {'Avg backwd':<12} {'Avg optim':<12} {'Avg Gs':<10}"
print(hdr2)
print("-" * 68)
for opt, rs in sorted(by_opt.items()):
    avg_its = sum(1000.0 / max(r["total_wall_ms"], 1) for r in rs) / len(rs)
    avg_bw = sum(r["timing"].get("backward", 0) for r in rs) / len(rs)
    avg_opt = sum(r["timing"].get("optimizer", 0) for r in rs) / len(rs)
    avg_gs = sum(r["num_gs"] for r in rs) / len(rs)
    print(f"{opt:<18} {len(rs):<6} {avg_its:<10.2f} {avg_bw:<12.2f} {avg_opt:<12.2f} {avg_gs:<10.0f}")

print("\n2. By optimizer + sparse_grad:")
by_opt_sparse = defaultdict(list)
for r in results:
    key = f"{r['optimizer']}+sparse={r['sparse_grad']}"
    by_opt_sparse[key].append(r)
for key, rs in sorted(by_opt_sparse.items()):
    avg_its = sum(1000.0 / max(r["total_wall_ms"], 1) for r in rs) / len(rs)
    avg_bw = sum(r["timing"].get("backward", 0) for r in rs) / len(rs)
    avg_opt = sum(r["timing"].get("optimizer", 0) for r in rs) / len(rs)
    avg_fwd = sum(r["timing"].get("forward", 0) for r in rs) / len(rs)
    avg_loss = sum(r["timing"].get("loss", 0) for r in rs) / len(rs)
    print(f"  {key:<30}: count={len(rs):<3}  it/s={avg_its:<7.2f}  fwd={avg_fwd:<6.2f}  loss={avg_loss:<7.2f}  bwd={avg_bw:<7.2f}  opt={avg_opt:<6.2f}")

print("\n3. Active fraction vs performance (sparse runs only):")
for r in results:
    if r["active_frac"] > 0:
        its = 1000.0 / max(r["total_wall_ms"], 1)
        print(f"  active={r['active_frac']*100:.1f}%  total_ms={r['total_wall_ms']:.1f}  it/s={its:.1f}  gs={r['num_gs']}")

print("\n4. Sparse optimizer sub-stages (when available):")
for r in results:
    if "selective_adam_prepare" in r["timing"] or "selective_adam_step" in r["timing"] or "rotation_normalize" in r["timing"]:
        prep = r["timing"].get("selective_adam_prepare", 0)
        step = r["timing"].get("selective_adam_step", 0)
        norm = r["timing"].get("rotation_normalize", 0)
        total_sub = prep + step + norm
        opt = r["timing"].get("optimizer", 0)
        run_name = os.path.basename(os.path.dirname(os.path.dirname(r["path"])))
        print(f"  {run_name:<30}: prep={prep:.3f} step={step:.3f} norm={norm:.3f} sub_total={total_sub:.3f} optimizer={opt:.3f}")

print("\n5. Average time distribution (all runs):")
timing_keys = ["forward", "loss", "backward", "optimizer", "utility", "reporting", "mutation", "schedule", "taming_scoring", "checkpoint"]
totals = {k: sum(r["timing"].get(k, 0) for r in results) for k in timing_keys}
total = sum(totals.values())
for k in sorted(totals, key=lambda x: totals[x], reverse=True):
    pct = totals[k] / total * 100 if total > 0 else 0
    print(f"  {k:<20}: {totals[k]:>8.2f}ms ({pct:.1f}%)")

print(f"\n6. Key observations:")
print(f"  - Total benchmark runs analyzed: {len(results)}")
print(f"  - Optimizer types: {set(r['optimizer'] for r in results)}")
print(f"  - Sparse_grad runs: {sum(1 for r in results if r['sparse_grad'])}")
print(f"  - Runs with active_fraction: {sum(1 for r in results if r['active_frac'] > 0)}")

print("\n7. Direct comparison: adam vs selective_adam (matching configs, 6K iters):")
runs_6k = [r for r in results if r["iters"] == 6000]
dense_6k = [r for r in runs_6k if r["optimizer"] == "adam" and not r["sparse_grad"]]
sparse_6k = [r for r in runs_6k if r["optimizer"] == "selective_adam" and r["sparse_grad"]]
if dense_6k and sparse_6k:
    d = dense_6k[0]
    s = sparse_6k[0]
    print(f"  adam (dense):  {1000.0/d['total_wall_ms']:.2f} it/s, bwd={d['timing'].get('backward',0):.2f}ms, opt={d['timing'].get('optimizer',0):.2f}ms")
    print(f"  selective_adam (sparse): {1000.0/s['total_wall_ms']:.2f} it/s, bwd={s['timing'].get('backward',0):.2f}ms, opt={s['timing'].get('optimizer',0):.2f}ms")
    print(f"  Speedup: {(1000.0/s['total_wall_ms'])/(1000.0/d['total_wall_ms'])-1:.1%}")
    print(f"  Optimizer speedup: {(d['timing'].get('optimizer',0)-s['timing'].get('optimizer',0))/d['timing'].get('optimizer',0):.1%}")
    print(f"  Backward speedup: {(d['timing'].get('backward',0)-s['timing'].get('backward',0))/d['timing'].get('backward',0):.1%}")
