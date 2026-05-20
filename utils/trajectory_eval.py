"""Trajectory evaluation utilities.

Saves estimated camera trajectories in TUM format, produces PNG plots,
and computes ATE/RPE metrics (with optional GT alignment via Umeyama SE(3)).
"""

from __future__ import annotations

import json
import os
from typing import Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix to quaternion [qx, qy, qz, qw]."""
    m = R
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [qx, qy, qz, qw] to 3×3 rotation matrix."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# TUM I/O
# ---------------------------------------------------------------------------

def save_tum_trajectory(out_path: str, timestamps_sec: np.ndarray, c2w_arr: np.ndarray) -> None:
    """Save N poses to TUM format: t tx ty tz qx qy qz qw per line."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# TUM trajectory — t tx ty tz qx qy qz qw\n")
        for t, T in zip(timestamps_sec, c2w_arr):
            xyz = T[:3, 3]
            q = _rot_to_quat(T[:3, :3])
            f.write(f"{t:.9f} {xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f} "
                    f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")


def load_tum_trajectory(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load TUM file, return (timestamps_sec [N], c2w_arr [N,4,4])."""
    ts, mats = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            t = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            R = _quat_to_rot(np.array([qx, qy, qz, qw]))
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            ts.append(t)
            mats.append(T)
    return np.array(ts), np.stack(mats)


def associate_by_timestamp(
    ts_est: np.ndarray,
    c2w_est: np.ndarray,
    ts_ref: np.ndarray,
    c2w_ref: np.ndarray,
    *,
    max_dt: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Nearest-neighbor timestamp association for trajectory metrics."""
    if len(ts_est) == 0 or len(ts_ref) == 0:
        return (
            np.asarray([], dtype=np.float64),
            np.empty((0, 4, 4), dtype=np.float64),
            np.empty((0, 4, 4), dtype=np.float64),
            {"n_matches": 0, "max_dt": float(max_dt)},
        )
    ref_order = np.argsort(ts_ref)
    ts_ref_sorted = ts_ref[ref_order]
    c2w_ref_sorted = c2w_ref[ref_order]
    est_out, ref_out, ts_out, dts = [], [], [], []
    for t, pose in zip(ts_est, c2w_est):
        idx = int(np.searchsorted(ts_ref_sorted, t))
        candidates = []
        if idx < len(ts_ref_sorted):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        if not candidates:
            continue
        best = min(candidates, key=lambda i: abs(float(ts_ref_sorted[i]) - float(t)))
        dt = float(ts_ref_sorted[best]) - float(t)
        if abs(dt) > max_dt:
            continue
        ts_out.append(float(t))
        est_out.append(pose)
        ref_out.append(c2w_ref_sorted[best])
        dts.append(dt)
    if not est_out:
        return (
            np.asarray([], dtype=np.float64),
            np.empty((0, 4, 4), dtype=np.float64),
            np.empty((0, 4, 4), dtype=np.float64),
            {"n_matches": 0, "max_dt": float(max_dt)},
        )
    dts_np = np.asarray(dts, dtype=np.float64)
    info = {
        "n_matches": len(est_out),
        "max_dt": float(max_dt),
        "dt_mean": float(dts_np.mean()),
        "dt_abs_p95": float(np.percentile(np.abs(dts_np), 95)),
        "dt_abs_max": float(np.max(np.abs(dts_np))),
    }
    return (
        np.asarray(ts_out, dtype=np.float64),
        np.stack(est_out).astype(np.float64),
        np.stack(ref_out).astype(np.float64),
        info,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_trajectory(
    out_path: str,
    c2w_arr: np.ndarray,
    *,
    title: Optional[str] = None,
    gt: Optional[np.ndarray] = None,
    metrics: Optional[dict] = None,
) -> None:
    """Top-down + side view of one trajectory, optional GT overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz = c2w_arr[:, :3, 3]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _plot_2d(axes[0], xyz[:, 0], xyz[:, 2], "X", "Z (forward)", "Top-down (XZ)", gt=gt)
    _plot_2d(axes[1], xyz[:, 0], xyz[:, 1], "X", "Y (up)", "Side view (XY)", gt=gt, invert_y=True)
    if title:
        fig.suptitle(title, fontsize=12)
    if metrics:
        _add_metrics_box(fig, metrics)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_trajectory_pair(
    out_path: str,
    c2w_a: np.ndarray,
    c2w_b: np.ndarray,
    *,
    labels: tuple[str, str] = ("A", "B"),
    title: Optional[str] = None,
    metrics: Optional[dict] = None,
) -> None:
    """Overlay two trajectories for comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz_a = c2w_a[:, :3, 3]
    xyz_b = c2w_b[:, :3, 3]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, xkey, ykey, xlabel, ylabel, view_title in [
        (axes[0], 0, 2, "X", "Z", "Top-down (XZ)"),
        (axes[1], 0, 1, "X", "Y", "Side view (XY)"),
    ]:
        ax.plot(xyz_a[:, xkey], xyz_a[:, ykey], lw=1.5, label=labels[0], color="steelblue")
        ax.plot(xyz_b[:, xkey], xyz_b[:, ykey], lw=1.5, label=labels[1], color="tomato", linestyle="--")
        ax.scatter(xyz_a[0, xkey], xyz_a[0, ykey], marker="o", s=60, color="steelblue", zorder=5)
        ax.scatter(xyz_b[0, xkey], xyz_b[0, ykey], marker="o", s=60, color="tomato", zorder=5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(view_title)
        ax.legend()
        ax.set_aspect("equal")
        if ykey == 1:
            ax.invert_yaxis()
    if title:
        fig.suptitle(title, fontsize=12)
    if metrics:
        _add_metrics_box(fig, metrics)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_2d(ax, xs, ys, xlabel, ylabel, title, *, gt=None, invert_y=False):
    ax.plot(xs, ys, lw=1.5, label="estimated", color="steelblue")
    ax.scatter(xs[0], ys[0], marker="o", s=60, color="steelblue", zorder=5)
    if gt is not None:
        gx, gy = gt[:, :3, 3][:, 0], gt[:, :3, 3][:, 2 if ylabel.startswith("Z") else 1]
        ax.plot(gx, gy, lw=1.5, linestyle="--", label="GT", color="tomato")
        ax.scatter(gx[0], gy[0], marker="o", s=60, color="tomato", zorder=5)
        ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal")
    if invert_y:
        ax.invert_yaxis()


def _trajectory_distance(c2w_arr: np.ndarray) -> float:
    xyz = c2w_arr[:, :3, 3]
    if len(xyz) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


def _fmt_metric(value, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        if not np.isfinite(float(value)):
            return "n/a"
        return f"{float(value):.4g}{suffix}"
    except Exception:
        return "n/a"


def _add_metrics_box(fig, metrics: dict) -> None:
    lines = [
        f"distance: {_fmt_metric(metrics.get('distance_m'), ' m')}",
        f"ATE RMSE: {_fmt_metric(metrics.get('ate_rmse'), ' m')}",
        f"ATE mean: {_fmt_metric(metrics.get('ate_mean'), ' m')}",
        f"RPE trans: {_fmt_metric(metrics.get('rpe_trans_rmse'), ' m')}",
        f"RPE rot: {_fmt_metric(metrics.get('rpe_rot_deg_rmse'), ' deg')}",
    ]
    if metrics.get("gt_distance_m") is not None:
        lines.insert(1, f"GT distance: {_fmt_metric(metrics.get('gt_distance_m'), ' m')}")
    fig.text(
        0.01,
        0.01,
        "\n".join(lines),
        ha="left",
        va="bottom",
        fontsize=9,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )


# ---------------------------------------------------------------------------
# Umeyama SE(3) alignment (no scale)
# ---------------------------------------------------------------------------

def _umeyama_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Align src positions [N,3] to dst [N,3] via SE(3) least-squares.

    Returns 4×4 transform T such that T @ src ≈ dst.
    """
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    H = src_c.T @ dst_c / n
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = mu_dst - R @ mu_src
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ---------------------------------------------------------------------------
# ATE / RPE metrics
# ---------------------------------------------------------------------------

def trajectory_metrics(c2w_est: np.ndarray, c2w_ref: np.ndarray) -> dict:
    """Compute ATE and RPE between estimated and reference trajectories.

    Both arrays are [N,4,4] world-from-camera matrices.
    Aligns est to ref via Umeyama SE(3) before computing ATE.
    Returns dict with ate_rmse, ate_mean, rpe_trans_rmse, rpe_rot_deg_rmse, n_frames.
    """
    n = min(len(c2w_est), len(c2w_ref))
    if n < 2:
        return {"ate_rmse": None, "ate_mean": None,
                "rpe_trans_rmse": None, "rpe_rot_deg_rmse": None, "n_frames": n,
                "distance_m": _trajectory_distance(c2w_est[:n]),
                "gt_distance_m": _trajectory_distance(c2w_ref[:n])}

    pos_est = c2w_est[:n, :3, 3]
    pos_ref = c2w_ref[:n, :3, 3]

    T_align = _umeyama_align(pos_est, pos_ref)
    pos_est_aligned = (T_align[:3, :3] @ pos_est.T).T + T_align[:3, 3]

    ate_err = np.linalg.norm(pos_est_aligned - pos_ref, axis=1)
    ate_rmse = float(np.sqrt(np.mean(ate_err ** 2)))
    ate_mean = float(np.mean(ate_err))

    # RPE: relative pose error between consecutive frames
    trans_errs, rot_errs = [], []
    R_align = T_align[:3, :3]
    for i in range(n - 1):
        # relative pose in reference
        dT_ref = np.linalg.inv(c2w_ref[i]) @ c2w_ref[i + 1]
        # relative pose in aligned estimate
        c2w_est_i = c2w_est[i].copy()
        c2w_est_i[:3, 3] = pos_est_aligned[i]
        c2w_est_i[:3, :3] = R_align @ c2w_est[i, :3, :3]
        c2w_est_i1 = c2w_est[i + 1].copy()
        c2w_est_i1[:3, 3] = pos_est_aligned[i + 1]
        c2w_est_i1[:3, :3] = R_align @ c2w_est[i + 1, :3, :3]
        dT_est = np.linalg.inv(c2w_est_i) @ c2w_est_i1

        err_T = np.linalg.inv(dT_ref) @ dT_est
        trans_errs.append(np.linalg.norm(err_T[:3, 3]))
        # rotation angle from trace
        cos_angle = np.clip((np.trace(err_T[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        rot_errs.append(np.degrees(np.arccos(cos_angle)))

    rpe_trans = float(np.sqrt(np.mean(np.array(trans_errs) ** 2)))
    rpe_rot = float(np.sqrt(np.mean(np.array(rot_errs) ** 2)))

    return {
        "ate_rmse": ate_rmse,
        "ate_mean": ate_mean,
        "rpe_trans_rmse": rpe_trans,
        "rpe_rot_deg_rmse": rpe_rot,
        "n_frames": n,
        "distance_m": _trajectory_distance(c2w_est[:n]),
        "gt_distance_m": _trajectory_distance(c2w_ref[:n]),
    }


# ---------------------------------------------------------------------------
# Camera list → numpy arrays
# ---------------------------------------------------------------------------

def _cams_to_c2w(cams) -> tuple[np.ndarray, np.ndarray]:
    """Extract timestamps (float seconds) and c2w [N,4,4] from camera list."""
    import torch
    ts, mats = [], []
    for cam in cams:
        # world-from-camera: column-major stored as R (3×3) and T (3,)
        # In the 3DGS convention cam.R is world-to-camera rotation,
        # cam.T is world-to-camera translation. Invert to get c2w.
        R_w2c = np.array(cam.R, dtype=np.float64)   # 3×3
        t_w2c = np.array(cam.T, dtype=np.float64)   # 3,
        # c2w
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_c2w
        T[:3, 3] = t_c2w
        mats.append(T)
        ts.append(float(getattr(cam, "_streaming_timestamp", getattr(cam, "fid", float(len(ts))))))
    return np.array(ts, dtype=np.float64), np.stack(mats)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_trajectory_eval(
    output_dir: str,
    train_cams: Sequence,
    *,
    gt_cams: Optional[Sequence] = None,
    gt_tum_path: Optional[str] = None,
    association_max_dt: float = 0.03,
    method_label: str = "estimated",
) -> None:
    """Save TUM trajectory, PNG plot, and metrics.json for one run."""
    os.makedirs(output_dir, exist_ok=True)

    if not train_cams:
        return

    ts, c2w = _cams_to_c2w(train_cams)

    tum_path = os.path.join(output_dir, "trajectory_tum.txt")
    save_tum_trajectory(tum_path, ts, c2w)

    gt_c2w = None
    association_info = None
    if gt_cams:
        _, gt_c2w = _cams_to_c2w(gt_cams)
    elif gt_tum_path:
        if os.path.exists(gt_tum_path):
            gt_ts, gt_all = load_tum_trajectory(gt_tum_path)
            matched_ts, c2w_matched, gt_c2w, association_info = associate_by_timestamp(
                ts,
                c2w,
                gt_ts,
                gt_all,
                max_dt=association_max_dt,
            )
            if len(matched_ts) > 0:
                ts = matched_ts
                c2w = c2w_matched
                save_tum_trajectory(os.path.join(output_dir, "trajectory_tum_associated.txt"), ts, c2w)
                save_tum_trajectory(os.path.join(output_dir, "groundtruth_tum_associated.txt"), ts, gt_c2w)
            else:
                gt_c2w = None
        else:
            print(f"[trajectory_eval] GT path not found: {gt_tum_path}", flush=True)

    if gt_c2w is not None:
        metrics = trajectory_metrics(c2w, gt_c2w)
    else:
        metrics = {"ate_rmse": None, "ate_mean": None,
                   "rpe_trans_rmse": None, "rpe_rot_deg_rmse": None,
                   "n_frames": len(train_cams),
                   "distance_m": _trajectory_distance(c2w),
                   "gt_distance_m": None}
    metrics["method"] = method_label
    metrics["n_train_cams"] = len(train_cams)
    if gt_tum_path:
        metrics["gt_tum_path"] = gt_tum_path
    if association_info:
        metrics["association"] = association_info

    png_path = os.path.join(output_dir, "trajectory.png")
    plot_trajectory(png_path, c2w, title=f"Trajectory — {method_label}", gt=gt_c2w, metrics=metrics)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"[trajectory_eval] → {output_dir}  "
        f"n={len(train_cams)} ATE={metrics.get('ate_rmse')}",
        flush=True,
    )
