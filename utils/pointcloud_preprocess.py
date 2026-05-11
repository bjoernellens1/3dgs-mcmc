import time
from dataclasses import dataclass

import numpy as np


@dataclass
class PointCloudPreprocessConfig:
    backend: str = "none"
    voxel_size: float = 0.0
    outlier_filter: str = "none"
    stat_nb_neighbors: int = 20
    stat_std_ratio: float = 2.0
    radius: float = 0.05
    min_neighbors: int = 4
    estimate_normals: bool = False
    force_regenerate: bool = False


def pointcloud_preprocess_config(**kwargs):
    return PointCloudPreprocessConfig(
        backend=str(kwargs.get("pointcloud_preprocess", "none")).lower(),
        voxel_size=float(kwargs.get("pcd_voxel_size", 0.0)),
        outlier_filter=str(kwargs.get("pcd_outlier_filter", "none")).lower(),
        stat_nb_neighbors=int(kwargs.get("pcd_stat_nb_neighbors", 20)),
        stat_std_ratio=float(kwargs.get("pcd_stat_std_ratio", 2.0)),
        radius=float(kwargs.get("pcd_radius", 0.05)),
        min_neighbors=int(kwargs.get("pcd_min_neighbors", 4)),
        estimate_normals=bool(kwargs.get("pcd_estimate_normals", False)),
        force_regenerate=bool(kwargs.get("pcd_force_regenerate", False)),
    )


def pointcloud_cache_suffix(config, extra_suffix=""):
    if config.backend == "none":
        return extra_suffix
    voxel = f"v{config.voxel_size:.3f}" if config.voxel_size > 0 else "vnone"
    filt = config.outlier_filter
    normals = "_normals" if config.estimate_normals else ""
    return f"_{config.backend}_{voxel}_{filt}{normals}{extra_suffix}"


def preprocess_pointcloud(points, colors, normals=None, config=None, label="pointcloud"):
    config = config or PointCloudPreprocessConfig()
    _validate_config(config)

    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    if normals is None:
        normals = np.zeros_like(points, dtype=np.float32)
    else:
        normals = np.asarray(normals, dtype=np.float32)

    if config.backend == "none":
        return points, colors, normals, {
            "backend": "none",
            "input_points": int(points.shape[0]),
            "output_points": int(points.shape[0]),
            "elapsed_sec": 0.0,
        }

    if config.backend != "open3d":
        raise ValueError("pointcloud_preprocess must be one of: none, open3d")

    _ensure_isatty_compat()
    try:
        import open3d as o3d
    except ModuleNotFoundError as exc:
        if exc.name != "open3d":
            raise RuntimeError(f"Open3D import failed because dependency '{exc.name}' is missing.") from exc
        raise RuntimeError(
            "--pointcloud_preprocess open3d requires Open3D in the project container. "
            "Rebuild the train image after the Dockerfile update."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Open3D import failed: {exc}") from exc

    start = time.perf_counter()
    input_count = int(points.shape[0])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0).astype(np.float64))
    if normals.shape == points.shape and np.any(np.isfinite(normals) & (np.abs(normals) > 0)):
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))

    if config.voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=config.voxel_size)

    if config.outlier_filter == "statistical" and len(pcd.points) > 1:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=min(max(1, config.stat_nb_neighbors), len(pcd.points) - 1),
            std_ratio=max(0.0, config.stat_std_ratio),
        )
    elif config.outlier_filter == "radius" and len(pcd.points) > 0:
        pcd, _ = pcd.remove_radius_outlier(
            nb_points=max(1, config.min_neighbors),
            radius=max(1e-8, config.radius),
        )

    if config.estimate_normals and len(pcd.points) > 0:
        normal_radius = config.radius if config.radius > 0 else max(config.voxel_size * 2.5, 0.05)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=max(normal_radius, 1e-8),
                max_nn=max(8, config.stat_nb_neighbors),
            )
        )
        pcd.normalize_normals()

    out_points = np.asarray(pcd.points, dtype=np.float32)
    out_colors = np.asarray(pcd.colors, dtype=np.float32)
    if pcd.has_normals():
        out_normals = np.asarray(pcd.normals, dtype=np.float32)
    else:
        out_normals = np.zeros_like(out_points, dtype=np.float32)

    elapsed = time.perf_counter() - start
    print(
        f"[pointcloud] {label}: preprocess=open3d "
        f"points={input_count}->{out_points.shape[0]} "
        f"voxel={config.voxel_size:g} outlier={config.outlier_filter} "
        f"normals={int(config.estimate_normals)} time={elapsed:.3f}s",
        flush=True,
    )

    if out_points.shape[0] == 0:
        raise RuntimeError(f"Open3D preprocessing removed all points for {label}.")

    return out_points, out_colors, out_normals, {
        "backend": "open3d",
        "input_points": input_count,
        "output_points": int(out_points.shape[0]),
        "elapsed_sec": float(elapsed),
    }


def _validate_config(config):
    if config.backend not in {"none", "open3d"}:
        raise ValueError("pointcloud_preprocess must be one of: none, open3d")
    if config.outlier_filter not in {"none", "statistical", "radius"}:
        raise ValueError("pcd_outlier_filter must be one of: none, statistical, radius")
    if config.voxel_size < 0:
        raise ValueError("pcd_voxel_size must be >= 0")


def _ensure_isatty_compat():
    # train.py replaces stdout/stderr with a timestamping wrapper that does not
    # expose isatty(). Open3D imports Dash/IPython, which assumes that method
    # exists during import even when no visualization API is used.
    import sys

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and not hasattr(stream, "isatty"):
            try:
                setattr(stream, "isatty", lambda: False)
            except Exception:
                pass
