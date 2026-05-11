import shutil
from pathlib import Path

import numpy as np

from scene.readers.common import storePly
from utils.pointcloud_preprocess import pointcloud_preprocess_config, preprocess_pointcloud
from utils.rgbd_frames import (
    depth_sanity,
    pointcloud_sanity,
    read_frame_records,
    rgbd_pointcloud_from_records,
    write_intrinsics,
    write_metadata,
    write_sequence_frame,
)


def reset_sequence_output(out, overwrite):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output folder is not empty: {out}. Use --overwrite to replace it.")
        shutil.rmtree(out)
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    return out


def write_rgbd_sequence_from_source(source, out, args, metadata):
    out = reset_sequence_output(out, overwrite=args.overwrite)

    intrinsics = None
    count = 0
    depth_reports = []
    metadata_tmp = out / "metadata.json.tmp"
    write_metadata(metadata_tmp, {**metadata, "complete": False})

    for frame in source:
        if intrinsics is None:
            intrinsics = frame.intrinsics
            write_intrinsics(out / "intrinsics.json", intrinsics)
        if count < 20:
            depth_reports.append(depth_sanity(frame.depth, frame.intrinsics, args.min_depth, args.max_depth))
        write_sequence_frame(out, frame, depth_scale=args.depth_scale)
        count += 1

    if count == 0 or intrinsics is None:
        raise RuntimeError("No RGB-D frames were written.")

    records = read_frame_records(out / "frames.jsonl")
    points, colors, normals = rgbd_pointcloud_from_records(
        root=out,
        records=records,
        intrinsics=intrinsics,
        depth_stride=args.depth_stride,
        init_frames=args.init_frames,
        max_points=args.max_init_points,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
    preprocess_cfg = pointcloud_preprocess_config(
        pointcloud_preprocess=getattr(args, "pointcloud_preprocess", "none"),
        pcd_voxel_size=getattr(args, "pcd_voxel_size", 0.0),
        pcd_outlier_filter=getattr(args, "pcd_outlier_filter", "none"),
        pcd_stat_nb_neighbors=getattr(args, "pcd_stat_nb_neighbors", 20),
        pcd_stat_std_ratio=getattr(args, "pcd_stat_std_ratio", 2.0),
        pcd_radius=getattr(args, "pcd_radius", 0.05),
        pcd_min_neighbors=getattr(args, "pcd_min_neighbors", 4),
        pcd_estimate_normals=getattr(args, "pcd_estimate_normals", False),
    )
    points, colors, normals, preprocess_summary = preprocess_pointcloud(
        points,
        colors,
        normals,
        config=preprocess_cfg,
        label="rgbd_sequence_export_init",
    )
    storePly(out / "init_rgbd.ply", points, np.clip(colors * 255.0, 0, 255), normals=normals)

    depth_summary = _summarize_depth(depth_reports)
    pcd_summary = pointcloud_sanity(points, records[: args.init_frames] if args.init_frames else records)
    final_metadata = {
        **metadata,
        "complete": True,
        "frame_count": count,
        "depth_scale": args.depth_scale,
        "depth_sanity": depth_summary,
        "pointcloud_sanity": pcd_summary,
        "pointcloud_preprocess": preprocess_summary,
    }
    write_metadata(out / "metadata.json", final_metadata)
    metadata_tmp.unlink(missing_ok=True)

    print(
        "[RGBD] depth valid={valid:.1%} median={median:.3f}m min={minv:.3f}m max={maxv:.3f}m".format(
            valid=depth_summary["valid_ratio"],
            median=depth_summary["median_m"],
            minv=depth_summary["min_m"],
            maxv=depth_summary["max_m"],
        )
    )
    print(
        "[RGBD] init points={num_points} bbox={bbox_min}->{bbox_max} camera_bbox={camera_bbox_min}->{camera_bbox_max} "
        "median_cam_dist={median_camera_distance:.3f}m behind={behind_camera_ratio:.1%}".format(**pcd_summary)
    )
    return out, count


def _summarize_depth(reports):
    if not reports:
        return {"valid_ratio": 0.0, "median_m": 0.0, "min_m": 0.0, "max_m": 0.0}
    return {
        "valid_ratio": float(np.mean([r["valid_ratio"] for r in reports])),
        "median_m": float(np.median([r["median_m"] for r in reports if r["valid_ratio"] > 0.0] or [0.0])),
        "min_m": float(np.min([r["min_m"] for r in reports if r["valid_ratio"] > 0.0] or [0.0])),
        "max_m": float(np.max([r["max_m"] for r in reports if r["valid_ratio"] > 0.0] or [0.0])),
    }
