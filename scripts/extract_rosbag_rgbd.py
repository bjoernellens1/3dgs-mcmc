#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene.dataset_readers import storePly
from utils.rgbd_frames import (
    read_frame_records,
    rgbd_pointcloud_from_records,
    write_intrinsics,
    write_metadata,
    write_sequence_frame,
)
from utils.rosbag_rgbd_source import RosbagFileRGBDSource


def _reset_output(out, overwrite):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise RuntimeError(f"Output folder is not empty: {out}. Use --overwrite to replace it.")
        shutil.rmtree(out)
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    return out


def _write_init_ply(out, intrinsics, args):
    records = read_frame_records(out / "frames.jsonl")
    points, colors, _ = rgbd_pointcloud_from_records(
        root=out,
        records=records,
        intrinsics=intrinsics,
        depth_stride=args.depth_stride,
        init_frames=args.init_frames,
        max_points=args.max_init_points,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
    storePly(out / "init_rgbd.ply", points, np.clip(colors * 255.0, 0, 255))


def parse_args():
    parser = argparse.ArgumentParser(description="Extract a normalized RGB-D sequence from a rosbag/rosbag2.")
    parser.add_argument("--bag", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rgb-topic", default="/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    parser.add_argument("--depth-camera-info-topic", default="")
    parser.add_argument("--pose-source", default="tf", choices=["tf", "pose", "odom"])
    parser.add_argument("--pose-topic", default="")
    parser.add_argument("--tf-topic", default="/tf")
    parser.add_argument("--tf-static-topic", default="/tf_static")
    parser.add_argument("--world-frame", default="map")
    parser.add_argument("--camera-frame", default="camera_color_optical_frame")
    parser.add_argument("--color-frame", default="")
    parser.add_argument("--depth-frame", default="")
    parser.add_argument("--depth-is-aligned-to-color", dest="depth_is_aligned_to_color", action="store_true", default=True)
    parser.add_argument("--no-depth-is-aligned-to-color", dest="depth_is_aligned_to_color", action="store_false")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-association-dt", type=float, default=0.03)
    parser.add_argument("--tf-tolerance", type=float, default=0.05)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=8.0)
    parser.add_argument("--depth-stride", type=int, default=4)
    parser.add_argument("--init-frames", type=int, default=300)
    parser.add_argument("--max-init-points", type=int, default=250000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out = _reset_output(args.out, overwrite=args.overwrite)
    source = RosbagFileRGBDSource(
        bag=args.bag,
        rgb_topic=args.rgb_topic,
        depth_topic=args.depth_topic,
        camera_info_topic=args.camera_info_topic,
        pose_source=args.pose_source,
        world_frame=args.world_frame,
        camera_frame=args.camera_frame,
        tf_topic=args.tf_topic,
        tf_static_topic=args.tf_static_topic,
        pose_topic=args.pose_topic,
        depth_camera_info_topic=args.depth_camera_info_topic,
        depth_frame=args.depth_frame,
        color_frame=args.color_frame,
        depth_is_aligned_to_color=args.depth_is_aligned_to_color,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        max_association_dt=args.max_association_dt,
        tf_tolerance=args.tf_tolerance,
        depth_scale=args.depth_scale,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )

    intrinsics = None
    count = 0
    for frame in source:
        if intrinsics is None:
            intrinsics = frame.intrinsics
            write_intrinsics(out / "intrinsics.json", intrinsics)
        write_sequence_frame(out, frame, depth_scale=args.depth_scale)
        count += 1

    if count == 0 or intrinsics is None:
        raise RuntimeError("No RGB-D frames were extracted from the rosbag.")

    _write_init_ply(out, intrinsics, args)
    write_metadata(
        out / "metadata.json",
        {
            "source": "rosbag",
            "bag": str(args.bag),
            "rgb_topic": args.rgb_topic,
            "depth_topic": args.depth_topic,
            "camera_info_topic": args.camera_info_topic,
            "depth_camera_info_topic": args.depth_camera_info_topic,
            "pose_source": args.pose_source,
            "pose_topic": args.pose_topic,
            "world_frame": args.world_frame,
            "camera_frame": args.camera_frame,
            "color_frame": args.color_frame or args.camera_frame,
            "depth_frame": args.depth_frame,
            "depth_is_aligned_to_color": args.depth_is_aligned_to_color,
            "frame_count": count,
            "depth_scale": args.depth_scale,
            "max_association_dt": args.max_association_dt,
        },
    )
    print(f"Wrote {count} RGB-D frames to {out}")


if __name__ == "__main__":
    main()
