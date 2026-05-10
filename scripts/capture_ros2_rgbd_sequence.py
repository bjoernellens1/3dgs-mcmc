#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.rgbd_sequence_writer import write_rgbd_sequence_from_source
from utils.ros2_live_rgbd_source import ROS2LiveRGBDSource


def parse_args():
    parser = argparse.ArgumentParser(description="Capture a normalized RGB-D sequence from live ROS2 topics.")
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
    parser.add_argument("--pose-frame", default="")
    parser.add_argument("--pose-is-camera-frame", action="store_true")
    parser.add_argument("--color-frame", default="")
    parser.add_argument("--depth-frame", default="")
    parser.add_argument("--depth-is-aligned-to-color", dest="depth_is_aligned_to_color", action="store_true", default=True)
    parser.add_argument("--no-depth-is-aligned-to-color", dest="depth_is_aligned_to_color", action="store_false")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-association-dt", type=float, default=0.03)
    parser.add_argument("--tf-tolerance", type=float, default=0.05)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=8.0)
    parser.add_argument("--assume-rectified", action="store_true")
    parser.add_argument("--fail-on-distortion", dest="fail_on_distortion", action="store_true", default=True)
    parser.add_argument("--no-fail-on-distortion", dest="fail_on_distortion", action="store_false")
    parser.add_argument("--rgb-encoding-override", default="", choices=["", "rgb8", "bgr8"])
    parser.add_argument("--depth-stride", type=int, default=4)
    parser.add_argument("--init-frames", type=int, default=300)
    parser.add_argument("--max-init-points", type=int, default=250000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    source = ROS2LiveRGBDSource(
        rgb_topic=args.rgb_topic,
        depth_topic=args.depth_topic,
        camera_info_topic=args.camera_info_topic,
        pose_source=args.pose_source,
        world_frame=args.world_frame,
        camera_frame=args.camera_frame,
        pose_frame=args.pose_frame,
        pose_is_camera_frame=args.pose_is_camera_frame,
        tf_topic=args.tf_topic,
        tf_static_topic=args.tf_static_topic,
        pose_topic=args.pose_topic,
        depth_camera_info_topic=args.depth_camera_info_topic,
        depth_frame=args.depth_frame,
        color_frame=args.color_frame,
        depth_is_aligned_to_color=args.depth_is_aligned_to_color,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        duration=args.duration,
        max_association_dt=args.max_association_dt,
        tf_tolerance=args.tf_tolerance,
        depth_scale=args.depth_scale,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        assume_rectified=args.assume_rectified,
        fail_on_distortion=args.fail_on_distortion,
        rgb_encoding_override=args.rgb_encoding_override,
    )

    out, count = write_rgbd_sequence_from_source(
        source,
        args.out,
        args,
        {
            "source": "ros2_live",
            "rgb_topic": args.rgb_topic,
            "depth_topic": args.depth_topic,
            "camera_info_topic": args.camera_info_topic,
            "depth_camera_info_topic": args.depth_camera_info_topic,
            "pose_source": args.pose_source,
            "pose_topic": args.pose_topic,
            "pose_frame": args.pose_frame,
            "pose_is_camera_frame": args.pose_is_camera_frame,
            "world_frame": args.world_frame,
            "camera_frame": args.camera_frame,
            "color_frame": args.color_frame or args.camera_frame,
            "depth_frame": args.depth_frame,
            "depth_is_aligned_to_color": args.depth_is_aligned_to_color,
            "max_association_dt": args.max_association_dt,
        },
    )
    print(f"Wrote {count} RGB-D frames to {out}")


if __name__ == "__main__":
    main()
