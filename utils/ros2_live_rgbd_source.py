import time

import numpy as np

from utils.rgbd_frames import (
    CameraIntrinsics,
    RGBDFrame,
    RGBDFrameSource,
    decode_color_image,
    decode_depth_image,
    reproject_depth_to_color,
)
from utils.rosbag_rgbd_source import _camera_info_to_intrinsics, _pose_to_matrix, _transform_to_matrix


def _optional_ros2():
    try:
        import rclpy
        from cv_bridge import CvBridge
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import CameraInfo, Image
        from tf2_msgs.msg import TFMessage
    except Exception as exc:
        raise RuntimeError(
            "Live ROS2 RGB-D capture requires optional ROS2 Python dependencies "
            "(rclpy, cv_bridge, sensor_msgs, geometry_msgs/nav_msgs, tf2_msgs). "
            "Install/use this script inside a ROS2 environment."
        ) from exc
    return rclpy, CvBridge, CameraInfo, Image, TFMessage, PoseStamped, Odometry


class ROS2LiveRGBDSource(RGBDFrameSource):
    def __init__(
        self,
        rgb_topic,
        depth_topic,
        camera_info_topic,
        pose_source="tf",
        world_frame="map",
        camera_frame="camera_color_optical_frame",
        tf_topic="/tf",
        tf_static_topic="/tf_static",
        pose_topic="",
        depth_camera_info_topic="",
        depth_frame="",
        color_frame="",
        depth_is_aligned_to_color=True,
        frame_stride=1,
        max_frames=0,
        duration=0.0,
        max_association_dt=0.03,
        tf_tolerance=0.05,
        depth_scale=1000.0,
        min_depth=0.1,
        max_depth=8.0,
    ):
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.camera_info_topic = camera_info_topic
        self.pose_source = pose_source
        self.world_frame = world_frame
        self.camera_frame = camera_frame
        self.tf_topic = tf_topic
        self.tf_static_topic = tf_static_topic
        self.pose_topic = pose_topic
        self.depth_camera_info_topic = depth_camera_info_topic
        self.depth_frame = depth_frame
        self.color_frame = color_frame or camera_frame
        self.depth_is_aligned_to_color = bool(depth_is_aligned_to_color)
        self.frame_stride = max(1, int(frame_stride))
        self.max_frames = max(0, int(max_frames))
        self.duration = max(0.0, float(duration))
        self.max_association_dt = float(max_association_dt)
        self.tf_tolerance = float(tf_tolerance)
        self.depth_scale = float(depth_scale)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)

    def __iter__(self):
        rclpy, CvBridge, CameraInfo, Image, TFMessage, PoseStamped, Odometry = _optional_ros2()
        from utils.rosbag_rgbd_source import _TfGraph, _msg_time

        rclpy.init(args=None)
        node = rclpy.create_node("rgbd_sequence_capture")
        bridge = CvBridge()
        tf_graph = _TfGraph()
        state = {
            "color_intrinsics": None,
            "depth_intrinsics": None,
            "depth_msgs": [],
            "pose_msgs": [],
            "rgb_count": 0,
            "emitted": 0,
            "queue": [],
        }

        def on_color_info(msg):
            state["color_intrinsics"] = _camera_info_to_intrinsics(
                msg,
                depth_scale=self.depth_scale,
                camera_frame=self.camera_frame,
                world_frame=self.world_frame,
            )

        def on_depth_info(msg):
            state["depth_intrinsics"] = _camera_info_to_intrinsics(
                msg,
                depth_scale=self.depth_scale,
                camera_frame=self.depth_frame,
                world_frame=self.world_frame,
            )

        def on_depth(msg):
            arr = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            state["depth_msgs"].append((_msg_time(msg, int(time.time() * 1e9)), arr, msg.encoding))
            state["depth_msgs"] = state["depth_msgs"][-200:]

        def on_pose(msg):
            pose = getattr(msg, "pose", msg)
            if hasattr(pose, "pose"):
                pose = pose.pose
            state["pose_msgs"].append((_msg_time(msg, int(time.time() * 1e9)), _pose_to_matrix(pose)))
            state["pose_msgs"] = state["pose_msgs"][-200:]

        def on_tf(msg):
            for t in msg.transforms:
                stamp = float(getattr(t.header.stamp, "sec", 0)) + float(getattr(t.header.stamp, "nanosec", 0)) * 1e-9
                tf_graph.add(stamp, str(t.header.frame_id), str(t.child_frame_id), _transform_to_matrix(t.transform))

        def on_rgb(msg):
            if state["color_intrinsics"] is None:
                return
            if state["rgb_count"] % self.frame_stride != 0:
                state["rgb_count"] += 1
                return
            state["rgb_count"] += 1

            rgb_time = _msg_time(msg, int(time.time() * 1e9))
            if not state["depth_msgs"]:
                return
            depth_times = np.array([d[0] for d in state["depth_msgs"]], dtype=np.float64)
            depth_idx = int(np.argmin(np.abs(depth_times - rgb_time)))
            if abs(depth_times[depth_idx] - rgb_time) > self.max_association_dt:
                return

            if self.pose_source == "tf":
                c2w = tf_graph.lookup(self.world_frame, self.camera_frame, rgb_time, tolerance=self.tf_tolerance)
                if c2w is None:
                    return
            else:
                if not state["pose_msgs"]:
                    return
                pose_times = np.array([p[0] for p in state["pose_msgs"]], dtype=np.float64)
                pose_idx = int(np.argmin(np.abs(pose_times - rgb_time)))
                if abs(pose_times[pose_idx] - rgb_time) > self.max_association_dt:
                    return
                c2w = state["pose_msgs"][pose_idx][1]

            rgb_raw = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            rgb = decode_color_image(rgb_raw, msg.encoding)
            depth_raw, depth_encoding = state["depth_msgs"][depth_idx][1], state["depth_msgs"][depth_idx][2]
            depth = decode_depth_image(depth_raw, depth_encoding)

            if not self.depth_is_aligned_to_color:
                if state["depth_intrinsics"] is None or not self.depth_frame:
                    return
                depth_to_color = tf_graph.lookup(self.color_frame, self.depth_frame, rgb_time, tolerance=self.tf_tolerance)
                if depth_to_color is None:
                    return
                depth = reproject_depth_to_color(
                    depth,
                    state["depth_intrinsics"],
                    state["color_intrinsics"],
                    depth_to_color,
                    min_depth=self.min_depth,
                    max_depth=self.max_depth,
                )

            state["queue"].append(
                RGBDFrame(
                    index=state["emitted"],
                    timestamp=rgb_time,
                    rgb=rgb,
                    depth=depth,
                    intrinsics=state["color_intrinsics"],
                    c2w=c2w,
                )
            )
            state["emitted"] += 1

        node.create_subscription(CameraInfo, self.camera_info_topic, on_color_info, 10)
        if self.depth_camera_info_topic:
            node.create_subscription(CameraInfo, self.depth_camera_info_topic, on_depth_info, 10)
        node.create_subscription(Image, self.depth_topic, on_depth, 10)
        node.create_subscription(TFMessage, self.tf_topic, on_tf, 50)
        node.create_subscription(TFMessage, self.tf_static_topic, on_tf, 10)
        if self.pose_topic:
            msg_type = Odometry if self.pose_source == "odom" else PoseStamped
            node.create_subscription(msg_type, self.pose_topic, on_pose, 50)
        node.create_subscription(Image, self.rgb_topic, on_rgb, 10)

        started = time.time()
        try:
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)
                while state["queue"]:
                    yield state["queue"].pop(0)
                if self.max_frames and state["emitted"] >= self.max_frames:
                    break
                if self.duration and time.time() - started >= self.duration:
                    break
        finally:
            node.destroy_node()
            rclpy.shutdown()
