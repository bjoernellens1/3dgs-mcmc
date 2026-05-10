from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from utils.rgbd_frames import (
    CameraIntrinsics,
    RGBDFrame,
    RGBDFrameSource,
    decode_color_image,
    decode_depth_image,
    reproject_depth_to_color,
)


def _optional_rosbags():
    try:
        from rosbags.highlevel import AnyReader
    except Exception as exc:
        raise RuntimeError(
            "Offline rosbag extraction requires the optional 'rosbags' package. "
            "Install it in the active environment with: pip install rosbags"
        ) from exc
    return AnyReader


def _stamp_to_sec(stamp):
    if stamp is None:
        return None
    sec = getattr(stamp, "sec", getattr(stamp, "secs", 0))
    nsec = getattr(stamp, "nanosec", getattr(stamp, "nsecs", 0))
    return float(sec) + float(nsec) * 1e-9


def _msg_time(msg, fallback_ns):
    header = getattr(msg, "header", None)
    t = _stamp_to_sec(getattr(header, "stamp", None))
    return t if t is not None else float(fallback_ns) * 1e-9


def _field(obj, *names, default=None):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _frame_id(obj):
    header = getattr(obj, "header", None)
    return str(getattr(header, "frame_id", ""))


def _quat_xyzw_to_rotmat(q):
    qx, qy, qz, qw = q
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ], dtype=np.float32)


def _transform_to_matrix(transform):
    trans = transform.translation
    rot = transform.rotation
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = _quat_xyzw_to_rotmat([
        float(rot.x),
        float(rot.y),
        float(rot.z),
        float(rot.w),
    ])
    mat[:3, 3] = [float(trans.x), float(trans.y), float(trans.z)]
    return mat


def _pose_to_matrix(pose):
    pos = pose.position
    rot = pose.orientation
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = _quat_xyzw_to_rotmat([
        float(rot.x),
        float(rot.y),
        float(rot.z),
        float(rot.w),
    ])
    mat[:3, 3] = [float(pos.x), float(pos.y), float(pos.z)]
    return mat


def _camera_info_to_intrinsics(msg, depth_scale, camera_frame="", world_frame=""):
    k = list(_field(msg, "k", "K"))
    return CameraIntrinsics(
        width=int(msg.width),
        height=int(msg.height),
        fx=float(k[0]),
        fy=float(k[4]),
        cx=float(k[2]),
        cy=float(k[5]),
        depth_scale=float(depth_scale),
        camera_frame=camera_frame or _frame_id(msg),
        world_frame=world_frame,
    )


def _image_msg_to_array(msg):
    encoding = str(msg.encoding)
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    data = bytes(msg.data)
    enc = encoding.lower()
    if enc in {"rgb8", "bgr8", "8uc3"}:
        bpp, dtype, channels = 3, np.uint8, 3
    elif enc in {"rgba8", "bgra8"}:
        bpp, dtype, channels = 4, np.uint8, 4
    elif enc in {"mono8", "8uc1"}:
        bpp, dtype, channels = 1, np.uint8, 1
    elif enc in {"16uc1", "mono16"}:
        bpp, dtype, channels = 2, np.uint16, 1
    elif enc == "32fc1":
        bpp, dtype, channels = 4, np.float32, 1
    else:
        raise ValueError(f"Unsupported sensor_msgs/Image encoding: {encoding}")

    rows = np.frombuffer(data, dtype=np.uint8).reshape(height, step)
    packed = rows[:, : width * bpp].copy()
    arr = np.frombuffer(packed.tobytes(), dtype=dtype)
    if bool(getattr(msg, "is_bigendian", False)) and arr.dtype.itemsize > 1:
        arr = arr.byteswap()
    if channels == 1:
        return arr.reshape(height, width), encoding
    return arr.reshape(height, width, channels), encoding


class _TfGraph:
    def __init__(self):
        self.transforms: List[Tuple[float, str, str, np.ndarray]] = []

    def add(self, stamp, parent, child, matrix):
        if parent and child:
            self.transforms.append((float(stamp), str(parent), str(child), np.asarray(matrix, dtype=np.float32)))

    def _nearest_edges(self, timestamp, tolerance):
        best: Dict[Tuple[str, str], Tuple[float, np.ndarray]] = {}
        for stamp, parent, child, mat in self.transforms:
            dt = abs(stamp - timestamp)
            # /tf_static is commonly stamped at zero and should remain valid.
            if stamp != 0.0 and tolerance is not None and dt > tolerance:
                continue
            key = (parent, child)
            if key not in best or dt < best[key][0]:
                best[key] = (dt, mat)
        graph: Dict[str, List[Tuple[str, np.ndarray]]] = {}
        for (parent, child), (_, mat) in best.items():
            graph.setdefault(parent, []).append((child, mat))
            graph.setdefault(child, []).append((parent, np.linalg.inv(mat)))
        return graph

    def lookup(self, parent_frame, child_frame, timestamp, tolerance=0.05):
        if parent_frame == child_frame:
            return np.eye(4, dtype=np.float32)
        graph = self._nearest_edges(float(timestamp), tolerance)
        queue = [(parent_frame, np.eye(4, dtype=np.float32))]
        seen = {parent_frame}
        while queue:
            frame, mat_to_frame = queue.pop(0)
            for next_frame, frame_to_next in graph.get(frame, []):
                if next_frame in seen:
                    continue
                mat_to_next = mat_to_frame @ frame_to_next
                if next_frame == child_frame:
                    return mat_to_next.astype(np.float32)
                seen.add(next_frame)
                queue.append((next_frame, mat_to_next))
        return None


class RosbagFileRGBDSource(RGBDFrameSource):
    def __init__(
        self,
        bag,
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
        max_association_dt=0.03,
        tf_tolerance=0.05,
        depth_scale=1000.0,
        min_depth=0.1,
        max_depth=8.0,
    ):
        self.bag = Path(bag)
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
        self.max_association_dt = float(max_association_dt)
        self.tf_tolerance = float(tf_tolerance)
        self.depth_scale = float(depth_scale)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)

    def __iter__(self):
        AnyReader = _optional_rosbags()
        rgb_msgs = []
        depth_msgs = []
        pose_msgs = []
        color_intrinsics = None
        depth_intrinsics = None
        tf_graph = _TfGraph()

        topics = {
            self.rgb_topic,
            self.depth_topic,
            self.camera_info_topic,
            self.tf_topic,
            self.tf_static_topic,
        }
        if self.pose_topic:
            topics.add(self.pose_topic)
        if self.depth_camera_info_topic:
            topics.add(self.depth_camera_info_topic)

        with AnyReader([self.bag]) as reader:
            connections = [c for c in reader.connections if c.topic in topics]
            if not connections:
                raise RuntimeError(f"No requested topics found in bag: {self.bag}")
            for conn, timestamp_ns, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, conn.msgtype)
                timestamp = _msg_time(msg, timestamp_ns)
                if conn.topic == self.rgb_topic:
                    arr, enc = _image_msg_to_array(msg)
                    rgb_msgs.append((timestamp, arr, enc))
                elif conn.topic == self.depth_topic:
                    arr, enc = _image_msg_to_array(msg)
                    depth_msgs.append((timestamp, arr, enc))
                elif conn.topic == self.camera_info_topic:
                    color_intrinsics = _camera_info_to_intrinsics(
                        msg,
                        depth_scale=self.depth_scale,
                        camera_frame=self.camera_frame,
                        world_frame=self.world_frame,
                    )
                elif conn.topic == self.depth_camera_info_topic:
                    depth_intrinsics = _camera_info_to_intrinsics(
                        msg,
                        depth_scale=self.depth_scale,
                        camera_frame=self.depth_frame,
                        world_frame=self.world_frame,
                    )
                elif conn.topic in {self.tf_topic, self.tf_static_topic}:
                    for t in getattr(msg, "transforms", []):
                        tf_stamp = _stamp_to_sec(getattr(t.header, "stamp", None))
                        if tf_stamp is None or (tf_stamp == 0.0 and conn.topic != self.tf_static_topic):
                            tf_stamp = timestamp
                        tf_graph.add(tf_stamp, str(t.header.frame_id), str(t.child_frame_id), _transform_to_matrix(t.transform))
                elif conn.topic == self.pose_topic:
                    pose = _field(msg, "pose")
                    if hasattr(pose, "pose"):
                        pose = pose.pose
                    pose_msgs.append((timestamp, _pose_to_matrix(pose)))

        if color_intrinsics is None:
            raise RuntimeError(f"No CameraInfo found on topic {self.camera_info_topic}")
        if not self.depth_is_aligned_to_color and depth_intrinsics is None:
            raise RuntimeError(
                "Unaligned depth requested, but no depth CameraInfo was found. "
                "Pass --depth-camera-info-topic and --depth-frame."
            )

        depth_times = np.array([t for t, _, _ in depth_msgs], dtype=np.float64)
        pose_times = np.array([t for t, _ in pose_msgs], dtype=np.float64) if pose_msgs else np.array([], dtype=np.float64)
        emitted = 0
        source_idx = 0
        for rgb_time, rgb_raw, rgb_encoding in rgb_msgs:
            if source_idx % self.frame_stride != 0:
                source_idx += 1
                continue
            source_idx += 1
            if len(depth_times) == 0:
                break
            depth_idx = int(np.argmin(np.abs(depth_times - rgb_time)))
            if abs(depth_times[depth_idx] - rgb_time) > self.max_association_dt:
                continue

            if self.pose_source == "tf":
                c2w = tf_graph.lookup(
                    self.world_frame,
                    self.camera_frame,
                    rgb_time,
                    tolerance=self.tf_tolerance,
                )
                if c2w is None:
                    continue
            else:
                if len(pose_times) == 0:
                    raise RuntimeError("pose_source is not tf, but no pose messages were loaded.")
                pose_idx = int(np.argmin(np.abs(pose_times - rgb_time)))
                if abs(pose_times[pose_idx] - rgb_time) > self.max_association_dt:
                    continue
                c2w = pose_msgs[pose_idx][1]

            depth_raw, depth_encoding = depth_msgs[depth_idx][1], depth_msgs[depth_idx][2]
            rgb = decode_color_image(rgb_raw, rgb_encoding)
            depth = decode_depth_image(depth_raw, depth_encoding)

            if not self.depth_is_aligned_to_color:
                if not self.depth_frame:
                    raise RuntimeError("Unaligned depth requested, but --depth-frame was not provided.")
                depth_to_color = tf_graph.lookup(
                    self.color_frame,
                    self.depth_frame,
                    rgb_time,
                    tolerance=self.tf_tolerance,
                )
                if depth_to_color is None:
                    raise RuntimeError(
                        f"Could not resolve TF transform {self.color_frame} <- {self.depth_frame}."
                    )
                depth = reproject_depth_to_color(
                    depth,
                    depth_intrinsics,
                    color_intrinsics,
                    depth_to_color,
                    min_depth=self.min_depth,
                    max_depth=self.max_depth,
                )

            yield RGBDFrame(
                index=emitted,
                timestamp=rgb_time,
                rgb=rgb,
                depth=depth,
                intrinsics=color_intrinsics,
                c2w=c2w,
            )
            emitted += 1
            if self.max_frames and emitted >= self.max_frames:
                break
