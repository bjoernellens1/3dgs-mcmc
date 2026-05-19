from bisect import bisect_left
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


def _camera_info_to_intrinsics(
    msg,
    depth_scale,
    camera_frame="",
    world_frame="",
    assume_rectified=False,
    fail_on_distortion=True,
):
    k = list(_field(msg, "k", "K"))
    p = list(_field(msg, "p", "P", default=[]))
    d = list(_field(msg, "d", "D", default=[]))
    distortion_model = str(_field(msg, "distortion_model", default=""))
    has_distortion = any(abs(float(v)) > 1e-12 for v in d)
    if has_distortion and not assume_rectified:
        message = (
            f"CameraInfo on frame '{_frame_id(msg)}' reports nonzero distortion "
            f"({distortion_model}, D={d}). Use a rectified image topic or pass "
            "--assume-rectified only if the stream is already rectified."
        )
        if fail_on_distortion:
            raise RuntimeError(message)
        print(f"[RGBD] Warning: {message}")

    intrinsics_source = "K"
    if len(p) >= 12 and abs(float(p[0])) > 1e-12 and abs(float(p[5])) > 1e-12:
        fx, fy, cx, cy = float(p[0]), float(p[5]), float(p[2]), float(p[6])
        intrinsics_source = "P_rectified"
    else:
        fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])

    return CameraIntrinsics(
        width=int(msg.width),
        height=int(msg.height),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        depth_scale=float(depth_scale),
        camera_frame=camera_frame or _frame_id(msg),
        world_frame=world_frame,
        distortion_model=distortion_model,
        D=[float(v) for v in d],
        intrinsics_source=intrinsics_source,
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
        self.transforms_by_edge: Dict[Tuple[str, str], List[Tuple[float, np.ndarray]]] = {}
        self.static_transforms: Dict[Tuple[str, str], np.ndarray] = {}
        self._sorted_edges = set()
        self._stamp_cache: Dict[Tuple[str, str], List[float]] = {}

    def add(self, stamp, parent, child, matrix, is_static=False):
        if parent and child:
            key = (str(parent), str(child))
            mat = np.asarray(matrix, dtype=np.float32)
            if is_static:
                self.static_transforms[key] = mat
                return
            self.transforms_by_edge.setdefault(key, []).append((float(stamp), mat))
            self._sorted_edges.discard(key)
            self._stamp_cache.pop(key, None)

    def _nearest_dynamic(self, key, timestamp, tolerance):
        entries = self.transforms_by_edge.get(key)
        if not entries:
            return None
        if key not in self._sorted_edges:
            entries.sort(key=lambda item: item[0])
            self._sorted_edges.add(key)
            self._stamp_cache[key] = [stamp for stamp, _ in entries]
        stamps = self._stamp_cache[key]
        idx = bisect_left(stamps, timestamp)
        candidates = []
        if idx < len(entries):
            candidates.append(entries[idx])
        if idx > 0:
            candidates.append(entries[idx - 1])
        if not candidates:
            return None
        stamp, mat = min(candidates, key=lambda item: abs(item[0] - timestamp))
        if tolerance is not None and abs(stamp - timestamp) > tolerance:
            return None
        return mat

    def _nearest_edges(self, timestamp, tolerance):
        best: Dict[Tuple[str, str], np.ndarray] = dict(self.static_transforms)
        for key in self.transforms_by_edge.keys():
            mat = self._nearest_dynamic(key, timestamp, tolerance)
            if mat is not None:
                best[key] = mat
        graph: Dict[str, List[Tuple[str, np.ndarray]]] = {}
        for (parent, child), mat in best.items():
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
        pose_frame="",
        pose_is_camera_frame=False,
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
        assume_rectified=False,
        fail_on_distortion=True,
        rgb_encoding_override="",
    ):
        self.bag = Path(bag)
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.camera_info_topic = camera_info_topic
        self.pose_source = pose_source
        self.world_frame = world_frame
        self.camera_frame = camera_frame
        self.pose_frame = pose_frame
        self.pose_is_camera_frame = bool(pose_is_camera_frame)
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
        self.assume_rectified = bool(assume_rectified)
        self.fail_on_distortion = bool(fail_on_distortion)
        self.rgb_encoding_override = rgb_encoding_override

    def __iter__(self):
        AnyReader = _optional_rosbags()
        pending_rgb = []
        depth_msgs = []
        pose_msgs = []
        color_intrinsics = None
        depth_intrinsics = None
        tf_graph = _TfGraph()
        emitted = 0
        rgb_seen = 0

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

        def nearest(buffer, timestamp):
            if not buffer:
                return None, None
            times = np.array([item[0] for item in buffer], dtype=np.float64)
            idx = int(np.argmin(np.abs(times - timestamp)))
            return idx, buffer[idx]

        def trim_buffers(current_time):
            cutoff = float(current_time) - max(self.max_association_dt, self.tf_tolerance) * 8.0
            depth_msgs[:] = [item for item in depth_msgs if item[0] >= cutoff]
            pose_msgs[:] = [item for item in pose_msgs if item[0] >= cutoff]
            pending_rgb[:] = [item for item in pending_rgb if item[0] >= cutoff]

        def pose_to_camera_c2w(pose_mat, timestamp):
            if self.pose_source == "tf" or self.pose_is_camera_frame:
                return pose_mat
            pose_frame = self.pose_frame
            if not pose_frame:
                raise RuntimeError(
                    "pose_source is pose/odom but --pose-frame was not provided. "
                    "Pass --pose-is-camera-frame only if the pose topic already publishes world_T_camera."
                )
            if pose_frame == self.camera_frame:
                return pose_mat
            pose_to_camera = tf_graph.lookup(
                pose_frame,
                self.camera_frame,
                timestamp,
                tolerance=self.tf_tolerance,
            )
            if pose_to_camera is None:
                return None
            return pose_mat @ pose_to_camera

        def try_emit_ready():
            nonlocal emitted
            made_progress = True
            while made_progress and pending_rgb:
                made_progress = False
                rgb_time, rgb_raw, rgb_encoding = pending_rgb[0]
                if color_intrinsics is None or not depth_msgs:
                    return
                depth_idx, depth_item = nearest(depth_msgs, rgb_time)
                if depth_item is None:
                    return
                if abs(depth_item[0] - rgb_time) > self.max_association_dt:
                    if depth_item[0] < rgb_time:
                        pending_rgb.pop(0)
                        made_progress = True
                    return

                if self.pose_source == "tf":
                    c2w = tf_graph.lookup(
                        self.world_frame,
                        self.camera_frame,
                        rgb_time,
                        tolerance=self.tf_tolerance,
                    )
                    if c2w is None:
                        return
                else:
                    if not pose_msgs:
                        return
                    _, pose_item = nearest(pose_msgs, rgb_time)
                    if pose_item is None:
                        return
                    if abs(pose_item[0] - rgb_time) > self.max_association_dt:
                        if pose_item[0] < rgb_time:
                            pending_rgb.pop(0)
                            made_progress = True
                        return
                    c2w = pose_to_camera_c2w(pose_item[1], rgb_time)
                    if c2w is None:
                        return

                depth_raw, depth_encoding = depth_item[1], depth_item[2]
                rgb = decode_color_image(rgb_raw, rgb_encoding, encoding_override=self.rgb_encoding_override)
                depth = decode_depth_image(depth_raw, depth_encoding)

                if not self.depth_is_aligned_to_color:
                    if depth_intrinsics is None or not self.depth_frame:
                        return
                    depth_to_color = tf_graph.lookup(
                        self.color_frame,
                        self.depth_frame,
                        rgb_time,
                        tolerance=self.tf_tolerance,
                    )
                    if depth_to_color is None:
                        return
                    depth = reproject_depth_to_color(
                        depth,
                        depth_intrinsics,
                        color_intrinsics,
                        depth_to_color,
                        min_depth=self.min_depth,
                        max_depth=self.max_depth,
                    )

                pending_rgb.pop(0)
                emitted_frame = RGBDFrame(
                    index=emitted,
                    timestamp=rgb_time,
                    rgb=rgb,
                    depth=depth,
                    intrinsics=color_intrinsics,
                    c2w=c2w,
                )
                emitted += 1
                made_progress = True
                yield emitted_frame

        with AnyReader([self.bag]) as reader:
            connections = [c for c in reader.connections if c.topic in topics]
            if not connections:
                raise RuntimeError(f"No requested topics found in bag: {self.bag}")
            for conn, timestamp_ns, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, conn.msgtype)
                timestamp = _msg_time(msg, timestamp_ns)
                if conn.topic == self.rgb_topic:
                    if rgb_seen % self.frame_stride == 0:
                        arr, enc = _image_msg_to_array(msg)
                        pending_rgb.append((timestamp, arr, enc))
                    rgb_seen += 1
                elif conn.topic == self.depth_topic:
                    arr, enc = _image_msg_to_array(msg)
                    depth_msgs.append((timestamp, arr, enc))
                elif conn.topic == self.camera_info_topic:
                    color_intrinsics = _camera_info_to_intrinsics(
                        msg,
                        depth_scale=self.depth_scale,
                        camera_frame=self.camera_frame,
                        world_frame=self.world_frame,
                        assume_rectified=self.assume_rectified,
                        fail_on_distortion=self.fail_on_distortion,
                    )
                elif conn.topic == self.depth_camera_info_topic:
                    depth_intrinsics = _camera_info_to_intrinsics(
                        msg,
                        depth_scale=self.depth_scale,
                        camera_frame=self.depth_frame,
                        world_frame=self.world_frame,
                        assume_rectified=self.assume_rectified,
                        fail_on_distortion=self.fail_on_distortion,
                    )
                elif conn.topic in {self.tf_topic, self.tf_static_topic}:
                    for t in getattr(msg, "transforms", []):
                        tf_stamp = _stamp_to_sec(getattr(t.header, "stamp", None))
                        if tf_stamp is None or (tf_stamp == 0.0 and conn.topic != self.tf_static_topic):
                            tf_stamp = timestamp
                        tf_graph.add(
                            tf_stamp,
                            str(t.header.frame_id),
                            str(t.child_frame_id),
                            _transform_to_matrix(t.transform),
                            is_static=(conn.topic == self.tf_static_topic),
                        )
                elif conn.topic == self.pose_topic:
                    pose = _field(msg, "pose")
                    if hasattr(pose, "pose"):
                        pose = pose.pose
                    pose_msgs.append((timestamp, _pose_to_matrix(pose)))
                trim_buffers(timestamp)
                for frame in try_emit_ready():
                    yield frame
                    if self.max_frames and emitted >= self.max_frames:
                        return

        if color_intrinsics is None:
            raise RuntimeError(f"No CameraInfo found on topic {self.camera_info_topic}")
        if not self.depth_is_aligned_to_color and depth_intrinsics is None:
            raise RuntimeError(
                "Unaligned depth requested, but no depth CameraInfo was found. "
                "Pass --depth-camera-info-topic and --depth-frame."
            )

        for frame in try_emit_ready():
            yield frame
            if self.max_frames and emitted >= self.max_frames:
                return
