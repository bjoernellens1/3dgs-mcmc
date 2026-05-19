import sys
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
path = '/mnt/cps_persistent1_shared/datasets/christian/embedding_map/bags/slam/kitchen1_slam'
typestore = get_typestore(Stores.ROS2_HUMBLE)
import numpy as np

def _quat_xyzw_to_rotmat(qx, qy, qz, qw):
    n = qx*qx + qy*qy + qz*qz + qw*qw
    if n < 1e-12: return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    return np.array([
        [1 - s*(qy*qy + qz*qz), s*(qx*qy - qz*qw), s*(qx*qz + qy*qw)],
        [s*(qx*qy + qz*qw), 1 - s*(qx*qx + qz*qz), s*(qy*qz - qx*qw)],
        [s*(qx*qz - qy*qw), s*(qy*qz + qx*qw), 1 - s*(qx*qx + qy*qy)],
    ], dtype=np.float32)

with Reader(path) as reader:
    conns = [c for c in reader.connections if c.topic == '/tf_static']
    for conn, ts, data in reader.messages(connections=conns):
        msg = typestore.deserialize_cdr(data, conn.msgtype)
        for tf in msg.transforms:
            p = tf.transform.translation
            q = tf.transform.rotation
            mat = np.eye(4, dtype=np.float32)
            mat[:3, :3] = _quat_xyzw_to_rotmat(q.x, q.y, q.z, q.w)
            mat[:3, 3] = [p.x, p.y, p.z]
            print(tf.header.frame_id, "->", tf.child_frame_id)
            print(np.array_str(mat, precision=4, suppress_small=True))
        break
