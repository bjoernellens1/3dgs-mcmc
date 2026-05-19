import sys
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

path = '/mnt/cps_persistent1_shared/datasets/christian/embedding_map/bags/slam/kitchen1_slam'

try:
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    seen = set()
    with Reader(path) as reader:
        conns = [c for c in reader.connections if c.topic in ('/camera/color/camera_info', '/camera/depth/camera_info')]
        for conn, ts, data in reader.messages(connections=conns):
            if conn.topic not in seen:
                seen.add(conn.topic)
                msg = typestore.deserialize_cdr(data, conn.msgtype)
                print(f"Topic: {conn.topic}")
                print(f"  Width: {msg.width}, Height: {msg.height}")
                print(f"  K: {msg.k}")
                print(f"  D: {msg.d}")
                print(f"  Distortion Model: {msg.distortion_model}")
                print(f"  P: {msg.p}")
            if len(seen) == 2:
                break
except Exception as e:
    print('Failed to read bag:', e)
