import cv2
import numpy as np
import io
from PIL import Image
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

path = '/mnt/cps_persistent1_shared/datasets/christian/embedding_map/bags/slam/kitchen1_slam'
typestore = get_typestore(Stores.ROS2_HUMBLE)

K = None
D = None
img_raw = None

with Reader(path) as reader:
    for conn, ts, data in reader.messages():
        if conn.topic == '/camera/color/camera_info' and K is None:
            msg = typestore.deserialize_cdr(data, conn.msgtype)
            K = np.array(msg.k).reshape(3, 3)
            D = np.array(msg.d)
        elif conn.topic == '/camera/color/image_raw/compressed' and img_raw is None:
            msg = typestore.deserialize_cdr(data, conn.msgtype)
            img_raw = np.array(Image.open(io.BytesIO(bytes(msg.data))))
        if K is not None and img_raw is not None:
            break

cv2.imwrite('raw.png', cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR))

# Undistort
new_cameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (img_raw.shape[1], img_raw.shape[0]), 0, (img_raw.shape[1], img_raw.shape[0]))
dst = cv2.undistort(img_raw, K, D, None, new_cameramtx)
cv2.imwrite('undistorted.png', cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))

# calculate difference
diff = np.abs(img_raw.astype(np.float32) - dst.astype(np.float32)).mean()
print(f"Mean pixel difference: {diff}")
print(f"Max difference: {np.abs(img_raw.astype(np.float32) - dst.astype(np.float32)).max()}")
