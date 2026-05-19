import sys
import numpy as np

def run_test():
    try:
        from rosbags.rosbag2 import Reader
        from rosbags.typesys import Stores, get_typestore
        import open3d as o3d
    except ImportError as e:
        print("Import error:", e)
        return

    path = '/mnt/cps_persistent1_shared/datasets/christian/embedding_map/bags/slam/kitchen1_slam'
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    
    color_msgs = []
    depth_msgs = []
    intrinsics = None

    with Reader(path) as reader:
        conns = [c for c in reader.connections if c.topic in ('/camera/color/image_raw/compressed', '/camera/depth/image_raw/compressed', '/camera/color/camera_info')]
        for conn, ts, data in reader.messages(connections=conns):
            if conn.topic == '/camera/color/camera_info' and intrinsics is None:
                msg = typestore.deserialize_cdr(data, conn.msgtype)
                intrinsics = msg.k
            elif conn.topic == '/camera/color/image_raw/compressed':
                msg = typestore.deserialize_cdr(data, conn.msgtype)
                color_msgs.append((ts, bytes(msg.data)))
            elif conn.topic == '/camera/depth/image_raw/compressed':
                msg = typestore.deserialize_cdr(data, conn.msgtype)
                depth_msgs.append((ts, bytes(msg.data)))
            if len(color_msgs) > 20 and len(depth_msgs) > 20:
                break
                
    # simple sync
    import io
    from PIL import Image
    
    fx, fy, cx, cy = intrinsics[0], intrinsics[4], intrinsics[2], intrinsics[5]
    width, height = 1280, 720
    odom_downscale = 4
    odom_width = width // odom_downscale
    odom_height = height // odom_downscale
    odom_fx = fx / odom_downscale
    odom_fy = fy / odom_downscale
    odom_cx = cx / odom_downscale
    odom_cy = cy / odom_downscale
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(odom_width, odom_height, odom_fx, odom_fy, odom_cx, odom_cy)
    option = o3d.pipelines.odometry.OdometryOption()
    jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

    def to_rgbd(rgb_bytes, depth_bytes):
        color_img = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
        depth_img = Image.open(io.BytesIO(depth_bytes))
        color_img = color_img.resize((odom_width, odom_height), Image.Resampling.BILINEAR)
        depth_img = depth_img.resize((odom_width, odom_height), Image.Resampling.NEAREST)
        color = o3d.geometry.Image(np.ascontiguousarray(np.asarray(color_img)))
        depth = o3d.geometry.Image(np.ascontiguousarray(np.asarray(depth_img)))
        return o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=1000.0, depth_trunc=5.0, convert_rgb_to_intensity=True)

    prev_rgbd = to_rgbd(color_msgs[0][1], depth_msgs[0][1])
    c2w = np.eye(4, dtype=np.float32)
    
    print("Centers:")
    print(c2w[:3, 3])
    for i in range(1, 10):
        curr_rgbd = to_rgbd(color_msgs[i][1], depth_msgs[i][1])
        success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(prev_rgbd, curr_rgbd, intrinsic, np.eye(4), jacobian, option)
        if success:
            c2w = c2w @ np.linalg.inv(trans)
            print(c2w[:3, 3])
        else:
            print("Failed")
        prev_rgbd = curr_rgbd

run_test()
