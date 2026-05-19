import sys
import numpy as np

def run_test():
    import open3d as o3d
    img_size = 200
    fx = fy = 100.0
    cx = cy = 100.0
    intrinsic = o3d.camera.PinholeCameraIntrinsic(img_size, img_size, fx, fy, cx, cy)

    # Let's create a more textured room with multiple planes
    def make_rgbd(cam_pos):
        depth = np.zeros((img_size, img_size), dtype=np.float32)
        color = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        # Room geometry: front wall at Z=5, floor at Y=-1, left wall at X=-2
        for v in range(img_size):
            for u in range(img_size):
                dir_x = (u - cx) / fx
                dir_y = (v - cy) / fy
                dir_z = 1.0
                
                # Intersection with front wall Z=5
                t_front = (5.0 - cam_pos[2]) / dir_z if dir_z > 0 else 999
                
                # Intersection with floor Y=-1 (camera looks down slightly or y is positive down)
                # cam_pos[1] + t * dir_y = -1  => t = (-1 - cam_pos[1])/dir_y
                # Wait, usually y is down in image coords, so floor is positive Y. Let's say floor at Y=2
                t_floor = (2.0 - cam_pos[1]) / dir_y if dir_y > 0 else 999
                
                # Intersection with left wall X=-2
                t_left = (-2.0 - cam_pos[0]) / dir_x if dir_x < 0 else 999
                
                t = min(t_front, t_floor, t_left)
                
                if 0 < t < 10.0:
                    depth[v, u] = t * dir_z
                    # Add texture based on world coordinates to allow ICP to match
                    P_world = cam_pos + t * np.array([dir_x, dir_y, dir_z])
                    # simple checkerboard
                    cx_tex = int(np.floor(P_world[0] * 5))
                    cy_tex = int(np.floor(P_world[1] * 5))
                    cz_tex = int(np.floor(P_world[2] * 5))
                    if (cx_tex + cy_tex + cz_tex) % 2 == 0:
                        color[v, u] = [255, 255, 255]
                    else:
                        color[v, u] = [100, 100, 100]
                        
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color), o3d.geometry.Image(depth),
            depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=True)

    src_rgbd = make_rgbd(np.array([0.0, 0.0, 0.0]))
    tgt_rgbd = make_rgbd(np.array([0.2, 0.0, 0.0])) # target camera is moved +0.2 in X

    option = o3d.pipelines.odometry.OdometryOption()
    success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
        src_rgbd, tgt_rgbd, intrinsic, np.eye(4), 
        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
    
    print("Success:", success)
    print("trans:\n", trans)

run_test()
