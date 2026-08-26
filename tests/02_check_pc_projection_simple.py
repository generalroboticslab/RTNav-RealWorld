"""Log one projected depth cloud to Rerun."""
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import rerun as rr
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "agent"))
sys.path.insert(0, str(ROOT / "rt_ovn" / "agents" / "rtnav"))

from patches import camera_geometry
from rtnav.modules.perception import camera_geometry as rtcg
RGB_TOPIC = "/obs/rgb"
DEPTH_TOPIC = "/obs/depth"
INFO_TOPIC = "/obs/camera_info"
POSE_TOPIC = "/obs/camera_pose"
GPS_TOPIC = "/obs/gps"
COMPASS_TOPIC = "/obs/compass"

# Stretch RE1: base footprint ~0.33 m, mast to ~1.4 m.
BASE_HALF = 0.17
BODY_H = 1.4

RERUN_HOST = "127.0.0.1"
RERUN_PORT = 9876
MIN_DEPTH_M = 0.3
MAX_DEPTH_M = 4.0
LOG_HZ = 5.0


def quat_to_R(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class Projector(Node):
    def __init__(self):
        super().__init__("pc_projection")
        self.rgb = None
        self.depth = None
        self.pose = None
        self.stamp = None
        self.gps = None
        self.yaw = 0.0
        self.have_K = False
        self.create_subscription(Image, RGB_TOPIC, self._rgb_cb, 1)
        self.create_subscription(Image, DEPTH_TOPIC, self._depth_cb, 1)
        self.create_subscription(CameraInfo, INFO_TOPIC, self._info_cb, 1)
        self.create_subscription(PoseStamped, POSE_TOPIC, self._pose_cb, 1)
        self.create_subscription(PointStamped, GPS_TOPIC, self._gps_cb, 1)
        self.create_subscription(Float32, COMPASS_TOPIC, self._compass_cb, 1)

    def _rgb_cb(self, msg):
        self.rgb = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)

    def _depth_cb(self, msg):
        self.depth = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
        self.stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _info_cb(self, msg):
        if not self.have_K:
            camera_geometry.capture_K(np.array(msg.k).reshape(3, 3))
            self.have_K = True

    def _pose_cb(self, msg):
        T = np.eye(4)
        T[:3, :3] = quat_to_R(msg.pose.orientation)
        T[:3, 3] = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.pose = T

    def _gps_cb(self, msg):
        self.gps = (msg.point.x, msg.point.y)

    def _compass_cb(self, msg):
        self.yaw = float(msg.data)

    def ready(self):
        return self.have_K and all(
            v is not None for v in (self.rgb, self.depth, self.pose, self.gps))


def project(node):
    """subscriber._on_sync + perception_thread._build_perception, same order."""
    gps, yaw = node.gps, node.yaw

    # subscriber side: pair this frame's extrinsic with its gps/compass.
    T_world_base_meas = camera_geometry.base_matrix(gps, yaw)
    camera_geometry.register_frame(
        gps, yaw, np.linalg.inv(T_world_base_meas) @ node.pose)

    # perception side: rebuild the camera pose through the patched builders.
    depth_m = rtcg.decode_depth_to_meters(node.depth, False, 0.0, 10.0)
    K = rtcg.build_K(0.0, depth_m.shape[1], depth_m.shape[0])
    T_world_base = rtcg.build_T_world_base(list(gps), float(yaw))
    T_world_cam = T_world_base @ rtcg.build_T_base_cam(None, True)
    pts, cols = rtcg.depth_to_pointcloud_world(
        depth_m, T_world_cam, K, MIN_DEPTH_M, MAX_DEPTH_M, rgb=node.rgb)
    return pts, (cols * 255).astype(np.uint8), T_world_cam


def main():
    rclpy.init()
    camera_geometry.install()
    node = Projector()

    rr.init("pc_projection", spawn=False)
    rr.connect_tcp(f"{RERUN_HOST}:{RERUN_PORT}")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Robot body + axes are fixed in the base frame; only world/robot's
    # transform moves, so log the shapes once as static.
    rr.log("world/robot/body", rr.Boxes3D(
        centers=[[0.0, 0.0, BODY_H / 2]],
        half_sizes=[[BASE_HALF, BASE_HALF, BODY_H / 2]],
        colors=[[80, 140, 255]],
    ), static=True)
    rr.log("world/robot/axes", rr.Arrows3D(
        origins=[[0, 0, 0]] * 3,
        vectors=[[0.4, 0, 0], [0, 0.4, 0], [0, 0, 0.4]],
        colors=[[255, 60, 60], [60, 255, 60], [60, 60, 255]],
        labels=["x fwd", "y left", "z up"],
    ), static=True)
    rr.log("world/camera/axes", rr.Arrows3D(
        origins=[[0, 0, 0]] * 3,
        vectors=[[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]],
        colors=[[255, 60, 60], [60, 255, 60], [60, 60, 255]],
    ), static=True)

    print(f"logging latest cloud to rerun at {RERUN_HOST}:{RERUN_PORT} — ctrl-c to quit")

    last_stamp, next_log = None, 0.0
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        now = time.monotonic()
        if not node.ready() or node.stamp == last_stamp or now < next_log:
            continue
        last_stamp, next_log = node.stamp, now + 1.0 / LOG_HZ

        pts, cols, T_world_cam = project(node)
        rr.log("world/points", rr.Points3D(pts, colors=cols, radii=0.01))
        rr.log("world/camera", rr.Transform3D(translation=T_world_cam[:3, 3],
                                              mat3x3=T_world_cam[:3, :3]))
        c, s = np.cos(node.yaw), np.sin(node.yaw)
        rr.log("world/robot", rr.Transform3D(
            translation=[node.gps[0], node.gps[1], 0.0],
            mat3x3=[[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        ))
        print(f"\rpts={len(pts)} odom=({node.gps[0]:+.2f},{node.gps[1]:+.2f},"
              f"{np.degrees(node.yaw):+.0f}deg)", end="", flush=True)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
