"""Synchronize Stretch ROS observations into rt_ovn shared state."""
import threading
import time
from collections import deque

import message_filters
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image

from patches import camera_geometry

OBS_NS = "/obs"
SYNC_SLOP_S = 0.02

_QOS_MAP = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def _image_to_numpy(msg):
    if msg.encoding == "rgb8":
        return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
    if msg.encoding == "32FC1":
        return np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width).copy()
    raise ValueError(f"unsupported encoding: {msg.encoding}")


def _pose_to_matrix(msg):
    T = np.eye(4)
    q = msg.pose.orientation
    T[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
    return T


class RealStretchSubscriber(Node):
    def __init__(
        self,
        shared_state=None,
        ns=OBS_NS,
        input_rotation_k=-1,
        on_observation=None,
    ):
        super().__init__("stretch_subscriber")
        if shared_state is None and on_observation is None:
            raise ValueError("shared_state or on_observation is required")
        self.shared_state = shared_state
        self._on_observation = on_observation
        self._input_rotation_k = int(input_rotation_k)
        if self._input_rotation_k not in (0, -1):
            raise ValueError("input_rotation_k must be 0 or -1 (90 degrees clockwise)")
        self._K = None
        self._step = 0
        self._odom_samples = deque(maxlen=120)
        self._odom_sequence = 0
        self._odom_frame_logged = False
        self._lock = threading.Lock()

        self.create_subscription(CameraInfo, f"{ns}/camera_info", self._info_cb, 1)
        self.create_subscription(OccupancyGrid, "/map", self._slam_map_cb, _QOS_MAP)
        self.create_subscription(Odometry, "/odom", self._odom_cb, 20)

        self._rgb = message_filters.Subscriber(self, Image, f"{ns}/rgb")
        self._depth = message_filters.Subscriber(self, Image, f"{ns}/depth")
        self._base_pose = message_filters.Subscriber(self, PoseStamped, f"{ns}/base_pose")
        self._pose = message_filters.Subscriber(self, PoseStamped, f"{ns}/camera_pose")
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._rgb, self._depth, self._base_pose, self._pose],
            queue_size=10, slop=SYNC_SLOP_S,
        )
        self._sync.registerCallback(self._on_sync)

    def _info_cb(self, msg):
        if self._K is None:
            self._K = np.array(msg.k, np.float32).reshape(3, 3)

    def _slam_map_cb(self, msg):
        if self.shared_state is not None:
            with self.shared_state.lock:
                self.shared_state.sensor.slam_map = msg

    @staticmethod
    def _stamp_ns(stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _odom_cb(self, msg):
        arrival_wall_ns = time.time_ns()
        arrival_monotonic_ns = time.monotonic_ns()
        q = msg.pose.pose.orientation
        yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
        p = msg.pose.pose.position
        sample = np.array([p.x, p.y, yaw], dtype=float)
        yaw_rate = float(msg.twist.twist.angular.z)
        if not self._odom_frame_logged:
            print(f"[subscriber] control pose: {msg.header.frame_id} -> "
                  f"{msg.child_frame_id} at /odom rate")
            self._odom_frame_logged = True
        with self._lock:
            self._odom_sequence += 1
            sequence = self._odom_sequence
            self._odom_samples.append((self._stamp_ns(msg.header.stamp), sample))
        if self.shared_state is not None:
            with self.shared_state.lock:
                sensor = self.shared_state.sensor
                sensor.control_odom = tuple(sample)
                sensor.control_yaw_rate = yaw_rate
                sensor.control_odom_meta = {
                    "source_stamp_ns": self._stamp_ns(msg.header.stamp),
                    "arrival_wall_ns": arrival_wall_ns,
                    "arrival_monotonic_ns": arrival_monotonic_ns,
                    "callback_sequence": sequence,
                }
                anchor = getattr(sensor, "map_odom_anchor", None)
                if anchor is not None:
                    sensor.planning_pose = tuple(
                        camera_geometry.odom_pose_to_map(sample, *anchor)
                    )

    def _on_sync(self, rgb_msg, depth_msg, base_msg, pose_msg):
        if self._K is None:
            return

        rgb = _image_to_numpy(rgb_msg)
        depth = _image_to_numpy(depth_msg)
        T_world_cam = _pose_to_matrix(pose_msg)
        rgb, depth, oriented_K, T_world_cam = camera_geometry.orient_rgbd(
            rgb, depth, self._K, T_world_cam, self._input_rotation_k)
        camera_geometry.capture_K(oriented_K)
        T_world_base = _pose_to_matrix(base_msg)
        gps = tuple(T_world_base[:2, 3])
        compass = R.from_matrix(T_world_base[:3, :3]).as_euler("xyz")[2]

        stamp_ns = self._stamp_ns(base_msg.header.stamp)
        with self._lock:
            self._step += 1
            step_id = self._step
            nearest = min(
                self._odom_samples,
                key=lambda item: abs(item[0] - stamp_ns),
                default=None,
            )
            map_pose = np.array([gps[0], gps[1], compass], dtype=float)
            if nearest is not None and abs(nearest[0] - stamp_ns) <= 100_000_000:
                map_odom_anchor = (tuple(map_pose), tuple(nearest[1]))
            else:
                map_odom_anchor = None

        camera_geometry.register_frame(
            gps, compass, np.linalg.inv(T_world_base) @ T_world_cam)

        if self.shared_state is not None:
            from rtnav.core.data_types import HabitatObservation

            with self.shared_state.lock:
                sensor = self.shared_state.sensor
                sensor.habitat_obs = HabitatObservation(
                    step_id=step_id, rgb=rgb, depth=depth,
                    gps=gps, compass=compass, timestamp=time.time(),
                )
                sensor.latest_odom = tuple(map_pose)
                if map_odom_anchor is not None:
                    sensor.map_odom_anchor = map_odom_anchor
                    sensor.planning_pose = tuple(map_pose)

        if self._on_observation is not None:
            self._on_observation({
                "step_id": step_id,
                "rgb": rgb,
                "depth": depth,
                "gps": gps,
                "compass": compass,
                "camera_matrix": oriented_K.copy(),
                "world_from_camera": T_world_cam.copy(),
                "timestamp": time.time(),
            })
