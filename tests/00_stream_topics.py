"""Verify the required ROS observation topics."""
import math
import time
from collections import defaultdict

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32

SECONDS = 5.0

# /map is latched on the ROS1 side; durability must match or it never arrives.
QOS_MAP = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

TOPICS = [
    ("/obs/rgb", Image, 1),
    ("/obs/depth", Image, 1),
    ("/obs/gps", PointStamped, 1),
    ("/obs/compass", Float32, 1),
    ("/obs/camera_pose", PoseStamped, 1),
    ("/obs/camera_info", CameraInfo, 1),
    ("/map", OccupancyGrid, QOS_MAP),
]


class Streamer(Node):
    def __init__(self):
        super().__init__("stream_topics")
        self.last = {}
        self.count = defaultdict(int)
        self.first_t = {}
        self.last_t = {}
        for topic, msg_type, qos in TOPICS:
            self.create_subscription(msg_type, topic, self._cb(topic), qos)

    def _cb(self, topic):
        def inner(msg):
            now = time.time()
            self.last[topic] = msg
            self.count[topic] += 1
            self.first_t.setdefault(topic, now)
            self.last_t[topic] = now
        return inner


def decode(msg):
    if msg.encoding == "rgb8":
        return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
    if msg.encoding == "32FC1":
        return np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
    return np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width)


def rpy_deg(q):
    roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x**2 + q.y**2))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def describe(topic, msg):
    if topic == "/obs/camera_info":
        K = np.array(msg.k).reshape(3, 3)
        return f"{msg.width}x{msg.height} fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}"
    if topic == "/obs/rgb":
        a = decode(msg)
        return f"{a.shape[1]}x{a.shape[0]} {msg.encoding} mean={a.mean():.0f}"
    if topic == "/obs/depth":
        a = decode(msg)
        v = np.isfinite(a) & (a > 0)
        rng = f"{a[v].min():.2f}-{a[v].max():.2f}m" if v.any() else "no valid px"
        return f"{a.shape[1]}x{a.shape[0]} {msg.encoding} valid={100*v.mean():.0f}% {rng}"
    if topic == "/obs/gps":
        return f"x={msg.point.x:+.3f} y={msg.point.y:+.3f}"
    if topic == "/obs/compass":
        return f"{math.degrees(msg.data):+.1f}deg"
    if topic == "/obs/camera_pose":
        p, (r, pi, y) = msg.pose.position, rpy_deg(msg.pose.orientation)
        return f"xyz=({p.x:+.2f},{p.y:+.2f},{p.z:+.2f}) rpy=({r:+.1f},{pi:+.1f},{y:+.1f})"
    if topic == "/map":
        i = msg.info
        return f"{i.width}x{i.height} @{i.resolution:.3f}m origin=({i.origin.position.x:+.2f},{i.origin.position.y:+.2f})"
    return ""


def main():
    rclpy.init()
    node = Streamer()
    print(f"collecting {SECONDS:.0f}s ...")
    deadline = time.time() + SECONDS
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

    print(f"\n{'TOPIC':<20}{'N':>6}{'HZ':>7}  DETAIL")
    missing = []
    for topic, _, _ in TOPICS:
        n = node.count[topic]
        if n == 0:
            print(f"{topic:<20}{'-':>6}{'-':>7}  MISSING")
            missing.append(topic)
            continue
        span = node.last_t[topic] - node.first_t[topic]
        hz = (n - 1) / span if span > 0 else 0.0
        print(f"{topic:<20}{n:>6}{hz:>7.1f}  {describe(topic, node.last[topic])}")

    node.destroy_node()
    rclpy.shutdown()
    print("\n" + ("MISSING: " + ", ".join(missing) if missing else "all topics OK"))


if __name__ == "__main__":
    main()
