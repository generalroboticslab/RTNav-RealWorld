"""Measure pose error from discrete base actions.

Usage: ``python tests/05_test_discrete_action_primitives.py 'F,L,F,R,F'``
"""
import math
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "utils"))
sys.path.insert(0, str(ROOT / "src" / "agent"))
sys.path.insert(0, str(ROOT / "rt_ovn" / "agents" / "rtnav"))

from rtnav.core.data_types import HabitatObservation

STOP, FORWARD, LEFT, RIGHT = 0, 1, 2, 3
TOKENS = {"S": STOP, "F": FORWARD, "L": LEFT, "R": RIGHT}
NAMES = {STOP: "STOP", FORWARD: "FWD", LEFT: "L", RIGHT: "R"}

SEQUENCE = sys.argv[1] if len(sys.argv) > 1 else "F,L,F,R,F"
SETTLE_S = 0.6
FORWARD_STEP_M = 0.25
TURN_DEG = 30.0
LINEAR_SPEED = 0.25
ANGULAR_SPEED_DEG = 30.0


class ObsSub(Node):
    """Feeds latest_odom and habitat_obs, and hosts RealRobotAPI's publisher."""

    def __init__(self, shared_state, ns="/obs"):
        super().__init__("discrete_primitives")
        self.ss = shared_state
        self.gps = None
        self.compass = 0.0
        self.depth = None
        self.step = 0
        self.create_subscription(PointStamped, f"{ns}/gps", self._gps_cb, 10)
        self.create_subscription(Float32, f"{ns}/compass", self._compass_cb, 10)
        self.create_subscription(Image, f"{ns}/depth", self._depth_cb, 1)

    def _publish(self):
        if self.gps is None or self.depth is None:
            return                      # wait for the first of each
        self.step += 1
        with self.ss.lock:
            self.ss.sensor.latest_odom = (self.gps[0], self.gps[1], self.compass)
            self.ss.sensor.habitat_obs = HabitatObservation(
                step_id=self.step, rgb=None, depth=self.depth,
                gps=self.gps, compass=self.compass, timestamp=time.time(),
            )

    def _gps_cb(self, msg):
        self.gps = (float(msg.point.x), float(msg.point.y))
        self._publish()

    def _compass_cb(self, msg):
        self.compass = float(msg.data)
        self._publish()

    def _depth_cb(self, msg):
        self.depth = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
        self._publish()


def wrap_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


def fmt(p):
    return f"({p[0]:+.3f}, {p[1]:+.3f}, {math.degrees(p[2]):+.1f}deg)"


def wait_for_obs(shared_state, timeout_s=5.0):
    """Block until ObsSub has both a pose and a depth frame."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with shared_state.lock:
            pose = shared_state.sensor.latest_odom
        if pose is not None:
            return pose
        time.sleep(0.05)
    raise SystemExit(
        "no pose+depth after 5s — is src/env/launch.py running "
        "and publishing /obs/gps, /obs/compass, /obs/depth?")


def main():
    actions = [TOKENS[t.strip().upper()] for t in SEQUENCE.split(",") if t.strip()]

    rclpy.init()
    import config
    from rtnav.core.shared_state import SharedState
    from patches.robot_api import GotoRobotAPI

    # Fixed-time settle so every primitive is measured over the same window.
    cfg = replace(config.build_config().nav, settle_s=SETTLE_S,
                  forward_step_m=FORWARD_STEP_M, turn_angle_deg=TURN_DEG,
                  linear_speed=LINEAR_SPEED, angular_speed_deg=ANGULAR_SPEED_DEG)

    shared_state = SharedState()
    shutdown = threading.Event()
    node = ObsSub(shared_state)
    robot = GotoRobotAPI(node, shared_state, cfg)
    robot.prepare()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    start = wait_for_obs(shared_state)
    print(f"start {fmt(start)}  sequence {SEQUENCE}  settle {SETTLE_S}s\n")

    rows = []
    try:
        for i, action in enumerate(actions, 1):
            with shared_state.lock:
                before = shared_state.sensor.latest_odom
            t0 = time.monotonic()
            robot.execute_action(action)
            dt = time.monotonic() - t0
            with shared_state.lock:
                after = shared_state.sensor.latest_odom

            move = math.hypot(after[0] - before[0], after[1] - before[1])
            dyaw = math.degrees(wrap_pi(after[2] - before[2]))
            bad = ((action == FORWARD and (abs(dyaw) > 15 or move < 0.15))
                   or (action in (LEFT, RIGHT) and move > 0.10))
            rows.append((NAMES[action], move, dyaw))
            print(f"[{i}/{len(actions)}] {NAMES[action]:<4} dt={dt:.2f}s "
                  f"move={move:.3f}m yaw={dyaw:+.1f}deg {fmt(after)}"
                  f"{'  <-- off' if bad else ''}")
    except KeyboardInterrupt:
        robot.emergency_stop()
        print("\ninterrupted — zero cmd_vel published")

    fwd = [r for r in rows if r[0] == "FWD"]
    turn = [r for r in rows if r[0] in ("L", "R")]
    print()
    if fwd:
        print(f"FWD  x{len(fwd)}  avg move={sum(r[1] for r in fwd)/len(fwd):.3f}m "
              f"(target {FORWARD_STEP_M})  avg |yaw drift|="
              f"{sum(abs(r[2]) for r in fwd)/len(fwd):.1f}deg")
    if turn:
        print(f"turn x{len(turn)}  avg yaw={sum(abs(r[2]) for r in turn)/len(turn):.1f}deg "
              f"(target {TURN_DEG})  avg slide={sum(r[1] for r in turn)/len(turn):.3f}m")
    with shared_state.lock:
        end = shared_state.sensor.latest_odom
    print(f"net drift  dxy={math.hypot(end[0]-start[0], end[1]-start[1]):.3f}m  "
          f"dyaw={math.degrees(wrap_pi(end[2]-start[2])):+.1f}deg")

    shutdown.set()
    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
