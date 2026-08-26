"""Stretch Twist-based robot API."""

import math
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped, Twist
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, Trigger

STOP, FORWARD, LEFT, RIGHT, LOOK_UP, LOOK_DOWN = 0, 1, 2, 3, 4, 5


@dataclass(frozen=True)
class RobotMotionConfig:
    """Stretch base motion for RTNav."""

    cmd_vel_topic: str = "/stretch/cmd_vel"
    cmd_rate_hz: float = 30.0
    forward_step_m: float = 0.25
    turn_angle_deg: float = 30.0
    linear_speed: float = 0.30
    angular_speed_deg: float = 30.0
    ramp_k_lin: float = 1.5
    ramp_k_ang: float = 3.0
    v_min: float = 0.05
    w_min_deg: float = 1.0
    settle_s: float = 0.0
    settle_max_s: float = 1.0
    settle_xy_eps_m: float = 0.003
    settle_yaw_eps_deg: float = 0.5
    primitive_timeout_s: float = 10.0
    control_pose_timeout_s: float = 2.0
    control_pose_wait_s: float = 2.0
    primitive_xy_tolerance_m: float = 0.05
    primitive_yaw_tolerance_deg: float = 4.0


class RealRobotAPI:
    """Blocking discrete primitives on /stretch/cmd_vel."""

    def __init__(self, node: Node, shared_state, cfg) -> None:
        self._node = node
        self._shared_state = shared_state
        self._cfg = cfg

        self._turn_angle_rad = math.radians(cfg.turn_angle_deg)
        self._angular_speed = math.radians(cfg.angular_speed_deg)
        self._w_min = math.radians(cfg.w_min_deg)
        self._settle_yaw_eps = math.radians(cfg.settle_yaw_eps_deg)
        self._cmd_period = 1.0 / cfg.cmd_rate_hz

        self._stall_timeout = 1.0
        self._dist_eps = 1e-3
        self._angle_eps = math.radians(0.5)

        self._aborted = False
        self._completed_successfully = False
        self._episode_done_event = threading.Event()
        self._pub_vel = node.create_publisher(Twist, cfg.cmd_vel_topic, 10)

        print("[RealRobotAPI] {} lin={:.2f}m/s ang={:.0f}deg/s".format(
            cfg.cmd_vel_topic, cfg.linear_speed, cfg.angular_speed_deg))

    def get_control_pose(self) -> Optional[Tuple[float, float, float]]:
        """Pose used by the active low-level controller."""
        return self._get_pose()

    def emergency_stop(self) -> None:
        """Zero the base and end the episode. Wired to estop.start() by the runner."""
        self._aborted = True
        self._publish_zero()
        self._episode_done_event.set()

    def reset_episode(self, episode_hash=None, preserve_step_id: bool = False) -> None:
        """episode_hash is in rt_ovn's signature but unused — one episode at a time."""
        self._episode_done_event.clear()
        self._aborted = False
        self._completed_successfully = False

    def set_episode_done(self) -> None:
        self._episode_done_event.set()

    @property
    def episode_done(self) -> bool:
        return self._episode_done_event.is_set() or self._aborted

    @property
    def target_completed(self) -> bool:
        return self._completed_successfully

    def send_done(self) -> None:
        """rt_ovn's /agent_done, with no simulator to notify: stop and mark done."""
        for _ in range(3):
            self._publish_zero()
            time.sleep(self._cmd_period)
        self._completed_successfully = True
        self._episode_done_event.set()
        print("[RealRobotAPI] episode done — base stopped")

    def get_latest_observation(self):
        with self._shared_state.lock:
            return self._shared_state.sensor.habitat_obs

    def _get_pose(self) -> Tuple[float, float, float]:
        with self._shared_state.lock:
            return getattr(self._shared_state.sensor, "control_odom", None)

    def _publish_vel(self, vx: float, vy: float, wz: float, _clip: bool = True) -> None:
        if self._aborted:
            vx = vy = wz = 0.0
        if _clip:
            vx = float(np.clip(vx, -self._cfg.linear_speed, self._cfg.linear_speed))
            wz = float(np.clip(wz, -self._angular_speed, self._angular_speed))
        t = Twist()
        t.linear.x = float(vx)
        t.angular.z = float(wz)
        try:
            self._pub_vel.publish(t)
        except Exception as exc:
            if not self._aborted:
                sys.stderr.write(f"[RealRobotAPI] _publish_vel: {exc}\n")
            self._aborted = True

    def _publish_zero(self) -> None:
        self._publish_vel(0.0, 0.0, 0.0, _clip=False)

    def drive(self, vx: float, wz: float) -> None:
        """One velocity command, for a controller doing its own timing."""
        self._publish_vel(vx, 0.0, wz)

    def _settle(self) -> None:
        """Hold zero until motion settles or the timeout expires."""
        if self._cfg.settle_s > 0.0:
            for _ in range(max(1, int(self._cfg.settle_s / self._cmd_period))):
                if self.episode_done:
                    return
                self._publish_zero()
                time.sleep(self._cmd_period)
            return

        deadline = time.monotonic() + self._cfg.settle_max_s
        still_streak = 0
        prev = self._get_pose()
        self._publish_zero()
        time.sleep(self._cmd_period)
        while time.monotonic() < deadline:
            if self.episode_done:
                return
            self._publish_zero()
            time.sleep(self._cmd_period)
            cur = self._get_pose()
            moved = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
            turned = abs(self._angle_diff(cur[2], prev[2]))
            prev = cur
            if moved < self._cfg.settle_xy_eps_m and turned < self._settle_yaw_eps:
                still_streak += 1
                if still_streak >= 2:
                    return
            else:
                still_streak = 0
        sys.stderr.write(
            f"\033[1;33m[nav] SETTLE CAP {self._cfg.settle_max_s:.2f}s — "
            f"base still moving at exit\033[0m\n")

    def execute_action(self, action: int):
        """Dispatch one blocking primitive; returns the latest observation."""
        if self._aborted:
            self._publish_zero()
            self.set_episode_done()
            return None

        if action == FORWARD:
            return self._run_forward()
        if action == LEFT:
            return self._run_turn(+1)
        if action == RIGHT:
            return self._run_turn(-1)
        if action in (LOOK_UP, LOOK_DOWN):
            return self._run_look(+1 if action == LOOK_UP else -1)

        self._publish_zero()
        return None

    def _run_forward(self):
        start_xy = np.array(self._get_pose()[:2], dtype=np.float64)
        stall_since = time.monotonic()
        prev_dist = 0.0

        for _ in range(int(10.0 / self._cmd_period)):
            if self.episode_done:
                break
            remaining = max(0.0, self._cfg.forward_step_m - prev_dist)
            self._publish_vel(
                max(self._cfg.v_min,
                    min(self._cfg.linear_speed, self._cfg.ramp_k_lin * remaining)),
                0.0, 0.0)
            time.sleep(self._cmd_period)
            dist = float(np.linalg.norm(np.array(self._get_pose()[:2]) - start_xy))
            if dist + self._dist_eps >= self._cfg.forward_step_m:
                break
            if abs(dist - prev_dist) < 5e-4:
                if time.monotonic() - stall_since > self._stall_timeout:
                    break
            else:
                stall_since = time.monotonic()
            prev_dist = dist

        self._settle()
        return self.get_latest_observation()

    def _run_turn(self, direction: int):
        start_yaw = float(self._get_pose()[2])
        prev_progress = 0.0
        stall_since = time.monotonic()

        for _ in range(int(10.0 / self._cmd_period)):
            if self.episode_done:
                break
            remaining = max(0.0, self._turn_angle_rad - prev_progress)
            self._publish_vel(0.0, 0.0, direction * max(
                self._w_min,
                min(self._angular_speed, self._cfg.ramp_k_ang * remaining)))
            time.sleep(self._cmd_period)
            progress = direction * self._angle_diff(
                float(self._get_pose()[2]), start_yaw)
            if progress + self._angle_eps >= self._turn_angle_rad:
                break
            if abs(progress - prev_progress) < math.radians(0.05):
                if time.monotonic() - stall_since > self._stall_timeout:
                    break
            else:
                stall_since = time.monotonic()
            prev_progress = progress

        self._settle()
        return self.get_latest_observation()

    def _run_look(self, direction: int):
        """Head pitch is not on /stretch/cmd_vel — no-op until pan/tilt is wired."""
        self._publish_zero()
        return self.get_latest_observation()

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        d = a - b
        return math.atan2(math.sin(d), math.cos(d))


def _primitive_goal(pose, action, forward_step_m, turn_angle_rad):
    """Convert one Habitat-style action to an absolute SE(2) goal."""
    x, y, yaw = map(float, pose)
    if action == FORWARD:
        x += forward_step_m * math.cos(yaw)
        y += forward_step_m * math.sin(yaw)
    elif action == LEFT:
        yaw += turn_angle_rad
    elif action == RIGHT:
        yaw -= turn_angle_rad
    return x, y, math.atan2(math.sin(yaw), math.cos(yaw))


class GotoRobotAPI(RealRobotAPI):
    """Nonblocking waypoints or discrete robot-local goto primitives."""

    def __init__(self, node: Node, shared_state, cfg) -> None:
        super().__init__(node, shared_state, cfg)
        self._control_pose = None
        self._control_pose_time = 0.0
        self._control_pose_version = 0
        self._control_pose_lock = threading.Lock()
        self._goal_lock = threading.Lock()
        self._goal_pending = False
        self._motion_seen = False
        self._at_goal_event = threading.Event()
        self.last_action_succeeded = True

        self._goal_pub = node.create_publisher(Pose, "/goto_controller/goal", 10)
        node.create_subscription(
            PoseStamped,
            "/state_estimator/pose_filtered",
            self._control_pose_cb,
            10,
        )
        node.create_subscription(
            Bool, "/goto_controller/at_goal", self._at_goal_cb, 10
        )
        self._nav_mode_cli = node.create_client(
            Trigger, "/switch_to_navigation_mode"
        )
        self._enable_cli = node.create_client(Trigger, "/goto_controller/enable")
        self._disable_cli = node.create_client(Trigger, "/goto_controller/disable")
        self._yaw_tracking_cli = node.create_client(
            SetBool, "/goto_controller/set_yaw_tracking"
        )

    def prepare(self, timeout_s=5.0, *, track_yaw=True) -> None:
        """Verify the bridge and put Stretch in navigation mode before moving."""
        for name, client in (
            ("/switch_to_navigation_mode", self._nav_mode_cli),
            ("/goto_controller/enable", self._enable_cli),
            ("/goto_controller/disable", self._disable_cli),
        ):
            if not client.wait_for_service(timeout_sec=timeout_s):
                raise RuntimeError(f"{name} unavailable — is src/env/launch.py running?")

        for name, client in (
            ("/switch_to_navigation_mode", self._nav_mode_cli),
            ("/goto_controller/disable", self._disable_cli),
        ):
            future = client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(
                self._node, future, timeout_sec=timeout_s
            )
            response = future.result()
            if response is None or not response.success:
                message = "timed out" if response is None else response.message
                raise RuntimeError(f"{name} failed: {message}")

        if self._yaw_tracking_cli.wait_for_service(timeout_sec=0.2):
            future = self._yaw_tracking_cli.call_async(
                SetBool.Request(data=bool(track_yaw))
            )
            rclpy.spin_until_future_complete(
                self._node, future, timeout_sec=timeout_s
            )
            response = future.result()
            if response is None or not response.success:
                message = "timed out" if response is None else response.message
                sys.stderr.write(
                    "[RealRobotAPI] could not set goto yaw tracking: "
                    f"{message}; continuing with the controller default\n"
                )
        elif not track_yaw:
            sys.stderr.write(
                "[RealRobotAPI] goto yaw-tracking service unavailable; "
                "continuing with streamed heading goals\n"
            )
        print("[RealRobotAPI] robot-local navigation controller ready")

    def _control_pose_cb(self, msg: PoseStamped) -> None:
        q = msg.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        with self._control_pose_lock:
            self._control_pose = (
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                yaw,
            )
            self._control_pose_time = time.monotonic()
            self._control_pose_version += 1

    def _at_goal_cb(self, msg: Bool) -> None:
        with self._goal_lock:
            if not self._goal_pending:
                return
            if not msg.data:
                self._motion_seen = True
            elif self._motion_seen:
                self._at_goal_event.set()

    def get_control_pose(self) -> Optional[Tuple[float, float, float]]:
        with self._control_pose_lock:
            if self._control_pose is None:
                return None
            if time.monotonic() - self._control_pose_time > self._cfg.control_pose_timeout_s:
                return None
            return self._control_pose

    def _wait_for_control_pose(self, after_version, timeout_s):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._control_pose_lock:
                fresh = (
                    self._control_pose is not None
                    and self._control_pose_version > after_version
                    and time.monotonic() - self._control_pose_time
                    <= self._cfg.control_pose_timeout_s
                )
                if fresh:
                    return self._control_pose
            if self.episode_done:
                return None
            time.sleep(0.02)
        return None

    def _call_trigger(self, client, name, timeout_s=2.0, abortable=True) -> bool:
        if not client.service_is_ready():
            sys.stderr.write(f"[RealRobotAPI] {name} unavailable\n")
            return False
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + timeout_s
        while not future.done() and time.monotonic() < deadline:
            if abortable and self.episode_done:
                return False
            time.sleep(0.01)
        response = future.result() if future.done() else None
        if response is None or not response.success:
            message = "timed out" if response is None else response.message
            sys.stderr.write(f"[RealRobotAPI] {name} failed: {message}\n")
            return False
        return True

    def stop_tracking(self) -> None:
        """Disable the local controller and explicitly zero the base."""
        for _ in range(3):
            self._publish_zero()
        if self._disable_cli.service_is_ready():
            self._call_trigger(
                self._disable_cli,
                "/goto_controller/disable",
                timeout_s=0.5,
                abortable=False,
            )
        for _ in range(3):
            self._publish_zero()

    def emergency_stop(self) -> None:
        super().emergency_stop()
        self.stop_tracking()

    def send_done(self) -> None:
        self.stop_tracking()
        super().send_done()

    def execute_action(self, action: int):
        if action not in (FORWARD, LEFT, RIGHT):
            return super().execute_action(action)
        if self.episode_done:
            self.stop_tracking()
            self.last_action_succeeded = False
            return None

        start = self.get_control_pose()
        if start is None:
            return self._fail_action("filtered control pose is missing or stale")
        target = _primitive_goal(
            start, action, self._cfg.forward_step_m, self._turn_angle_rad
        )
        goal = Pose()
        goal.position.x, goal.position.y = target[:2]
        goal.orientation.z = math.sin(target[2] / 2.0)
        goal.orientation.w = math.cos(target[2] / 2.0)

        deadline = time.monotonic() + self._cfg.primitive_timeout_s
        while time.monotonic() < deadline:
            if not self._call_trigger(
                self._enable_cli, "/goto_controller/enable"
            ):
                return self._fail_action("could not enable goto controller")
            with self._goal_lock:
                self._goal_pending = True
                self._motion_seen = False
                self._at_goal_event.clear()
            with self._control_pose_lock:
                pose_version = self._control_pose_version
            self._goal_pub.publish(goal)

            while not self._at_goal_event.wait(0.05):
                if self.episode_done:
                    return self._fail_action("primitive interrupted")
                if self.get_control_pose() is None:
                    return self._fail_action("filtered control pose became stale")
                if time.monotonic() >= deadline:
                    return self._fail_action("goto controller timed out")

            with self._goal_lock:
                self._goal_pending = False
            current = self._wait_for_control_pose(
                pose_version,
                min(
                    self._cfg.control_pose_wait_s,
                    max(0.0, deadline - time.monotonic()),
                ),
            )
            if current is None:
                return self._fail_action(
                    "no fresh filtered control pose after completion"
                )
            xy_error = math.hypot(target[0] - current[0], target[1] - current[1])
            yaw_error = abs(self._angle_diff(target[2], current[2]))
            if (xy_error <= self._cfg.primitive_xy_tolerance_m
                    and yaw_error <= math.radians(
                        self._cfg.primitive_yaw_tolerance_deg
                    )):
                self.last_action_succeeded = True
                return self.get_latest_observation()
            print(f"[RealRobotAPI] correcting primitive residual: "
                  f"{xy_error:.2f}m, {math.degrees(yaw_error):.1f}deg")

        return self._fail_action("goto controller timed out")

    def _fail_action(self, reason):
        with self._goal_lock:
            self._goal_pending = False
        self.last_action_succeeded = False
        self.stop_tracking()
        sys.stderr.write(f"[RealRobotAPI] primitive failed: {reason}\n")
        return None
