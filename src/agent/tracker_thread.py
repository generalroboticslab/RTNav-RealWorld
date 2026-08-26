"""Path-heading follower for the planner's FMM path."""
import math
import threading
import time

import numpy as np

from navigation_safety import path_is_free, planning_free


def wrap(angle):
    """Radians to (-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def accumulate_left_turn(turned, previous, current):
    """Accumulate net left rotation without counting odometry corrections twice."""
    return max(0.0, turned + wrap(current - previous))


def stream_ready(shared_state, min_frames=1):
    """Fresh observations are flowing and at least one reached mapping."""
    with shared_state.lock:
        return (
            shared_state.sensor.habitat_obs is not None
            and getattr(shared_state.sensor, "control_odom", None) is not None
            and getattr(shared_state.sensor, "map_odom_anchor", None) is not None
            and shared_state.perception.perception_version >= min_frames
            and shared_state.mapping.mapping_output is not None
        )


def verified_target_ready(shared_state):
    """Whether target navigation should preempt the opening scan."""
    with shared_state.lock:
        blocked = shared_state.target.target_node_blacklist_ids
        return any(
            target.get("vlm_confirmed")
            and (target.get("node_id") is None or int(target["node_id"]) not in blocked)
            for target in shared_state.target.target_goals
        )


def adaptive_lookahead(path, xy, near_m, far_m, straight_tolerance_m):
    """Use far_m only when the corresponding FMM path segment is straight."""
    closest = int(np.argmin(np.linalg.norm(path - xy, axis=1)))
    tail = path[closest:]
    if len(tail) < 2:
        return path[-1]

    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(tail, axis=0), axis=1))]

    def point_at(distance):
        indices = np.flatnonzero(arc >= distance)
        return int(indices[0]) if indices.size else len(tail) - 1

    near_index = point_at(near_m)
    far_index = point_at(far_m)
    chord = tail[far_index] - tail[0]
    chord_length = float(np.linalg.norm(chord))
    if far_index == near_index or chord_length < far_m * 0.8:
        return tail[near_index]

    offsets = tail[:far_index + 1] - tail[0]
    cross = np.abs(chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0])
    return tail[far_index] if np.max(cross / chord_length) <= straight_tolerance_m else tail[near_index]


def slew(current, target, max_delta):
    return current + float(np.clip(target - current, -max_delta, max_delta))


def damped_yaw_rate(error, measured_rate, gain, damping_s, limit):
    return float(np.clip(gain * error - damping_s * measured_rate, -limit, limit))


def map_points_to_odom(points, map_pose, odom_pose):
    """Transform map-frame xy points into the smooth odom frame."""
    points = np.asarray(points, dtype=float)
    map_pose = np.asarray(map_pose, dtype=float)
    odom_pose = np.asarray(odom_pose, dtype=float)
    angle = odom_pose[2] - map_pose[2]
    c, s = np.cos(angle), np.sin(angle)
    delta = points - map_pose[:2]
    return np.column_stack((
        odom_pose[0] + c * delta[:, 0] - s * delta[:, 1],
        odom_pose[1] + s * delta[:, 0] + c * delta[:, 1],
    ))


def map_yaw_to_odom(yaw, map_pose, odom_pose):
    return wrap(yaw + odom_pose[2] - map_pose[2])


def frontier_alignment_error(pose, remaining, goal_yaw, anchor, max_distance):
    """Desired odom-frame turn near a frontier, independent of its FMM path."""
    if (pose is None or remaining is None or remaining >= max_distance
            or goal_yaw is None or anchor is None):
        return None
    return wrap(map_yaw_to_odom(goal_yaw, *anchor) - pose[2])


def path_tracking_error(path, xy, heading_distance):
    """Return smoothed path heading and signed lateral error."""
    segments = path[1:] - path[:-1]
    lengths2 = np.sum(segments * segments, axis=1)
    valid = lengths2 > 1e-9
    t = np.zeros(len(segments))
    t[valid] = np.clip(
        np.sum((xy - path[:-1][valid]) * segments[valid], axis=1)
        / lengths2[valid],
        0.0,
        1.0,
    )
    projections = path[:-1] + t[:, None] * segments
    segment_index = int(np.argmin(np.linalg.norm(projections - xy, axis=1)))
    nearest = projections[segment_index]

    distances = np.linalg.norm(np.diff(path[segment_index:], axis=0), axis=1)
    ahead = np.flatnonzero(np.cumsum(distances) >= heading_distance)
    end = segment_index + 1 + (int(ahead[0]) if ahead.size else len(distances) - 1)
    tangent = path[end] - nearest
    if np.linalg.norm(tangent) < 1e-6:
        tangent = segments[segment_index]
    tangent /= max(np.linalg.norm(tangent), 1e-6)
    lateral_error = tangent[0] * (xy[1] - nearest[1]) - tangent[1] * (xy[0] - nearest[0])
    return math.atan2(tangent[1], tangent[0]), float(lateral_error)


def path_curvature(path, xy, near_distance, far_distance):
    """Average heading change per metre ahead of the robot."""
    near_yaw, _ = path_tracking_error(path, xy, near_distance)
    far_yaw, _ = path_tracking_error(path, xy, far_distance)
    return abs(wrap(far_yaw - near_yaw)) / max(
        far_distance - near_distance, 1e-3
    )


class TrackerThread(threading.Thread):
    """Drive path_provider()'s route waypoint by waypoint until the end."""

    def __init__(self, shared_state, shutdown_event, *, robot_api, cfg, path_provider):
        super().__init__(name="TrackerThread", daemon=True)
        self._shared_state = shared_state
        self._shutdown = shutdown_event
        self._robot_api = robot_api
        self._cfg = cfg
        self._path_provider = path_provider

        self.target_xy = None  # the waypoint being driven at, for viz
        self.command = (0.0, 0.0, "idle")
        self._turning_in_place = False
        self.heading_error = 0.0
        self.lateral_error = 0.0
        self.path_curvature = 0.0
        self.desired_yaw_rate = 0.0
        self.measured_yaw_rate = 0.0
        self._yaw_command = 0.0
        self._open_lookahead = False
        self.odom_read_wall_ns = None
        self.odom_sequence = None
        self.odom_callback_age_ms = None

    def _drive(self, vx, wz, mode):
        previous_mode = self.command[2]
        self.command = (float(vx), float(wz), mode)
        self._robot_api.drive(vx, wz)
        if mode == "path_blocked" and previous_mode != mode:
            print(
                f"[tracker] path blocked within "
                f"{max(self._cfg.lookahead_m, self._cfg.open_lookahead_m):.1f}m — stopping"
            )

    def _pose(self):
        with self._shared_state.lock:
            return getattr(self._shared_state.sensor, "control_odom", None)

    def spin(self):
        """Rotate in place to seed the map."""
        period = 1.0 / self._cfg.rate_hz
        target = math.radians(self._cfg.spin_degrees)
        if target <= 0.0:
            return

        print("[tracker] waiting for observation stream")
        while not stream_ready(self._shared_state, self._cfg.stream_warmup_frames):
            if self._shutdown.wait(period):
                return
        pose = self._pose()
        while pose is None:
            if self._shutdown.wait(period):
                return
            pose = self._pose()
        deadline = time.monotonic() + self._cfg.spin_timeout_s
        turned, previous = 0.0, pose[2]
        print(f"[tracker] spin {self._cfg.spin_degrees:.0f}deg")

        while turned < target and time.monotonic() < deadline:
            if self._shutdown.is_set():
                break
            if verified_target_ready(self._shared_state):
                print("[tracker] verified target interrupted bootstrap spin")
                break
            pose = self._pose()
            if pose is None:
                self._drive(0.0, 0.0, "spin_pose_stale")
                time.sleep(period)
                continue
            yaw = pose[2]
            turned = accumulate_left_turn(turned, previous, yaw)
            previous = yaw
            rate = math.radians(self._cfg.turn_rate_deg)
            self._drive(0.0, rate, "spin")
            time.sleep(period)

        self._drive(0.0, 0.0, "spin_done")
        print(f"[tracker] spin done ({math.degrees(turned):.0f}deg)")

    def _lookahead(self, path, xy):
        tolerance = (
            self._cfg.open_path_exit_tolerance_m
            if self._open_lookahead else self._cfg.open_path_tolerance_m
        )
        target = adaptive_lookahead(
            path,
            xy,
            self._cfg.lookahead_m,
            self._cfg.open_lookahead_m,
            tolerance,
        )
        self._open_lookahead = (
            np.linalg.norm(target - xy)
            > (self._cfg.lookahead_m + self._cfg.open_lookahead_m) / 2.0
        )
        return target

    def run(self):
        period = 1.0 / self._cfg.rate_hz
        turn_in_place = math.radians(self._cfg.turn_in_place_deg)
        resume_forward = math.radians(self._cfg.resume_forward_deg)
        turn_rate = math.radians(self._cfg.turn_rate_deg)
        self.spin()
        with self._shared_state.lock:
            self._shared_state.system.bootstrap_spin_complete = True

        while not self._shutdown.is_set():
            time.sleep(period)
            map_path = self._path_provider()
            read_wall_ns = time.time_ns()
            read_monotonic_ns = time.monotonic_ns()
            with self._shared_state.lock:
                pose = getattr(self._shared_state.sensor, "control_odom", None)
                odom_meta = getattr(
                    self._shared_state.sensor, "control_odom_meta", None
                )
                anchor = getattr(self._shared_state.sensor, "map_odom_anchor", None)
                obstacle_map = self._shared_state.mapping.obstacle_map
                goal = self._shared_state.nav.goal_xy
                goal_yaw = getattr(self._shared_state.nav, "goal_yaw", None)
                measured_yaw_rate = float(getattr(
                    self._shared_state.sensor, "control_yaw_rate", 0.0
                ))
            self.measured_yaw_rate = measured_yaw_rate
            self.odom_read_wall_ns = read_wall_ns
            self.odom_sequence = (
                None if odom_meta is None else odom_meta["callback_sequence"]
            )
            self.odom_callback_age_ms = (
                None if odom_meta is None else
                (read_monotonic_ns - odom_meta["arrival_monotonic_ns"]) / 1e6
            )
            path = (None if map_path is None or anchor is None else
                    map_points_to_odom(map_path, *anchor))
            odom_goal = (None if goal is None or anchor is None else
                         map_points_to_odom([goal], *anchor)[0])
            xy = None if pose is None else np.asarray(pose[:2], float)
            remaining = (None if odom_goal is None or xy is None
                         else float(np.linalg.norm(odom_goal - xy)))
            map_xy = (
                None if xy is None or anchor is None else
                map_points_to_odom([xy], anchor[1], anchor[0])[0]
            )
            alpha = frontier_alignment_error(
                pose,
                remaining,
                goal_yaw,
                anchor,
                self._cfg.frontier_alignment_dist_m,
            )
            if alpha is not None:
                self.heading_error = alpha
                if abs(alpha) > math.radians(
                    self._cfg.frontier_heading_tolerance_deg
                ):
                    desired = damped_yaw_rate(
                        alpha, measured_yaw_rate, self._cfg.turn_kp,
                        self._cfg.heading_damping_s, turn_rate,
                    )
                    self._yaw_command = slew(
                        self._yaw_command,
                        desired,
                        math.radians(self._cfg.angular_accel_deg) * period,
                    )
                    self._drive(0.0, self._yaw_command, "frontier_alignment")
                    continue
            if map_path is not None and map_xy is not None and obstacle_map is not None:
                if not path_is_free(
                    map_path,
                    map_xy,
                    planning_free(obstacle_map),
                    obstacle_map.xy_to_px,
                    max_distance_m=max(
                        self._cfg.lookahead_m,
                        self._cfg.open_lookahead_m,
                    ),
                ):
                    self.target_xy = None
                    self._turning_in_place = False
                    self._yaw_command = 0.0
                    self._drive(0.0, 0.0, "path_blocked")
                    continue
            if (remaining is None or remaining < self._cfg.goal_tol_m
                    or path is None or len(path) < 2):
                self.target_xy = None
                self._turning_in_place = False
                self.heading_error = 0.0
                self.lateral_error = 0.0
                self.path_curvature = 0.0
                self.desired_yaw_rate = 0.0
                self._yaw_command = 0.0
                self._open_lookahead = False
                self._drive(0.0, 0.0, "idle")
                continue

            target = self._lookahead(np.asarray(path, float), xy)
            target_index = int(np.argmin(np.linalg.norm(path - target, axis=1)))
            self.target_xy = tuple(np.asarray(map_path)[target_index])
            path_yaw, lateral_error = path_tracking_error(
                path, xy, self._cfg.path_heading_distance_m
            )
            curvature = path_curvature(
                path,
                xy,
                self._cfg.path_heading_distance_m,
                self._cfg.curvature_horizon_m,
            )
            self.lateral_error = lateral_error
            self.path_curvature = curvature
            correction = math.atan2(
                self._cfg.cross_track_gain * lateral_error,
                self._cfg.cross_track_softening_mps + self._cfg.v_max,
            )
            alpha = wrap(path_yaw - correction - pose[2])
            self.heading_error = alpha

            if abs(alpha) > turn_in_place:
                self._turning_in_place = True
            if self._turning_in_place:
                if abs(alpha) <= resume_forward:
                    self._turning_in_place = False
                else:
                    self.desired_yaw_rate = damped_yaw_rate(
                        alpha, measured_yaw_rate, self._cfg.turn_kp,
                        self._cfg.heading_damping_s, turn_rate,
                    )
                    self._yaw_command = self.desired_yaw_rate
                    self._drive(
                        0.0, self._yaw_command, "turn_in_place"
                    )
                    continue

            braking_distance = max(0.0, remaining - self._cfg.goal_tol_m)
            speed = self._cfg.v_max * min(
                1.0, braking_distance / self._cfg.slow_radius_m
            )
            speed *= max(0.0, math.cos(alpha))
            speed /= 1.0 + self._cfg.curvature_slowdown_m * curvature
            speed *= max(
                self._cfg.cross_track_min_speed_fraction,
                1.0 - abs(lateral_error) / self._cfg.cross_track_slowdown_m,
            )
            desired = damped_yaw_rate(
                alpha, measured_yaw_rate, self._cfg.heading_gain,
                self._cfg.heading_damping_s, turn_rate,
            )
            if abs(alpha) < math.radians(self._cfg.steering_deadband_deg):
                desired = damped_yaw_rate(
                    0.0, measured_yaw_rate, 0.0,
                    self._cfg.heading_damping_s, turn_rate,
                )
            self.desired_yaw_rate = desired
            self._yaw_command = slew(
                self._yaw_command,
                desired,
                math.radians(self._cfg.angular_accel_deg) * period,
            )
            self._drive(speed, self._yaw_command, "path_tracking")
