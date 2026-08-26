"""FMM planning on rt_ovn's obstacle map."""
import math
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import skimage.morphology

from navigation_safety import planning_free

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "rt_ovn/agents/baseline_l3mvn/L3MVN/envs/utils"))
from fmm_planner import FMMPlanner

STOP, FORWARD, LEFT, RIGHT = 0, 1, 2, 3  # RealRobotAPI's contract
WAIT = -1  # keep navigation active while waiting for orientation/map state
NAMES = {STOP: "STOP", FORWARD: "FWD", LEFT: "L", RIGHT: "R"}


def wrap(angle):
    """Radians to (-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def cells(meters, ppm):
    """Metres to map cells, at least one."""
    return max(1, int(round(meters * ppm)))


def trace_path(field, start, limit=1000):
    """Trace cells down an FMM field."""
    row, col = start
    path = [(row, col)]
    for _ in range(limit):
        r0, c0 = max(0, row - 1), max(0, col - 1)
        window = field[r0:row + 2, c0:col + 2]
        dr, dc = np.unravel_index(np.argmin(window), window.shape)
        if (r0 + dr, c0 + dc) == (row, col):
            break
        row, col = r0 + dr, c0 + dc
        path.append((row, col))
    return path


def _line_cells(start, end):
    count = max(abs(end[0] - start[0]), abs(end[1] - start[1])) + 1
    rows = np.rint(np.linspace(start[0], end[0], count)).astype(int)
    cols = np.rint(np.linspace(start[1], end[1], count)).astype(int)
    return list(zip(rows, cols))


def simplify_path(path, clearance, min_clearance):
    """Straighten only chords that remain safely clear of obstacles."""
    if len(path) < 3:
        return path
    simplified = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        farthest = anchor + 1
        # ponytail: O(n²) visibility scan; replace if paths exceed the 1000-cell cap.
        for candidate in range(anchor + 2, len(path)):
            line = _line_cells(path[anchor], path[candidate])
            if min(clearance[cell] for cell in line) < min_clearance:
                break
            farthest = candidate
        line = _line_cells(path[anchor], path[farthest])
        simplified.extend(line[1:])
        anchor = farthest
    return simplified


def stable_path(previous, proposed, pose_xy, free, xy_to_px,
                improvement_m, endpoint_tolerance_m):
    """Keep a safe current route unless replanning is meaningfully shorter."""
    if previous is None or len(previous) < 2:
        return proposed
    closest = int(np.argmin(np.linalg.norm(previous - pose_xy, axis=1)))
    current = previous[closest:]
    if len(current) < 2 or np.linalg.norm(current[-1] - proposed[-1]) > endpoint_tolerance_m:
        return proposed
    px = xy_to_px(current).astype(int)
    inside = (
        (px[:, 0] >= 0) & (px[:, 0] < free.shape[1])
        & (px[:, 1] >= 0) & (px[:, 1] < free.shape[0])
    )
    if not np.all(inside) or not np.all(free[px[:, 1], px[:, 0]]):
        return proposed

    current_length = np.linalg.norm(current[0] - pose_xy) + np.linalg.norm(
        np.diff(current, axis=0), axis=1
    ).sum()
    proposed_length = np.linalg.norm(proposed[0] - pose_xy) + np.linalg.norm(
        np.diff(proposed, axis=0), axis=1
    ).sum()
    return proposed if proposed_length + improvement_m < current_length else current


def add_boundary(mat, value=1.0):
    """L3MVN pads by one cell so FMM cannot escape the crop."""
    out = np.full((mat.shape[0] + 2, mat.shape[1] + 2), value, dtype=float)
    out[1:-1, 1:-1] = mat
    return out


def reachable_goal_map(traversible, robot, goal, radius, max_projection):
    """Return the requested goal, or its nearest reachable approach cell."""
    _, labels = cv2.connectedComponents(traversible.astype(np.uint8))
    robot_label = labels[robot]
    if robot_label == 0:
        return None, False
    reachable = labels == robot_label
    requested = np.zeros_like(traversible, dtype=bool)
    requested[goal] = True
    requested = skimage.morphology.binary_dilation(
        requested, skimage.morphology.disk(radius)
    )
    if np.any(requested & reachable):
        return requested & reachable, not bool(reachable[goal])

    cells_rc = np.argwhere(reachable)
    nearest = cells_rc[np.argmin(np.sum((cells_rc - goal) ** 2, axis=1))]
    if np.linalg.norm(nearest - goal) > max_projection:
        return None, False
    projected = np.zeros_like(traversible, dtype=bool)
    projected[tuple(nearest)] = True
    projected = skimage.morphology.binary_dilation(
        projected, skimage.morphology.disk(radius)
    )
    return projected & reachable, True


class PlannerThread(threading.Thread):
    """Turns shared_state.nav.goal_xy into discrete actions on robot_api."""

    def __init__(self, shared_state, shutdown_event, *, robot_api, cfg,
                 publish_bootstrap_complete=True):
        super().__init__(name="PlannerThread", daemon=True)
        self._shared_state = shared_state
        self._shutdown = shutdown_event
        self._robot_api = robot_api
        self._cfg = cfg
        self._publish_bootstrap_complete = publish_bootstrap_complete

        self.short_term_goal = None
        self.path_xy = None
        self.fmm_field = None
        self._projected_log_key = None
        self.fmm_origin = (0, 0)

        self._bootstrap_done = False
        self._goal_id = -1
        self._best_rho = math.inf
        self._best_t = 0.0
        self._reset_requested = threading.Event()

    def reset_episode(self):
        """Called by the runner on an episode boundary."""
        self._reset_requested.set()

    def _read_goal(self):
        with self._shared_state.lock:
            nav = self._shared_state.nav
            goal = nav.goal_xy
            return (
                tuple(goal) if goal else None,
                getattr(nav, "goal_yaw", None),
                int(nav.nav_id),
                str(nav.status),
            )

    def _finish(self, reason):
        with self._shared_state.lock:
            nav = self._shared_state.nav
            if int(nav.nav_id) == self._goal_id:
                nav.status = "arrived" if reason == "success" else "failed"
                nav.goal_xy = None
                nav.failure_reason = None if reason == "success" else reason
        self._goal_id = -1
        self.path_xy = None
        print(f"[planner] {reason}")

    def _plan(self, pose, goal, goal_yaw=None):
        """Return ``(action, distance)`` or ``(None, distance)``."""
        self.short_term_goal = self.fmm_field = None
        rho = math.hypot(goal[0] - pose[0], goal[1] - pose[1])
        with self._shared_state.lock:
            omap = self._shared_state.mapping.obstacle_map
        full_free = planning_free(omap)
        (r_col, r_row), (g_col, g_row) = omap.xy_to_px(
            np.array([pose[:2], goal], dtype=float))
        if not (
            0 <= r_col < full_free.shape[1]
            and 0 <= r_row < full_free.shape[0]
            and 0 <= g_col < full_free.shape[1]
            and 0 <= g_row < full_free.shape[0]
        ):
            return None, rho
        success_dist = (
            self._cfg.frontier_alignment_dist_m
            if goal_yaw is not None else self._cfg.success_dist_m
        )
        if rho < success_dist and full_free[g_row, g_col]:
            if (goal_yaw is not None and
                    abs(wrap(goal_yaw - pose[2])) >
                    math.radians(self._cfg.frontier_heading_tolerance_deg)):
                return WAIT, rho
            return STOP, rho

        margin = int(self._cfg.crop_margin_m * omap.ppm)
        row0 = max(0, min(r_row, g_row) - margin)
        col0 = max(0, min(r_col, g_col) - margin)
        crop = np.s_[row0:min(omap.size, max(r_row, g_row) + margin + 1),
                     col0:min(omap.size, max(r_col, g_col) + margin + 1)]
        robot = (r_row - row0 + 1, r_col - col0 + 1)  # +1 for add_boundary
        goal_px = (g_row - row0 + 1, g_col - col0 + 1)

        crop_free = full_free[crop]
        traversible = add_boundary(crop_free, 0.0)
        traversible[robot] = 1.0  # the base may occupy an inflated/rounding cell

        goal_map, projected = reachable_goal_map(
            traversible,
            robot,
            goal_px,
            cells(self._cfg.goal_radius_m, omap.ppm),
            cells(self._cfg.max_goal_projection_m, omap.ppm),
        )
        if goal_map is None:
            return None, rho
        if projected:
            approach = np.argwhere(goal_map)[0]
            projection_m = (
                math.hypot(
                    approach[0] - goal_px[0], approach[1] - goal_px[1]
                ) / omap.ppm
            )
            log_key = self._goal_id, round(projection_m, 2)
            if log_key != self._projected_log_key:
                print(
                    f"[planner] goal blocked/disconnected; using reachable "
                    f"approach {projection_m:.2f}m away"
                )
                self._projected_log_key = log_key
        goal_map = goal_map.astype(float)

        planner = FMMPlanner(traversible,
                             step_size=cells(self._cfg.step_m, omap.ppm))
        clearance = cv2.distanceTransform(
            traversible.astype(np.uint8), cv2.DIST_L2, 5
        )
        desired_clearance = cells(self._cfg.clearance_m, omap.ppm)
        clearance_penalty = np.clip(
            1.0 - clearance / desired_clearance, 0.0, 1.0
        ) * (self._cfg.clearance_cost_m * omap.ppm)
        planner.around = np.maximum(planner.around, clearance_penalty)
        planner.set_multi_goal(goal_map)
        stg_row, stg_col, _replan, stop = planner.get_short_term_goal(list(robot))

        self.fmm_field = planner.fmm_dist
        self.fmm_origin = (row0 - 1, col0 - 1)  # add_boundary shifted by one
        if stop:
            self.path_xy = None
            return (STOP if projected else None), rho
        self.short_term_goal = tuple(omap.px_to_xy(np.array(  # undo crop + padding
            [[stg_col - 1 + col0, stg_row - 1 + row0]], float))[0])
        path_cells = trace_path(planner.fmm_dist, robot)
        path_cells = simplify_path(path_cells, clearance, desired_clearance)
        proposed_path = omap.px_to_xy(np.array(
            [[c - 1 + col0, r - 1 + row0]
             for r, c in path_cells], float))
        path = stable_path(
            self.path_xy,
            proposed_path,
            np.asarray(pose[:2], dtype=float),
            full_free,
            omap.xy_to_px,
            self._cfg.path_replan_improvement_m,
            self._cfg.path_endpoint_tolerance_m,
        )
        self.path_xy = path

        # Map rows point opposite world y.
        relative = wrap(math.atan2(-(stg_row - robot[0]), stg_col - robot[1]) - pose[2])
        half_turn = math.radians(self._cfg.turn_angle_deg) / 2.0
        if relative > half_turn:
            return LEFT, rho
        if relative < -half_turn:
            return RIGHT, rho
        return FORWARD, rho

    def _bootstrap_spin(self):
        """Opening turn-in-place, so the map has something before the first plan."""
        print(f"[planner] bootstrap spin: {self._cfg.bootstrap_turns} LEFT turns")
        start = time.time()
        completed = 0
        for _ in range(self._cfg.bootstrap_turns):
            if (self._shutdown.is_set() or self._robot_api.episode_done
                    or time.time() - start > self._cfg.bootstrap_max_s):
                break
            self._robot_api.execute_action(LEFT)
            if not getattr(self._robot_api, "last_action_succeeded", True):
                print("[planner] bootstrap spin failed — base stopped")
                self._shutdown.set()
                return
            completed += 1
        if completed != self._cfg.bootstrap_turns:
            if not self._shutdown.is_set() and not self._robot_api.episode_done:
                print(f"[planner] bootstrap spin timed out after {completed} turns")
                self._shutdown.set()
            return
        self._bootstrap_done = True
        with self._shared_state.lock:
            self._shared_state.frontier.frontier_output = None
            if self._publish_bootstrap_complete:
                self._shared_state.system.bootstrap_spin_complete = True
        print(f"[planner] bootstrap spin complete: {completed} turns "
              f"({time.time() - start:.1f}s)")

    def run(self):
        from rtnav.utils.task_gate import wait_for_task_ready

        while not self._shutdown.is_set():
            if not wait_for_task_ready(self._shared_state, "planner", self._shutdown):
                return
            if self._reset_requested.is_set():
                self._reset_requested.clear()
                self._bootstrap_done = False
                self._goal_id = -1
                self.short_term_goal = self.fmm_field = None
                self.path_xy = None
                print("[planner] episode reset")
            pose = None
            get_control_pose = getattr(self._robot_api, "get_control_pose", None)
            if get_control_pose is not None:
                pose = get_control_pose()
            else:
                with self._shared_state.lock:
                    sensor = self._shared_state.sensor
                    pose = getattr(sensor, "planning_pose", sensor.latest_odom)
            if pose is None:
                self._shutdown.wait(0.05)
                continue

            if not self._bootstrap_done:
                self._bootstrap_spin()
                continue

            goal, goal_yaw, nav_id, status = self._read_goal()
            if goal is None or status != "navigating":
                self.path_xy = None
                self._shutdown.wait(0.1)
                continue
            if nav_id != self._goal_id:
                self._goal_id, self._best_rho, self._best_t = nav_id, math.inf, time.time()
                self.path_xy = None

            action, rho = self._plan(pose, goal, goal_yaw)
            if action is None:
                self._finish("no_progress")
                continue
            if action == STOP:
                self._finish("success")
                continue

            # Give up when nothing closes the distance, so Target picks another
            # goal instead of the robot grinding at this one.
            now = time.time()
            if rho < self._best_rho - 0.01:
                self._best_rho, self._best_t = rho, now
            elif now - self._best_t > self._cfg.no_progress_s:
                tracker = getattr(self, "tracker", None)
                tracker_blocked = (
                    tracker is not None
                    and getattr(tracker, "command", (0.0, 0.0, "idle"))[2]
                    == "path_blocked"
                )
                self._finish("path_blocked" if tracker_blocked else "no_progress")
                continue

            if action == WAIT:
                self._shutdown.wait(0.1)
                continue

            if self._cfg.log_actions:
                print(f"[planner] {NAMES[action]} rho={rho:.2f}m pose=("
                      f"{pose[0]:+.2f},{pose[1]:+.2f},{math.degrees(pose[2]):+.0f}deg)")
            self._robot_api.execute_action(action)  # blocks until the move ends
            if not getattr(self._robot_api, "last_action_succeeded", True):
                self._finish("controller_failed")
