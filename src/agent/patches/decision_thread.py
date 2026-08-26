"""Carry the selected frontier's outward heading into navigation."""
import math

import numpy as np


def frontier_heading(frontier_output, obstacle_map, goal_xy, robot_xy=None):
    centroids = getattr(frontier_output, "frontier_centroids", ())
    directions = getattr(frontier_output, "frontier_unexplored_directions", ())
    if not len(centroids):
        return None
    xy = np.asarray(obstacle_map.px_to_xy(np.asarray(centroids)), dtype=float)
    index = int(np.argmin(np.linalg.norm(xy - np.asarray(goal_xy), axis=1)))
    direction = directions[index] if index < len(directions) else None
    if direction is not None:
        direction = np.asarray(direction, dtype=float)
        if np.linalg.norm(direction) > 1e-6:
            return math.atan2(direction[1], direction[0])

    # A snapped navigable goal normally sits just inside the frontier. If the
    # local explored/unexplored estimate was empty, at least face its boundary.
    direction = xy[index] - np.asarray(goal_xy, dtype=float)
    if np.linalg.norm(direction) <= 1e-6 and robot_xy is not None:
        direction = xy[index] - np.asarray(robot_xy, dtype=float)
    if np.linalg.norm(direction) <= 1e-6:
        return None
    return math.atan2(direction[1], direction[0])


def install():
    from rtnav.modules.decision.decision_thread import DecisionThread

    original_send = DecisionThread._send_goal
    original_flush = DecisionThread._flush_nav

    def send_goal(self, goal, source, nav):
        original_send(self, goal, source, nav)
        nav.goal_yaw = None
        if source != "frontier":
            return
        with self.shared_state.lock:
            output = self.shared_state.frontier.frontier_output
            obstacle_map = self.shared_state.mapping.obstacle_map
            sensor = self.shared_state.sensor
            robot_pose = getattr(sensor, "planning_pose", sensor.latest_odom)
        if output is not None and obstacle_map is not None:
            nav.goal_yaw = frontier_heading(
                output,
                obstacle_map,
                goal[0],
                None if robot_pose is None else robot_pose[:2],
            )

    def flush_nav(self, update):
        original_flush(self, update)
        if update.dirty:
            with self.shared_state.lock:
                self.shared_state.nav.goal_yaw = getattr(update, "goal_yaw", None)

    DecisionThread._send_goal = send_goal
    DecisionThread._flush_nav = flush_nav
