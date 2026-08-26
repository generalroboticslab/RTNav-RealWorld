"""Route targets outside explored FOW through a nearby frontier."""
import cv2
import numpy as np

from navigation_safety import planning_free
from rtnav.modules.mapping.obstacle_map.utils import find_navigable_goal

APPROACH_WAYPOINT = "approach_waypoint"
NAVIGABLE_STANDOFF = "navigable_standoff"


def near_explored(explored, x, y, radius):
    """Whether an explored cell lies within a circular pixel neighborhood."""
    radius = max(0, int(radius))
    y0, y1 = max(0, y - radius), min(explored.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(explored.shape[1], x + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return False
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    return bool(np.any(explored[y0:y1, x0:x1] & disk))


def nearest_frontier_goal(target_xy, obstacle_map, frontier_output):
    if frontier_output is None:
        return None
    choices = []
    centroids = getattr(frontier_output, "frontier_centroids", None)
    if centroids is None:
        return None
    for centroid in centroids:
        nav = obstacle_map.find_navigable_frontier_goal(centroid)
        if nav is None:
            continue
        distance = np.linalg.norm(np.asarray(nav[0], dtype=float) - target_xy)
        choices.append((distance, nav))
    return min(choices, key=lambda choice: choice[0])[1] if choices else None


def approach_waypoint_target(target, nav_xy):
    """Build temporary goal metadata without demoting the confirmed target."""
    waypoint = dict(target)
    waypoint["x"], waypoint["y"] = nav_xy
    waypoint["surface_x"], waypoint["surface_y"] = nav_xy
    waypoint["temporary_target"] = True
    waypoint[APPROACH_WAYPOINT] = True
    return waypoint


def confirmed_target_standoff(target, obstacle_map, robot_xy, max_distance_m):
    """Project a close confirmed target onto the nearest safe navigable cell."""
    if (
        not target.get("vlm_confirmed")
        or target.get("temporary_target")
        or robot_xy is None
    ):
        return None
    center_xy = np.asarray((target["x"], target["y"]), dtype=float)
    if np.linalg.norm(center_xy - np.asarray(robot_xy, dtype=float)) > max_distance_m:
        return None

    target_xy = np.asarray(
        (target.get("surface_x", target["x"]), target.get("surface_y", target["y"])),
        dtype=float,
    )
    free = planning_free(obstacle_map)
    robot_nav = find_navigable_goal(
        obstacle_map,
        tuple(robot_xy),
        search_radius_m=0.5,
        navigable=free,
    )
    if robot_nav is None:
        return None
    _count, labels = cv2.connectedComponents(np.asarray(free, dtype=np.uint8))
    robot_label = labels[robot_nav[1][1], robot_nav[1][0]]
    free = labels == robot_label
    target_px = obstacle_map.xy_to_px(target_xy.reshape(1, 2))[0].astype(int)
    target_in_bounds = (
        0 <= target_px[0] < free.shape[1]
        and 0 <= target_px[1] < free.shape[0]
    )
    if target_in_bounds and free[target_px[1], target_px[0]]:
        return None
    nav = find_navigable_goal(
        obstacle_map,
        tuple(target_xy),
        search_radius_m=max_distance_m,
        navigable=free,
    )
    if nav is None:
        return None
    stand = dict(target)
    stand["surface_x"], stand["surface_y"] = nav[0]
    stand[NAVIGABLE_STANDOFF] = True
    return stand


def is_final_target_approach(current_target, current_goal):
    if current_target is None or current_target.get("temporary_target"):
        return False
    if (
        not current_goal
        or len(current_goal) < 3
        or not isinstance(current_goal[2], dict)
    ):
        return False
    goal_target = current_goal[2].get("target") or {}
    return bool(goal_target.get(APPROACH_WAYPOINT))


def install() -> None:
    from rtnav.modules.decision.target_strategy import TargetStrategy

    original_init = TargetStrategy.__init__
    original_make_goal = TargetStrategy._make_goal
    original_finish_temporary = TargetStrategy.finish_temporary_target

    def init(self, shared_state, decision_cfg, camera_cfg):
        original_init(self, shared_state, decision_cfg, camera_cfg)
        stop = float(decision_cfg.target_stop_distance_m)
        self.TARGET_COMMIT_DIST_M = stop
        self.TEMPORARY_TARGET_REACHED_DIST_M = stop
        self._fow_tolerance_m = float(decision_cfg.target_fow_tolerance_m)

    def make_goal(self, target, obstacle_map, robot_xy=None):
        target_xy = np.asarray(self._target_surface_xy(target), dtype=float)
        stand = confirmed_target_standoff(
            target,
            obstacle_map,
            robot_xy,
            float(self._camera_cfg.max_depth),
        )
        if stand is not None:
            nav_xy = self._target_surface_xy(stand)
            print(
                f"[Target] close confirmed target is not navigable; stopping at "
                f"nearest safe point ({nav_xy[0]:.1f}, {nav_xy[1]:.1f})"
            )
            return original_make_goal(self, stand, obstacle_map, robot_xy)

        px = obstacle_map.xy_to_px(target_xy.reshape(1, 2))[0].astype(int)
        x, y = int(px[0]), int(px[1])
        explored = obstacle_map.explored
        radius = round(self._fow_tolerance_m * obstacle_map.ppm)
        if near_explored(explored, x, y, radius):
            return original_make_goal(self, target, obstacle_map, robot_xy)

        with self.shared_state.lock:
            frontier_output = self.shared_state.frontier.frontier_output
        nav = nearest_frontier_goal(target_xy, obstacle_map, frontier_output)
        if nav is None:
            print("[Target] outside explored FOW; waiting for a frontier")
            return None

        nav_xy = tuple(float(value) for value in nav[0])
        goal_target = approach_waypoint_target(target, nav_xy)
        print(
            f"[Target] confirmed target remains locked; approaching frontier waypoint "
            f"({nav_xy[0]:.1f}, {nav_xy[1]:.1f})"
        )
        return original_make_goal(self, goal_target, obstacle_map, robot_xy)

    def finish_temporary_target(self, reached, reason=""):
        if is_final_target_approach(self._current_target, self._current_goal):
            waypoint_xy = self._current_goal[0]
            outcome = "reached" if reached else f"failed ({reason or 'unknown'})"
            print(
                f"[Target] approach waypoint {outcome} at "
                f"({waypoint_xy[0]:.2f},{waypoint_xy[1]:.2f}); "
                "retaining confirmed target and replanning"
            )
            self._current_goal = None
            return
        original_finish_temporary(self, reached, reason)

    TargetStrategy.__init__ = init
    TargetStrategy._make_goal = make_goal
    TargetStrategy.finish_temporary_target = finish_temporary_target
