"""Bound frontier stickiness by cumulative, not per-update, movement."""
import numpy as np

MAX_ANCHOR_MISSES = 3


def anchor_has_match(anchor, points, radius):
    points = np.asarray(points, dtype=float)
    return bool(len(points) and np.min(np.linalg.norm(points - anchor, axis=1)) <= radius)


def hold_missing_frontier(misses, cached):
    return misses < MAX_ANCHOR_MISSES and cached is not None


def install():
    from rtnav.modules.decision.frontier_strategy import FrontierStrategy

    original_remember = FrontierStrategy._remember_choice
    original_refresh = FrontierStrategy.refresh_goal
    original_clear = FrontierStrategy._clear_sticky_state
    original_finish = FrontierStrategy._finish_selection

    def remember(self, idx, nav_xy, centroids_xy, robot_xy):
        original_remember(self, idx, nav_xy, centroids_xy, robot_xy)
        self._sticky_anchor_xy = self._last_frontier_xy.copy()
        self._sticky_anchor_misses = 0

    def finish(self, result):
        self._sticky_last_goal = result
        return original_finish(self, result)

    def refresh(self, context):
        anchor = getattr(self, "_sticky_anchor_xy", None)
        output = context.get("frontier_output")
        obstacle_map = context.get("obstacle_map")
        if anchor is not None and output is not None and obstacle_map is not None:
            centroids = np.asarray(output.frontier_centroids)
            points = obstacle_map.px_to_xy(centroids) if len(centroids) else ()
            if not anchor_has_match(anchor, points, self._sticky_match_m):
                self._sticky_anchor_misses = getattr(
                    self, "_sticky_anchor_misses", 0
                ) + 1
                cached = getattr(self, "_sticky_last_goal", None)
                if hold_missing_frontier(self._sticky_anchor_misses, cached):
                    return cached, "frontier temporarily missing"
                return None, "frontier moved beyond sticky anchor"
        self._sticky_anchor_misses = 0
        goal, reason = original_refresh(self, context)
        if goal is not None:
            self._sticky_last_goal = goal
        return goal, reason

    def clear(self):
        original_clear(self)
        self._sticky_anchor_xy = None
        self._sticky_anchor_misses = 0
        self._sticky_last_goal = None

    FrontierStrategy._remember_choice = remember
    FrontierStrategy._finish_selection = finish
    FrontierStrategy.refresh_goal = refresh
    FrontierStrategy._clear_sticky_state = clear
