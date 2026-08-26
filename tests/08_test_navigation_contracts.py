"""Fast navigation and mapping contract checks; no robot required."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "agent"))
sys.path.insert(0, str(ROOT / "rt_ovn" / "agents" / "rtnav"))
sys.modules.setdefault("skfmm", SimpleNamespace())
from patches.vlfm_frontier_detector import keep_wide_frontiers, merge_nearby
from patches.decision_thread import frontier_heading
from patches.frontier_strategy import anchor_has_match, hold_missing_frontier
from patches.camera_geometry import odom_pose_to_map
from patches.target_strategy import (
    approach_waypoint_target,
    confirmed_target_standoff,
    install as install_target_strategy,
    is_final_target_approach,
    near_explored,
    nearest_frontier_goal,
)
from config import build_config
import l3mvn_fmm_planner_thread as planner_module
from l3mvn_fmm_planner_thread import (
    STOP,
    WAIT,
    PlannerThread,
    reachable_goal_map,
    simplify_path,
    stable_path,
)
from navigation_safety import path_is_free, slam_planning_free
from rtnav.modules.decision.decision_thread import should_refresh_frontier_goal
from rtnav.modules.mapping.obstacle_map.utils import is_goal_blocked
from slam_occupancy_map import (
    SlamOccupancyMap,
    close_small_fow_gaps,
    reveal_occupied_endpoints,
)
from tracker_thread import (
    accumulate_left_turn,
    adaptive_lookahead,
    damped_yaw_rate,
    frontier_alignment_error,
    map_points_to_odom,
    map_yaw_to_odom,
    path_curvature,
    path_tracking_error,
    slew,
    stream_ready,
    verified_target_ready,
)


points = np.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0], [2.0, 0.0]])
keep = merge_nearby(points, 0.3)
assert len(keep) == 2
assert np.allclose(points[keep[0]], [0.2, 0.0])
assert np.allclose(points[keep[1]], [2.0, 0.0])

control_path = map_points_to_odom(
    [[10.0, 21.0]], [10.0, 20.0, np.pi / 2], [1.0, 2.0, 0.0]
)
assert np.allclose(control_path, [[2.0, 2.0]])
assert np.isclose(map_yaw_to_odom(np.pi / 2, [0, 0, np.pi / 2], [0, 0, 0]), 0)
assert np.isclose(frontier_alignment_error(
    (0, 0, 0), 0.2, np.pi / 2,
    ((0, 0, 0), (0, 0, 0)), 0.3,
), np.pi / 2)
assert frontier_alignment_error(
    (0, 0, 0), 0.3, np.pi / 2,
    ((0, 0, 0), (0, 0, 0)), 0.3,
) is None
planning_pose = odom_pose_to_map(
    [2.0, 2.0, np.pi / 4], [10.0, 20.0, np.pi / 2], [1.0, 2.0, 0.0]
)
assert np.allclose(planning_pose, [10.0, 21.0, 3 * np.pi / 4])

boundaries = [
    np.array([[[0.0, 0.0]], [[0.1, 0.0]], [[0.2, 0.0]]]),
    np.array([[[0.0, 0.0]], [[0.0, 0.2]], [[0.0, 0.4]], [[0.0, 0.6]]]),
]
assert keep_wide_frontiers(boundaries, 0.5).tolist() == [1]


class Map:
    def px_to_xy(self, points):
        return np.asarray(points, dtype=float)

    def find_navigable_frontier_goal(self, centroid):
        xy = np.asarray(centroid, dtype=float)
        return tuple(xy), tuple(centroid)


class Output:
    frontier_centroids = np.array([(1, 0), (8, 0)])
    frontier_unexplored_directions = [np.array([0, 1]), np.array([1, 0])]


goal = nearest_frontier_goal(np.array([10.0, 0.0]), Map(), Output())
assert goal[0] == (8.0, 0.0)
assert np.isclose(frontier_heading(Output(), Map(), (8.0, 0.0)), 0.0)

class MissingDirectionOutput:
    frontier_centroids = np.array([(2, 1)])
    frontier_unexplored_directions = [None]

assert np.isclose(
    frontier_heading(MissingDirectionOutput(), Map(), (1.0, 1.0)), 0.0
)
assert np.isclose(
    frontier_heading(MissingDirectionOutput(), Map(), (2.0, 1.0), (2.0, 0.0)),
    np.pi / 2,
)

standoff_free = np.zeros((9, 9), bool)
standoff_free[4, :3] = True
standoff_free[4, 4] = True  # closer to the target, but unreachable from the robot
standoff_map = SimpleNamespace(
    size=9,
    ppm=1,
    navigable=np.ones((9, 9), bool),
    _planning_free=standoff_free,
    xy_to_px=lambda points: np.asarray(points, dtype=int),
    px_to_xy=lambda points: np.asarray(points, dtype=float),
)
close_confirmed = {
    "x": 4.0, "y": 4.0, "surface_x": 4.0, "surface_y": 4.0,
    "temporary_target": False, "vlm_confirmed": True,
}
stand = confirmed_target_standoff(close_confirmed, standoff_map, (0.0, 4.0), 4.0)
assert close_confirmed["surface_x"] == 4.0 and not close_confirmed.get("navigable_standoff")
assert stand["temporary_target"] is False and stand["navigable_standoff"]
assert np.allclose((stand["surface_x"], stand["surface_y"]), (2.0, 4.0))
assert confirmed_target_standoff(
    close_confirmed, standoff_map, (-1.0, 4.0), 4.0
) is None

explored = np.zeros((9, 9), dtype=bool)
explored[4, 6] = True
assert not near_explored(explored, 4, 4, 1)
assert near_explored(explored, 4, 4, 2)

confirmed = {"x": 10.0, "y": 2.0, "temporary_target": False, "vlm_confirmed": True}
waypoint = approach_waypoint_target(confirmed, (4.0, 1.0))
assert confirmed["temporary_target"] is False
assert waypoint["temporary_target"] is True and waypoint["approach_waypoint"] is True
assert is_final_target_approach(
    confirmed, ((4.0, 1.0), (4, 1), {"target": waypoint})
)
assert not is_final_target_approach(
    {**confirmed, "temporary_target": True},
    ((4.0, 1.0), (4, 1), {"target": waypoint}),
)
install_target_strategy()
from rtnav.modules.decision.target_strategy import TargetStrategy
strategy = TargetStrategy.__new__(TargetStrategy)
strategy._current_target = confirmed.copy()
strategy._current_goal = ((4.0, 1.0), (4, 1), {"target": waypoint})
strategy.finish_temporary_target(reached=True)
assert strategy._current_target["temporary_target"] is False
assert strategy._current_goal is None
strategy._camera_cfg = SimpleNamespace(max_depth=4.0)
goal = strategy._make_goal(close_confirmed, standoff_map, (0.0, 4.0))
assert np.allclose(goal[0], (2.0, 4.0))
assert goal[2]["target"]["navigable_standoff"]
assert goal[2]["target"]["temporary_target"] is False

slam_map = SlamOccupancyMap.__new__(SlamOccupancyMap)
slam_map._cfg = SimpleNamespace(hybrid_fow_range_m=4.0)
slam_map.ppm = 30
points = np.array([
    [1.0, -1.0],  # left edge
    [2.0, 0.0],   # center; the farther center point wins
    [1.0, 0.0],
    [1.0, 1.0],   # right edge
    [5 * np.cos(np.pi / 8), 5 * np.sin(np.pi / 8)],  # capped bin stays unknown
    [0.0, 1.0],   # outside the 90-degree field of view
])
ranges = slam_map._depth_ray_ranges_px(points, 0, 0, 0, 90, 10, num_rays=5)
assert np.allclose(ranges, [30 * np.sqrt(2), 0, 60, 0, 30 * np.sqrt(2)])
assert np.count_nonzero(ranges) == 3  # missing angular bins remain unknown
assert not slam_map._depth_ray_ranges_px(
    None, 0, 0, 0, 90, 10, num_rays=5
).any()
observed = np.zeros((7, 7), bool)
observed[3, 3] = True
slam_obstacles = np.zeros_like(observed)
slam_obstacles[3, 4] = True
slam_obstacles[3, 6] = True
visible = reveal_occupied_endpoints(observed, slam_obstacles)
assert visible[3, 4] and not visible[3, 6]

class Lock:
    def __enter__(self): return self
    def __exit__(self, *_): pass


state = SimpleNamespace(
    lock=Lock(),
    sensor=SimpleNamespace(
        habitat_obs=object(), control_odom=(0, 0, 0),
        map_odom_anchor=((0, 0, 0), (0, 0, 0)),
    ),
    perception=SimpleNamespace(perception_version=9),
    mapping=SimpleNamespace(mapping_output=None),
)
assert not stream_ready(state, 10)
state.mapping.mapping_output = object()
assert not stream_ready(state, 10)
state.perception.perception_version = 10
assert stream_ready(state, 10)

state.target = SimpleNamespace(target_goals=[], target_node_blacklist_ids=set())
assert not verified_target_ready(state)
state.target.target_goals = [{"vlm_confirmed": True, "node_id": 4}]
assert verified_target_ready(state)
state.target.target_node_blacklist_ids.add(4)
assert not verified_target_ready(state)

turned = accumulate_left_turn(0.0, np.deg2rad(179), np.deg2rad(-179))
assert np.isclose(np.rad2deg(turned), 2.0)
turned = accumulate_left_turn(turned, np.deg2rad(-179), np.deg2rad(179))
assert np.isclose(np.rad2deg(turned), 0.0)

straight = np.array([[x, 0.0] for x in np.linspace(0.0, 1.2, 13)])
assert adaptive_lookahead(straight, straight[0], 0.3, 0.9, 0.08)[0] >= 0.9
corner = np.array([[0.0, 0.0], [0.3, 0.0], [0.5, 0.0], [0.5, 0.4], [0.5, 0.8]])
assert np.allclose(adaptive_lookahead(corner, corner[0], 0.3, 0.9, 0.08), [0.3, 0.0])
gentle = np.array([[0.0, 0.0], [0.3, 0.0], [0.6, 0.3], [0.9, 0.3]])
assert np.allclose(adaptive_lookahead(gentle, gentle[0], 0.3, 0.9, 0.08), [0.3, 0.0])
assert np.allclose(adaptive_lookahead(gentle, gentle[0], 0.3, 0.9, 0.15), [0.9, 0.3])
heading, lateral = path_tracking_error(straight, np.array([0.3, 0.2]), 0.3)
assert np.isclose(heading, 0.0)
assert np.isclose(lateral, 0.2)
assert np.isclose(path_curvature(straight, straight[0], 0.3, 0.9), 0.0)
assert np.isclose(slew(-0.35, 0.35, 0.05), -0.30)
assert np.isclose(slew(0.02, 0.0, 0.05), 0.0)
assert np.isclose(damped_yaw_rate(0.2, 0.4, 1.0, 0.25, 1.0), 0.1)
assert np.isclose(damped_yaw_rate(2.0, 0.0, 1.0, 0.25, 0.5), 0.5)

clearance = np.full((12, 12), 10.0)
zigzag = [(5, 1), (4, 2), (5, 3), (4, 4), (5, 5)]
assert simplify_path(zigzag, clearance, 5.0) == [
    (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)
]
clearance[5, 3] = 0.0
assert (5, 3) not in simplify_path(
    [(5, 1), (4, 2), (3, 3), (4, 4), (5, 5)], clearance, 5.0
)

old = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
new = np.array([[1.0, 5.0], [2.0, 5.1], [3.0, 5.0]])
free = np.ones((10, 10), dtype=bool)
identity = lambda points: np.asarray(points)
assert np.array_equal(stable_path(old, new, old[0], free, identity, 0.15, 0.3), old)
free[5, 2] = False
assert np.array_equal(stable_path(old, new, old[0], free, identity, 0.15, 0.3), new)
assert not path_is_free(old, old[0], free, identity)
free[5, 2] = True
assert path_is_free(old, old[0], free, identity)
assert path_is_free(old, old[-1], free, identity)
free[5, 3] = False
assert path_is_free(old, old[0], free, identity, max_distance_m=0.9)
assert not path_is_free(old, old[0], free, identity, max_distance_m=2.0)

explored = np.zeros((7, 9), bool)
explored[3, 1:8] = True
explored[3, 4] = False
assert close_small_fow_gaps(explored)[3, 4]
explored[3, 3:6] = False
assert not close_small_fow_gaps(explored)[3, 4]

explored = np.array([[1, 0, 1, 0]], bool)
blocked = np.array([[0, 0, 1, 0]], bool)
assert np.array_equal(
    slam_planning_free(explored, blocked),
    np.array([[1, 0, 0, 0]], bool),
)
traversible = np.zeros((9, 12), bool)
traversible[4, 1:5] = True
near_goal, projected = reachable_goal_map(traversible, (4, 1), (4, 6), 1, 2)
assert near_goal is not None and projected
traversible[4, 5] = True
near_goal, projected = reachable_goal_map(traversible, (4, 1), (4, 6), 1, 2)
assert near_goal is not None and projected  # neighbour is free; exact goal is not
assert reachable_goal_map(traversible, (4, 1), (4, 10), 1, 2)[0] is None

projected_free = np.zeros((12, 12), bool)
projected_free[4, 1:6] = True
projected_map = SimpleNamespace(
    size=12,
    ppm=1,
    _planning_free=projected_free,
    xy_to_px=lambda points: np.asarray(points, dtype=int),
    px_to_xy=lambda points: np.asarray(points, dtype=float),
)
projected_planner = PlannerThread.__new__(PlannerThread)
projected_planner._shared_state = SimpleNamespace(
    lock=Lock(), mapping=SimpleNamespace(obstacle_map=projected_map)
)
projected_planner._cfg = SimpleNamespace(
    success_dist_m=0.15,
    frontier_alignment_dist_m=0.3,
    frontier_heading_tolerance_deg=15.0,
    crop_margin_m=5.0,
    goal_radius_m=0.2,
    max_goal_projection_m=2.0,
    step_m=0.25,
    clearance_m=0.45,
    clearance_cost_m=0.4,
    path_replan_improvement_m=0.15,
    path_endpoint_tolerance_m=0.3,
)
projected_planner._goal_id = 1
projected_planner._projected_log_key = None
projected_planner.path_xy = None

class StoppedFMMPlanner:
    def __init__(self, traversible, **_):
        self.around = np.zeros_like(traversible, dtype=float)
        self.fmm_dist = np.zeros_like(traversible, dtype=float)

    def set_multi_goal(self, _):
        pass

    def get_short_term_goal(self, robot):
        return robot[0], robot[1], False, True

original_fmm_planner = planner_module.FMMPlanner
planner_module.FMMPlanner = StoppedFMMPlanner
try:
    assert projected_planner._plan(
        (4.0, 4.0, 0.0), (4.2, 4.0), np.pi / 2
    )[0] == WAIT
    assert projected_planner._plan(
        (4.0, 4.0, np.pi / 2), (4.2, 4.0), np.pi / 2
    )[0] == STOP
    assert projected_planner._plan((4.0, 4.0, 0.0), (6.0, 4.0))[0] == STOP
    assert projected_planner._plan((4.0, 4.0, 0.0), (20.0, 4.0))[0] is None
finally:
    planner_module.FMMPlanner = original_fmm_planner

goal_map = SimpleNamespace(
    _combined_blocked=np.array([[False, True]]),
    xy_to_px=lambda points: np.asarray(points, dtype=int),
)
assert is_goal_blocked(goal_map, (1, 0))
assert is_goal_blocked(goal_map, (2, 0))
assert should_refresh_frontier_goal(0.10, 0.80, True)
assert not should_refresh_frontier_goal(0.10, 0.80, False)
cfg = build_config()
assert cfg.tracker.frontier_alignment_dist_m == cfg.planner.frontier_alignment_dist_m
assert anchor_has_match(np.array([0.0, 0.0]), [[0.7, 0.0]], 0.8)
assert not anchor_has_match(np.array([0.0, 0.0]), [[0.9, 0.0]], 0.8)
assert hold_missing_frontier(1, object())
assert hold_missing_frontier(2, object())
assert not hold_missing_frontier(3, object())
print("navigation contracts: ok")
