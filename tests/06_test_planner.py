"""Click-to-drive planner test. Drives the base unless ``--dry-run``."""
import argparse
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import rclpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "agent"))
sys.path.insert(0, str(ROOT / "src" / "utils"))
sys.path.insert(0, str(ROOT / "rt_ovn" / "agents" / "rtnav"))

import config
import estop
import patches

patches.install()  # before rtnav binds geometry builders

from l3mvn_fmm_planner_thread import PlannerThread
from patches.robot_api import RealRobotAPI
from rtnav.core.shared_state import SharedState
from rtnav.modules.mapping.mapping_thread import MappingThread
from rtnav.modules.perception.perception_thread import PerceptionThread
from rtnav.tools.visualization.obstacle_map_renderer import (
    render_region,
    world_to_view_px,
)
from subscriber import RealStretchSubscriber

WINDOW = "planner - left-click a goal, q to quit"
VIEW_RADIUS_M = 6.0
ROBOT_BGR = (0, 0, 255)
GOAL_BGR = (255, 0, 255)
WAYPOINT_BGR = (0, 180, 0)
PATH_BGR = (0, 0, 0)


class DryRunAPI:
    """robot_api stand-in: the planner runs, the base does not."""

    episode_done = False

    def execute_action(self, action):
        time.sleep(0.5)  # stands in for a real primitive's duration

    def emergency_stop(self):
        pass


def draw_field(image, planner, bounds):
    """Tint the view with the planner's FMM field: blue near the goal, red far.

    The field covers the robot-goal crop, so it only overlaps part of the view.
    """
    field = planner.fmm_field
    if field is None:
        return
    row0, col0 = planner.fmm_origin
    x_min, y_min = bounds[0], bounds[1]
    r0, c0 = max(row0, y_min), max(col0, x_min)
    r1 = min(row0 + field.shape[0], y_min + image.shape[0])
    c1 = min(col0 + field.shape[1], x_min + image.shape[1])
    if r1 <= r0 or c1 <= c0:
        return

    patch = field[r0 - row0:r1 - row0, c0 - col0:c1 - col0]
    # Obstacles carry max+1, which would flatten everything else; clip them off.
    scale = max(float(np.percentile(patch, 90)), 1e-6)
    heat = cv2.applyColorMap(
        (np.clip(patch / scale, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    view = image[r0 - y_min:r1 - y_min, c0 - x_min:c1 - x_min]
    image[r0 - y_min:r1 - y_min, c0 - x_min:c1 - x_min] = cv2.addWeighted(
        heat, 0.45, view, 0.55, 0.0)


def set_goal(shared_state, xy):
    with shared_state.lock:
        nav = shared_state.nav
        nav.goal_xy = (float(xy[0]), float(xy[1]))
        nav.nav_id += 1
        nav.status = "navigating"
    print(f"[test] goal ({xy[0]:+.2f}, {xy[1]:+.2f})")


def start_threads(shared_state, shutdown, robot_api, cfg):
    """Perception and mapping feed the map; the planner drives to the goal.

    Returns the planner, whose short_term_goal the window draws.
    """
    planner = PlannerThread(shared_state, shutdown,
                            robot_api=robot_api, cfg=cfg.planner)
    for thread in (PerceptionThread(shared_state, shutdown, cfg),
                   MappingThread(shared_state, shutdown, cfg),
                   planner):
        thread.start()
    shared_state.task_ready.set()  # every worker blocks on the task gate
    return planner


def main():
    parser = argparse.ArgumentParser(description="Click a goal, watch the planner")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan and draw, but never command the base")
    dry_run = parser.parse_args().dry_run

    cfg = config.build_config()
    shared_state = SharedState()
    shutdown = threading.Event()

    rclpy.init()
    subscriber = RealStretchSubscriber(shared_state)
    if dry_run:
        robot_api = DryRunAPI()
        cfg = replace(cfg, planner=replace(cfg.planner, bootstrap_turns=0))
        print("[test] DRY RUN — planning only, the base will not move")
    else:
        robot_api = RealRobotAPI(subscriber, shared_state, cfg.nav)
        estop.start(robot_api.emergency_stop, shutdown)
    shared_state.system.robot_api = robot_api

    planner = start_threads(shared_state, shutdown, robot_api, cfg)
    threading.Thread(target=rclpy.spin, args=(subscriber,), daemon=True).start()

    view = {}  # what the last frame was rendered against, to undo world_to_view_px

    def on_click(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and view:
            px = np.array([[x + view["bounds"][0], y + view["bounds"][1]]], float)
            set_goal(shared_state, view["map"].px_to_xy(px)[0])

    cv2.namedWindow(WINDOW)
    cv2.waitKey(1)  # Qt builds the window on its own thread; bind after it exists
    cv2.setMouseCallback(WINDOW, on_click)
    print("[test] left-click a goal once the map has filled in")

    while rclpy.ok() and not shutdown.is_set():
        with shared_state.lock:
            omap = shared_state.mapping.obstacle_map
            goal = shared_state.nav.goal_xy
            odom = shared_state.sensor.latest_odom
        image, bounds = render_region(omap, VIEW_RADIUS_M)
        view["map"], view["bounds"] = omap, bounds

        draw_field(image, planner, bounds)  # under the markers
        path = planner.path_xy
        if path is not None and len(path) >= 2:
            pts = np.array([world_to_view_px(omap, xy, bounds) for xy in path],
                           dtype=np.int32)
            # Clip per point: cv2 drops a whole segment if either end is off-image.
            np.clip(pts[:, 0], 0, image.shape[1] - 1, out=pts[:, 0])
            np.clip(pts[:, 1], 0, image.shape[0] - 1, out=pts[:, 1])
            cv2.polylines(image, [pts.reshape(-1, 1, 2)], False, PATH_BGR, 2,
                          cv2.LINE_AA)
        if odom is not None:
            cv2.circle(image, world_to_view_px(omap, odom[:2], bounds), 5,
                       ROBOT_BGR, -1, cv2.LINE_AA)
        if goal is not None:
            cv2.drawMarker(image, world_to_view_px(omap, goal, bounds), GOAL_BGR,
                           cv2.MARKER_CROSS, 16, 2)
        waypoint = planner.short_term_goal  # what the last plan steered to
        if waypoint is not None and odom is not None:
            cv2.line(image, world_to_view_px(omap, odom[:2], bounds),
                     world_to_view_px(omap, waypoint, bounds),
                     WAYPOINT_BGR, 2, cv2.LINE_AA)
        cv2.imshow(WINDOW, image)
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break

    shutdown.set()
    cv2.destroyAllWindows()
    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
