"""Click-to-drive tracker test. Drives the base unless ``--dry-run``."""
import argparse
import csv
import json
import math
import sys
import threading
import time
from collections import deque
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
from tracker_thread import TrackerThread

WINDOW = "path follower - left-click a goal, q to quit"
VIEW_RADIUS_M = 6.0
ROBOT_BGR = (0, 0, 255)
GOAL_BGR = (255, 0, 255)
PATH_BGR = (0, 0, 0)          # what the planner asked for
TRAIL_BGR = (200, 0, 0)       # where the base actually went
TARGET_BGR = (0, 180, 0)
TRAIL_LEN = 600


def point_to_polyline_distance(point, path):
    """True cross-track distance (nearest segment, not nearest waypoint)."""
    point, path = np.asarray(point, float), np.asarray(path, float)
    starts, vectors = path[:-1], np.diff(path, axis=0)
    lengths2 = np.einsum("ij,ij->i", vectors, vectors)
    u = np.divide(np.einsum("ij,ij->i", point - starts, vectors), lengths2,
                  out=np.zeros_like(lengths2), where=lengths2 > 1e-12)
    projections = starts + np.clip(u, 0.0, 1.0)[:, None] * vectors
    return float(np.linalg.norm(projections - point, axis=1).min())


def route_length(points):
    points = np.asarray(points, float)
    return (float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            if len(points) >= 2 else 0.0)


class MeasuredAPI:
    """Transparent drive wrapper exposing the commands used by the tracker."""

    def __init__(self, api):
        self.api = api
        self.last_command = (0.0, 0.0)

    def drive(self, vx, wz):
        self.last_command = (float(vx), float(wz))
        return self.api.drive(vx, wz)

    def __getattr__(self, name):
        return getattr(self.api, name)


class PlanOnlyAPI:
    """The planner's actions go nowhere; TrackerThread owns the base."""

    episode_done = False

    def execute_action(self, action):
        time.sleep(0.1)  # paces replanning

    def drive(self, vx, wz):
        pass


def summarize_trial(trial, status, cfg):
    """Return comparable metrics for one goal."""
    if trial is None or len(trial["samples"]) < 2:
        return None
    samples = trial["samples"]
    errors = np.asarray([s["cross_track_m"] for s in samples])
    trail = np.asarray([[s["x_m"], s["y_m"]] for s in samples])
    commands = np.asarray([[s["vx_mps"], s["wz_radps"]] for s in samples])
    duration = samples[-1]["elapsed_s"]
    driven = route_length(trail)
    initial_distance = float(np.linalg.norm(np.asarray(trial["goal"]) - trail[0]))
    final_error = float(np.linalg.norm(np.asarray(trial["goal"]) - trail[-1]))
    moving = np.abs(commands[:, 0]) > 1e-3
    turning = (np.abs(commands[:, 0]) <= 1e-3) & (np.abs(commands[:, 1]) > 1e-3)
    return {
        "trial": trial["number"], "status": status, "samples": len(samples),
        "duration_s": round(duration, 3), "initial_goal_distance_m": round(initial_distance, 4),
        "final_goal_error_m": round(final_error, 4), "distance_driven_m": round(driven, 4),
        "path_efficiency": round(initial_distance / max(driven, 1e-9), 4),
        "cross_track_rmse_m": round(float(np.sqrt(np.mean(errors ** 2))), 4),
        "cross_track_mean_m": round(float(errors.mean()), 4),
        "cross_track_p95_m": round(float(np.percentile(errors, 95)), 4),
        "cross_track_max_m": round(float(errors.max()), 4),
        "mean_forward_speed_mps": round(float(np.abs(commands[moving, 0]).mean()), 4) if moving.any() else 0.0,
        "turn_in_place_fraction": round(float(turning.mean()), 4),
        "lookahead_m": cfg.lookahead_m, "v_max_mps": cfg.v_max,
        "turn_rate_degps": cfg.turn_rate_deg, "slow_radius_m": cfg.slow_radius_m,
        "turn_in_place_deg": cfg.turn_in_place_deg,
    }


def write_trial(trial, summary, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"trial_{trial['number']:03d}_{int(time.time())}"
    with (output_dir / f"{stem}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=trial["samples"][0].keys())
        writer.writeheader()
        writer.writerows(trial["samples"])
    with (output_dir / "summary.jsonl").open("a") as stream:
        stream.write(json.dumps(summary, sort_keys=True) + "\n")
    print("[test] " + " ".join((
        f"trial={summary['trial']}", f"status={summary['status']}",
        f"RMSE={summary['cross_track_rmse_m'] * 100:.1f}cm",
        f"p95={summary['cross_track_p95_m'] * 100:.1f}cm",
        f"goal={summary['final_goal_error_m'] * 100:.1f}cm",
        f"efficiency={summary['path_efficiency']:.2f}",
        f"turn-only={summary['turn_in_place_fraction'] * 100:.0f}%")))
    print(f"[test] wrote {output_dir / (stem + '.csv')}")


def draw_route(image, omap, bounds, route, colour, width):
    """Polyline of world XY points, clipped to the view."""
    if route is None or len(route) < 2:
        return
    pts = np.array([world_to_view_px(omap, xy, bounds) for xy in route], np.int32)
    # Clip per point: cv2 drops a whole segment if either end is off-image.
    np.clip(pts[:, 0], 0, image.shape[1] - 1, out=pts[:, 0])
    np.clip(pts[:, 1], 0, image.shape[0] - 1, out=pts[:, 1])
    cv2.polylines(image, [pts.reshape(-1, 1, 2)], False, colour, width, cv2.LINE_AA)


def set_goal(shared_state, xy):
    with shared_state.lock:
        nav = shared_state.nav
        nav.goal_xy = (float(xy[0]), float(xy[1]))
        nav.nav_id += 1
        nav.status = "navigating"
    print(f"[test] goal ({xy[0]:+.2f}, {xy[1]:+.2f})")


def start_mapping(shared_state, shutdown, cfg):
    """Perception and mapping, so there is a map to spin up and plan against."""
    for thread in (PerceptionThread(shared_state, shutdown, cfg),
                   MappingThread(shared_state, shutdown, cfg)):
        thread.start()
    shared_state.task_ready.set()  # every worker blocks on the task gate


def start_driving(shared_state, shutdown, robot_api, cfg):
    """The planner plans; the tracker spins to seed the map, then drives."""
    planner = PlannerThread(
        shared_state, shutdown, robot_api=PlanOnlyAPI(),
        # The tracker drives, so the planner's own spin would be a no-op.
        cfg=replace(cfg.planner, bootstrap_turns=0),
        publish_bootstrap_complete=False)
    tracker = TrackerThread(shared_state, shutdown, robot_api=robot_api,
                            cfg=cfg.tracker, path_provider=lambda: planner.path_xy)
    planner.start()
    tracker.start()
    return planner, tracker


def main():
    parser = argparse.ArgumentParser(description="Click a goal, watch the follower")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan and draw, but never command the base")
    parser.add_argument("--lookahead", type=float, help="pure-pursuit lookahead (m)")
    parser.add_argument("--v-max", type=float, help="maximum forward speed (m/s)")
    parser.add_argument("--turn-rate", type=float, help="yaw limit (deg/s)")
    parser.add_argument("--slow-radius", type=float, help="goal slowdown radius (m)")
    parser.add_argument("--turn-in-place", type=float, help="turn-only threshold (deg)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "pd_follower",
                        help="CSV samples and summary.jsonl destination")
    args = parser.parse_args()
    dry_run = args.dry_run

    cfg = config.build_config()
    overrides = {"lookahead_m": args.lookahead, "v_max": args.v_max,
                 "turn_rate_deg": args.turn_rate, "slow_radius_m": args.slow_radius,
                 "turn_in_place_deg": args.turn_in_place}
    cfg = replace(cfg, tracker=replace(
        cfg.tracker, **{key: value for key, value in overrides.items() if value is not None}))
    for name in ("lookahead_m", "v_max", "turn_rate_deg", "slow_radius_m"):
        if getattr(cfg.tracker, name) <= 0:
            parser.error(f"{name} must be positive")
    if not 0 < cfg.tracker.turn_in_place_deg <= 180:
        parser.error("turn_in_place_deg must be in (0, 180]")
    print("[test] tuning "
          f"lookahead={cfg.tracker.lookahead_m:.2f}m "
          f"v_max={cfg.tracker.v_max:.2f}m/s "
          f"turn_rate={cfg.tracker.turn_rate_deg:.0f}deg/s "
          f"slow_radius={cfg.tracker.slow_radius_m:.2f}m "
          f"turn_in_place={cfg.tracker.turn_in_place_deg:.0f}deg")
    shared_state = SharedState()
    shutdown = threading.Event()

    rclpy.init()
    subscriber = RealStretchSubscriber(shared_state)
    if dry_run:
        robot_api = PlanOnlyAPI()
        # Nothing turns, so the spin would just burn its timeout.
        cfg = replace(cfg, tracker=replace(cfg.tracker, spin_degrees=0.0))
        print("[test] DRY RUN — planning only, the base will not move")
    else:
        robot_api = RealRobotAPI(subscriber, shared_state, cfg.nav)
        estop.start(robot_api.emergency_stop, shutdown)
    robot_api = MeasuredAPI(robot_api)
    shared_state.system.robot_api = robot_api

    start_mapping(shared_state, shutdown, cfg)
    threading.Thread(target=rclpy.spin, args=(subscriber,), daemon=True).start()

    planner, tracker = start_driving(shared_state, shutdown, robot_api, cfg)

    view = {}  # what the last frame was rendered against, to undo world_to_view_px
    trail = deque(maxlen=TRAIL_LEN)
    errors = []  # cross-track samples for the goal in progress
    trial = None
    trial_number = 0

    def finish_trial(status):
        nonlocal trial
        summary = summarize_trial(trial, status, cfg.tracker)
        if summary is not None:
            write_trial(trial, summary, args.output_dir)
        trial = None

    def on_click(event, x, y, _flags, _param):
        nonlocal trial, trial_number
        if event == cv2.EVENT_LBUTTONDOWN and view:
            px = np.array([[x + view["bounds"][0], y + view["bounds"][1]]], float)
            goal_xy = view["map"].px_to_xy(px)[0]
            if trial is not None:
                finish_trial("interrupted")
            set_goal(shared_state, goal_xy)
            trial_number += 1
            trial = {"number": trial_number, "goal": tuple(goal_xy),
                     "start": time.monotonic(), "samples": []}
            trail.clear()
            errors.clear()

    cv2.namedWindow(WINDOW)
    cv2.waitKey(1)  # Qt builds the window on its own thread; bind after it exists
    cv2.setMouseCallback(WINDOW, on_click)
    print("[test] left-click a goal once the map has filled in")

    while rclpy.ok() and not shutdown.is_set():
        with shared_state.lock:
            omap = shared_state.mapping.obstacle_map
            goal = shared_state.nav.goal_xy
            odom = shared_state.sensor.latest_odom
            nav_status = str(shared_state.nav.status)
        image, bounds = render_region(omap, VIEW_RADIUS_M)
        view["map"], view["bounds"] = omap, bounds

        path = planner.path_xy
        if odom is not None:
            trail.append(odom[:2])
        draw_route(image, omap, bounds, path, PATH_BGR, 2)
        draw_route(image, omap, bounds, trail, TRAIL_BGR, 1)

        if path is not None and len(path) >= 2 and odom is not None:
            # Distance to the nearest point of the plan: how far off it drove.
            error = point_to_polyline_distance(odom[:2], path)
            errors.append(error)
            if trial is not None:
                vx, wz = robot_api.last_command
                trial["samples"].append({
                    "elapsed_s": round(time.monotonic() - trial["start"], 4),
                    "x_m": float(odom[0]), "y_m": float(odom[1]),
                    "yaw_rad": float(odom[2]), "cross_track_m": error,
                    "goal_error_m": float(np.linalg.norm(np.asarray(trial["goal"]) - odom[:2])),
                    "vx_mps": vx, "wz_radps": wz,
                    "target_x_m": float(tracker.target_xy[0]) if tracker.target_xy else math.nan,
                    "target_y_m": float(tracker.target_xy[1]) if tracker.target_xy else math.nan,
                })
        if goal is None and trial is not None:  # arrived or gave up — score it
            finish_trial(nav_status)
            errors.clear()
        if errors:
            off = np.asarray(errors) * 100.0
            cv2.putText(image, f"cross-track {off[-1]:3.0f}cm  mean {off.mean():3.0f}cm"
                        f"  max {off.max():3.0f}cm", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        if odom is not None:
            cv2.circle(image, world_to_view_px(omap, odom[:2], bounds), 5,
                       ROBOT_BGR, -1, cv2.LINE_AA)
        if goal is not None:
            cv2.drawMarker(image, world_to_view_px(omap, goal, bounds), GOAL_BGR,
                           cv2.MARKER_CROSS, 16, 2)
        target = tracker.target_xy  # the waypoint the controller is driving at
        if target is not None and odom is not None:
            cv2.line(image, world_to_view_px(omap, odom[:2], bounds),
                     world_to_view_px(omap, target, bounds),
                     TARGET_BGR, 2, cv2.LINE_AA)
        cv2.imshow(WINDOW, image)
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break

    if trial is not None:
        finish_trial("quit")
    shutdown.set()
    cv2.destroyAllWindows()
    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
