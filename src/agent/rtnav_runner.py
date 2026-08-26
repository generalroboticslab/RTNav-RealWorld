"""Real-world RtNav runner with explicit /obs/* ingestion.

This mirrors the canonical rt_ovn runner lifecycle while wiring in the
real-topic subscriber adapter.
"""

import argparse
import os
import signal
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor


def _bootstrap_paths() -> None:
    """Make `rtnav` and our own modules importable."""
    root = Path(__file__).resolve().parents[2]
    for path in (root / "rt_ovn" / "agents" / "rtnav",
                 root / "src" / "utils",
                 root / "src" / "agent"):
        sys.path.insert(0, str(path))


_bootstrap_paths()

# The RTNav image installs these extensions at top level, while the upstream
# wrappers import them relative to their packages. Alias them without changing
# the pinned rt_ovn submodule.
import perception_accel as _perception_accel
import sg_accel as _sg_accel
sys.modules["rtnav.modules.perception.cpp_accel.perception_accel"] = _perception_accel
sys.modules["rtnav.modules.scenegraph.cpp_accel.sg_accel"] = _sg_accel

import config
import estop
import patches
from subscriber import RealStretchSubscriber
from rtnav.config.model_paths import get_llm_model_dir
from rtnav.core.shared_state import SharedState
from rtnav.modules.decision.decision_thread import DecisionThread
from rtnav.modules.frontier.frontier_detection_thread import FrontierDetectionThread
from rtnav.modules.mapping.mapping_thread import MappingThread
from rtnav.modules.perception.detector_thread import OpenVocabDetectorThread
from rtnav.modules.perception.perception_thread import PerceptionThread
from rtnav.modules.scenegraph.scene_graph_thread import SceneGraphThread
from rtnav.task.episode_goal import GoalParser
from rtnav.tools.visualization.detection_visualizer_thread import WebDetectionVisualizerThread
from rtnav.tools.visualization.habitat_obs_map_web_viz_thread import (
    HabitatObsMapWebVizThread,
)
from rtnav.tools.visualization.map_visualizer_thread import MapVisualizerBase
from rtnav.tools.visualization.obstacle_map_renderer import render_region, world_to_view_px
from rtnav.utils.vllm_utils import ensure_vllm_server


VLLM_GPU_MEM = 0.4  # leaves room for OWLv2 + MobileSAM in Thor's unified memory
MAP_VIZ_PORT = 8766


class RealRtNavAgent(object):
    """Main real-world RtNav runtime: build once, shut down cleanly."""

    def __init__(self, cfg, verbose=False, reasoning=True):
        self.cfg = cfg
        self.shared_state = SharedState()
        self.shutdown_event = threading.Event()

        # main() builds these after construction — they need the shared_state that lives here.
        self._robot_api = None
        self.navigation = None
        self._vlm_logger = None

        self._goal_parser = None
        if reasoning:
            from rtnav.tools.visualization.vlm_decision_logger import VLMDecisionLogger

            self._vlm_logger = VLMDecisionLogger()
            self.shared_state.system.vlm_logger = self._vlm_logger
            self._load_models()
        else:
            # GoalParser.set_goal normally opens the task gate every worker waits on.
            self.shared_state.task_ready.set()
            print("[agent] reasoning OFF — no vLLM, no DecisionThread")
        self._build_workers(verbose=bool(verbose), reasoning=reasoning)
        self._start_workers()

        print("[agent] ready env={} threads=[{}]".format(
            self.cfg.env_name, ", ".join(t.name for t in self.threads)))

    def _load_models(self):
        os.environ.setdefault("RTNAV_VLLM_MAX_NUM_SEQS", "1")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        ensure_vllm_server(get_llm_model_dir(), gpu_mem=VLLM_GPU_MEM, max_wait=600.0)
        print(f"[agent] vLLM gpu_memory_utilization={VLLM_GPU_MEM}")
        print("[agent] LLM ready")
        self._goal_parser = GoalParser.build()

    def _build_workers(self, verbose=False, reasoning=True):
        ss, se, cfg = self.shared_state, self.shutdown_event, self.cfg
        DecisionThread.FRONTIER_GOAL_REFRESH_M = (
            cfg.decision.frontier_position_threshold
        )

        self.perception = PerceptionThread(ss, se, cfg, verbose=verbose)
        self.mapping = MappingThread(ss, se, cfg, verbose=verbose)
        self.frontier = FrontierDetectionThread(ss, se, cfg)
        self.detector = OpenVocabDetectorThread(ss, se, cfg)
        self.scene_graph = SceneGraphThread(ss, se, cfg)
        self.decision = DecisionThread(ss, se, cfg) if reasoning else None

        self.threads = [self.perception, self.mapping, self.frontier,
                        self.detector, self.scene_graph]

    def _start_workers(self):
        for thread in self.threads:
            thread.start()
        if self.decision is not None:
            self.decision.start()

    def reset_episode(
        self,
        episode_hash=None,
        goal_info=None,
        scene_id=None,
        episode_id=None,
        output_dir=None,
    ):
        self.shared_state.task_ready.clear()
        self.shared_state.reset_episode()

        self.perception.reset_episode()
        self.detector.reset_episode()
        self.mapping.reset_episode()
        self.scene_graph.reset_episode()
        self.frontier.reset_episode()
        if self.navigation is not None:
            self.navigation.reset_episode()
        if self.decision is not None:
            self.decision.reset_episode()

        logger = self.shared_state.system.vlm_logger
        if logger is not None:
            logger.new_episode(scene_id, episode_id, output_dir=output_dir)

        if self._robot_api is not None:
            self._robot_api.reset_episode(episode_hash, preserve_step_id=True)
        if goal_info is not None and self._goal_parser is not None:
            self._goal_parser.set_goal(self.shared_state, goal_info)

    def shutdown(self, timeout=3.0):
        self.shutdown_event.set()
        if self.decision is not None:
            self.decision.shutdown()
        for thread in self.threads:
            thread.join(timeout=timeout)
        alive = [t.name for t in self.threads if t.is_alive()]
        if alive:
            print(f"[agent] WARNING: threads still alive after shutdown: {alive}")
        if self._vlm_logger is not None:
            self._vlm_logger.shutdown()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-env RtNav bring-up harness")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--map-viz-web", action="store_true", help="Start HabitatObstacleMap web viz")
    p.add_argument("--det-viz-web", action="store_true", help="Start detection web viz")
    p.add_argument(
        "--no-record",
        action="store_true",
        help="Skip recording this run under experiments/",
    )
    p.add_argument(
        "--no-reasoning",
        action="store_true",
        help=(
            "Skip vLLM and the DecisionThread: perception, mapping, frontiers "
            "and detection only, with nothing picking a goal. Ignores --target."
        ),
    )
    p.add_argument(
        "--target",
        type=str,
        default="chair",
        help="Goal target label for real-world task injection (for example: chair)",
    )
    p.add_argument(
        "--enable-navigation",
        action="store_true",
        help=(
            "Attach the navigation driver thread and let it publish velocity "
            "commands to /stretch/cmd_vel. Defaults OFF for safety. Actions "
            "run unprompted; press ENTER at any time to e-stop."
        ),
    )
    p.add_argument(
        "--controller",
        choices=("step", "track"),
        default="track",
        help=(
            "How the planner reaches the base. 'track' continuously follows "
            "the FMM path with forward-only velocity commands (default). "
            "'step' waits for each 25cm/30deg primitive and is intended for "
            "comparison/debugging."
        ),
    )
    p.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Exit after N seconds (0 = run until Ctrl-C)",
    )
    return p.parse_args()


class _PlanOnly:
    """robot_api stand-in for --controller track: the tracker owns the base."""

    episode_done = False

    def execute_action(self, action):
        time.sleep(0.1)  # paces replanning


def _build_navigation(agent, robot_api, controller):
    """Returns (thread the runner resets per episode, threads to start).

    'step' drives L3MVN's single discrete action per plan. 'track' hands the
    whole path to tracker_thread, so the planner stops driving — and its opening
    spin goes with it, since the tracker does its own.
    """
    from l3mvn_fmm_planner_thread import PlannerThread

    if controller == "step":
        planner = PlannerThread(agent.shared_state, agent.shutdown_event,
                                robot_api=robot_api, cfg=agent.cfg.planner)
        return planner, [planner]

    from tracker_thread import TrackerThread

    planner_cfg = replace(
        agent.cfg.planner,
        bootstrap_turns=0,
        success_dist_m=agent.cfg.tracker.goal_tol_m,
    )
    planner = PlannerThread(agent.shared_state, agent.shutdown_event,
                            robot_api=_PlanOnly(),
                            cfg=planner_cfg,
                            publish_bootstrap_complete=False)
    tracker = TrackerThread(agent.shared_state, agent.shutdown_event,
                            robot_api=robot_api, cfg=agent.cfg.tracker,
                            path_provider=lambda: planner.path_xy)
    planner.tracker = tracker
    return planner, [planner, tracker]


def _enable_navigation(agent, controller):
    """Attach the robot API, e-stop and planner. Returns the cmd_vel node."""
    from patches.robot_api import GotoRobotAPI

    node = rclpy.create_node("rtnav_cmd_vel_node")
    robot_api = GotoRobotAPI(node, agent.shared_state, agent.cfg.nav)
    # Also for track: this switches navigation mode and disables any stale
    # goto goal. TrackerThread sends only forward-only Twist commands.
    robot_api.prepare(track_yaw=controller == "step")
    agent._robot_api = robot_api
    agent.shared_state.system.robot_api = robot_api
    estop.start(robot_api.emergency_stop, agent.shutdown_event)

    agent.navigation, threads = _build_navigation(agent, robot_api, controller)
    for thread in threads:
        thread.start()  # _start_workers already ran without them
    agent.threads.extend(threads)
    print(f"[rtnav_runner] navigation ENABLED ({controller}) — "
          "/stretch/cmd_vel will be driven")
    return node


class _PathMapVisualizer(MapVisualizerBase):
    """Decision map with the FMM path."""

    def __init__(self, shared_state, path_provider):
        super().__init__(shared_state)
        self._path_provider = path_provider

    def _render(self, state, view_radius_m):
        if state is None or state["map"] is None:
            return None
        obstacle_map = state["map"]
        image, state["view_bounds"] = render_region(obstacle_map, view_radius_m)
        x0, y0, x1, y1 = state["view_bounds"]
        image.fill(255)
        explored = obstacle_map.explored[y0:y1, x0:x1] > 0
        blocked = obstacle_map.traversability[y0:y1, x0:x1] < 0.1
        occupied = obstacle_map.occupancy[y0:y1, x0:x1] > 0
        image[explored] = (200, 255, 200)
        image[blocked] = (100, 100, 100)
        if getattr(obstacle_map, "is_slam_occupancy", False):
            slam = obstacle_map._slam_occupancy_full[y0:y1, x0:x1] & occupied
            voxel = obstacle_map._depth_occupancy[y0:y1, x0:x1] & occupied
            image[slam] = (0, 0, 0)
            image[voxel] = (255, 255, 0)
            legend_width = 72
            legend_x = max(8, image.shape[1] - legend_width)
            for i, (label, color) in enumerate((
                    ("SLAM", (0, 0, 0)),
                    ("VOXEL", (255, 255, 0)))):
                y = 14 + 16 * i
                cv2.rectangle(
                    image, (legend_x, y - 8), (legend_x + 10, y + 2), color, -1
                )
                cv2.putText(image, label, (legend_x + 15, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, (0, 0, 0), 1)
        else:
            image[occupied] = (0, 0, 0)
        scale_px = int(5.0 * obstacle_map.ppm)
        bar_y = image.shape[0] - 15
        cv2.line(image, (10, bar_y), (10 + scale_px, bar_y), (0, 0, 0), 2)
        cv2.putText(image, "5m", (12, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), 1)
        self._draw_frontiers(image, state)
        self._draw_target_nodes(image, state)
        self._draw_goal(image, state)
        self._draw_robot(image, state)
        path = self._path_provider()
        if image is None or path is None or len(path) < 2:
            return image

        spacing = max(1, int(round(0.25 * obstacle_map.ppm)))
        waypoints = np.asarray(path)[::spacing]
        if not np.array_equal(waypoints[-1], path[-1]):
            waypoints = np.vstack((waypoints, path[-1]))
        points = np.array([world_to_view_px(state["map"], xy, state["view_bounds"])
                           for xy in waypoints], np.int32)
        np.clip(points[:, 0], 0, image.shape[1] - 1, out=points[:, 0])
        np.clip(points[:, 1], 0, image.shape[0] - 1, out=points[:, 1])
        for point in points:
            cv2.circle(image, tuple(point), 3, (255, 100, 0), -1, cv2.LINE_AA)
        return image


class _PathViz(HabitatObsMapWebVizThread):
    """Web map with the FMM path."""

    def __init__(self, path_provider, **kwargs):
        super().__init__(**kwargs)
        self._path_provider = path_provider

    def _render_map_panel(self, target_h):
        get = self._shared_state_getter
        shared_state = get() if get is not None else None
        if shared_state is None:
            return None
        if (self._map_helper is None
                or self._map_helper.shared_state is not shared_state):
            self._map_helper = _PathMapVisualizer(shared_state, self._path_provider)
        return super()._render_map_panel(target_h)


def _start_recorder(agent, target):
    """Start one experiment recording."""
    from recorder import DemoRecorder

    path_map = _PathMapVisualizer(
        agent.shared_state,
        lambda: getattr(agent.navigation, "path_xy", None),
    )
    recorder = DemoRecorder(
        agent.shared_state,
        target,
        agent.cfg.camera.name,
        root=Path(__file__).resolve().parents[2] / "experiments",
        topdown_renderer=path_map.render_decision_frame_clean,
        planner_debug_getter=lambda: agent.navigation,
    )
    recorder.start()
    return recorder


def _start_viz(agent, args):
    """The web viewers the flags asked for."""
    viz_threads = []
    if args.map_viz_web:
        map_viz = _PathViz(
            lambda: getattr(agent.navigation, "path_xy", None),
            habitat_map_getter=lambda: agent.frontier.detector._habitat_map,
            shared_state_getter=lambda: agent.shared_state,
            shutdown_event=agent.shutdown_event,
            port=MAP_VIZ_PORT,
        )
        map_viz.start()
        viz_threads.append(map_viz)
        print(f"[rtnav_runner] map viz — http://localhost:{map_viz.port}")
    if args.det_viz_web:
        det_viz = WebDetectionVisualizerThread(agent.shared_state, agent.shutdown_event)
        det_viz.start()
        viz_threads.append(det_viz)
        print(f"[rtnav_runner] detection viz — http://localhost:{det_viz.port}")
    return viz_threads


def main() -> None:
    args = _parse_args()
    cfg = config.build_config()
    patches.install()
    rclpy.init()

    agent = RealRtNavAgent(cfg, verbose=args.verbose, reasoning=not args.no_reasoning)
    subscriber = RealStretchSubscriber(
        agent.shared_state, input_rotation_k=agent.cfg.camera.input_rotation_k)
    nodes = [subscriber]

    if args.enable_navigation:
        nodes.append(_enable_navigation(agent, args.controller))
    else:
        print("[rtnav_runner] navigation DISABLED (dry) — see --enable-navigation")

    executor = MultiThreadedExecutor(num_threads=4)
    for node in nodes:
        executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True, name="rtnav_obs_spin")
    spin.start()

    def request_shutdown(*_):
        agent.shutdown_event.set()
        if agent._robot_api is not None:
            agent._robot_api.emergency_stop()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    viz_threads = _start_viz(agent, args)
    target = "" if args.no_reasoning else args.target.strip()
    recorder = None if args.no_record else _start_recorder(agent, target)
    if target:
        agent.reset_episode(
            goal_info={"target": target},
            scene_id="real_world",
            episode_id=str(int(time.time())),
            output_dir=str(recorder.dir) if recorder is not None else None,
        )
    print("[rtnav_runner] started")

    deadline = time.monotonic() + args.max_seconds if args.max_seconds > 0 else None
    found = False
    while not agent.shutdown_event.is_set():
        if deadline is not None and time.monotonic() >= deadline:
            print(f"[rtnav_runner] max runtime reached: {args.max_seconds:.1f}s")
            break
        if agent._robot_api is not None and agent._robot_api.episode_done:
            found = bool(agent._robot_api.target_completed)
            if found and recorder is not None:
                recorder.mark_found()
            outcome = "target committed" if found else "navigation stopped"
            print(f"[rtnav_runner] {outcome} — saving result and exiting")
            break
        time.sleep(0.05)

    request_shutdown()
    if recorder is not None:
        recorder.finalize(found)
    print("[rtnav_runner] shutting down")
    executor.shutdown()
    spin.join(timeout=3.0)
    for node in nodes:
        node.destroy_node()
    agent.shutdown(timeout=3.0)
    for thread in viz_threads:
        thread.join(timeout=3.0)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
