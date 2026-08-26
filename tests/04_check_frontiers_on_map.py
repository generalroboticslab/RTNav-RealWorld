"""Run frontier detection and serve its map at http://localhost:8766."""
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "agent"))
sys.path.insert(0, str(ROOT / "rt_ovn" / "agents" / "rtnav"))

import config
import patches

patches.install()  # before rtnav binds geometry builders

from rtnav.core.shared_state import SharedState
from rtnav.modules.frontier.frontier_detection_thread import FrontierDetectionThread
from rtnav.modules.mapping.mapping_thread import MappingThread
from rtnav.modules.perception.perception_thread import PerceptionThread
from rtnav.tools.visualization.habitat_obs_map_web_viz_thread import (
    HabitatObsMapWebVizThread,
)
from subscriber import RealStretchSubscriber

VIZ_PORT = 8766
LOG_HZ = 5.0


def start_threads():
    """The three rtnav threads the runner starts, plus the subscriber feeding them."""
    cfg = config.build_config()
    shared_state = SharedState()
    shutdown = threading.Event()

    frontier = FrontierDetectionThread(shared_state, shutdown, cfg)
    for thread in (PerceptionThread(shared_state, shutdown, cfg),
                   MappingThread(shared_state, shutdown, cfg),
                   frontier):
        thread.start()
    shared_state.task_ready.set()  # every worker blocks on the task gate

    rclpy.init()
    return shared_state, RealStretchSubscriber(shared_state), shutdown, frontier


def start_viz(shared_state, frontier, shutdown):
    """rt_ovn's dual-panel viewer, the one rtnav_runner starts with --map-viz-web."""
    def agent_xy():
        with shared_state.lock:
            odom = shared_state.sensor.latest_odom
        return None if odom is None else (odom[0], odom[1])

    viz = HabitatObsMapWebVizThread(
        habitat_map_getter=lambda: frontier.detector._habitat_map,
        shared_state_getter=lambda: shared_state,
        agent_xy_getter=agent_xy,
        shutdown_event=shutdown,
        port=VIZ_PORT,
        view_size=800,
        fps=4,
    )
    viz.start()
    return viz


def status(shared_state, habitat_map):
    """Cloud size, both explored counts, frontier count, odom."""
    with shared_state.lock:
        perception = shared_state.perception.perception_output
        obstacle_map = shared_state.mapping.obstacle_map
        frontiers = shared_state.frontier.frontier_output
        odom = shared_state.sensor.latest_odom
    if frontiers is None:  # last thread in the chain to produce
        return "waiting for perception -> mapping -> frontier"

    x, y, yaw = odom
    return (f"vox={len(perception.points_world)} "
            f"rtnav_expl={int(np.sum(obstacle_map.explored))} "
            f"vlfm_expl={int(np.sum(habitat_map.explored_area))} "
            f"frontiers={len(frontiers.frontier_clusters)} "
            f"odom=({x:+.2f},{y:+.2f},{np.degrees(yaw):+.0f}deg)")


def main():
    shared_state, node, shutdown, frontier = start_threads()
    viz = start_viz(shared_state, frontier, shutdown)
    print(f"vlfm + rtnav maps -> http://localhost:{viz.port} — ctrl-c to quit")

    next_log = 0.0
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        now = time.monotonic()
        if now < next_log:
            continue
        next_log = now + 1.0 / LOG_HZ
        print(f"\r{status(shared_state, frontier.detector._habitat_map)}",
              end="", flush=True)

    shutdown.set()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
