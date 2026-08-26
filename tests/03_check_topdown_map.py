"""Log confirmed voxels and the resulting top-down map."""
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import rerun as rr
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "agent"))
sys.path.insert(0, str(ROOT / "rt_ovn" / "agents" / "rtnav"))

import config
import patches
import voxel_grid as vg

patches.install()  # before rtnav binds geometry builders

from rtnav.core.shared_state import SharedState
from rtnav.modules.mapping.obstacle_map.obstacle_map import ObstacleMap
from rtnav.modules.perception.perception_thread import PerceptionThread
from subscriber import RealStretchSubscriber
RERUN_HOST = "127.0.0.1"
RERUN_PORT = 9876
LOG_HZ = 5.0
BASE_HALF = 0.17
BODY_H = 1.4
VIEW_CELLS = 400   # fixed window, so rerun's 2D view never goes stale
RERUN_POINT_VOXEL_M = 0.10
MAX_RERUN_POINTS = 30_000

UNKNOWN = (110, 110, 110)
FREE = (245, 245, 245)
OCCUPIED = (220, 60, 60)
FRONTIER = (60, 220, 120)


def downsample_for_rerun(points, colors):
    """One representative per display voxel; never changes mapping input."""
    points = np.asarray(points)
    colors = np.asarray(colors)
    if len(points) == 0:
        return points, colors
    cells = np.floor(points / RERUN_POINT_VOXEL_M).astype(np.int32)
    _, first = np.unique(cells, axis=0, return_index=True)
    if len(first) > MAX_RERUN_POINTS:
        take = np.linspace(0, len(first) - 1, MAX_RERUN_POINTS).astype(np.int64)
        first = first[take]
    return points[first], colors[first]


def layers(m):
    """free / occupied / frontier masks. Frontier = free cells touching unknown."""
    expl = np.asarray(m.explored).astype(bool)
    occ = np.asarray(m.occupancy).astype(bool)
    free = expl & ~occ
    frontier = free & ndimage.binary_dilation(~expl)
    return free & ~frontier, occ, frontier


def window(m):
    """Fixed-size window centred on the explored region."""
    ys, xs = np.nonzero(m.explored)
    h, w = m.explored.shape
    half = VIEW_CELLS // 2
    cy = int(np.clip((ys.min() + ys.max()) // 2, half, h - half))
    cx = int(np.clip((xs.min() + xs.max()) // 2, half, w - half))
    return slice(cy - half, cy + half), slice(cx - half, cx + half)


def map_image(masks, sl):
    free, occ, frontier = (a[sl] for a in masks)
    img = np.full(free.shape + (3,), UNKNOWN, np.uint8)
    img[free] = FREE
    img[occ] = OCCUPIED
    img[frontier] = FRONTIER
    return img


def log_static():
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/robot/body", rr.Boxes3D(
        centers=[[0.0, 0.0, BODY_H / 2]],
        half_sizes=[[BASE_HALF, BASE_HALF, BODY_H / 2]],
        colors=[[80, 140, 255]],
    ), static=True)


def log_robot(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    rr.log("world/robot", rr.Transform3D(
        translation=[x, y, 0.0],
        mat3x3=[[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
    ))


def log_map_3d(omap, masks):
    """Map cells as points at their world position, one entity per layer."""
    for name, mask, color, z in (("free", masks[0], FREE, 0.0),
                                 ("occupied", masks[1], OCCUPIED, 0.02),
                                 ("frontier", masks[2], FRONTIER, 0.04)):
        ys, xs = np.nonzero(mask)
        world = omap.px_to_xy(np.c_[xs, ys].astype(float))
        rr.log(f"world/map/{name}", rr.Points3D(
            np.c_[world, np.full(len(world), z)],
            colors=[list(color)], radii=0.5 / omap.ppm))


def start_perception():
    """PerceptionThread + the subscriber feeding it."""
    shared_state = SharedState()
    shutdown = threading.Event()
    perception = PerceptionThread(shared_state, shutdown, config.build_config())
    perception.start()
    shared_state.task_ready.set()  # every worker blocks on the task gate
    rclpy.init()
    return shared_state, RealStretchSubscriber(shared_state), shutdown, perception


def latest(shared_state):
    with shared_state.lock:
        return (shared_state.perception.perception_output,
                shared_state.perception.perception_version,
                shared_state.sensor.latest_odom)


def main():
    cfg = config.build_config()
    shared_state, node, shutdown, perception = start_perception()

    # The map MappingThread would build. patches/perception_thread has already
    # confirmed and accumulated the point cloud in its voxel grid.
    omap = ObstacleMap(size=cfg.mapping.obstacle_map_size,
                       pixels_per_meter=cfg.mapping.pixels_per_meter,
                       config=cfg.mapping)

    rr.init("topdown_map", spawn=False)
    rr.connect_tcp(f"{RERUN_HOST}:{RERUN_PORT}")
    log_static()
    print(f"cloud + map -> rerun at {RERUN_HOST}:{RERUN_PORT} — ctrl-c to quit")

    version, next_log = -1, 0.0
    while rclpy.ok():
        try:
            rclpy.spin_once(node, timeout_sec=0.05)
        except KeyboardInterrupt:
            break
        now = time.monotonic()
        if now < next_log:
            continue
        out, v, odom = latest(shared_state)
        if out is None or odom is None or v == version:
            continue
        version, next_log = v, now + 1.0 / LOG_HZ

        mout = omap.update_from_perception(out)  # what MappingThread does
        display_points, display_colors = downsample_for_rerun(
            perception.voxels, perception.voxel_colors
        )
        rr.log("world/points", rr.Points3D(
            display_points, colors=display_colors,
            radii=RERUN_POINT_VOXEL_M / 3,
        ))
        x, y, yaw = odom
        log_robot(x, y, yaw)

        counts = "waiting"
        if np.any(mout.explored):
            masks = layers(mout)
            log_map_3d(omap, masks)
            rr.log("map/occupancy", rr.Image(map_image(masks, window(mout))))
            counts = (f"{int(masks[1].sum())}occ/{int(masks[0].sum())}free/"
                      f"{int(masks[2].sum())}frontier")
        print(f"\rpoints={len(perception.voxels)} display={len(display_points)} "
              f"map={counts} "
              f"odom=({x:+.2f},{y:+.2f},{np.degrees(yaw):+.0f}deg)",
              end="", flush=True)

    shutdown.set()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
