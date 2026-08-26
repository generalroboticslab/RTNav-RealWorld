"""Shared collision checks for planning and path tracking."""
import numpy as np


def slam_planning_free(explored_free, blocked):
    """Only accumulated depth-explored cells are safe to plan through."""
    blocked = np.asarray(blocked, bool)
    return np.asarray(explored_free, bool) & ~blocked


def planning_free(obstacle_map):
    """Return one stable map view of cells currently safe to traverse."""
    free = getattr(obstacle_map, "_planning_free", None)
    if free is not None:
        return free

    explored_free = (
        np.asarray(obstacle_map.navigable, bool)
        & np.asarray(obstacle_map.explored, bool)
    )
    return explored_free


def path_is_free(
    path_xy,
    pose_xy,
    free,
    xy_to_px,
    start_clearance_px=0,
    max_distance_m=None,
):
    """Check map cells along the immediate remaining path."""
    path = np.asarray(path_xy, dtype=float)
    if path.ndim != 2 or len(path) < 2:
        return False

    closest = int(np.argmin(np.linalg.norm(path - np.asarray(pose_xy), axis=1)))
    remaining = path[closest:]
    if max_distance_m is not None and len(remaining) > 1:
        arc = np.r_[
            0.0,
            np.cumsum(np.linalg.norm(np.diff(remaining, axis=0), axis=1)),
        ]
        beyond = np.flatnonzero(arc > max_distance_m)
        if beyond.size:
            remaining = remaining[:max(2, int(beyond[0]) + 1)]
    points = np.asarray(xy_to_px(remaining), dtype=int)
    robot = np.asarray(xy_to_px(np.asarray([pose_xy], dtype=float))[0], dtype=int)
    cells = []
    for start, end in zip(points[:-1], points[1:]):
        count = int(np.max(np.abs(end - start))) + 1
        cells.append(np.rint(np.linspace(start, end, count)).astype(int))
    pixels = np.concatenate(cells, axis=0) if cells else points

    inside = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < free.shape[1])
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < free.shape[0])
    )
    if not np.all(inside):
        return False
    safe = np.asarray(free, bool)[pixels[:, 1], pixels[:, 0]]
    at_robot = np.max(np.abs(pixels - robot), axis=1) <= start_clearance_px
    return bool(np.all(safe | at_robot))
