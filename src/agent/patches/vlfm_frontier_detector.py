"""Extract VLFM frontiers from the authoritative map."""
import cv2
import numpy as np


def merge_nearby(points, radius):
    """Return one existing point index per nearby connected group."""
    points = np.asarray(points, dtype=float)
    if radius <= 0 or len(points) < 2:
        return np.arange(len(points))

    # ponytail: O(n^2) is fine for tens of frontiers; use a spatial index for hundreds.
    linked = np.linalg.norm(points[:, None] - points[None, :], axis=2) <= radius
    remaining = set(range(len(points)))
    keep = []
    while remaining:
        stack = [remaining.pop()]
        group = set(stack)
        while stack:
            neighbors = set(np.flatnonzero(linked[stack.pop()])) & remaining
            remaining -= neighbors
            group |= neighbors
            stack.extend(neighbors)
        indices = np.fromiter(group, dtype=int)
        center = points[indices].mean(axis=0)
        keep.append(indices[np.argmin(np.linalg.norm(points[indices] - center, axis=1))])
    return np.asarray(keep, dtype=int)


def keep_wide_frontiers(boundaries_xy, min_width_m):
    """Keep candidates whose own ordered boundary has sufficient metric arc length."""
    if min_width_m <= 0:
        return np.arange(len(boundaries_xy))

    keep = []
    for i, boundary in enumerate(boundaries_xy):
        points = np.asarray(boundary, dtype=float).reshape(-1, 2)
        if len(points) < 2:
            continue
        length_m = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
        if length_m >= min_width_m:
            keep.append(i)
    return np.asarray(keep, dtype=int)


def resample(hmap, omap, mask):
    """Redraw a mask from rtnav's grid onto the VLFM one."""
    out = np.zeros_like(hmap._map)
    ys, xs = np.nonzero(mask)
    px = hmap._xy_to_px(omap.px_to_xy(np.c_[xs, ys].astype(float)))
    px = px[(px[:, 0] >= 0) & (px[:, 0] < out.shape[1])
            & (px[:, 1] >= 0) & (px[:, 1] < out.shape[0])]
    out[px[:, 1], px[:, 0]] = True
    return out


def mirror_obstacle_map(hmap, omap):
    """Mirror occupancy, navigability, and explored masks onto VLFM's grid."""
    occupancy, explored = omap.occupancy.copy(), omap.explored.copy()

    hmap._map = resample(hmap, omap, occupancy)
    hmap._navigable_map = resample(hmap, omap, omap.navigable > 0)

    explored = resample(hmap, omap, explored)
    explored &= hmap._navigable_map
    if not explored.any():
        hmap.explored_area.fill(False)
        hmap._frontiers_px = np.array([])
        hmap.frontier_boundaries = []
        hmap.frontier_unexplored_directions = []
        hmap.frontiers = np.array([])
        return
    _, labels = cv2.connectedComponents(explored.astype(np.uint8))
    robot = hmap._xy_to_px(np.asarray(omap._last_robot_pose[:2]).reshape(1, 2))[0]
    ys, xs = np.nonzero(explored)
    nearest = np.argmin((xs - robot[0]) ** 2 + (ys - robot[1]) ** 2)
    hmap.explored_area = labels == labels[ys[nearest], xs[nearest]]

    hmap._frontiers_px = hmap._get_frontiers()
    hmap.frontiers = (hmap._px_to_xy(hmap._frontiers_px)
                      if len(hmap._frontiers_px) else np.array([]))


def install() -> None:
    """Rebind the detector's observation push to mirror rtnav's map."""
    from rtnav.modules.frontier.frontier_detection_thread import FrontierDetectionThread
    from rtnav.modules.frontier.vlfm_frontier_detector import VLFMFrontierDetector

    def push_observations(self):
        with self.shared_state.lock:
            obstacle_map = self.shared_state.mapping.obstacle_map
        if obstacle_map is None or not obstacle_map.explored.any():
            return
        mirror_obstacle_map(self._habitat_map, obstacle_map)
        frontiers = np.asarray(self._habitat_map.frontiers)
        frontiers_px = np.asarray(self._habitat_map._frontiers_px)
        boundaries = list(self._habitat_map.frontier_boundaries)
        directions = list(self._habitat_map.frontier_unexplored_directions)
        if not (len(frontiers) == len(frontiers_px) == len(boundaries) == len(directions)):
            raise RuntimeError("frontier candidate arrays are not aligned")
        keep = keep_wide_frontiers(
            boundaries,
            self.frontier_min_width_m,
        )
        frontiers = frontiers[keep]
        frontiers_px = frontiers_px[keep]
        boundaries = [boundaries[i] for i in keep]
        directions = [directions[i] for i in keep]
        keep = merge_nearby(
            frontiers, float(getattr(self, "frontier_merge_radius_m", 0.0))
        )
        self._habitat_map.frontiers = frontiers[keep]
        self._habitat_map._frontiers_px = frontiers_px[keep]
        self._habitat_map.frontier_boundaries = [boundaries[i] for i in keep]
        self._habitat_map.frontier_unexplored_directions = [directions[i] for i in keep]

    VLFMFrontierDetector._push_observations = push_observations

    original_init = FrontierDetectionThread.__init__

    def init(self, shared_state, shutdown_event, cfg):
        original_init(self, shared_state, shutdown_event, cfg)
        self.detector.frontier_merge_radius_m = cfg.frontier.vlfm_frontier_merge_radius_m
        self.detector.frontier_min_width_m = cfg.frontier.vlfm_min_frontier_width_m

    FrontierDetectionThread.__init__ = init
    print("[patches] frontier map mirrors the authoritative obstacle map")
