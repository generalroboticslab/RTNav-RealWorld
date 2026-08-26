"""ObstacleMap-compatible view of Hector's ROS OccupancyGrid."""
import time

import cv2
import numpy as np

from navigation_safety import slam_planning_free
from rtnav.core.data_types import MappingOutput
from rtnav.modules.mapping.obstacle_map.obstacle_map import ObstacleMap


def reveal_occupied_endpoints(observed, occupied, radius_px=1):
    """Include wall cells bordering revealed free space."""
    radius_px = max(1, int(radius_px))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1)
    )
    adjacent = cv2.dilate(np.asarray(observed, np.uint8), kernel).astype(bool)
    return np.asarray(occupied, bool) & adjacent


def close_small_fow_gaps(explored):
    """Fill one-cell seams between accumulated observations."""
    return cv2.morphologyEx(
        np.asarray(explored, np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    ).astype(bool)


class SlamOccupancyMap(ObstacleMap):
    """Pure SLAM occupancy/free/unknown geometry for every map consumer."""

    is_slam_occupancy = True

    def __init__(self, shared_state, *args, use_depth_obstacles=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_state = shared_state
        self._use_depth_obstacles = bool(use_depth_obstacles)
        self._slam_occupancy = np.zeros((self.size, self.size), np.uint8)
        self._slam_occupancy_full = np.zeros((self.size, self.size), bool)
        self._depth_occupancy = np.zeros((self.size, self.size), bool)
        self._depth_admitted = np.zeros((self.size, self.size), bool)
        self._combined_occupancy = np.zeros((self.size, self.size), bool)
        self._combined_blocked = np.zeros((self.size, self.size), bool)
        self._slam_known = np.zeros((self.size, self.size), bool)
        self._slam_explored_seen = np.zeros((self.size, self.size), bool)
        self._slam_margin_px = 0
        self._last_slam_msg = None
        self.navigable.fill(0)
        empty = np.zeros((self.size, self.size), bool)
        self._planning_free = empty

    @property
    def occupancy(self):
        return self._slam_occupancy

    def clear_explored(self):
        super().clear_explored()
        self._slam_explored_seen.fill(False)
        self._depth_occupancy.fill(False)
        self._depth_admitted.fill(False)

    def _depth_ray_ranges_px(self, points_xy, x, y, yaw, fov_deg, r, num_rays=180):
        range_m = min(float(r), float(self._cfg.hybrid_fow_range_m))
        return super()._depth_ray_ranges_px(
            points_xy, x, y, yaw, fov_deg, range_m, num_rays
        )

    def update_from_perception(self, perception_out):
        with self.shared_state.lock:
            msg = getattr(self.shared_state.sensor, "slam_map", None)
        T = perception_out.robot_pose
        yaw = np.arctan2(T[1, 0], T[0, 0])
        self._last_robot_pose = (float(T[0, 3]), float(T[1, 3]), float(yaw))
        if msg is not None:
            if msg is not self._last_slam_msg:
                self._load_grid(msg)
                self._last_slam_msg = msg
        self._update_combined_obstacles(perception_out)
        visibility = getattr(perception_out, "visibility_points_world", None)
        if visibility is None:
            visibility = perception_out.points_world
        cameras, fovs, ranges = [], [], []
        for name, camera_T in perception_out.camera_extrinsics.items():
            meta = perception_out.camera_meta.get(name)
            if meta is None:
                continue
            cameras.append((
                float(camera_T[0, 3]),
                float(camera_T[1, 3]),
                float(np.arctan2(camera_T[1, 2], camera_T[0, 2])),
            ))
            fovs.append(float(meta.get("fov_deg", 90.0)))
            ranges.append(float(meta.get("range_m", 3.0) or 3.0))
        visibility_xy = (
            np.asarray(visibility)[:, :2] if visibility is not None and len(visibility)
            else np.empty((0, 2))
        )
        self._update_explored(cameras, fovs, ranges, visibility_xy)
        self._refresh_explored()
        return MappingOutput(
            occupancy=self.occupancy.copy(),
            traversability=self.traversability.copy(),
            explored=self.explored.copy(),
            navigable=self.navigable.copy(),
            robot_pose_xyyaw=self._last_robot_pose,
            timestamp=time.time(),
        )

    def _load_grid(self, msg):
        values = np.asarray(msg.data, np.int16).reshape(
            int(msg.info.height), int(msg.info.width)
        )
        origin = msg.info.origin
        q = origin.orientation
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        scale = float(msg.info.resolution) * self.ppm
        c, s = np.cos(yaw), np.sin(yaw)
        # ROS rows point opposite RTNav y; cell centers need the half-cell offset.
        transform = np.array([
            [scale * c, -scale * s,
             self.origin_px[0] + self.ppm * origin.position.x
             + 0.5 * scale * (c - s)],
            [-scale * s, -scale * c,
             self.origin_px[1] - self.ppm * origin.position.y
             - 0.5 * scale * (s + c)],
        ], dtype=np.float64)
        warp = lambda image: cv2.warpAffine(
            image.astype(np.uint8), transform, (self.size, self.size),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        known = warp(values >= 0)
        occupied = warp(values > 0)
        self._slam_occupancy_full = occupied
        self._slam_known = known

    def _update_combined_obstacles(self, perception_out):
        if self._use_depth_obstacles:
            points = np.asarray(perception_out.points_world)
            current = np.zeros_like(self._depth_occupancy)
            if len(points):
                T = perception_out.robot_pose
                base = T[:3, 3]
                dz = points[:, 2] - base[2]
                obstacle = (
                    (dz >= self._cfg.hybrid_depth_min_height_m)
                    & (dz <= self._cfg.hybrid_depth_max_height_m)
                )
                px = self.xy_to_px(points[obstacle, :2])
                current[px[:, 1], px[:, 0]] = True
                distance = np.linalg.norm(points[:, :2] - base[:2], axis=1)
                near_px = self.xy_to_px(points[
                    obstacle & (distance <= self._cfg.hybrid_depth_max_range_m), :2
                ])
                self._depth_admitted[near_px[:, 1], near_px[:, 0]] = True
            self._depth_occupancy = current & self._depth_admitted

        self._combined_occupancy = self._slam_occupancy_full | self._depth_occupancy
        margin_px = max(0, int(round(float(self._cfg.safety_margin) * self.ppm)))
        self._slam_margin_px = margin_px
        if margin_px:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1)
            )
            self._combined_blocked = cv2.dilate(
                self._combined_occupancy.astype(np.uint8), kernel
            ).astype(bool)
        else:
            self._combined_blocked = self._combined_occupancy
        self._obstacle_mask = self._combined_blocked

    def _refresh_explored(self):
        observed = ((self._fow_raw > 0) if self._cfg.slam_clip_to_depth_fow
                    else np.ones_like(self._slam_known))
        visible_occupied = reveal_occupied_endpoints(
            observed,
            self._combined_occupancy if self._use_depth_obstacles
            else self._slam_occupancy_full,
            self._slam_margin_px + 1,
        )
        visible_combined_blocked = reveal_occupied_endpoints(
            observed, self._combined_blocked, self._slam_margin_px + 1
        )
        self._slam_explored_seen |= self._slam_known & observed
        self._slam_explored_seen = (
            close_small_fow_gaps(self._slam_explored_seen) & self._slam_known
        )
        self._slam_occupancy = (
            visible_occupied
        ).astype(np.uint8)
        # Reveal SLAM geometry only where depth has actually observed. The
        # planning snapshot below still applies the full obstacle mask to any
        # explored cells, while unseen space remains non-traversable.
        self.navigable = (~visible_combined_blocked).astype(np.uint8)
        self.explored = (
            self._slam_explored_seen
            & ~visible_combined_blocked
        ).astype(np.uint8)
        self._traversability_u8 = np.where(
            visible_combined_blocked, 0, 255
        ).astype(np.uint8)
        self._invalidate_derived_caches()
        explored_free = self.navigable.astype(bool) & self.explored.astype(bool)
        planning_free = slam_planning_free(
            explored_free, self._combined_blocked
        )
        # Reference replacement is atomic and the array is not mutated by the
        # next update, so planner and tracker each see one consistent map tick.
        self._planning_free = planning_free
