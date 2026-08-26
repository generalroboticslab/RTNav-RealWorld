"""Feed mapping confirmed voxels and current visibility points."""
from dataclasses import replace

import numpy as np

import voxel_grid as vg

def install() -> None:
    """Publish confirmed voxels, plus current points for visibility rays."""
    from rtnav.modules.perception.perception_thread import PerceptionThread

    init = PerceptionThread.__init__
    build_perception = PerceptionThread._build_perception

    def patched_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.voxel_grid = vg.VoxelGrid()
        self.voxels = np.empty((0, 3))
        self.voxel_colors = np.empty((0, 3), np.uint8)

    def patched_build_perception(self, obs):
        perception, T_world_base, camera = build_perception(self, obs)
        live_points = np.asarray(perception.points_world)
        visibility_points = live_points[vg.reject_flying_pixels(camera)]
        obstacle_points = visibility_points[
            vg.keep_in_height_band(visibility_points)
        ]
        self.voxels, self.voxel_colors = self.voxel_grid.add(
            obstacle_points,
            vg.colors_by_projection(obstacle_points, camera),
            camera,
        )
        perception = replace(
            perception,
            points_world=self.voxels,
        )
        perception.point_colors = self.voxel_colors
        perception.visibility_points_world = visibility_points
        return perception, T_world_base, camera

    PerceptionThread.__init__ = patched_init
    PerceptionThread._build_perception = patched_build_perception
    print("[patches] perception cloud accumulated in a confirmed voxel grid")
