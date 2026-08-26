"""Stretch-only runtime adapters for rt_ovn files.

Each module here is named after the upstream file or interface it adapts.

    camera_geometry          measured K + per-frame extrinsic, not habitat's
                             synthesized FOV and fixed camera mount
    perception_thread        point cloud accumulated in a confirmed voxel grid;
                             current points are retained only for visibility
    mapping_thread           depth map or pure Hector occupancy for all consumers
    vlfm_frontier_detector   frontier maps mirrored from rtnav's ObstacleMap,
                             not a second one projected from raw depth
    robot_api                Twist on /stretch/cmd_vel with speed caps and
                             ramp/settle, not sim VelocityCommand

install() applies the local runtime adapters before rtnav binds those names;
robot_api is instead passed to the runner as a constructor argument. 
With them in place the stock threads are corrected for a real Stretch, so nothing downstream needs changing.
"""
from . import (
    camera_geometry,
    decision_thread,
    frontier_strategy,
    mapping_thread,
    perception_thread,
    target_strategy,
    vlfm_frontier_detector,
)

_installed = False


def install():
    """Apply the monkey-patches. Viz patches are installed by their callers."""
    global _installed
    if _installed:
        return
    camera_geometry.install()
    decision_thread.install()
    frontier_strategy.install()
    perception_thread.install()
    mapping_thread.install()
    target_strategy.install()
    vlfm_frontier_detector.install()
    _installed = True
