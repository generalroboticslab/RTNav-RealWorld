"""Real-world rt_ovn configuration."""
import math
from dataclasses import field
from typing import Tuple

from rtnav.config import configclass
from rtnav.config.environments.hm3d_config import (
    HM3DMappingConfig,
    HM3DSceneGraphConfig,
)
from rtnav.config.modules.camera_cfg import CameraConfig
from rtnav.config.modules.decision_cfg import DecisionConfig
from rtnav.config.modules.detection_cfg import DetectionConfig
from rtnav.config.modules.frontier_cfg import FrontierConfig
from patches.robot_api import RobotMotionConfig

HFOV_DEG = math.degrees(2 * math.atan(480 / 2 / 605.35))


@configclass
class StretchCameraConfig(CameraConfig):
    """Head-mounted D435i."""

    name: str = "stretch_realsense"
    width: int = 480
    height: int = 640
    hfov_deg: float = HFOV_DEG
    input_rotation_k: int = -1
    position_base: Tuple[float, float, float] = (0.0, 0.0, 1.30)
    depth_normalized: bool = False
    sensor_depth_min_m: float = 0.0
    sensor_depth_max_m: float = 10.0
    min_depth: float = 0.3
    max_depth: float = 4.0
    depth_hole_area_thresh_px2: int = 0


@configclass
class StretchFrontierConfig(FrontierConfig):
    """Obstacle band at robot scale rather than HM3D's simulated agent."""

    vlfm_min_obstacle_height: float = 0.4
    vlfm_max_obstacle_height: float = 1.9
    vlfm_floor_drop_height: float = -0.20
    vlfm_use_local_height_range: bool = False
    vlfm_area_thresh_m2: float = 10.0
    vlfm_min_frontier_width_m: float = 0.1
    vlfm_frontier_merge_radius_m: float = 0.75
    vlfm_update_every_n_steps: int = 1
    update_every_n_obs: int = 1


@configclass
class StretchDetectionConfig(DetectionConfig):
    threshold: float = 0.4


@configclass
class StretchDecisionConfig(DecisionConfig):
    frontier_position_threshold: float = 1.50
    target_stop_distance_m: float = 0.50
    target_fow_tolerance_m: float = 0.50


@configclass
class StretchMappingConfig(HM3DMappingConfig):
    """Mapping fed by the already-confirmed perception voxel grid."""

    map_source: str = "hybrid"  # "depth", "slam", or "hybrid"
    safety_margin: float = 0.2
    slam_clip_to_depth_fow: bool = True
    hybrid_depth_min_height_m: float = 0.30
    hybrid_depth_max_height_m: float = 1.80
    hybrid_depth_max_range_m: float = 4.0
    hybrid_fow_range_m: float = 4.0


@configclass
class StretchPlannerConfig:
    """FMM planner parameters in metres."""

    turn_angle_deg: float = 30.0
    success_dist_m: float = 0.50
    frontier_alignment_dist_m: float = 0.30
    frontier_heading_tolerance_deg: float = 15.0
    crop_margin_m: float = 2.0

    bootstrap_turns: int = 11
    bootstrap_max_s: float = 45.0
    no_progress_s: float = 20.0

    step_m: float = 0.25
    goal_radius_m: float = 0.20
    max_goal_projection_m: float = 0.50
    clearance_m: float = 0.45
    clearance_cost_m: float = 0.40
    path_replan_improvement_m: float = 0.15
    path_endpoint_tolerance_m: float = 0.30
    log_actions: bool = True


@configclass
class StretchTrackerConfig:
    """Path-heading follower."""

    rate_hz: float = 30.0
    stream_warmup_frames: int = 10
    lookahead_m: float = 0.30
    open_lookahead_m: float = 0.90
    open_path_tolerance_m: float = 0.08
    open_path_exit_tolerance_m: float = 0.15
    path_heading_distance_m: float = 0.45
    heading_gain: float = 1.0
    heading_damping_s: float = 0.10
    cross_track_gain: float = 0.75
    cross_track_softening_mps: float = 0.15
    curvature_horizon_m: float = 0.90
    curvature_slowdown_m: float = 0.40
    cross_track_slowdown_m: float = 0.30
    cross_track_min_speed_fraction: float = 0.50
    steering_deadband_deg: float = 3.0
    angular_accel_deg: float = 60.0
    turn_rate_deg: float = 25.0
    v_max: float = 0.30
    slow_radius_m: float = 0.45
    goal_tol_m: float = 0.15
    frontier_alignment_dist_m: float = 0.30
    frontier_heading_tolerance_deg: float = 15.0
    spin_degrees: float = 360.0
    spin_timeout_s: float = 30.0
    turn_in_place_deg: float = 50.0
    resume_forward_deg: float = 10.0
    turn_kp: float = 1.0


@configclass
class RealWorldConfig:
    """Live Stretch. Mapping and scenegraph reuse HM3D's tuning, as OVON does."""

    env_name: str = "stretch-real"
    use_sim_time: bool = False

    camera: StretchCameraConfig = field(default_factory=StretchCameraConfig)
    frontier: StretchFrontierConfig = field(default_factory=StretchFrontierConfig)
    nav: RobotMotionConfig = field(default_factory=RobotMotionConfig)
    planner: StretchPlannerConfig = field(default_factory=StretchPlannerConfig)
    tracker: StretchTrackerConfig = field(default_factory=StretchTrackerConfig)
    mapping: StretchMappingConfig = field(default_factory=StretchMappingConfig)
    scenegraph: HM3DSceneGraphConfig = field(default_factory=HM3DSceneGraphConfig)
    decision: StretchDecisionConfig = field(default_factory=StretchDecisionConfig)
    detection: StretchDetectionConfig = field(default_factory=StretchDetectionConfig)


def build_config():
    return RealWorldConfig()
