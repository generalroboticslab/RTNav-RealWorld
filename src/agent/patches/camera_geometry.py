"""Patch rt_ovn's camera geometry with real Stretch values.

rtnav rebuilds the camera pose as build_T_world_base(gps, compass) @
build_T_base_cam(), so the extrinsic must belong to the same frame as the
gps/compass it multiplies. register_frame() stores each frame's extrinsic
keyed by its gps/compass; build_T_world_base looks it up, build_T_base_cam
returns it. A frame nobody registered raises.
"""
import threading
from collections import OrderedDict

import numpy as np

K = None
_extrinsics = OrderedDict()   # (x, y, yaw) -> T_base_cam, newest last
_pending = threading.local()  # extrinsic matched to the frame being rebuilt
_MAX_FRAMES = 64


def orient_rgbd(rgb, depth, k, T_world_cam, rotation_k):
    """Rotate RGB-D, intrinsics, and optical frame together."""
    if rotation_k == 0:
        return rgb, depth, np.asarray(k, np.float64), np.asarray(T_world_cam, np.float64)
    if rotation_k != -1:
        raise ValueError("input_rotation_k must be 0 or -1 (90 degrees clockwise)")
    if rgb.shape[:2] != depth.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch: {rgb.shape[:2]} != {depth.shape[:2]}")

    image_height = depth.shape[0]
    old_k = np.asarray(k, np.float64)
    new_k = np.array([
        [old_k[1, 1], 0.0, image_height - 1 - old_k[1, 2]],
        [0.0, old_k[0, 0], old_k[0, 2]],
        [0.0, 0.0, 1.0],
    ])

    new_to_old = np.array([[0.0, 1.0, 0.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0]])
    new_T_world_cam = np.asarray(T_world_cam, np.float64).copy()
    new_T_world_cam[:3, :3] = new_T_world_cam[:3, :3] @ new_to_old
    return (np.ascontiguousarray(np.rot90(rgb, k=-1)),
            np.ascontiguousarray(np.rot90(depth, k=-1)),
            new_k, new_T_world_cam)


def base_matrix(gps, compass):
    """Yaw-only world-from-base pose. Extrinsics must be built against this."""
    c, s = np.cos(compass), np.sin(compass)
    T = np.eye(4)
    T[0, 0], T[0, 1] = c, -s
    T[1, 0], T[1, 1] = s, c
    T[0, 3], T[1, 3] = gps[0], gps[1]
    return T


def odom_pose_to_map(odom_pose, map_anchor, odom_anchor):
    """Propagate a synchronized map pose using smooth odometry motion."""
    odom_pose = np.asarray(odom_pose, dtype=float)
    map_anchor = np.asarray(map_anchor, dtype=float)
    odom_anchor = np.asarray(odom_anchor, dtype=float)
    angle = map_anchor[2] - odom_anchor[2]
    c, s = np.cos(angle), np.sin(angle)
    delta = odom_pose[:2] - odom_anchor[:2]
    return np.array([
        map_anchor[0] + c * delta[0] - s * delta[1],
        map_anchor[1] + s * delta[0] + c * delta[1],
        (map_anchor[2] + odom_pose[2] - odom_anchor[2] + np.pi)
        % (2 * np.pi) - np.pi,
    ])


def capture_K(k):
    global K
    K = np.asarray(k, dtype=np.float64)


def register_frame(gps, compass, T_base_cam):
    """Store the extrinsic that exactly reconstructs this frame's camera pose."""
    _extrinsics[_key(gps, compass)] = np.asarray(T_base_cam, dtype=np.float64)
    while len(_extrinsics) > _MAX_FRAMES:
        _extrinsics.popitem(last=False)


def _key(gps, compass):
    return (round(gps[0], 6), round(gps[1], 6), round(compass, 6))


def _build_T_world_base(gps, compass):
    _pending.T_base_cam = _extrinsics[_key(gps, compass)]
    return base_matrix(gps, compass)


def _build_T_base_cam(*args, **kwargs):
    return _pending.T_base_cam


def _build_K(*args, **kwargs):
    return K


def install():
    import rtnav.core.data_types as dt
    import rtnav.modules.perception.camera_geometry as cg
    import rtnav.modules.perception.perception_thread as pt

    # perception_thread imports these by name at module load, so patching cg
    # alone would not reach it — rebind both.
    for mod in (cg, pt):
        mod.build_T_base_cam = _build_T_base_cam
        mod.build_T_world_base = _build_T_world_base
        mod.build_K = _build_K

    orig_init = dt.CameraFrame.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.rotation_k = 0

    dt.CameraFrame.__init__ = __init__
    print("[camera_geometry] patched build_T_base_cam / build_T_world_base / build_K "
          "+ CameraFrame.rotation_k=0")
