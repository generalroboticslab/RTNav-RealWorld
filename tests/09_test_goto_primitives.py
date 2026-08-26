"""Small geometry check for robot-local discrete navigation goals."""
import math
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "agent"))
sys.path.insert(0, str(ROOT / "rt_ovn" / "agents" / "rtnav"))

from std_msgs.msg import Bool

from patches.robot_api import FORWARD, LEFT, RIGHT, GotoRobotAPI, _primitive_goal


def close(actual, expected, eps=1e-9):
    assert all(abs(a - e) < eps for a, e in zip(actual, expected)), (
        actual,
        expected,
    )


close(_primitive_goal((1, 2, math.pi / 2), FORWARD, 0.25, math.pi / 6),
      (1, 2.25, math.pi / 2))
close(_primitive_goal((1, 2, 0), LEFT, 0.25, math.pi / 6),
      (1, 2, math.pi / 6))
close(_primitive_goal((1, 2, -math.pi), RIGHT, 0.25, math.pi / 6),
      (1, 2, 5 * math.pi / 6))

api = GotoRobotAPI.__new__(GotoRobotAPI)
api._goal_lock = threading.Lock()
api._goal_pending = True
api._motion_seen = False
api._at_goal_event = threading.Event()
api._at_goal_cb(Bool(data=True))
assert not api._at_goal_event.is_set(), "stale at_goal completed a new primitive"
api._at_goal_cb(Bool(data=False))
api._at_goal_cb(Bool(data=True))
assert api._at_goal_event.is_set(), "false→true completion was not accepted"

api._control_pose_lock = threading.Lock()
api._control_pose = (1.0, 2.0, 3.0)
api._control_pose_time = time.monotonic()
api._control_pose_version = 2
api._cfg = SimpleNamespace(control_pose_timeout_s=2.0)
api._episode_done_event = threading.Event()
api._aborted = False
assert api._wait_for_control_pose(1, 0.01) == api._control_pose
assert api._wait_for_control_pose(2, 0.01) is None
print("goto primitive goals OK")
