"""Selective ros1_bridge parameter_bridge for our /obs/* topics.

parameter_bridge reads its topic list from the ROS1 param server (not --ros-args
-p, not config_file:=), so we emit rosparam-shaped yaml, `rosparam load` it, then
exec the bridge bare. Keys: topics / services_{1_to_2,2_to_1}. Each `topics:`
entry bridges both directions; data flows wherever an upstream publisher exists.
"""
import os
import tempfile

TOPICS = [
    ("/obs/rgb",                 "sensor_msgs/msg/Image"),
    ("/obs/depth",               "sensor_msgs/msg/Image"),
    ("/obs/gps",                 "geometry_msgs/msg/PointStamped"),
    ("/obs/compass",             "std_msgs/msg/Float32"),
    ("/obs/base_pose",           "geometry_msgs/msg/PoseStamped"),
    ("/obs/camera_pose",         "geometry_msgs/msg/PoseStamped"),
    ("/obs/camera_info",         "sensor_msgs/msg/CameraInfo"),
    # Robot-local feedback and controller used for FMM waypoint tracking.
    ("/state_estimator/pose_filtered", "geometry_msgs/msg/PoseStamped"),
    ("/goto_controller/goal",    "geometry_msgs/msg/Pose"),
    ("/goto_controller/at_goal", "std_msgs/msg/Bool"),
    ("/tf",                      "tf2_msgs/msg/TFMessage"),
    ("/tf_static",               "tf2_msgs/msg/TFMessage"),
    ("/map",                     "nav_msgs/msg/OccupancyGrid"),
    ("/odom",                    "nav_msgs/msg/Odometry"),
    # patches/robot_api.py streams Twist here.
    ("/stretch/cmd_vel",         "geometry_msgs/msg/Twist"),
]

# Mode switch used by patches/robot_api.py before driving.
SERVICES_2_TO_1 = [
    ("/switch_to_navigation_mode", "std_srvs/Trigger"),
    ("/goto_controller/enable",    "std_srvs/Trigger"),
    ("/goto_controller/disable",   "std_srvs/Trigger"),
    ("/goto_controller/set_yaw_tracking", "std_srvs/SetBool"),
]


def _write_yaml():
    lines = ["topics:"]
    for topic, msgtype in TOPICS:
        qs = 100 if topic in ("/tf", "/tf_static") else 20 if topic == "/odom" else 2
        lines += [
            f"  - topic: {topic}",
            f"    type: {msgtype}",
            f"    queue_size: {qs}",
        ]
        # Hector /map is LATCHED → TRANSIENT_LOCAL, else late subs miss it.
        if topic == "/map":
            lines += [
                "    qos:",
                "      durability: transient_local",
                "      reliability: reliable",
                "      history: keep_last",
                "      depth: 1",
            ]
    if SERVICES_2_TO_1:
        lines.append("services_2_to_1:")
        for svc, srvtype in SERVICES_2_TO_1:
            lines += [
                f"  - service: {svc}",
                f"    type: {srvtype}",
            ]
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    f.write("\n".join(lines) + "\n")
    f.close()
    return f.name


CMD = """
. /opt/humble.sh
rosparam load {yaml}
exec ros2 run ros1_bridge parameter_bridge
""".format(yaml=_write_yaml())

if __name__ == "__main__":
    os.execvp("bash", ["bash", "-c", CMD])
