"""ROS1 observation producer: home-robot raw topics → /obs/* namespace.

    /obs/rgb           sensor_msgs/Image           (rgb8)
    /obs/depth         sensor_msgs/Image           (32FC1, metres)
    /obs/gps           geometry_msgs/PointStamped  (xy in map frame)
    /obs/compass       std_msgs/Float32            (yaw in map frame, rad)
    /obs/base_pose     geometry_msgs/PoseStamped   (map ← base_link)
    /obs/camera_pose   geometry_msgs/PoseStamped   (map ← camera_color_optical_frame)
    /obs/camera_info   sensor_msgs/CameraInfo
"""
import threading
import cv2
import message_filters
import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import PointStamped, PoseStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Float32


COLOR_TOPIC = "/camera/color/image_raw/compressed"
DEPTH_TOPIC = "/camera/aligned_depth_to_color/image_raw/compressedDepth"
COLOR_INFO_TOPIC = "/camera/color/camera_info"
MAP_FRAME = "map"
CAMERA_FRAME = "camera_color_optical_frame"
BASE_FRAME = "base_link"
OBS_NS = "/obs"

SYNC_SLOP_S = 0.05
DEPTH_NEAR_VAL = 0.3  # stretch_ai uses 0.5, increase this as needed
DEPTH_FAR_VAL = 4.0 # stretch_ai uses 2.0, decrease this as needed


def tf_to_matrix(t):
    q = t.transform.rotation
    p = t.transform.translation
    M = np.eye(4)
    M[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    M[:3, 3] = [p.x, p.y, p.z]
    return M


def tf_to_xyt(t):
    q = t.transform.rotation
    p = t.transform.translation
    yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
    return np.array([p.x, p.y, yaw], dtype=np.float32)


def relative_pose_xyt(curr, start):
    dx = curr[0] - start[0]
    dy = curr[1] - start[1]
    c, s = np.cos(-start[2]), np.sin(-start[2])
    gps_x = dx * c - dy * s
    gps_y = dx * s + dy * c
    dtheta = (curr[2] - start[2] + np.pi) % (2 * np.pi) - np.pi
    return (
        np.array([gps_x, gps_y], dtype=np.float32),
        np.array([dtheta], dtype=np.float32),
    )


def fix_depth(depth):
    """Verbatim port of home_robot.utils.image.Camera.fix_depth."""
    depth = depth.copy()
    depth[depth > DEPTH_FAR_VAL] = 0
    depth[depth < DEPTH_NEAR_VAL] = 0
    return depth


def _numpy_to_image_msg(arr, encoding, stamp, frame_id):
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = arr.shape[0]
    msg.width = arr.shape[1]
    msg.encoding = encoding
    msg.is_bigendian = 0
    channels = arr.shape[2] if arr.ndim == 3 else 1
    msg.step = arr.shape[1] * arr.dtype.itemsize * channels
    msg.data = arr.tobytes()
    return msg


class ObsNode:
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf_buffer)

        self.K = None
        # Zero = map origin, so /obs/gps is absolute map coords matching
        # /obs/camera_pose (else rt_ovn mixes frames and offsets the cloud).
        self.episode_start_xyt = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self._rx_count = 0
        self._tf_drop_count = 0
        self._lock = threading.Lock()

        self.pub_rgb = rospy.Publisher(f"{OBS_NS}/rgb", Image, queue_size=2, tcp_nodelay=True)
        self.pub_depth = rospy.Publisher(f"{OBS_NS}/depth", Image, queue_size=2, tcp_nodelay=True)
        self.pub_gps = rospy.Publisher(f"{OBS_NS}/gps", PointStamped, queue_size=2)
        self.pub_compass = rospy.Publisher(f"{OBS_NS}/compass", Float32, queue_size=2)
        self.pub_base_pose = rospy.Publisher(f"{OBS_NS}/base_pose", PoseStamped, queue_size=2)
        self.pub_cam_pose = rospy.Publisher(f"{OBS_NS}/camera_pose", PoseStamped, queue_size=2)
        self.pub_cam_info = rospy.Publisher(f"{OBS_NS}/camera_info", CameraInfo, queue_size=2)

        rospy.Subscriber(COLOR_INFO_TOPIC, CameraInfo, self._info_cb, queue_size=1)
        rgb_sub = message_filters.Subscriber(
            COLOR_TOPIC, CompressedImage,
            queue_size=2, buff_size=2**22, tcp_nodelay=True,
        )
        dpt_sub = message_filters.Subscriber(
            DEPTH_TOPIC, CompressedImage,
            queue_size=2, buff_size=2**22, tcp_nodelay=True,
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, dpt_sub], queue_size=10, slop=SYNC_SLOP_S,
        )
        self._sync.registerCallback(self._on_image_pair)

    def _info_cb(self, msg):
        if self.K is None:
            self.K = np.array(msg.K, dtype=np.float32).reshape(3, 3)

    def _decode_rgb(self, msg):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return None if bgr is None else np.ascontiguousarray(bgr[:, :, ::-1])

    def _decode_depth(self, msg):
        # compressedDepth: 12-byte header {uint32 format, float32 A, float32 B} + PNG.
        #   format=0: raw 16-bit mm (/1000 → m).
        #   format=1: inverse-depth, depth = A/(raw - B) (blind /1000 distorts far).
        if len(msg.data) <= 12:
            return None
        fmt = int.from_bytes(msg.data[:4], byteorder="little")
        quant_a = float(np.frombuffer(msg.data[4:8], dtype=np.float32)[0])
        quant_b = float(np.frombuffer(msg.data[8:12], dtype=np.float32)[0])
        if not getattr(self, "_depth_fmt_logged", False):
            print(f"[obs_node] compressedDepth format={fmt} "
                  f"(0=raw mm, 1=inv-depth), a={quant_a:.3f}, b={quant_b:.3f}")
            self._depth_fmt_logged = True
        arr = np.frombuffer(msg.data[12:], dtype=np.uint8)
        d16 = cv2.imdecode(arr, cv2.IMREAD_ANYDEPTH)
        if d16 is None:
            return None
        if fmt == 1:
            depth = np.zeros_like(d16, dtype=np.float32)
            valid = d16 > 0
            depth[valid] = quant_a / (d16[valid].astype(np.float32) - quant_b)
            return depth
        return d16.astype(np.float32) / 1000.0

    def _lookup_tf(self, target, source, stamp):
        try:
            return self.tf_buffer.lookup_transform(
                target, source, stamp, rospy.Duration(0.1),
            )
        except (tf2_ros.LookupException,
                tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException):
            return None

    def _on_image_pair(self, rgb_msg, dpt_msg):
        if self.K is None:
            return
        stamp = rgb_msg.header.stamp

        cam_tf = self._lookup_tf(MAP_FRAME, CAMERA_FRAME, stamp)
        base_tf = self._lookup_tf(MAP_FRAME, BASE_FRAME, stamp)
        if cam_tf is None or base_tf is None:
            with self._lock:
                self._tf_drop_count += 1
            return

        rgb = self._decode_rgb(rgb_msg)
        depth_raw = self._decode_depth(dpt_msg)
        if rgb is None or depth_raw is None:
            return

        depth = fix_depth(depth_raw)  # range clip only (no temporal filter)

        cam_T = tf_to_matrix(cam_tf)
        base_xyt = tf_to_xyt(base_tf)
        gps, compass = relative_pose_xyt(base_xyt, self.episode_start_xyt)

        with self._lock:
            self._rx_count += 1

        self._publish(stamp, rgb, depth, cam_T, gps, compass)

    def _publish(self, stamp, rgb, depth, cam_T, gps, compass):
        self.pub_rgb.publish(_numpy_to_image_msg(rgb, "rgb8", stamp, CAMERA_FRAME))
        depth_c = np.ascontiguousarray(depth, dtype=np.float32)
        self.pub_depth.publish(_numpy_to_image_msg(depth_c, "32FC1", stamp, CAMERA_FRAME))

        gps_msg = PointStamped()
        gps_msg.header.stamp = stamp
        gps_msg.header.frame_id = MAP_FRAME
        gps_msg.point.x = float(gps[0])
        gps_msg.point.y = float(gps[1])
        gps_msg.point.z = 0.0
        self.pub_gps.publish(gps_msg)

        c_msg = Float32(); c_msg.data = float(compass[0])
        self.pub_compass.publish(c_msg)

        base = PoseStamped()
        base.header.stamp = stamp
        base.header.frame_id = MAP_FRAME
        base.pose.position.x = float(gps[0])
        base.pose.position.y = float(gps[1])
        base.pose.orientation.z = float(np.sin(compass[0] / 2))
        base.pose.orientation.w = float(np.cos(compass[0] / 2))
        self.pub_base_pose.publish(base)

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = MAP_FRAME
        pose.pose.position.x = float(cam_T[0, 3])
        pose.pose.position.y = float(cam_T[1, 3])
        pose.pose.position.z = float(cam_T[2, 3])
        qx, qy, qz, qw = R.from_matrix(cam_T[:3, :3]).as_quat()
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)
        self.pub_cam_pose.publish(pose)

        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = CAMERA_FRAME
        info.height = rgb.shape[0]
        info.width = rgb.shape[1]
        info.K = self.K.flatten().tolist()
        info.distortion_model = "plumb_bob"
        info.D = [0.0] * 5
        self.pub_cam_info.publish(info)


def main():
    rospy.init_node("obs_node")
    node = ObsNode()
    print(f"obs_node publishing on {OBS_NS}/*. Ctrl-C to quit.")
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        with node._lock:
            print(
                f"rx={node._rx_count:6d}  "
                f"tf_drops={node._tf_drop_count:4d}  "
                f"K_ready={node.K is not None}",
                flush=True,
            )
        rate.sleep()


if __name__ == "__main__":
    main()
