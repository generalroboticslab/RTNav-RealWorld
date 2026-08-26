"""Display synchronized RGB and depth."""
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

RGB_TOPIC = "/obs/rgb"
DEPTH_TOPIC = "/obs/depth"
DTYPE = {
    "rgb8": (np.uint8, 3),
    "bgr8": (np.uint8, 3),
    "16UC1": (np.uint16, 1),
    "32FC1": (np.float32, 1),
}


def decode(msg):
    dtype, ch = DTYPE[msg.encoding]
    shape = (msg.height, msg.width) if ch == 1 else (msg.height, msg.width, ch)
    return np.frombuffer(msg.data, dtype=dtype).reshape(shape)


class Viewer(Node):
    def __init__(self):
        super().__init__("verify_camera_simple")
        self.rgb = None
        self.depth = None
        self.n_rgb = 0
        self.n_depth = 0
        self.create_subscription(Image, RGB_TOPIC, self._rgb_cb, 1)
        self.create_subscription(Image, DEPTH_TOPIC, self._depth_cb, 1)

    def _rgb_cb(self, msg):
        img = decode(msg)
        self.rgb = np.ascontiguousarray(img[:, :, ::-1] if msg.encoding == "rgb8" else img)
        self.n_rgb += 1

    def _depth_cb(self, msg):
        d = decode(msg).astype(np.float32)
        self.depth = d / 1000.0 if msg.encoding == "16UC1" else d
        self.n_depth += 1


def panels(rgb, depth):
    valid = np.isfinite(depth) & (depth > 0)
    d_u8 = np.zeros(depth.shape, np.uint8)
    lo, hi = (np.percentile(depth[valid], (2, 98)) if valid.any() else (0.0, 0.0))
    if hi > lo:
        d_u8[valid] = (255 * np.clip((depth[valid] - lo) / (hi - lo), 0, 1)).astype(np.uint8)
    d_color = cv2.cvtColor(d_u8, cv2.COLOR_GRAY2BGR)
    if d_color.shape[:2] != rgb.shape[:2]:
        d_color = cv2.resize(d_color, (rgb.shape[1], rgb.shape[0]))
    cv2.putText(rgb, "RGB", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(d_color, f"DEPTH {lo:.2f}-{hi:.2f}m", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return np.hstack([rgb, d_color])


def main():
    rclpy.init()
    node = Viewer()
    print(f"{RGB_TOPIC} + {DEPTH_TOPIC} — q or ESC to quit")
    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    waiting = True
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.02)
        if node.rgb is None or node.depth is None:
            continue
        if waiting:
            print(f"streaming: rgb={node.n_rgb} depth={node.n_depth}")
            waiting = False
        cv2.imshow("camera", panels(node.rgb.copy(), node.depth.copy()))
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
