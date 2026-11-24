#!/usr/bin/env python3
"""Register rectified depth into the color image using CameraInfo and /tf_static."""

from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage

from .sam3_crate_mapper import stamp_ns, transform_matrix


class DepthToColorAlign(Node):
    def __init__(self):
        super().__init__("depth_to_color_align")
        self.declare_parameter("max_sync_dt", 0.03)
        self.depth_info = None
        self.color_info = None
        self.edges = {}
        self.rgb_stamps = deque(maxlen=10)
        self.pending_depth = deque(maxlen=10)

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        static_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.publisher = self.create_publisher(
            Image, "/camera/camera/aligned_depth_to_color/image_raw", image_qos
        )
        self.create_subscription(
            Image, "/camera/camera/color/image_raw", self.rgb_callback, image_qos)
        self.create_subscription(
            Image, "/camera/camera/depth/image_rect_raw", self.depth_callback, image_qos)
        self.create_subscription(
            CameraInfo, "/camera/camera/depth/camera_info", self.depth_info_callback, image_qos)
        self.create_subscription(
            CameraInfo, "/camera/camera/color/camera_info", self.color_info_callback, image_qos)
        self.create_subscription(TFMessage, "/tf_static", self.tf_callback, static_qos)
        self.get_logger().info("Depth-to-color aligner waiting for calibration and images")

    def depth_info_callback(self, msg):
        self.depth_info = msg
        self.retry()

    def color_info_callback(self, msg):
        self.color_info = msg
        self.retry()

    def tf_callback(self, msg):
        for item in msg.transforms:
            t, q = item.transform.translation, item.transform.rotation
            matrix = transform_matrix([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])
            self.edges.setdefault(item.header.frame_id, {})[item.child_frame_id] = matrix
            self.edges.setdefault(item.child_frame_id, {})[item.header.frame_id] = np.linalg.inv(matrix)
        self.retry()

    def rgb_callback(self, msg):
        self.rgb_stamps.append((stamp_ns(msg), msg.header))
        self.retry()

    def depth_callback(self, msg):
        self.pending_depth.append(msg)
        self.retry()

    def lookup_transform(self, source, target):
        queue = deque([(source, np.eye(4))])
        visited = {source}
        while queue:
            frame, transform = queue.popleft()
            if frame == target:
                return transform
            for neighbor, edge in self.edges.get(frame, {}).items():
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, edge @ transform))
        return None

    def retry(self):
        if self.depth_info is None or self.color_info is None or not self.rgb_stamps:
            return
        depth_frame = self.depth_info.header.frame_id
        color_frame = self.color_info.header.frame_id
        color_from_depth = self.lookup_transform(depth_frame, color_frame)
        if color_from_depth is None:
            return
        tolerance_ns = int(float(self.get_parameter("max_sync_dt").value) * 1e9)
        while self.pending_depth:
            depth_msg = self.pending_depth[0]
            depth_stamp = stamp_ns(depth_msg)
            nearest = min(self.rgb_stamps, key=lambda item: abs(item[0] - depth_stamp))
            if abs(nearest[0] - depth_stamp) > tolerance_ns:
                if nearest[0] < depth_stamp:
                    return
                self.pending_depth.popleft()
                continue
            self.pending_depth.popleft()
            self.publisher.publish(self.align(depth_msg, nearest[1], color_from_depth))

    def align(self, msg, rgb_header, color_from_depth):
        depth = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        depth = depth[:, :msg.width * 2].copy().view(np.uint16).reshape(msg.height, msg.width)
        rows, cols = np.nonzero(depth)
        z = depth[rows, cols].astype(np.float64) * 0.001
        kd = np.asarray(self.depth_info.k).reshape(3, 3)
        points = np.vstack(((cols - kd[0, 2]) * z / kd[0, 0],
                            (rows - kd[1, 2]) * z / kd[1, 1], z,
                            np.ones_like(z)))
        color_points = color_from_depth @ points
        x, y, color_z = color_points[:3]
        valid_z = color_z > 0
        xn, yn = x / color_z, y / color_z

        # Project into the raw color image, including its plumb-bob distortion.
        distortion = list(self.color_info.d) + [0.0] * (5 - len(self.color_info.d))
        k1, k2, p1, p2, k3 = distortion[:5]
        radius2 = xn * xn + yn * yn
        radial = 1 + k1 * radius2 + k2 * radius2 ** 2 + k3 * radius2 ** 3
        xd = xn * radial + 2 * p1 * xn * yn + p2 * (radius2 + 2 * xn * xn)
        yd = yn * radial + p1 * (radius2 + 2 * yn * yn) + 2 * p2 * xn * yn
        kc = np.asarray(self.color_info.k).reshape(3, 3)
        out_cols = np.rint(kc[0, 0] * xd + kc[0, 2]).astype(np.int32)
        out_rows = np.rint(kc[1, 1] * yd + kc[1, 2]).astype(np.int32)
        height, width = self.color_info.height, self.color_info.width
        valid = valid_z & (out_cols >= 0) & (out_cols < width) & (out_rows >= 0) & (out_rows < height)
        flat_indices = out_rows[valid] * width + out_cols[valid]
        millimeters = np.rint(color_z[valid] * 1000).clip(1, 65535).astype(np.uint16)
        aligned = np.full(height * width, np.iinfo(np.uint16).max, dtype=np.uint16)
        np.minimum.at(aligned, flat_indices, millimeters)
        aligned[aligned == np.iinfo(np.uint16).max] = 0
        aligned = aligned.reshape(height, width)

        output = Image()
        output.header = rgb_header
        output.header.frame_id = self.color_info.header.frame_id
        output.height, output.width = height, width
        output.encoding = "16UC1"
        output.is_bigendian = False
        output.step = width * 2
        output.data = aligned.tobytes()
        return output


def main(args=None):
    rclpy.init(args=args)
    node = DepthToColorAlign()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
