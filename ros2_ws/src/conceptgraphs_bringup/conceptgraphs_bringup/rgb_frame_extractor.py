#!/usr/bin/env python3
"""Save every Nth ROS Image frame to disk."""

from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class RGBFrameExtractor(Node):
    def __init__(self):
        super().__init__("rgb_frame_extractor")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("output_dir", "rgb_frames")
        self.declare_parameter("frame_step", 20)
        self.declare_parameter("jpeg_quality", 95)

        self.output_dir = Path(
            str(self.get_parameter("output_dir").value)
        ).expanduser().resolve()
        self.frame_step = int(self.get_parameter("frame_step").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        if self.frame_step < 1:
            raise ValueError("frame_step must be at least 1")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.received = 0
        self.saved = 0
        image_topic = str(self.get_parameter("image_topic").value)
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.subscription = self.create_subscription(
            Image, image_topic, self.image_callback, image_qos
        )
        self.get_logger().info(
            f"Reading {image_topic}; saving every {self.frame_step} frames to "
            f"{self.output_dir}"
        )

    @staticmethod
    def image_to_bgr(msg: Image) -> np.ndarray:
        channels_by_encoding = {
            "bgr8": 3,
            "rgb8": 3,
            "bgra8": 4,
            "rgba8": 4,
            "mono8": 1,
        }
        channels = channels_by_encoding.get(msg.encoding.lower())
        if channels is None:
            raise ValueError(f"unsupported image encoding: {msg.encoding}")

        row_bytes = msg.width * channels
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        if len(raw) < msg.step * msg.height:
            raise ValueError("image buffer is shorter than height * step")
        rows = raw[: msg.step * msg.height].reshape(msg.height, msg.step)
        image = rows[:, :row_bytes].reshape(msg.height, msg.width, channels)

        encoding = msg.encoding.lower()
        if encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if encoding == "bgra8":
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if encoding == "mono8":
            return image[:, :, 0]
        return image

    def image_callback(self, msg: Image):
        self.received += 1
        if self.received % self.frame_step != 0:
            return

        try:
            image = self.image_to_bgr(msg)
            timestamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            filename = self.output_dir / (
                f"frame_{self.received:06d}_{timestamp_ns}.jpg"
            )
            written = cv2.imwrite(
                str(filename), image,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )
            if not written:
                raise RuntimeError("cv2.imwrite returned false")
            self.saved += 1
            self.get_logger().info(
                f"Saved #{self.saved}: source frame {self.received} -> {filename.name}"
            )
        except Exception as exc:
            self.get_logger().error(
                f"Could not save source frame {self.received}: {exc}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = RGBFrameExtractor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f"Finished: received {node.received} frames, saved {node.saved} images"
        )
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
