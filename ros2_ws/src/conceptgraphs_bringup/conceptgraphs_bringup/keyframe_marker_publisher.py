#!/usr/bin/env python3
"""Publish extracted View positions and optical viewing directions in RViz."""

import json
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

from .keyframe_extractor import quaternion_to_matrix
from .paths import default_dataset


class KeyframeMarkerPublisher(Node):
    def __init__(self):
        super().__init__("keyframe_marker_publisher")
        self.declare_parameter(
            "views_path",
            str(default_dataset() / "keyframes/views.jsonl"),
        )
        self.declare_parameter("arrow_length", 0.7)
        path = Path(self.get_parameter("views_path").value)
        if not path.is_file():
            raise FileNotFoundError(f"View dataset not found: {path}")
        with path.open("r", encoding="utf-8") as stream:
            self.views = [json.loads(line) for line in stream if line.strip()]
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.create_publisher(MarkerArray, "/keyframe_markers", qos)
        self.timer = self.create_timer(1.0, self.publish_markers)
        self.publish_markers()
        self.get_logger().info(f"Loaded {len(self.views)} Views from {path}")

    @staticmethod
    def _point(values):
        point = Point()
        point.x, point.y, point.z = (float(value) for value in values)
        return point

    def publish_markers(self):
        array = MarkerArray()
        arrow_length = float(self.get_parameter("arrow_length").value)
        for index, view in enumerate(self.views):
            pose = view["pose"]
            origin = np.asarray(pose["translation_xyz"], dtype=np.float64)
            # Optical cameras look along +Z in camera_color_optical_frame.
            direction = quaternion_to_matrix(pose["quaternion_xyzw"]) @ np.array([0.0, 0.0, 1.0])
            arrow = Marker()
            arrow.header.frame_id = view["frame_id"]
            arrow.header.stamp = self.get_clock().now().to_msg()
            arrow.ns, arrow.id = "keyframe_view_directions", index
            arrow.type, arrow.action = Marker.ARROW, Marker.ADD
            arrow.pose.orientation.w = 1.0
            arrow.points = [self._point(origin), self._point(origin + arrow_length * direction)]
            arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.07, 0.13, 0.18
            color = {"head": (0.1, 0.9, 1.0), "middle": (1.0, 0.8, 0.1),
                     "tail": (1.0, 0.2, 0.7)}.get(view.get("aisle_section"), (0.8, 0.8, 0.8))
            arrow.color.r, arrow.color.g, arrow.color.b = color
            arrow.color.a = 0.95
            arrow.lifetime = Duration(seconds=0).to_msg()
            array.markers.append(arrow)
        self.publisher.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = KeyframeMarkerPublisher()
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
