#!/usr/bin/env python3
"""Publish confirmed crate positions from CSV as persistent RViz markers."""

import csv
from pathlib import Path

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray


from .paths import project_root


class CrateMarkerPublisher(Node):
    def __init__(self):
        super().__init__("crate_marker_publisher")
        self.declare_parameter("csv_path", str(project_root() / "data/crate_clusters.csv"))
        self.declare_parameter("frame_id", "camera_init")
        self.declare_parameter("marker_scale", 0.45)
        self.declare_parameter("z_offset", 0.0)
        csv_path = Path(self.get_parameter("csv_path").value)
        if not csv_path.is_file():
            raise FileNotFoundError(f"Crate cluster CSV not found: {csv_path}")
        with csv_path.open(newline="", encoding="utf-8") as stream:
            self.crates = list(csv.DictReader(stream))

        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.create_publisher(MarkerArray, "/yellow_crate_markers", qos)
        self.timer = self.create_timer(1.0, self.publish_markers)
        self.publish_markers()
        self.get_logger().info(f"Loaded {len(self.crates)} confirmed crates from {csv_path}")

    def publish_markers(self):
        frame = self.get_parameter("frame_id").value
        scale = float(self.get_parameter("marker_scale").value)
        z_offset = float(self.get_parameter("z_offset").value)
        markers = MarkerArray()
        for row in self.crates:
            crate_id = int(row["crate_id"])
            x, y = float(row["world_x_m"]), float(row["world_y_m"])
            z = float(row["world_z_m"]) + z_offset

            sphere = Marker()
            sphere.header.frame_id = frame
            sphere.ns, sphere.id = "confirmed_crates", crate_id
            sphere.type, sphere.action = Marker.SPHERE, Marker.ADD
            sphere.pose.position.x, sphere.pose.position.y, sphere.pose.position.z = x, y, z
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = scale
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = 0.0, 1.0, 0.0, 0.95
            sphere.lifetime = Duration(seconds=0).to_msg()
            markers.markers.append(sphere)

            label = Marker()
            label.header.frame_id = frame
            label.ns, label.id = "confirmed_crate_labels", crate_id
            label.type, label.action = Marker.TEXT_VIEW_FACING, Marker.ADD
            label.pose.position.x, label.pose.position.y = x, y
            label.pose.position.z = z + scale
            label.pose.orientation.w = 1.0
            label.scale.z = 0.28
            label.color.r, label.color.g, label.color.b, label.color.a = 1.0, 1.0, 1.0, 1.0
            label.text = f"Crate {crate_id}"
            label.lifetime = Duration(seconds=0).to_msg()
            markers.markers.append(label)
        self.publisher.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = CrateMarkerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
