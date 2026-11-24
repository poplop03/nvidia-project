#!/usr/bin/env python3
"""Publish valid M4 centroids, 3D boxes and labels for RViz QA."""

import json
from pathlib import Path

import rclpy
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray


from .paths import default_dataset


class Observation3DMarkerPublisher(Node):
    def __init__(self):
        super().__init__("observation_3d_marker_publisher")
        self.declare_parameter(
            "observations_path",
            str(default_dataset() / "observations/observations_3d.jsonl"))
        path = Path(self.get_parameter("observations_path").value)
        if not path.is_file():
            raise FileNotFoundError(f"M4 observations not found: {path}")
        with path.open("r", encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        self.observations = [record for record in records
                             if record["projection_status"] == "valid"]
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.create_publisher(
            MarkerArray, "/observation_3d_markers", qos)
        self.timer = self.create_timer(1.0, self.publish_markers)
        self.publish_markers()
        self.get_logger().info(f"Loaded {len(self.observations)} valid M4 observations")

    @staticmethod
    def _point(values):
        point = Point()
        point.x, point.y, point.z = (float(value) for value in values)
        return point

    @staticmethod
    def _color(label):
        return (1.0, 0.2, 0.15) if label.startswith("red_") else (1.0, 0.85, 0.05)

    def _base(self, observation, namespace, marker_id, marker_type):
        marker = Marker()
        marker.header.frame_id = observation["frame_id"]
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns, marker.id = namespace, marker_id
        marker.type, marker.action = marker_type, Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime = Duration(seconds=0).to_msg()
        marker.color.r, marker.color.g, marker.color.b = self._color(
            observation["canonical_label"])
        marker.color.a = 0.95
        return marker

    def publish_markers(self):
        result = MarkerArray()
        edges = ((0, 1), (1, 3), (3, 2), (2, 0),
                 (4, 5), (5, 7), (7, 6), (6, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7))
        for index, observation in enumerate(self.observations):
            centroid = observation["centroid_3d"]
            sphere = self._base(observation, "observation_centroids", index,
                                Marker.SPHERE)
            sphere.pose.position = self._point(centroid)
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.18
            result.markers.append(sphere)

            minimum, maximum = observation["bbox_3d_min"], observation["bbox_3d_max"]
            corners = [[x, y, z] for z in (minimum[2], maximum[2])
                       for y in (minimum[1], maximum[1])
                       for x in (minimum[0], maximum[0])]
            box = self._base(observation, "observation_bboxes", index,
                             Marker.LINE_LIST)
            box.scale.x = 0.035
            for start, end in edges:
                box.points.extend((self._point(corners[start]), self._point(corners[end])))
            result.markers.append(box)

            label = self._base(observation, "observation_labels", index,
                               Marker.TEXT_VIEW_FACING)
            label.pose.position = self._point((centroid[0], centroid[1], maximum[2] + 0.2))
            label.scale.z = 0.18
            label.color.r = label.color.g = label.color.b = 1.0
            label.text = f"{observation['observation_id']}\n{observation['canonical_label']}"
            result.markers.append(label)
        self.publisher.publish(result)


def main(args=None):
    rclpy.init(args=args)
    node = Observation3DMarkerPublisher()
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
