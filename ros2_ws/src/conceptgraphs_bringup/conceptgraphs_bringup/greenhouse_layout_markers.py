#!/usr/bin/env python3
"""Publish greenhouse and aisle layout annotations for RViz."""

from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

from .greenhouse_layout import Aisle, load_layout


def _display_name(identifier: str) -> str:
    return identifier.replace("_", "-").title()


class GreenhouseLayoutMarkerPublisher(Node):
    def __init__(self):
        super().__init__("greenhouse_layout_markers")
        default_path = (Path(get_package_share_directory("conceptgraphs_bringup")) /
                        "config/greenhouse_layout.yaml")
        self.declare_parameter("layout_path", str(default_path))
        self.layout = load_layout(self.get_parameter("layout_path").value)
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.create_publisher(MarkerArray, "/greenhouse_layout_markers", qos)
        self.timer = self.create_timer(1.0, self.publish_markers)
        self.publish_markers()

    def _base(self, namespace: str, marker_id: int, marker_type: int) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.layout.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns, marker.id = namespace, marker_id
        marker.type, marker.action = marker_type, Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime = Duration(seconds=0).to_msg()
        return marker

    @staticmethod
    def _point(xy, z=0.08):
        point = Point()
        point.x, point.y, point.z = float(xy[0]), float(xy[1]), z
        return point

    def _label(self, marker_id: int, text: str, xy, z: float) -> Marker:
        marker = self._base("layout_labels", marker_id, Marker.TEXT_VIEW_FACING)
        marker.pose.position = self._point(xy, z)
        marker.scale.z = 0.45
        marker.color.r = marker.color.g = marker.color.b = marker.color.a = 1.0
        marker.text = text
        return marker

    @staticmethod
    def _anchor(aisle: Aisle):
        points = aisle.polygon_xy or aisle.centerline
        if points:
            return (sum(p[0] for p in points) / len(points),
                    sum(p[1] for p in points) / len(points))
        return (0.0, 0.0)

    def publish_markers(self):
        result = MarkerArray()
        anchors = [self._anchor(aisle) for aisle in self.layout.aisles]
        result.markers.append(self._label(
            0, _display_name(self.layout.greenhouse_id), anchors[0], 1.2
        ))
        for index, aisle in enumerate(self.layout.aisles):
            anchor = anchors[index]
            result.markers.append(self._label(
                100 + index, _display_name(aisle.id), anchor, 0.7
            ))
            if aisle.polygon_xy:
                polygon = self._base("aisle_polygon", 200 + index, Marker.LINE_STRIP)
                polygon.scale.x = 0.12
                polygon.color.r, polygon.color.g, polygon.color.b, polygon.color.a = 0.1, 0.9, 0.2, 0.95
                polygon.points = [self._point(p) for p in aisle.polygon_xy]
                polygon.points.append(self._point(aisle.polygon_xy[0]))
                result.markers.append(polygon)
            if aisle.centerline:
                line = self._base("aisle_centerline", 300 + index, Marker.LINE_STRIP)
                line.scale.x = 0.08
                line.color.r, line.color.g, line.color.b, line.color.a = 0.1, 0.6, 1.0, 1.0
                line.points = [self._point(p, 0.12) for p in aisle.centerline]
                result.markers.append(line)
            for offset, name, point, color in (
                (0, "Head", aisle.head_point, (0.2, 1.0, 1.0)),
                (1, "Tail", aisle.tail_point, (1.0, 0.3, 0.8)),
            ):
                if point is None:
                    continue
                sphere = self._base("aisle_endpoints", 400 + index * 2 + offset, Marker.SPHERE)
                sphere.pose.position = self._point(point, 0.18)
                sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.35
                sphere.color.r, sphere.color.g, sphere.color.b = color
                sphere.color.a = 1.0
                result.markers.append(sphere)
                result.markers.append(self._label(500 + index * 2 + offset, name, point, 0.55))
        self.publisher.publish(result)


def main(args=None):
    rclpy.init(args=args)
    node = GreenhouseLayoutMarkerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
