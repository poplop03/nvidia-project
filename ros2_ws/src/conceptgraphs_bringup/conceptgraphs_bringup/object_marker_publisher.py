#!/usr/bin/env python3
"""Publish M5 Objects and their member Observation links for RViz QA."""
import json
from pathlib import Path
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray


from .paths import default_dataset


class ObjectMarkerPublisher(Node):
    def __init__(self):
        super().__init__("object_marker_publisher")
        base = str(default_dataset())
        self.declare_parameter("objects_path", base + "/objects/objects.jsonl")
        self.declare_parameter("observations_path", base + "/observations/observations_3d.jsonl")
        self.objects = self._read(Path(self.get_parameter("objects_path").value))
        self.observations = {x["observation_id"]: x for x in self._read(Path(self.get_parameter("observations_path").value))}
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.create_publisher(MarkerArray, "/object_association_markers", qos)
        self.timer = self.create_timer(1.0, self.publish)
        self.publish()
        self.get_logger().info(f"Loaded {len(self.objects)} M5 Objects")

    @staticmethod
    def _read(path):
        if not path.is_file(): raise FileNotFoundError(path)
        return [json.loads(x) for x in path.read_text().splitlines() if x]

    @staticmethod
    def _point(values):
        p = Point(); p.x, p.y, p.z = map(float, values); return p

    def _marker(self, obj, namespace, marker_id, marker_type):
        marker = Marker(); marker.header.frame_id = obj["frame_id"]
        marker.header.stamp = self.get_clock().now().to_msg(); marker.ns = namespace
        marker.id = marker_id; marker.type = marker_type; marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.color.r, marker.color.g, marker.color.b = ((1.0, 0.2, 0.15)
            if obj["canonical_label"].startswith("red_") else (1.0, 0.85, 0.05))
        marker.color.a = 0.95; return marker

    def publish(self):
        array = MarkerArray(); edges = ((0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7))
        for index, obj in enumerate(self.objects):
            center = self._marker(obj, "object_centroids", index, Marker.SPHERE)
            center.pose.position = self._point(obj["centroid_3d"]); center.scale.x = center.scale.y = center.scale.z = 0.25
            array.markers.append(center)
            lo, hi = obj["bbox_3d_min"], obj["bbox_3d_max"]
            corners = [[x,y,z] for z in (lo[2],hi[2]) for y in (lo[1],hi[1]) for x in (lo[0],hi[0])]
            box = self._marker(obj, "object_bboxes", index, Marker.LINE_LIST); box.scale.x = 0.045
            for a,b in edges: box.points.extend((self._point(corners[a]), self._point(corners[b])))
            array.markers.append(box)
            label = self._marker(obj, "object_labels", index, Marker.TEXT_VIEW_FACING)
            label.pose.position = self._point((obj["centroid_3d"][0], obj["centroid_3d"][1], hi[2]+0.3)); label.scale.z = 0.22
            label.color.r = label.color.g = label.color.b = 1.0
            label.text = f"{obj['object_id']} {obj['canonical_label']}\n{obj['observation_count']} obs / {obj['position_quality']}"
            array.markers.append(label)
            links = self._marker(obj, "object_observation_links", index, Marker.LINE_LIST); links.scale.x = 0.025; links.color.a = 0.65
            observations = self._marker(obj, "object_observations", index, Marker.SPHERE_LIST); observations.scale.x = observations.scale.y = observations.scale.z = 0.10
            for oid in obj["observation_ids"]:
                p = self._point(self.observations[oid]["centroid_3d"]); observations.points.append(p)
                links.points.extend((self._point(obj["centroid_3d"]), p))
            array.markers.extend((observations, links))
        self.publisher.publish(array)


def main(args=None):
    rclpy.init(args=args); node = ObjectMarkerPublisher()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


if __name__ == "__main__": main()
