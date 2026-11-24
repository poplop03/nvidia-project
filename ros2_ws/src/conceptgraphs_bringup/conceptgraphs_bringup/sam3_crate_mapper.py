#!/usr/bin/env python3
"""Map text-prompted SAM3 detections into 3D and publish persistent RViz markers."""

from collections import deque
import base64
import colorsys
import hashlib
import json
from pathlib import Path
import queue
import struct
import subprocess
import threading

import numpy as np
import cv2
import rclpy
import yaml
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray

from .paths import default_sam3_model, resolve_worker_python
from .sam3_prompt_config import load_sam3_prompt_config


def stamp_ns(msg):
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


def transform_matrix(translation, quaternion):
    x, y, z, w = quaternion
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ]
    matrix[:3, 3] = translation
    return matrix


def load_prompt_texts(config_path):
    """Load the shared offline/online SAM3 prompt list."""
    config = load_sam3_prompt_config(config_path)
    return [entry["text"] for entry in config["prompts"]]


def marker_color_for_label(label):
    """Return a stable, semantically meaningful RGB color for a SAM3 label."""
    normalized = str(label).strip().lower().replace("_", " ")
    semantic_colors = (
        ("yellow", (0.10, 1.0, 0.10)),
        ("red", (1.0, 0.0, 0.65)),
        ("blue", (0.05, 0.30, 1.0)),
        ("ladder", (0.72, 0.20, 0.95)),
        ("wheelbarrow", (0.0, 1.0, 1.0)),
    )
    for cue, color in semantic_colors:
        if cue in normalized:
            return color

    # New prompts still receive a repeatable, saturated color across runs.
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") / 65535.0
    return colorsys.hsv_to_rgb(hue, 0.75, 0.95)


class Sam3CrateMapper(Node):
    def __init__(self):
        super().__init__("sam3_crate_mapper")
        defaults = {
            "model": str(default_sam3_model()),
            "worker_python": resolve_worker_python(),
            "worker_script": "",
            "prompt_config_path": "",
            "prompts": ["yellow plastic container on the ground", "red plastic container on the ground"],
            "confidence": 0.4,
            "device": "auto",
            "cluster_radius": 1.5,
            "min_observations": 2,
            "min_cluster_confidence": 0.6,
            "max_depth": 15.0,
            "max_distance": 3.0,
            "min_depth_samples": 10,
            "marker_scale": 0.4,
            "marker_z_offset": 0.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        prompt_config_path = self.get_parameter("prompt_config_path").value
        prompts = (load_prompt_texts(prompt_config_path) if prompt_config_path
                   else list(self.get_parameter("prompts").value))

        self.camera_k = None
        self.static_edges = {}
        self.odometry = deque(maxlen=300)
        self.rgb_pending, self.depth_pending = {}, {}
        self.received = {"rgb": 0, "depth": 0, "info": 0, "odom": 0,
                         "tf_static": 0, "queued": 0}
        self.jobs = queue.Queue(maxsize=1)
        self.clusters = []
        self.lock = threading.Lock()

        reliable = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=20,
                              reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.VOLATILE)
        # This recorded bag replays /tf_static as volatile. The mapper starts
        # before bag playback, so a volatile subscription is compatible and safe.
        static_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                                reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.VOLATILE)
        marker_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                                reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.marker_pub = self.create_publisher(
            MarkerArray, "/yellow_crate_markers", marker_qos
        )
        self.debug_pub = self.create_publisher(Image, "/sam3/debug_image", 10)
        self.create_subscription(Image, "/camera/camera/color/image_raw", self.rgb_callback, reliable)
        self.create_subscription(Image, "/camera/camera/aligned_depth_to_color/image_raw", self.depth_callback, reliable)
        self.create_subscription(CameraInfo, "/camera/camera/color/camera_info", self.info_callback, reliable)
        self.create_subscription(Odometry, "/Odometry", self.odom_callback, reliable)
        self.create_subscription(TFMessage, "/tf_static", self.tf_static_callback, static_qos)

        script = self.get_parameter("worker_script").value
        if not script:
            from ament_index_python.packages import get_package_share_directory
            script = str(Path(get_package_share_directory("conceptgraphs_bringup")) / "scripts/sam3_inference_worker.py")
        command = [self.get_parameter("worker_python").value, script,
                   "--model", self.get_parameter("model").value,
                   "--conf", str(self.get_parameter("confidence").value),
                   "--device", self.get_parameter("device").value,
                   "--imgsz", "640"]
        for prompt in prompts:
            command.extend(["--prompt", prompt])
        self.worker = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()
        self.get_logger().info(
            f"SAM3 semantic mapper prompts: {prompts}"
        )

    def info_callback(self, msg):
        self.camera_k = list(msg.k)
        self.received["info"] += 1
        if self.received["info"] == 1:
            self.get_logger().info("Received CameraInfo")
        self.retry_pending()

    def odom_callback(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.odometry.append((stamp_ns(msg), transform_matrix(
            [p.x, p.y, p.z], [q.x, q.y, q.z, q.w])))
        self.received["odom"] += 1
        if self.received["odom"] == 1:
            self.get_logger().info("Received Odometry")
        self.retry_pending()

    def tf_static_callback(self, msg):
        for item in msg.transforms:
            t, q = item.transform.translation, item.transform.rotation
            self.static_edges[(item.header.frame_id, item.child_frame_id)] = transform_matrix(
                [t.x, t.y, t.z], [q.x, q.y, q.z, q.w])
        self.received["tf_static"] += 1
        if self.body_to_camera() is not None:
            self.get_logger().info("Camera static transform chain is ready")
        self.retry_pending()

    def rgb_callback(self, msg):
        self.received["rgb"] += 1
        if self.received["rgb"] == 1:
            self.get_logger().info("Received first RGB image")
        key = stamp_ns(msg)
        self.rgb_pending[key] = msg
        self.try_pair(key)

    def depth_callback(self, msg):
        self.received["depth"] += 1
        if self.received["depth"] == 1:
            self.get_logger().info("Received first aligned depth image")
        key = stamp_ns(msg)
        self.depth_pending[key] = msg
        self.try_pair(key)

    def body_to_camera(self):
        required = [("base_link", "camera_link"),
                    ("camera_link", "camera_color_frame"),
                    ("camera_color_frame", "camera_color_optical_frame")]
        if any(edge not in self.static_edges for edge in required):
            return None
        result = np.eye(4)
        for edge in required:
            result = result @ self.static_edges[edge]
        return result

    def retry_pending(self):
        for key in list(set(self.rgb_pending) & set(self.depth_pending)):
            self.try_pair(key)

    def try_pair(self, key):
        if key not in self.rgb_pending or key not in self.depth_pending or self.camera_k is None or not self.odometry:
            return
        body_to_camera = self.body_to_camera()
        if body_to_camera is None:
            return
        rgb_msg, depth_msg = self.rgb_pending.pop(key), self.depth_pending.pop(key)
        for pending in (self.rgb_pending, self.depth_pending):
            while len(pending) > 10:
                pending.pop(next(iter(pending)))
        _, world_to_body = min(self.odometry, key=lambda item: abs(item[0] - key))
        if rgb_msg.encoding not in ("rgb8", "bgr8") or depth_msg.encoding != "16UC1":
            self.get_logger().error("Expected rgb8/bgr8 RGB and 16UC1 aligned depth")
            return
        rgb = np.frombuffer(rgb_msg.data, np.uint8).reshape(rgb_msg.height, rgb_msg.step)[:, :rgb_msg.width*3]
        rgb = rgb.reshape(rgb_msg.height, rgb_msg.width, 3).copy()
        if rgb_msg.encoding == "rgb8":
            rgb = rgb[:, :, ::-1].copy()
        depth = np.frombuffer(depth_msg.data, np.uint8).reshape(depth_msg.height, depth_msg.step)
        depth = depth[:, :depth_msg.width*2].copy().view(np.uint16).reshape(depth_msg.height, depth_msg.width)
        job = (key, rgb, depth, world_to_body @ body_to_camera)
        try:
            self.jobs.put_nowait(job)
        except queue.Full:
            try:
                self.jobs.get_nowait()
            except queue.Empty:
                pass
            self.jobs.put_nowait(job)
        self.received["queued"] += 1
        if self.received["queued"] == 1 or self.received["queued"] % 30 == 0:
            self.get_logger().info(
                f"Queued synchronized RGB-depth frame for SAM3 "
                f"(total={self.received['queued']})"
            )

    @staticmethod
    def read_exact(stream, size):
        data = bytearray()
        while len(data) < size:
            chunk = stream.read(size - len(data))
            if not chunk:
                raise EOFError("SAM3 worker stopped")
            data.extend(chunk)
        return bytes(data)

    def worker_loop(self):
        while rclpy.ok():
            try:
                key, rgb, depth, world_from_camera = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            header = json.dumps({
                "stamp_ns": key, "height": rgb.shape[0], "width": rgb.shape[1],
                "camera_k": self.camera_k, "world_from_camera": world_from_camera.tolist(),
                "depth_scale_m": 0.001,
                "min_depth": 0.15,
                "max_depth": float(self.get_parameter("max_depth").value),
                "max_distance": float(self.get_parameter("max_distance").value),
                "min_depth_samples": int(self.get_parameter("min_depth_samples").value),
            }).encode()
            try:
                self.worker.stdin.write(struct.pack("!I", len(header)) + header + rgb.tobytes() + depth.tobytes())
                self.worker.stdin.flush()
                size = struct.unpack("!I", self.read_exact(self.worker.stdout, 4))[0]
                result = json.loads(self.read_exact(self.worker.stdout, size))
                self.publish_debug_image(result.get("debug_jpeg", ""), key)
                self.get_logger().info(
                    f"SAM3 frame: raw_masks={result.get('raw_masks', 0)}, "
                    f"valid_3d={len(result['detections'])}"
                )
                counts = {}
                for item in result["detections"]:
                    counts[item.get("label", "unknown")] = counts.get(
                        item.get("label", "unknown"), 0) + 1
                self.get_logger().info(f"3D semantic detections: {counts}")
                self.update_clusters(result["detections"])
            except (BrokenPipeError, EOFError, OSError, json.JSONDecodeError) as error:
                self.get_logger().error(f"SAM3 worker communication failed: {error}")
                return

    def publish_debug_image(self, encoded, key):
        if not encoded:
            return
        image = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        msg = Image()
        msg.header.frame_id = "camera_color_optical_frame"
        msg.header.stamp.sec = key // 1_000_000_000
        msg.header.stamp.nanosec = key % 1_000_000_000
        msg.height, msg.width = image.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = msg.width * 3
        msg.data = image.tobytes()
        self.debug_pub.publish(msg)

    def update_clusters(self, detections):
        radius = float(self.get_parameter("cluster_radius").value)
        for detection in detections:
            label = detection.get("label", "unknown")
            point = np.array([detection["x"], detection["y"], detection["z"]])
            same_label = [cluster for cluster in self.clusters if cluster["label"] == label]
            nearest = min(same_label,
                          key=lambda c: np.linalg.norm(point[:2] - c["position"][:2]),
                          default=None)
            if nearest is None or np.linalg.norm(point[:2] - nearest["position"][:2]) > radius:
                self.clusters.append({"position": point, "points": [point], "observations": 1,
                                      "confidence": detection["confidence"], "label": label})
            else:
                nearest["points"].append(point)
                nearest["observations"] += 1
                nearest["confidence"] = max(nearest["confidence"], detection["confidence"])
                nearest["position"] = np.median(nearest["points"], axis=0)
        self.publish_markers()

    def publish_markers(self):
        array = MarkerArray()
        minimum = int(self.get_parameter("min_observations").value)
        minimum_confidence = float(self.get_parameter("min_cluster_confidence").value)
        scale = float(self.get_parameter("marker_scale").value)
        z_offset = float(self.get_parameter("marker_z_offset").value)
        stamp = self.get_clock().now().to_msg()
        for index, cluster in enumerate(self.clusters):
            if (cluster["observations"] < minimum
                    or cluster["confidence"] < minimum_confidence):
                continue
            marker = Marker()
            marker.header.frame_id, marker.header.stamp = "camera_init", stamp
            marker.ns = "sam3_semantic_objects"
            marker.id, marker.type, marker.action = index, Marker.SPHERE, Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = float(cluster["position"][0])
            marker.pose.position.y = float(cluster["position"][1])
            marker.pose.position.z = float(cluster["position"][2]) + z_offset
            marker.scale.x = marker.scale.y = marker.scale.z = scale
            marker.color.r, marker.color.g, marker.color.b = marker_color_for_label(
                cluster["label"])
            marker.color.a = 0.95
            array.markers.append(marker)
        self.marker_pub.publish(array)
        if detections_count := len(array.markers):
            self.get_logger().info(f"Published {detections_count} confirmed semantic marker(s)")

    def destroy_node(self):
        if self.worker.poll() is None:
            self.worker.terminate()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Sam3CrateMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
