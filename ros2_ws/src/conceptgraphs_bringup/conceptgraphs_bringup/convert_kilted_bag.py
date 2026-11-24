#!/usr/bin/env python3
"""Convert the processed Kilted NVIDIA bag into the Humble pipeline format."""

import argparse
import bisect
from collections import deque
from pathlib import Path

from mcap.reader import make_reader
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tf2_msgs.msg import TFMessage

from .sam3_crate_mapper import stamp_ns, transform_matrix


TOPIC_TYPES = {
    "/camera/camera/color/image_raw": (Image, "sensor_msgs/msg/Image"),
    "/camera/camera/color/camera_info": (CameraInfo, "sensor_msgs/msg/CameraInfo"),
    "/camera/camera/depth/image_rect_raw": (Image, "sensor_msgs/msg/Image"),
    "/camera/camera/depth/camera_info": (CameraInfo, "sensor_msgs/msg/CameraInfo"),
    "/cloud_registered": (PointCloud2, "sensor_msgs/msg/PointCloud2"),
    "/cloud_registered_body": (PointCloud2, "sensor_msgs/msg/PointCloud2"),
    "/Odometry": (Odometry, "nav_msgs/msg/Odometry"),
    "/tf": (TFMessage, "tf2_msgs/msg/TFMessage"),
    "/tf_static": (TFMessage, "tf2_msgs/msg/TFMessage"),
}
ALIGNED_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"


def iter_mcap(path: Path, topics=None):
    with path.open("rb") as stream:
        reader = make_reader(stream)
        for schema, channel, record in reader.iter_messages(topics=topics, log_time_order=True):
            yield schema, channel, record


def find_mcap(source: Path) -> Path:
    paths = sorted(source.glob("*.mcap")) if source.is_dir() else [source]
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one MCAP file in {source}, found {len(paths)}")
    return paths[0]


def static_edges(mcap_path: Path):
    edges = {}
    for _, _, record in iter_mcap(mcap_path, ["/tf_static"]):
        message = deserialize_message(record.data, TFMessage)
        for item in message.transforms:
            t, q = item.transform.translation, item.transform.rotation
            matrix = transform_matrix((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))
            edges.setdefault(item.header.frame_id, {})[item.child_frame_id] = matrix
            edges.setdefault(item.child_frame_id, {})[item.header.frame_id] = np.linalg.inv(matrix)
    return edges


def lookup_transform(edges, source, target):
    queue = deque([(source, np.eye(4))])
    visited = {source}
    while queue:
        frame, matrix = queue.popleft()
        if frame == target:
            return matrix
        for neighbor, edge in edges.get(frame, {}).items():
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, edge @ matrix))
    return None


def collect_alignment_inputs(mcap_path: Path):
    color_info = depth_info = None
    rgb_headers = {}
    topics = ["/camera/camera/color/image_raw", "/camera/camera/color/camera_info",
              "/camera/camera/depth/camera_info"]
    for _, channel, record in iter_mcap(mcap_path, topics):
        cls = TOPIC_TYPES[channel.topic][0]
        message = deserialize_message(record.data, cls)
        if channel.topic.endswith("color/image_raw"):
            rgb_headers[stamp_ns(message)] = message.header
        elif channel.topic.endswith("color/camera_info") and color_info is None:
            color_info = message
        elif channel.topic.endswith("depth/camera_info") and depth_info is None:
            depth_info = message
    if color_info is None or depth_info is None or not rgb_headers:
        raise ValueError("Missing RGB or camera calibration required for depth alignment")
    edges = static_edges(mcap_path)
    color_from_depth = lookup_transform(
        edges, depth_info.header.frame_id, color_info.header.frame_id)
    if color_from_depth is None:
        raise ValueError("No static TF chain from depth optical frame to color optical frame")
    return color_info, depth_info, color_from_depth, rgb_headers


def nearest_header(headers, timestamp_ns, tolerance_ns=30_000_000):
    stamps = sorted(headers)
    index = bisect.bisect_left(stamps, timestamp_ns)
    candidates = stamps[max(0, index - 1):min(len(stamps), index + 1)]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda value: abs(value - timestamp_ns))
    return headers[nearest] if abs(nearest - timestamp_ns) <= tolerance_ns else None


def align_depth(message, rgb_header, color_info, depth_info, color_from_depth):
    row = np.frombuffer(message.data, np.uint8).reshape(message.height, message.step)
    dtype = np.dtype(">u2" if message.is_bigendian else "<u2")
    depth = row[:, :message.width * 2].copy().view(dtype).reshape(message.height, message.width)
    rows, cols = np.nonzero(depth)
    z = depth[rows, cols].astype(np.float64) * 0.001
    kd = np.asarray(depth_info.k).reshape(3, 3)
    points = np.vstack(((cols - kd[0, 2]) * z / kd[0, 0],
                        (rows - kd[1, 2]) * z / kd[1, 1], z, np.ones_like(z)))
    color_points = color_from_depth @ points
    x, y, color_z = color_points[:3]
    valid_z = color_z > 0
    xn, yn = x / color_z, y / color_z
    distortion = list(color_info.d) + [0.0] * (5 - len(color_info.d))
    k1, k2, p1, p2, k3 = distortion[:5]
    radius2 = xn * xn + yn * yn
    radial = 1 + k1 * radius2 + k2 * radius2 ** 2 + k3 * radius2 ** 3
    xd = xn * radial + 2 * p1 * xn * yn + p2 * (radius2 + 2 * xn * xn)
    yd = yn * radial + p1 * (radius2 + 2 * yn * yn) + 2 * p2 * xn * yn
    kc = np.asarray(color_info.k).reshape(3, 3)
    out_cols = np.rint(kc[0, 0] * xd + kc[0, 2]).astype(np.int32)
    out_rows = np.rint(kc[1, 1] * yd + kc[1, 2]).astype(np.int32)
    height, width = color_info.height, color_info.width
    valid = valid_z & (out_cols >= 0) & (out_cols < width) & (out_rows >= 0) & (out_rows < height)
    flat = out_rows[valid] * width + out_cols[valid]
    millimeters = np.rint(color_z[valid] * 1000).clip(1, 65535).astype(np.uint16)
    aligned = np.full(height * width, np.iinfo(np.uint16).max, dtype=np.uint16)
    np.minimum.at(aligned, flat, millimeters)
    aligned[aligned == np.iinfo(np.uint16).max] = 0

    output = Image()
    output.header = rgb_header
    output.header.frame_id = color_info.header.frame_id
    output.height, output.width = height, width
    output.encoding = "16UC1"
    output.is_bigendian = False
    output.step = width * 2
    output.data = aligned.reshape(height, width).tobytes()
    return output


def normalize_message(topic, message):
    if topic == "/Odometry":
        if message.header.frame_id == "odom":
            message.header.frame_id = "camera_init"
        if message.child_frame_id == "base_link":
            message.child_frame_id = "body"
    elif topic == "/cloud_registered":
        if message.header.frame_id == "odom":
            message.header.frame_id = "camera_init"
    elif topic == "/cloud_registered_body":
        if message.header.frame_id == "base_link":
            message.header.frame_id = "body"
    elif topic == "/tf":
        for item in message.transforms:
            if item.header.frame_id == "odom":
                item.header.frame_id = "camera_init"
            if item.child_frame_id == "base_link":
                item.child_frame_id = "body"
    return message


def convert(source: Path, output: Path):
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    mcap_path = find_mcap(source)
    color_info, depth_info, color_from_depth, rgb_headers = collect_alignment_inputs(mcap_path)

    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(output), storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    for topic, (_, type_name) in TOPIC_TYPES.items():
        writer.create_topic(rosbag2_py.TopicMetadata(
            name=topic, type=type_name, serialization_format="cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(
        name=ALIGNED_TOPIC, type="sensor_msgs/msg/Image", serialization_format="cdr"))

    counts = {topic: 0 for topic in TOPIC_TYPES}
    aligned_count = 0
    for _, channel, record in iter_mcap(mcap_path, TOPIC_TYPES):
        topic = channel.topic
        cls = TOPIC_TYPES[topic][0]
        message = normalize_message(topic, deserialize_message(record.data, cls))
        writer.write(topic, serialize_message(message), record.log_time)
        counts[topic] += 1
        if topic == "/camera/camera/depth/image_rect_raw":
            header = nearest_header(rgb_headers, stamp_ns(message))
            if header is not None:
                aligned = align_depth(message, header, color_info, depth_info, color_from_depth)
                writer.write(ALIGNED_TOPIC, serialize_message(aligned), record.log_time)
                aligned_count += 1
    del writer
    return counts, aligned_count


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Kilted bag directory or MCAP")
    parser.add_argument("--output", required=True, type=Path, help="New Humble bag directory")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    counts, aligned_count = convert(args.input.resolve(), args.output.resolve())
    print(f"Created Humble bag: {args.output.resolve()}")
    for topic, count in counts.items():
        print(f"{topic}: {count}")
    print(f"{ALIGNED_TOPIC}: {aligned_count}")


if __name__ == "__main__":
    main()
