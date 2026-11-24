#!/usr/bin/env python3
"""Deterministically extract synchronized RGB-D keyframes from a ROS 2 bag."""

import argparse
import bisect
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rosbag2_py
import yaml
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage

from .greenhouse_layout import Aisle, GreenhouseLayout, load_layout


NSEC = 1_000_000_000


@dataclass(frozen=True)
class PoseSample:
    timestamp_ns: int
    translation: Tuple[float, float, float]
    quaternion: Tuple[float, float, float, float]
    frame_id: str
    child_frame_id: str


@dataclass(frozen=True)
class Candidate:
    rgb_timestamp_ns: int
    depth_timestamp_ns: int
    camera_info_timestamp_ns: int
    pose: PoseSample
    distance_from_previous_m: float
    yaw_delta_from_previous_deg: float


def stamp_ns(message) -> int:
    return message.header.stamp.sec * NSEC + message.header.stamp.nanosec


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_yaw(quaternion: Sequence[float]) -> float:
    x, y, z, w = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    x, y, z, w = quaternion
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm == 0.0:
        raise ValueError("Quaternion must not be zero")
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def matrix_to_quaternion(rotation: np.ndarray) -> Tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x, y, z = 0.25 * s, (rotation[0, 1] + rotation[1, 0]) / s, (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x, y, z = (rotation[0, 1] + rotation[1, 0]) / s, 0.25 * s, (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x, y, z = (rotation[0, 2] + rotation[2, 0]) / s, (rotation[1, 2] + rotation[2, 1]) / s, 0.25 * s
    quaternion = np.array([x, y, z, w], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


def transform_matrix(translation: Sequence[float], quaternion: Sequence[float]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = quaternion_to_matrix(quaternion)
    result[:3, 3] = translation
    return result


def nearest_index(sorted_values: Sequence[int], value: int) -> Optional[int]:
    if not sorted_values:
        return None
    index = bisect.bisect_left(sorted_values, value)
    choices = []
    if index < len(sorted_values):
        choices.append(index)
    if index > 0:
        choices.append(index - 1)
    return min(choices, key=lambda item: (abs(sorted_values[item] - value), item))


def should_select_keyframe(current: PoseSample, previous: Optional[PoseSample],
                           distance_threshold_m: float, yaw_threshold_deg: float,
                           minimum_time_interval_sec: float) -> Tuple[bool, float, float]:
    if previous is None:
        return True, 0.0, 0.0
    elapsed = (current.timestamp_ns - previous.timestamp_ns) / NSEC
    distance = math.dist(current.translation, previous.translation)
    yaw_delta = abs(math.degrees(normalize_angle(
        quaternion_yaw(current.quaternion) - quaternion_yaw(previous.quaternion)
    )))
    epsilon = 1e-9
    selected = (elapsed + epsilon >= minimum_time_interval_sec and
                (distance + epsilon >= distance_threshold_m or
                 yaw_delta + epsilon >= yaw_threshold_deg))
    return selected, distance, yaw_delta


def project_onto_centerline(point_xy: Sequence[float], aisle: Aisle) -> Optional[float]:
    """Return normalized arclength s for the closest point on an aisle centerline."""
    if len(aisle.centerline) < 2:
        return None
    point = np.asarray(point_xy[:2], dtype=np.float64)
    segments = []
    total = 0.0
    for start, end in zip(aisle.centerline, aisle.centerline[1:]):
        a, b = np.asarray(start), np.asarray(end)
        length = float(np.linalg.norm(b - a))
        if length > 0.0:
            segments.append((a, b, length, total))
            total += length
    if total == 0.0:
        return None
    best_distance, best_s = math.inf, 0.0
    for start, end, length, offset in segments:
        vector = end - start
        fraction = float(np.clip(np.dot(point - start, vector) / (length * length), 0.0, 1.0))
        projected = start + fraction * vector
        distance = float(np.linalg.norm(point - projected))
        if distance < best_distance:
            best_distance = distance
            best_s = (offset + fraction * length) / total
    return best_s


def aisle_section(coordinate_s: Optional[float]) -> Optional[str]:
    if coordinate_s is None:
        return None
    if coordinate_s < 1.0 / 3.0:
        return "head"
    if coordinate_s < 2.0 / 3.0:
        return "middle"
    return "tail"


def image_to_array(message: Image) -> np.ndarray:
    if message.encoding in ("rgb8", "bgr8"):
        row = np.frombuffer(message.data, np.uint8).reshape(message.height, message.step)
        image = row[:, :message.width * 3].reshape(message.height, message.width, 3).copy()
        return image[:, :, ::-1].copy() if message.encoding == "rgb8" else image
    if message.encoding == "16UC1":
        row = np.frombuffer(message.data, np.uint8).reshape(message.height, message.step)
        raw = row[:, :message.width * 2].copy()
        dtype = np.dtype(">u2" if message.is_bigendian else "<u2")
        return raw.view(dtype).reshape(message.height, message.width).astype(np.uint16, copy=False)
    raise ValueError(f"Unsupported image encoding: {message.encoding}")


def camera_info_dict(message: CameraInfo) -> dict:
    return {
        "timestamp_ns": stamp_ns(message), "frame_id": message.header.frame_id,
        "width": message.width, "height": message.height,
        "distortion_model": message.distortion_model,
        "d": list(message.d), "k": list(message.k), "r": list(message.r), "p": list(message.p),
        "binning_x": message.binning_x, "binning_y": message.binning_y,
        "roi": {"x_offset": message.roi.x_offset, "y_offset": message.roi.y_offset,
                "height": message.roi.height, "width": message.roi.width,
                "do_rectify": message.roi.do_rectify},
    }


def open_reader(bag_path: str) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    return reader


def _merge_config(base: dict, overrides: dict) -> dict:
    """Recursively merge dataset overrides without duplicating pipeline settings."""
    result = dict(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path, _seen=None) -> dict:
    path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError(f"Cyclic extractor config inheritance: {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Extractor config must be a mapping")
    parent = config.pop("extends", None)
    if parent is None:
        return config
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _merge_config(load_config(parent_path, seen), config)


def collect_metadata(config: dict) -> dict:
    topics = config["topics"]
    reader = open_reader(config["bag_path"])
    rgb, depth, info, poses, static_edges = [], [], [], [], {}
    counts = Counter()
    relevant = {topics["rgb"], topics["depth"], topics["camera_info"],
                topics["odometry"], topics["tf_static"]}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in relevant:
            continue
        counts[topic] += 1
        if topic == topics["rgb"]:
            rgb.append(stamp_ns(deserialize_message(data, Image)))
        elif topic == topics["depth"]:
            depth.append(stamp_ns(deserialize_message(data, Image)))
        elif topic == topics["camera_info"]:
            info.append(stamp_ns(deserialize_message(data, CameraInfo)))
        elif topic == topics["odometry"]:
            message = deserialize_message(data, Odometry)
            p, q = message.pose.pose.position, message.pose.pose.orientation
            poses.append(PoseSample(stamp_ns(message), (p.x, p.y, p.z),
                                    (q.x, q.y, q.z, q.w), message.header.frame_id,
                                    message.child_frame_id))
        else:
            message = deserialize_message(data, TFMessage)
            for item in message.transforms:
                t, q = item.transform.translation, item.transform.rotation
                static_edges[(item.header.frame_id, item.child_frame_id)] = transform_matrix(
                    (t.x, t.y, t.z), (q.x, q.y, q.z, q.w))
    return {"rgb": sorted(rgb), "depth": sorted(depth), "info": sorted(info),
            "poses": sorted(poses, key=lambda pose: pose.timestamp_ns),
            "static_edges": static_edges, "topic_counts": dict(counts)}


def camera_static_transform(config: dict, edges: Dict[Tuple[str, str], np.ndarray]) -> np.ndarray:
    alias = config["frames"]["body_static_alias"]
    camera = config["frames"]["camera"]
    chain = [(alias, "camera_link"), ("camera_link", "camera_color_frame"),
             ("camera_color_frame", camera)]
    missing = [edge for edge in chain if edge not in edges]
    if missing:
        raise ValueError(f"Missing static transform edges: {missing}")
    result = np.eye(4)
    for edge in chain:
        result = result @ edges[edge]
    return result


def select_candidates(config: dict, metadata: dict, start_offset_sec: float,
                      duration_sec: Optional[float]) -> Tuple[List[Candidate], dict]:
    selection = config["selection"]
    tolerance_ns = int(float(selection["maximum_sync_delta_ms"]) * 1_000_000)
    rgb_stamps, depth_stamps, info_stamps = metadata["rgb"], metadata["depth"], metadata["info"]
    poses = metadata["poses"]
    pose_stamps = [pose.timestamp_ns for pose in poses]
    if not rgb_stamps:
        raise ValueError("Bag contains no RGB frames")
    start_ns = rgb_stamps[0] + int(start_offset_sec * NSEC)
    end_ns = None if duration_sec is None else start_ns + int(duration_sec * NSEC)
    failures = Counter()
    candidates = []
    previous_pose = None
    evaluated = 0
    for rgb_stamp in rgb_stamps:
        if rgb_stamp < start_ns or (end_ns is not None and rgb_stamp >= end_ns):
            continue
        evaluated += 1
        depth_index = nearest_index(depth_stamps, rgb_stamp)
        info_index = nearest_index(info_stamps, rgb_stamp)
        pose_index = nearest_index(pose_stamps, rgb_stamp)
        missing = []
        if depth_index is None or abs(depth_stamps[depth_index] - rgb_stamp) > tolerance_ns:
            missing.append("missing_depth")
        if info_index is None or abs(info_stamps[info_index] - rgb_stamp) > tolerance_ns:
            missing.append("missing_calibration")
        if pose_index is None or abs(pose_stamps[pose_index] - rgb_stamp) > tolerance_ns:
            missing.append("missing_pose")
        if missing:
            failures.update(missing)
            continue
        pose = poses[pose_index]
        selected, distance, yaw_delta = should_select_keyframe(
            pose, previous_pose, float(selection["distance_threshold_m"]),
            float(selection["yaw_threshold_deg"]),
            float(selection["minimum_time_interval_sec"]))
        if not selected:
            failures["not_keyframe"] += 1
            continue
        candidates.append(Candidate(rgb_stamp, depth_stamps[depth_index],
                                    info_stamps[info_index], pose, distance, yaw_delta))
        previous_pose = pose
    orphan_depth = sum(1 for value in depth_stamps
                       if (idx := nearest_index(rgb_stamps, value)) is None or
                       abs(rgb_stamps[idx] - value) > tolerance_ns)
    failures["missing_rgb"] = orphan_depth
    for reason in ("missing_rgb", "missing_depth", "missing_calibration",
                   "missing_pose", "not_keyframe"):
        failures.setdefault(reason, 0)
    return candidates, {"rgb_frames_evaluated": evaluated, "skip_reasons": dict(failures)}


def collect_selected_messages(config: dict, candidates: Sequence[Candidate]) -> dict:
    topics = config["topics"]
    wanted = {
        topics["rgb"]: {item.rgb_timestamp_ns for item in candidates},
        topics["depth"]: {item.depth_timestamp_ns for item in candidates},
        topics["camera_info"]: {item.camera_info_timestamp_ns for item in candidates},
    }
    result = {"rgb": {}, "depth": {}, "info": {}}
    output_keys = {topics["rgb"]: ("rgb", Image), topics["depth"]: ("depth", Image),
                   topics["camera_info"]: ("info", CameraInfo)}
    reader = open_reader(config["bag_path"])
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in wanted:
            continue
        key, message_type = output_keys[topic]
        message = deserialize_message(data, message_type)
        timestamp = stamp_ns(message)
        if timestamp in wanted[topic] and timestamp not in result[key]:
            result[key][timestamp] = message
    return result


def atomic_write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_dataset(config: dict, layout: GreenhouseLayout, candidates: Sequence[Candidate],
                  messages: dict, selection_report: dict, metadata: dict,
                  start_offset_sec: float, duration_sec: Optional[float]) -> dict:
    target = Path(config["output_dir"]).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        rgb_dir, depth_dir = temp / "keyframes/rgb", temp / "keyframes/depth"
        info_dir, report_dir = temp / "keyframes/camera_info", temp / "reports"
        for directory in (rgb_dir, depth_dir, info_dir, report_dir):
            directory.mkdir(parents=True, exist_ok=True)
        aisle_by_id = {aisle.id: aisle for aisle in layout.aisles}
        assignment_config = config.get("aisle_assignment", {})
        assignment_method = assignment_config.get("method", "polygon")
        maximum_assignment_distance = float(
            assignment_config.get("maximum_distance_m", 0.0)
        )
        if assignment_method not in {"polygon", "polygon_then_nearest"}:
            raise ValueError(
                "aisle_assignment.method must be 'polygon' or "
                "'polygon_then_nearest'"
            )
        body_to_camera = camera_static_transform(config, metadata["static_edges"])
        records, distances = [], []
        jpeg_quality = int(config["image_output"]["jpeg_quality"])
        for index, candidate in enumerate(candidates, 1):
            view_id = f"view_{index:06d}"
            rgb_message = messages["rgb"].get(candidate.rgb_timestamp_ns)
            depth_message = messages["depth"].get(candidate.depth_timestamp_ns)
            info_message = messages["info"].get(candidate.camera_info_timestamp_ns)
            if rgb_message is None or depth_message is None or info_message is None:
                raise RuntimeError(f"Second pass could not load all messages for {view_id}")
            rgb = image_to_array(rgb_message)
            depth = image_to_array(depth_message)
            rgb_rel = f"keyframes/rgb/{view_id}.jpg"
            depth_rel = f"keyframes/depth/{view_id}.png"
            info_rel = f"keyframes/camera_info/{view_id}.json"
            if not cv2.imwrite(str(temp / rgb_rel), rgb,
                               [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
                raise RuntimeError(f"Failed to write RGB image for {view_id}")
            if not cv2.imwrite(str(temp / depth_rel), depth):
                raise RuntimeError(f"Failed to write depth image for {view_id}")
            atomic_write_json(temp / info_rel, camera_info_dict(info_message))

            world_body = transform_matrix(candidate.pose.translation, candidate.pose.quaternion)
            world_camera = world_body @ body_to_camera
            camera_translation = tuple(float(x) for x in world_camera[:3, 3])
            camera_quaternion = matrix_to_quaternion(world_camera[:3, :3])
            body_xy = candidate.pose.translation[:2]
            if assignment_method == "polygon_then_nearest":
                aisle_id, aisle_distance, aisle_assignment_method = (
                    layout.assign_nearest_aisle(
                        *body_xy, maximum_assignment_distance
                    )
                )
            else:
                aisle_id = layout.assign_aisle(*body_xy)
                aisle_distance = 0.0 if aisle_id is not None else None
                aisle_assignment_method = (
                    "polygon" if aisle_id is not None else "unassigned"
                )
            assigned_aisle = aisle_by_id.get(aisle_id)
            coordinate_s = (
                project_onto_centerline(body_xy, assigned_aisle)
                if assigned_aisle is not None else None
            )
            record = {
                "view_id": view_id, "dataset_id": config["dataset_id"],
                "greenhouse_id": layout.greenhouse_id,
                "aisle_id": aisle_id,
                "aisle_assignment": {
                    "method": aisle_assignment_method,
                    "distance_m": aisle_distance,
                },
                "timestamp_ns": candidate.rgb_timestamp_ns,
                "frame_id": layout.frame_id,
                "camera_frame_id": config["frames"]["camera"],
                "rgb_path": rgb_rel, "depth_path": depth_rel,
                "camera_info_path": info_rel,
                "pose": {"translation_xyz": list(camera_translation),
                         "quaternion_xyzw": list(camera_quaternion)},
                "body_pose": {"translation_xyz": list(candidate.pose.translation),
                              "quaternion_xyzw": list(candidate.pose.quaternion)},
                "aisle_coordinate_s": coordinate_s,
                "aisle_section": aisle_section(coordinate_s),
                "travel_pass": "unknown",
                "sync_delta_ms": {
                    "depth": abs(candidate.depth_timestamp_ns - candidate.rgb_timestamp_ns) / 1e6,
                    "camera_info": abs(candidate.camera_info_timestamp_ns - candidate.rgb_timestamp_ns) / 1e6,
                    "pose": abs(candidate.pose.timestamp_ns - candidate.rgb_timestamp_ns) / 1e6,
                },
                "selection": {"distance_from_previous_m": candidate.distance_from_previous_m,
                              "yaw_delta_from_previous_deg": candidate.yaw_delta_from_previous_deg},
                "processing_status": "extracted",
            }
            if index > 1:
                distances.append(candidate.distance_from_previous_m)
            records.append(record)

        if records:
            turn_index = max(range(len(records)), key=lambda i: (
                -1.0 if records[i]["aisle_coordinate_s"] is None else records[i]["aisle_coordinate_s"]
            ))
            for index, record in enumerate(records):
                record["travel_pass"] = "forward" if index <= turn_index else "return"
        views_path = temp / "keyframes/views.jsonl"
        with views_path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        atomic_write_json(temp / "keyframes/views.json", records)

        resolved_config = dict(config)
        resolved_config["extraction_window"] = {
            "start_offset_sec": start_offset_sec, "duration_sec": duration_sec}
        (temp / "pipeline_config.yaml").write_text(
            yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8")
        manifest = {
            "schema_version": "1.0", "dataset_id": config["dataset_id"],
            "greenhouse_id": layout.greenhouse_id, "aisle_ids": [a.id for a in layout.aisles],
            "frame_id": layout.frame_id, "view_count": len(records),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_bag": str(Path(config["bag_path"]).resolve()),
            "views_file": "keyframes/views.jsonl",
            "views_json_file": "keyframes/views.json",
        }
        atomic_write_json(temp / "manifest.json", manifest)
        report = {
            "dataset_id": config["dataset_id"], "status": "complete",
            "source_topic_counts": metadata["topic_counts"],
            "rgb_frames_evaluated": selection_report["rgb_frames_evaluated"],
            "views_written": len(records),
            "skip_reasons": selection_report["skip_reasons"],
            "selection_triggers": {
                "first_view": 1 if candidates else 0,
                "distance_only": sum(
                    item.distance_from_previous_m >= float(config["selection"]["distance_threshold_m"])
                    and item.yaw_delta_from_previous_deg < float(config["selection"]["yaw_threshold_deg"])
                    for item in candidates[1:]),
                "rotation_only": sum(
                    item.distance_from_previous_m < float(config["selection"]["distance_threshold_m"])
                    and item.yaw_delta_from_previous_deg >= float(config["selection"]["yaw_threshold_deg"])
                    for item in candidates[1:]),
                "distance_and_rotation": sum(
                    item.distance_from_previous_m >= float(config["selection"]["distance_threshold_m"])
                    and item.yaw_delta_from_previous_deg >= float(config["selection"]["yaw_threshold_deg"])
                    for item in candidates[1:]),
            },
            "synchronization_failures": sum(
                count for reason, count in selection_report["skip_reasons"].items()
                if reason.startswith("missing_") or reason.startswith("orphan_")),
            "mean_distance_between_views_m": (
                float(sum(distances) / len(distances)) if distances else 0.0),
            "minimum_distance_between_views_m": min(distances) if distances else 0.0,
            "maximum_distance_between_views_m": max(distances) if distances else 0.0,
            "aisle_assignment": {
                "method": assignment_method,
                "maximum_distance_m": maximum_assignment_distance,
                "counts": dict(Counter(
                    record["aisle_assignment"]["method"] for record in records
                )),
                "unassigned_views": sum(
                    record["aisle_id"] is None for record in records
                ),
            },
            "extraction_window": resolved_config["extraction_window"],
        }
        atomic_write_json(report_dir / "extraction_report.json", report)

        backup = target.with_name(f".{target.name}.previous")
        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            os.replace(target, backup)
        os.replace(temp, target)
        if backup.exists():
            shutil.rmtree(backup)
        return report
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def parse_args(argv=None):
    default = Path(get_package_share_directory("conceptgraphs_bringup")) / "config/keyframe_extractor.yaml"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default)
    parser.add_argument("--bag-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--layout-path")
    parser.add_argument("--distance-threshold-m", type=float)
    parser.add_argument("--yaw-threshold-deg", type=float)
    parser.add_argument("--minimum-time-interval-sec", type=float)
    parser.add_argument("--maximum-sync-delta-ms", type=float)
    parser.add_argument("--start-offset-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    for argument, key in ((args.bag_path, "bag_path"), (args.output_dir, "output_dir"),
                          (args.layout_path, "layout_path")):
        if argument is not None:
            config[key] = argument
    for argument, key in (
            (args.distance_threshold_m, "distance_threshold_m"),
            (args.yaw_threshold_deg, "yaw_threshold_deg"),
            (args.minimum_time_interval_sec, "minimum_time_interval_sec"),
            (args.maximum_sync_delta_ms, "maximum_sync_delta_ms")):
        if argument is not None:
            config["selection"][key] = argument
    from .paths import resolve_config_path, resolve_project_path
    config["layout_path"] = str(resolve_config_path(config["layout_path"]))
    config["bag_path"] = str(resolve_project_path(config["bag_path"]))
    config["output_dir"] = str(resolve_project_path(config["output_dir"]))
    layout = load_layout(config["layout_path"])
    if layout.frame_id != config["frames"]["map"]:
        raise ValueError("Layout and extractor map frames do not match")
    print("Pass 1/2: indexing timestamps, odometry and static transforms...", flush=True)
    metadata = collect_metadata(config)
    unexpected_frames = sorted({pose.frame_id for pose in metadata["poses"]} -
                               {config["frames"]["map"]})
    unexpected_children = sorted({pose.child_frame_id for pose in metadata["poses"]} -
                                 {config["frames"]["odometry_child"]})
    if unexpected_frames or unexpected_children:
        raise ValueError(
            f"Unexpected odometry frames: parents={unexpected_frames}, children={unexpected_children}"
        )
    candidates, selection_report = select_candidates(
        config, metadata, args.start_offset_sec, args.duration_sec)
    print(f"Selected {len(candidates)} keyframes; pass 2/2: loading selected messages...", flush=True)
    messages = collect_selected_messages(config, candidates)
    report = write_dataset(config, layout, candidates, messages, selection_report,
                           metadata, args.start_offset_sec, args.duration_sec)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
