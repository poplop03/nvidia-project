#!/usr/bin/env python3
"""Remove rear-following people and transient voxels from registered clouds."""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2


def stamp_ns(message):
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def rotation_matrix(quaternion):
    """Return body-to-world rotation for an xyzw quaternion."""
    x, y, z, w = quaternion
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("Odometry quaternion has zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def rear_sector_mask(points_world, translation, quaternion, lidar_offset_body,
                     sector_angle_deg, min_distance, max_distance, min_z, max_z):
    """Return points in a full-angle sector behind the LiDAR optical centre."""
    body_to_world = rotation_matrix(quaternion)
    lidar_world = (np.asarray(translation) +
                   body_to_world @ np.asarray(lidar_offset_body, dtype=np.float64))
    points_lidar = (points_world - lidar_world) @ body_to_world
    horizontal_distance = np.hypot(points_lidar[:, 0], points_lidar[:, 1])
    # Angle is zero directly behind the robot and grows toward either side.
    angle_from_rear = np.arctan2(np.abs(points_lidar[:, 1]), -points_lidar[:, 0])
    half_angle_rad = math.radians(float(sector_angle_deg) / 2.0)
    return (
        (points_lidar[:, 0] < 0.0) &
        (angle_from_rear <= half_angle_rad) &
        (horizontal_distance >= min_distance) &
        (horizontal_distance <= max_distance) &
        (points_lidar[:, 2] >= min_z) & (points_lidar[:, 2] <= max_z)
    )


class VoxelPersistence:
    """Require a voxel to be observed from distinct robot positions."""

    def __init__(self, voxel_size, minimum_observations,
                 minimum_observer_translation, stale_after_scans):
        self.voxel_size = float(voxel_size)
        self.minimum_observations = int(minimum_observations)
        self.minimum_observer_translation = float(minimum_observer_translation)
        self.stale_after_scans = int(stale_after_scans)
        self.states = {}
        self.scan_index = 0

    def update(self, points, observer_xy):
        if not len(points):
            self.scan_index += 1
            return np.zeros(0, dtype=bool)
        self.scan_index += 1
        voxels = np.floor(points / self.voxel_size).astype(np.int64)
        unique, inverse = np.unique(voxels, axis=0, return_inverse=True)
        observer = np.asarray(observer_xy, dtype=np.float64)
        accepted = np.zeros(len(unique), dtype=bool)
        minimum_distance_sq = self.minimum_observer_translation ** 2

        for index, voxel in enumerate(unique):
            key = tuple(int(value) for value in voxel)
            state = self.states.get(key)
            if state is None:
                state = [1, observer.copy(), self.scan_index]
                self.states[key] = state
            else:
                delta = observer - state[1]
                if float(delta @ delta) >= minimum_distance_sq:
                    state[0] += 1
                    state[1] = observer.copy()
                state[2] = self.scan_index
            accepted[index] = state[0] >= self.minimum_observations

        if self.scan_index % 50 == 0:
            oldest = self.scan_index - self.stale_after_scans
            self.states = {key: value for key, value in self.states.items()
                           if value[2] >= oldest}
        return accepted[inverse]


@dataclass(frozen=True)
class Pose:
    timestamp_ns: int
    translation: tuple
    quaternion: tuple


def point_records_and_xyz(message):
    fields = {field.name: field for field in message.fields}
    if any(name not in fields for name in ("x", "y", "z")):
        raise ValueError("PointCloud2 must contain x, y and z fields")
    if any(fields[name].datatype != fields[name].FLOAT32 for name in ("x", "y", "z")):
        raise ValueError("PointCloud2 x, y and z fields must be float32")

    byte_order = ">" if message.is_bigendian else "<"
    float_dtype = np.dtype(f"{byte_order}f4")
    raw = memoryview(message.data)
    record_chunks, xyz_chunks = [], []
    for row in range(max(1, message.height)):
        start = row * message.row_step
        available = max(0, len(raw) - start) // message.point_step
        count = min(message.width, available) if message.width > 0 else available
        if count <= 0:
            continue
        row_bytes = raw[start:start + count * message.point_step]
        record_chunks.append(np.frombuffer(row_bytes, dtype=f"V{message.point_step}"))
        xyz = np.column_stack([
            np.ndarray((count,), dtype=float_dtype, buffer=row_bytes,
                       offset=fields[name].offset, strides=(message.point_step,))
            for name in ("x", "y", "z")
        ])
        xyz_chunks.append(xyz)
    records = (np.concatenate(record_chunks) if record_chunks
               else np.empty(0, dtype=f"V{message.point_step}"))
    xyz = np.concatenate(xyz_chunks) if xyz_chunks else np.empty((0, 3), dtype=np.float32)
    return records, xyz


class RegisteredCloudDynamicFilter(Node):
    def __init__(self):
        super().__init__("registered_cloud_dynamic_filter")
        defaults = {
            "input_cloud_topic": "/cloud_registered",
            "odometry_topic": "/Odometry",
            "output_cloud_topic": "/cloud_registered_static",
            "maximum_pose_delta_ms": 100.0,
            "enable_rear_roi": True,
            # TF: base_link -> laser_frame = [0.345, 0.0, 0.1], identity rotation.
            "lidar_offset_x_m": 0.345,
            "lidar_offset_y_m": 0.0,
            "lidar_offset_z_m": 0.1,
            "rear_sector_angle_deg": 30.0,
            "rear_min_distance_m": 0.0,
            "rear_max_distance_m": 100.0,
            "rear_min_z": -1.0,
            "rear_max_z": 2.0,
            # Persistence is intentionally opt-in. Registered scans in this bag
            # overlap sparsely while the robot moves slowly, so an aggressive
            # default can hide the map during warm-up. Rear ROI is the safe
            # default for the known people-following-behind failure mode.
            "enable_persistence": False,
            "voxel_size_m": 0.30,
            "minimum_observations": 2,
            "minimum_observer_translation_m": 0.10,
            "stale_after_scans": 500,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.poses = deque(maxlen=200)
        self.cloud_count = 0
        self.persistence = VoxelPersistence(
            self.get_parameter("voxel_size_m").value,
            self.get_parameter("minimum_observations").value,
            self.get_parameter("minimum_observer_translation_m").value,
            self.get_parameter("stale_after_scans").value,
        )
        input_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=20,
                               reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.VOLATILE)
        output_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                                reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.VOLATILE)
        self.publisher = self.create_publisher(
            PointCloud2, self.get_parameter("output_cloud_topic").value, output_qos)
        self.create_subscription(
            Odometry, self.get_parameter("odometry_topic").value,
            self.odometry_callback, input_qos)
        self.create_subscription(
            PointCloud2, self.get_parameter("input_cloud_topic").value,
            self.cloud_callback, input_qos)
        self.get_logger().info(
            "Dynamic cloud filter: %s -> %s; rear_sector=%s persistence=%s" % (
                self.get_parameter("input_cloud_topic").value,
                self.get_parameter("output_cloud_topic").value,
                self.get_parameter("enable_rear_roi").value,
                self.get_parameter("enable_persistence").value,
            ))

    def odometry_callback(self, message):
        p = message.pose.pose.position
        q = message.pose.pose.orientation
        self.poses.append(Pose(stamp_ns(message), (p.x, p.y, p.z),
                               (q.x, q.y, q.z, q.w)))

    def nearest_pose(self, timestamp_ns):
        if not self.poses:
            return None
        pose = min(self.poses, key=lambda item: abs(item.timestamp_ns - timestamp_ns))
        maximum_delta = float(self.get_parameter("maximum_pose_delta_ms").value) * 1e6
        return pose if abs(pose.timestamp_ns - timestamp_ns) <= maximum_delta else None

    def cloud_callback(self, message):
        pose = self.nearest_pose(stamp_ns(message))
        if pose is None:
            self.get_logger().warning("Skipping cloud without synchronized odometry")
            return
        try:
            records, xyz = point_records_and_xyz(message)
        except ValueError as error:
            self.get_logger().error(str(error))
            return
        finite = np.isfinite(xyz).all(axis=1)
        keep = finite.copy()
        rear_count = 0
        if self.get_parameter("enable_rear_roi").value:
            rear = rear_sector_mask(
                xyz, pose.translation, pose.quaternion,
                (float(self.get_parameter("lidar_offset_x_m").value),
                 float(self.get_parameter("lidar_offset_y_m").value),
                 float(self.get_parameter("lidar_offset_z_m").value)),
                float(self.get_parameter("rear_sector_angle_deg").value),
                float(self.get_parameter("rear_min_distance_m").value),
                float(self.get_parameter("rear_max_distance_m").value),
                float(self.get_parameter("rear_min_z").value),
                float(self.get_parameter("rear_max_z").value),
            )
            rear_count = int((rear & finite).sum())
            keep &= ~rear
        candidate_indices = np.flatnonzero(keep)
        if self.get_parameter("enable_persistence").value:
            persistent = self.persistence.update(
                xyz[candidate_indices], pose.translation[:2])
            keep[:] = False
            keep[candidate_indices[persistent]] = True

        selected = records[keep]
        output = deepcopy(message)
        output.height = 1
        output.width = len(selected)
        output.row_step = output.width * output.point_step
        output.is_dense = True
        output.data = selected.tobytes()
        self.publisher.publish(output)

        self.cloud_count += 1
        if self.cloud_count == 1 or self.cloud_count % 20 == 0:
            self.get_logger().info(
                f"Dynamic filter cloud={self.cloud_count}: input={len(records)}, "
                f"rear_removed={rear_count}, output={len(selected)}, "
                f"tracked_voxels={len(self.persistence.states)}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = RegisteredCloudDynamicFilter()
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
