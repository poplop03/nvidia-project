#!/usr/bin/env python3
"""Remove PointCloud2 points outside a configurable world-Z interval."""

from copy import deepcopy

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2


class ZPassthroughFilter(Node):
    def __init__(self):
        super().__init__("z_passthrough_filter")
        self.declare_parameter("input_topic", "/cloud_registered")
        self.declare_parameter("output_topic", "/cloud_filtered")
        self.declare_parameter("z_min", -1.3)
        self.declare_parameter("z_max", 2.0)
        self.received_clouds = 0

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.publisher = self.create_publisher(PointCloud2, output_topic, output_qos)
        self.subscription = self.create_subscription(
            PointCloud2, input_topic, self.filter_cloud, input_qos
        )
        self.get_logger().info(
            f"Filtering {input_topic} -> {output_topic}; "
            f"z=[{self.get_parameter('z_min').value}, "
            f"{self.get_parameter('z_max').value}] m"
        )

    def filter_cloud(self, msg: PointCloud2):
        z_field = next((field for field in msg.fields if field.name == "z"), None)
        if z_field is None or z_field.datatype != z_field.FLOAT32:
            self.get_logger().error("PointCloud2 must contain a float32 'z' field")
            return

        # Registered clouds in this dataset are unorganized. Respect row padding
        # as well so the node also works with organized PointCloud2 messages.
        records = []
        finite_z_chunks = []
        input_points = 0
        byte_order = ">" if msg.is_bigendian else "<"
        z_dtype = np.dtype(f"{byte_order}f4")
        raw = memoryview(msg.data)
        z_min = float(self.get_parameter("z_min").value)
        z_max = float(self.get_parameter("z_max").value)

        for row in range(max(1, msg.height)):
            start = row * msg.row_step
            # Some FAST-LIO /Laser_map messages keep the allocated cloud width
            # while serializing fewer bytes.  Limit the view to the points that
            # are actually present instead of letting NumPy build an invalid
            # strided array and killing this node.
            available_bytes = max(0, len(raw) - start)
            serialized_points = available_bytes // msg.point_step
            # FAST-LIO's accumulated map may carry width=0/row_step=0 even
            # though data contains a valid unorganized cloud.
            row_points = (min(msg.width, serialized_points)
                          if msg.width > 0 else serialized_points)
            if row_points == 0:
                continue
            end = start + row_points * msg.point_step
            row_bytes = raw[start:end]
            points = np.frombuffer(row_bytes, dtype=f"V{msg.point_step}")
            z = np.ndarray(
                shape=(row_points,),
                dtype=z_dtype,
                buffer=row_bytes,
                offset=z_field.offset,
                strides=(msg.point_step,),
            )
            keep = np.isfinite(z) & (z >= z_min) & (z <= z_max)
            records.append(points[keep])
            finite_z_chunks.append(z[np.isfinite(z)])
            input_points += row_points

        kept = np.concatenate(records) if records else np.empty(0, dtype=f"V{msg.point_step}")
        output = deepcopy(msg)
        output.height = 1
        output.width = len(kept)
        output.row_step = output.width * output.point_step
        output.is_dense = True
        output.data = kept.tobytes()
        self.publisher.publish(output)
        self.received_clouds += 1
        if self.received_clouds == 1 or self.received_clouds % 10 == 0:
            finite_z = (np.concatenate(finite_z_chunks)
                        if finite_z_chunks else np.array([]))
            input_range = (f"input_z=[{finite_z.min():.2f}, {finite_z.max():.2f}]"
                           if finite_z.size else "input_z=n/a")
            self.get_logger().info(
                f"Z filter cloud={self.received_clouds}: input={input_points}, "
                f"kept={len(kept)}, {input_range}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = ZPassthroughFilter()
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
