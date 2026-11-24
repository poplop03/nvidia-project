#!/usr/bin/env python3
"""Remove Livox points inside a short box behind the robot."""

import rclpy
from livox_ros_driver2.msg import CustomMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class RearPersonFilter(Node):
    def __init__(self):
        super().__init__("rear_person_filter")
        self.declare_parameter("enabled", True)
        self.declare_parameter("min_x", -2.0)
        self.declare_parameter("max_x", -0.25)
        self.declare_parameter("half_width", 0.75)
        self.declare_parameter("max_distance", 2.2)
        self.scans = 0
        self.removed_total = 0
        if self.get_parameter("min_x").value > self.get_parameter("max_x").value:
            raise ValueError("rear min_x must be less than or equal to max_x")

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.publisher = self.create_publisher(CustomMsg, "/livox/lidar_filtered", qos)
        self.create_subscription(CustomMsg, "/livox/lidar", self.callback, qos)
        self.get_logger().info(
            "Rear filter: min_x=%.2f, max_x=%.2f, half_width=%.2f, max_distance=%.2f"
            % (self.get_parameter("min_x").value,
               self.get_parameter("max_x").value,
               self.get_parameter("half_width").value,
               self.get_parameter("max_distance").value)
        )

    def callback(self, msg):
        if not self.get_parameter("enabled").value:
            self.publisher.publish(msg)
            return

        min_x = float(self.get_parameter("min_x").value)
        max_x = float(self.get_parameter("max_x").value)
        half_width = float(self.get_parameter("half_width").value)
        max_distance_squared = float(self.get_parameter("max_distance").value) ** 2

        filtered_points = [
            point for point in msg.points
            if not self.should_remove(
                point.x, point.y, point.z, min_x, max_x,
                half_width, max_distance_squared)
        ]
        removed = len(msg.points) - len(filtered_points)
        output = CustomMsg()
        output.header = msg.header
        output.timebase = msg.timebase
        output.point_num = len(filtered_points)
        output.lidar_id = msg.lidar_id
        output.rsvd = msg.rsvd
        output.points = filtered_points
        self.publisher.publish(output)

        self.scans += 1
        self.removed_total += removed
        if self.scans == 1 or self.scans % 50 == 0:
            self.get_logger().info(
                f"Rear filter scan={self.scans}: kept={len(filtered_points)}, "
                f"removed={removed}, removed_total={self.removed_total}"
            )

    @staticmethod
    def should_remove(x, y, z, min_x, max_x, half_width, max_distance_squared):
        inside_box = min_x <= x <= max_x and abs(y) <= half_width
        close_enough = x * x + y * y + z * z <= max_distance_squared
        return inside_box and close_enough


def main(args=None):
    rclpy.init(args=args)
    node = RearPersonFilter()
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
