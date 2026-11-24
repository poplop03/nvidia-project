"""Visualize an extracted M2 View dataset without replaying its rosbag."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from conceptgraphs_bringup.paths import default_dataset


def generate_launch_description():
    share = Path(get_package_share_directory("conceptgraphs_bringup"))
    default_dataset_dir = default_dataset()
    views_path = LaunchConfiguration("views_path")
    layout_path = LaunchConfiguration("layout_path")
    rviz = Node(package="rviz2", executable="rviz2",
                arguments=["-d", str(share / "config/conceptgraphs_map.rviz")],
                parameters=[{"use_sim_time": False}], output="screen")
    return LaunchDescription([
        DeclareLaunchArgument("views_path", default_value=str(default_dataset_dir / "keyframes/views.jsonl")),
        DeclareLaunchArgument("layout_path", default_value=str(share / "config/greenhouse_layout.yaml")),
        Node(package="conceptgraphs_bringup", executable="greenhouse_layout_markers",
             parameters=[{"layout_path": layout_path, "use_sim_time": False}], output="screen"),
        Node(package="conceptgraphs_bringup", executable="keyframe_marker_publisher",
             parameters=[{"views_path": views_path, "use_sim_time": False}], output="screen"),
        rviz,
        RegisterEventHandler(OnProcessExit(
            target_action=rviz,
            on_exit=[EmitEvent(event=Shutdown(reason="RViz closed"))],
        )),
    ])
