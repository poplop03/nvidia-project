"""Play one RGB topic from a bag and save every Nth frame as JPEG."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from conceptgraphs_bringup.paths import bags_dir


def generate_launch_description():
    dataset_dir = bags_dir()
    bag_path = LaunchConfiguration("bag_path")
    image_topic = LaunchConfiguration("image_topic")

    extractor = Node(
        package="conceptgraphs_bringup",
        executable="rgb_frame_extractor",
        name="rgb_frame_extractor",
        output="screen",
        parameters=[{
            "image_topic": image_topic,
            "output_dir": LaunchConfiguration("output_dir"),
            "frame_step": ParameterValue(LaunchConfiguration("frame_step"), value_type=int),
            "jpeg_quality": ParameterValue(LaunchConfiguration("jpeg_quality"), value_type=int),
        }],
    )
    bag_player = ExecuteProcess(
        cmd=[
            "ros2", "bag", "play", bag_path, "--storage", "mcap",
            "--rate", LaunchConfiguration("rate"), "--disable-keyboard-controls",
            "--topics", image_topic,
        ],
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "bag_path",
            default_value=str(dataset_dir / "mid360_camera_20260910_150208_processed_fov"),
        ),
        DeclareLaunchArgument(
            "output_dir",
            default_value=str(dataset_dir / "extracted_rgb"),
        ),
        DeclareLaunchArgument(
            "image_topic", default_value="/camera/camera/color/image_raw"
        ),
        DeclareLaunchArgument("frame_step", default_value="20"),
        DeclareLaunchArgument("jpeg_quality", default_value="95"),
        DeclareLaunchArgument("rate", default_value="1.0"),
        extractor,
        TimerAction(period=1.0, actions=[bag_player]),
        RegisterEventHandler(OnProcessExit(
            target_action=bag_player,
            on_exit=[TimerAction(
                period=1.0,
                actions=[EmitEvent(event=Shutdown(reason="RGB extraction complete"))],
            )],
        )),
    ])
