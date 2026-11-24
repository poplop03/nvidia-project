"""Convert a raw Livox+D455 bag into a reusable mapping/SAM3 bag."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from conceptgraphs_bringup.paths import bags_dir, project_root


def generate_launch_description():
    project = project_root()
    bag_root = bags_dir()
    input_bag = LaunchConfiguration("input_bag")
    output_bag = LaunchConfiguration("output_bag")
    rate = LaunchConfiguration("rate")
    fast_lio_config = LaunchConfiguration("fast_lio_config")
    use_sim_time = LaunchConfiguration("use_sim_time")
    fov_filter_enabled = LaunchConfiguration("fov_filter_enabled")
    fov_angle_min_deg = LaunchConfiguration("fov_angle_min_deg")
    fov_angle_max_deg = LaunchConfiguration("fov_angle_max_deg")

    fast_lio = Node(
        package="fast_lio",
        executable="fastlio_mapping",
        name="fastlio_mapping",
        output="screen",
        parameters=[fast_lio_config, {
            "use_sim_time": use_sim_time,
            "preprocess.fov_filter_enabled": ParameterValue(fov_filter_enabled, value_type=bool),
            "preprocess.fov_angle_min_deg": ParameterValue(fov_angle_min_deg, value_type=float),
            "preprocess.fov_angle_max_deg": ParameterValue(fov_angle_max_deg, value_type=float),
        }],
    )
    depth_align = Node(
        package="conceptgraphs_bringup",
        executable="depth_to_color_align",
        name="depth_to_color_align",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time, "max_sync_dt": 0.03}],
    )
    recorder = ExecuteProcess(
        cmd=[
            "ros2", "bag", "record", "--storage", "mcap", "-o", output_bag,
            "/camera/camera/color/image_raw",
            "/camera/camera/color/camera_info",
            "/camera/camera/aligned_depth_to_color/image_raw",
            "/cloud_registered",
            "/Laser_map",
            "/Odometry",
            "/path",
            "/tf",
            "/tf_static",
        ],
        output="screen",
    )
    # Delay playback so FAST-LIO, the aligner and recorder have subscribed.
    bag_player = ExecuteProcess(
        cmd=[
            "ros2", "bag", "play", input_bag, "--storage", "mcap",
            "--clock", "--rate", rate, "--disable-keyboard-controls",
            "--topics",
            "/livox/lidar", "/livox/imu",
            "/camera/camera/color/image_raw",
            "/camera/camera/color/camera_info",
            "/camera/camera/depth/image_rect_raw",
            "/camera/camera/depth/camera_info",
            "/tf_static",
        ],
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "input_bag",
            default_value=str(bag_root / "mid360_camera_20260910_153058_humble"),
            description="Humble-compatible raw bag directory",
        ),
        DeclareLaunchArgument(
            "output_bag",
            default_value=str(bag_root / "mid360_camera_20260910_153058_processed"),
            description="New processed bag directory (must not already exist)",
        ),
        DeclareLaunchArgument("rate", default_value="1.0"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("fov_filter_enabled", default_value="true"),
        DeclareLaunchArgument("fov_angle_min_deg", default_value="-150.0"),
        DeclareLaunchArgument("fov_angle_max_deg", default_value="150.0"),
        DeclareLaunchArgument(
            "fast_lio_config",
            default_value=str(project / "livox_ws/src/FAST_LIO_ROS2/config/mid360.yaml"),
        ),
        fast_lio,
        depth_align,
        recorder,
        TimerAction(period=3.0, actions=[bag_player]),
        RegisterEventHandler(OnProcessExit(
            target_action=bag_player,
            on_exit=[TimerAction(
                period=3.0,
                actions=[EmitEvent(event=Shutdown(reason="Dataset processing complete"))],
            )],
        )),
    ])
