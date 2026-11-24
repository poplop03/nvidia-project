"""Launch the recorded dataset, 2D maps, lifecycle manager, and RViz."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from conceptgraphs_bringup.paths import (
    bags_dir,
    default_dataset,
    default_sam3_model,
    project_root,
    resolve_worker_python,
)


def generate_launch_description():
    share = Path(get_package_share_directory("conceptgraphs_bringup"))
    dataset_default = default_dataset()
    default_bag = bags_dir() / "conceptgraphs_bag_humble.mcap"

    bag_path = LaunchConfiguration("bag_path")
    rate = LaunchConfiguration("rate")
    use_sim_time = LaunchConfiguration("use_sim_time")
    start_rviz = LaunchConfiguration("start_rviz")
    z_min = LaunchConfiguration("z_min")
    z_max = LaunchConfiguration("z_max")
    lidar_map_topic = LaunchConfiguration("lidar_map_topic")
    start_sam3 = LaunchConfiguration("start_sam3")
    sam3_model = LaunchConfiguration("sam3_model")
    sam3_python = LaunchConfiguration("sam3_python")
    sam3_device = LaunchConfiguration("sam3_device")
    sam3_confidence = LaunchConfiguration("sam3_confidence")
    sam3_prompt_config = LaunchConfiguration("sam3_prompt_config")
    crate_min_observations = LaunchConfiguration("crate_min_observations")
    start_crate_markers = LaunchConfiguration("start_crate_markers")
    crate_csv = LaunchConfiguration("crate_csv")
    marker_scale = LaunchConfiguration("marker_scale")
    marker_z_offset = LaunchConfiguration("marker_z_offset")
    crate_cluster_radius = LaunchConfiguration("crate_cluster_radius")
    crate_min_confidence = LaunchConfiguration("crate_min_confidence")
    start_greenhouse_layout = LaunchConfiguration("start_greenhouse_layout")
    greenhouse_layout_path = LaunchConfiguration("greenhouse_layout_path")
    start_static_maps = LaunchConfiguration("start_static_maps")
    occupancy_map_yaml = LaunchConfiguration("occupancy_map_yaml")
    costmap_yaml = LaunchConfiguration("costmap_yaml")
    start_keyframe_markers = LaunchConfiguration("start_keyframe_markers")
    views_path = LaunchConfiguration("views_path")
    start_observation_3d_markers = LaunchConfiguration("start_observation_3d_markers")
    observations_3d_path = LaunchConfiguration("observations_3d_path")
    start_object_markers = LaunchConfiguration("start_object_markers")
    objects_path = LaunchConfiguration("objects_path")
    start_dynamic_cloud_filter = LaunchConfiguration("start_dynamic_cloud_filter")
    dynamic_filter_input_topic = LaunchConfiguration("dynamic_filter_input_topic")
    dynamic_filter_output_topic = LaunchConfiguration("dynamic_filter_output_topic")
    dynamic_filter_enable_persistence = LaunchConfiguration("dynamic_filter_enable_persistence")
    dynamic_filter_rear_sector_angle_deg = LaunchConfiguration("dynamic_filter_rear_sector_angle_deg")
    dynamic_filter_rear_max_distance_m = LaunchConfiguration("dynamic_filter_rear_max_distance_m")

    cloud_filter = Node(
        package="conceptgraphs_bringup",
        executable="z_passthrough_filter",
        name="z_passthrough_filter",
        output="screen",
        parameters=[{
            "input_topic": lidar_map_topic,
            "output_topic": "/cloud_filtered",
            "z_min": z_min,
            "z_max": z_max,
            "use_sim_time": use_sim_time,
        }],
    )
    dynamic_cloud_filter = Node(
        package="conceptgraphs_bringup",
        executable="registered_cloud_dynamic_filter",
        name="registered_cloud_dynamic_filter",
        output="screen",
        condition=IfCondition(start_dynamic_cloud_filter),
        parameters=[{
            "input_cloud_topic": dynamic_filter_input_topic,
            "output_cloud_topic": dynamic_filter_output_topic,
            "odometry_topic": "/Odometry",
            "enable_persistence": ParameterValue(
                dynamic_filter_enable_persistence, value_type=bool),
            "rear_sector_angle_deg": ParameterValue(
                dynamic_filter_rear_sector_angle_deg, value_type=float),
            "rear_max_distance_m": ParameterValue(
                dynamic_filter_rear_max_distance_m, value_type=float),
            "use_sim_time": use_sim_time,
        }],
    )
    crate_mapper = Node(
        package="conceptgraphs_bringup",
        executable="sam3_crate_mapper",
        name="sam3_crate_mapper",
        output="screen",
        condition=IfCondition(start_sam3),
        parameters=[{
            "model": sam3_model,
            "worker_python": sam3_python,
            "device": ParameterValue(sam3_device, value_type=str),
            "confidence": sam3_confidence,
            "prompt_config_path": sam3_prompt_config,
            "min_observations": crate_min_observations,
            "marker_scale": marker_scale,
            "marker_z_offset": marker_z_offset,
            "cluster_radius": crate_cluster_radius,
            "min_cluster_confidence": crate_min_confidence,
            "use_sim_time": use_sim_time,
        }],
    )
    crate_markers = Node(
        package="conceptgraphs_bringup",
        executable="crate_marker_publisher",
        name="crate_marker_publisher",
        output="screen",
        condition=IfCondition(start_crate_markers),
        parameters=[{
            "csv_path": crate_csv,
            "marker_scale": marker_scale,
            "z_offset": marker_z_offset,
            "use_sim_time": use_sim_time,
        }],
    )
    greenhouse_layout = Node(
        package="conceptgraphs_bringup",
        executable="greenhouse_layout_markers",
        name="greenhouse_layout_markers",
        output="screen",
        condition=IfCondition(start_greenhouse_layout),
        parameters=[{
            "layout_path": greenhouse_layout_path,
            "use_sim_time": use_sim_time,
        }],
    )
    keyframe_markers = Node(
        package="conceptgraphs_bringup",
        executable="keyframe_marker_publisher",
        name="keyframe_marker_publisher",
        output="screen",
        condition=IfCondition(start_keyframe_markers),
        parameters=[{"views_path": views_path, "use_sim_time": use_sim_time}],
    )
    observation_3d_markers = Node(
        package="conceptgraphs_bringup",
        executable="observation_3d_marker_publisher",
        name="observation_3d_marker_publisher",
        output="screen",
        condition=IfCondition(start_observation_3d_markers),
        parameters=[{"observations_path": observations_3d_path,
                     "use_sim_time": use_sim_time}],
    )

    object_markers = Node(
        package="conceptgraphs_bringup",
        executable="object_marker_publisher",
        name="object_marker_publisher",
        output="screen",
        condition=IfCondition(start_object_markers),
        parameters=[{"objects_path": objects_path,
                     "observations_path": observations_3d_path,
                     "use_sim_time": use_sim_time}],
    )

    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        condition=IfCondition(start_static_maps),
        parameters=[{
            "yaml_filename": occupancy_map_yaml,
            "frame_id": "camera_init",
            "topic_name": "map",
            "use_sim_time": use_sim_time,
        }],
    )

    costmap_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="costmap_server",
        output="screen",
        condition=IfCondition(start_static_maps),
        parameters=[{
            "yaml_filename": costmap_yaml,
            "frame_id": "camera_init",
            "topic_name": "costmap",
            "use_sim_time": use_sim_time,
        }],
    )
    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_maps",
        output="screen",
        condition=IfCondition(start_static_maps),
        parameters=[{
            "autostart": True,
            "node_names": ["map_server", "costmap_server"],
            "use_sim_time": use_sim_time,
            "bond_timeout": 0.0,
        }],
    )
    bag_player = ExecuteProcess(
        cmd=["ros2", "bag", "play", bag_path, "--storage", "mcap", "--clock",
             "--rate", rate, "--disable-keyboard-controls"],
        output="screen",
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", str(share / "config/conceptgraphs_map.rviz")],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(start_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument("bag_path", default_value=str(default_bag),
                              description="Absolute path to the Humble-compatible MCAP"),
        DeclareLaunchArgument("rate", default_value="1.0", description="Bag playback rate"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("start_rviz", default_value="true"),
        DeclareLaunchArgument("z_min", default_value="-1.3",
                              description="Lowest point-cloud Z to display (metres)"),
        DeclareLaunchArgument("z_max", default_value="2.0",
                              description="Highest point-cloud Z to display (metres)"),
        DeclareLaunchArgument(
            "lidar_map_topic",
            default_value="/cloud_registered",
            description="World-frame LiDAR cloud to filter; RViz accumulates the filtered frames",
        ),
        DeclareLaunchArgument(
            "start_dynamic_cloud_filter", default_value="false",
            description="Remove rear-following people and transient registered-cloud voxels",
        ),
        DeclareLaunchArgument(
            "dynamic_filter_input_topic", default_value="/cloud_registered",
        ),
        DeclareLaunchArgument(
            "dynamic_filter_output_topic", default_value="/cloud_registered_static",
        ),
        DeclareLaunchArgument(
            "dynamic_filter_enable_persistence", default_value="false",
            description="Experimental cross-view voxel persistence; rear ROI remains active",
        ),
        DeclareLaunchArgument(
            "dynamic_filter_rear_sector_angle_deg", default_value="30.0",
            description="Full opening angle of the removal sector behind the LiDAR centre",
        ),
        DeclareLaunchArgument(
            "dynamic_filter_rear_max_distance_m", default_value="100.0",
            description="Maximum range of the rear sector; 100 m removes the full recorded wedge",
        ),
        DeclareLaunchArgument("start_sam3", default_value="false",
                              description="Run online yellow-crate detection and mapping"),
        DeclareLaunchArgument("sam3_model", default_value=str(default_sam3_model())),
        DeclareLaunchArgument("sam3_python", default_value=resolve_worker_python()),
        DeclareLaunchArgument("sam3_device", default_value="0",
                              description="SAM3 device: auto, cpu, or CUDA device such as 0"),
        DeclareLaunchArgument("sam3_confidence", default_value="0.2"),
        DeclareLaunchArgument(
            "sam3_prompt_config",
            default_value=str(share / "config/sam3_prompts.yaml"),
            description="Shared YAML containing sam3.prompts for online SAM3",
        ),
        DeclareLaunchArgument("crate_min_observations", default_value="2"),
        DeclareLaunchArgument("crate_cluster_radius", default_value="1.5",
                              description="XY association radius for repeated crate observations (metres)"),
        DeclareLaunchArgument("crate_min_confidence", default_value="0.6",
                              description="A cluster needs at least one observation at this confidence"),
        DeclareLaunchArgument("start_crate_markers", default_value="false",
                              description="Publish dataset-specific confirmed crate markers from CSV"),
        DeclareLaunchArgument("crate_csv", default_value=str(project_root() / "data/crate_clusters.csv")),
        DeclareLaunchArgument("marker_scale", default_value="1.2"),
        DeclareLaunchArgument("marker_z_offset", default_value="0.0"),
        DeclareLaunchArgument("start_greenhouse_layout", default_value="true",
                              description="Publish greenhouse and aisle layout markers"),
        DeclareLaunchArgument(
            "greenhouse_layout_path",
            default_value=str(share / "config/greenhouse_layout.yaml"),
            description="Greenhouse layout YAML file",
        ),
        DeclareLaunchArgument(
            "start_static_maps",
            default_value="false",
            description="Load dataset-specific occupancy and costmap YAML files",
        ),
        DeclareLaunchArgument(
            "occupancy_map_yaml",
            default_value=str(share / "maps/occupancy_grid.yaml"),
            description="Occupancy map YAML; only used when start_static_maps is true",
        ),
        DeclareLaunchArgument(
            "costmap_yaml",
            default_value=str(share / "maps/costmap_2d.yaml"),
            description="Costmap YAML; only used when start_static_maps is true",
        ),
        DeclareLaunchArgument(
            "start_keyframe_markers",
            default_value="false",
            description="Publish M2 View poses from an extracted views.jsonl",
        ),
        DeclareLaunchArgument(
            "views_path",
            default_value=str(dataset_default / "keyframes/views.jsonl"),
            description="Path to the extracted M2 views.jsonl",
        ),
        DeclareLaunchArgument(
            "start_observation_3d_markers",
            default_value="false",
            description="Publish valid M4 centroid, bbox and label markers",
        ),
        DeclareLaunchArgument(
            "observations_3d_path",
            default_value=str(dataset_default / "observations/observations_3d.jsonl"),
            description="Path to M4 observations_3d.jsonl",
        ),
        DeclareLaunchArgument(
            "start_object_markers", default_value="false",
            description="Publish M5 Object and Observation association markers",
        ),
        DeclareLaunchArgument(
            "objects_path",
            default_value=str(dataset_default / "objects/objects.jsonl"),
            description="Path to M5 objects.jsonl",
        ),
        map_server,
        costmap_server,
        lifecycle_manager,
        dynamic_cloud_filter,
        cloud_filter,
        crate_mapper,
        crate_markers,
        greenhouse_layout,
        keyframe_markers,
        observation_3d_markers,
        object_markers,
        bag_player,
        rviz,
        RegisterEventHandler(OnProcessExit(
            target_action=rviz,
            on_exit=[EmitEvent(event=Shutdown(reason="RViz closed"))],
        )),
    ])
