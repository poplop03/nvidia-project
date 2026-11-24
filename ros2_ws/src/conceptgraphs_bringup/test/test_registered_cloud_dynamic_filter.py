import math

import numpy as np

from conceptgraphs_bringup.registered_cloud_dynamic_filter import (
    VoxelPersistence, rear_sector_mask,
)


def test_rear_sector_uses_robot_pose_not_fixed_world_axes():
    points = np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
    yaw_90 = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    mask = rear_sector_mask(points, (0.0, 0.0, 0.0), yaw_90,
                            (0.0, 0.0, 0.0), 60.0, 0.25, 2.0, -1.0, 2.0)
    assert mask.tolist() == [True, False, False]


def test_rear_sector_respects_angle_distance_and_height():
    points = np.array([
        [-2.0, 0.0, 0.0],    # directly behind
        [-2.0, 1.0, 0.0],    # 26.6 degrees behind
        [-2.0, 2.0, 0.0],    # 45 degrees, outside a 30 degree half-angle
        [-4.0, 0.0, 0.0],    # too far
        [-2.0, 0.0, 2.5],    # too high
        [2.0, 0.0, 0.0],     # in front
    ])
    mask = rear_sector_mask(points, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
                            (0.0, 0.0, 0.0), 60.0, 0.25, 3.0, -1.0, 2.0)
    assert mask.tolist() == [True, True, False, False, False, False]


def test_rear_sector_origin_is_lidar_not_body():
    points = np.array([
        [0.20, 0.0, 0.10],  # 0.145 m behind LiDAR: below minimum range
        [0.00, 0.0, 0.10],  # 0.345 m behind LiDAR: removed
    ])
    mask = rear_sector_mask(points, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
                            (0.345, 0.0, 0.10), 30.0, 0.25, 3.0, -1.0, 2.0)
    assert mask.tolist() == [False, True]


def test_voxel_requires_observations_from_distinct_robot_positions():
    persistence = VoxelPersistence(0.2, 3, 0.25, 100)
    points = np.array([[1.02, 2.01, 0.0], [5.0, 5.0, 0.0]])
    assert not persistence.update(points, (0.0, 0.0)).any()
    assert not persistence.update(points, (0.1, 0.0)).any()
    assert not persistence.update(points, (0.3, 0.0)).any()
    accepted = persistence.update(points, (0.6, 0.0))
    assert accepted.tolist() == [True, True]


def test_moving_voxel_does_not_gain_static_persistence():
    persistence = VoxelPersistence(0.2, 3, 0.25, 100)
    for index in range(5):
        point = np.array([[float(index), 0.0, 0.0]])
        assert not persistence.update(point, (index * 0.3, 0.0)).any()
