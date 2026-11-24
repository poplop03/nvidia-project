import numpy as np
import pytest

from conceptgraphs_bringup.observation_projection import (
    project_arrays, validate_output,
)


def config():
    return {
        "frames": {"map": "camera_init", "camera": "camera_color_optical_frame"},
        "projection": {
            "version": "test", "depth_scale_m": 0.001,
            "minimum_depth_m": 0.15, "maximum_depth_m": 15.0,
            "maximum_object_distance_m": 5.0, "minimum_depth_samples": 1,
            "mask_erosion_kernel_px": 1, "outlier_absolute_m": 0.2,
            "outlier_relative": 0.1,
        },
    }


def observation():
    return {"observation_id": "obs_000001_001", "view_id": "view_000001",
            "aisle_id": "aisle_01", "projection_status": "pending"}


def view():
    return {"view_id": "view_000001", "frame_id": "camera_init",
            "camera_frame_id": "camera_color_optical_frame", "aisle_id": "aisle_01",
            "pose": {"translation_xyz": [10.0, 0.0, 0.0],
                     "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}}


def camera_info():
    return {"frame_id": "camera_color_optical_frame", "width": 3, "height": 3,
            "k": [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]}


def test_projects_camera_points_through_view_extrinsic():
    mask = np.zeros((3, 3), np.uint8)
    mask[1, 1] = 255
    depth = np.full((3, 3), 2000, np.uint16)
    result = project_arrays(observation(), view(), mask, depth, camera_info(), config())
    assert result["projection_status"] == "valid"
    assert result["centroid_3d"] == pytest.approx([10.0, 0.0, 2.0])
    assert result["bbox_3d_min"] == pytest.approx([10.0, 0.0, 2.0])
    assert result["median_depth_m"] == pytest.approx(2.0)
    assert result["valid_depth_ratio"] == pytest.approx(1.0)


def test_rejects_insufficient_depth_but_keeps_record():
    cfg = config()
    cfg["projection"]["minimum_depth_samples"] = 2
    mask = np.zeros((3, 3), np.uint8)
    mask[1, 1] = 255
    depth = np.full((3, 3), 2000, np.uint16)
    result = project_arrays(observation(), view(), mask, depth, camera_info(), cfg)
    assert result["projection_status"] == "rejected"
    assert result["rejection_reason"] == "insufficient_valid_depth"
    assert result["centroid_3d"] is None


def test_rejects_camera_frame_mismatch():
    info = camera_info()
    info["frame_id"] = "laser_frame"
    result = project_arrays(observation(), view(), np.ones((3, 3), np.uint8),
                            np.full((3, 3), 1000, np.uint16), info, config())
    assert result["projection_status"] == "rejected"
    assert result["rejection_reason"] == "camera_info_frame_mismatch"


def test_validation_requires_exactly_one_output_per_input():
    valid = project_arrays(observation(), view(), np.ones((3, 3), np.uint8),
                           np.full((3, 3), 1000, np.uint16), camera_info(), config())
    assert validate_output([observation()], [valid])["valid"]
    with pytest.raises(ValueError, match="exactly"):
        validate_output([observation()], [])
