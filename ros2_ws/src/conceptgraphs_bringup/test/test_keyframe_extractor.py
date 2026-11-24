import math

import pytest

from conceptgraphs_bringup.greenhouse_layout import Aisle
from conceptgraphs_bringup.keyframe_extractor import (
    PoseSample, aisle_section, nearest_index, project_onto_centerline,
    should_select_keyframe,
)


def pose(timestamp_ns, x=0.0, yaw_deg=0.0):
    yaw = math.radians(yaw_deg)
    return PoseSample(timestamp_ns, (x, 0.0, 0.0),
                      (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)),
                      "camera_init", "body")


def test_selects_first_distance_and_rotation_keyframes():
    selected, _, _ = should_select_keyframe(pose(0), None, 0.5, 15.0, 0.3)
    assert selected
    selected, distance, _ = should_select_keyframe(pose(1_000_000_000, 0.5), pose(0), 0.5, 15.0, 0.3)
    assert selected and distance == pytest.approx(0.5)
    selected, _, yaw = should_select_keyframe(pose(1_000_000_000, yaw_deg=15), pose(0), 0.5, 15.0, 0.3)
    assert selected and yaw == pytest.approx(15.0)


def test_minimum_interval_and_wrapped_yaw():
    assert not should_select_keyframe(pose(100_000_000, 2.0), pose(0), 0.5, 15.0, 0.3)[0]
    selected, _, yaw = should_select_keyframe(
        pose(1_000_000_000, yaw_deg=-179), pose(0, yaw_deg=179), 0.5, 15.0, 0.3)
    assert not selected
    assert yaw == pytest.approx(2.0)


def test_centerline_projection_and_sections():
    aisle = Aisle("aisle_01", (), ((1.0, 0.3), (30.0, 0.3)), (1.0, 0.3), (30.0, 0.3))
    assert project_onto_centerline((1.0, 4.0), aisle) == pytest.approx(0.0)
    assert project_onto_centerline((15.5, -2.0), aisle) == pytest.approx(0.5)
    assert project_onto_centerline((30.0, 0.3), aisle) == pytest.approx(1.0)
    assert aisle_section(0.1) == "head"
    assert aisle_section(0.5) == "middle"
    assert aisle_section(0.9) == "tail"


def test_nearest_index_is_deterministic():
    assert nearest_index([], 5) is None
    assert nearest_index([0, 10], 5) == 0
    assert nearest_index([0, 10], 9) == 1
