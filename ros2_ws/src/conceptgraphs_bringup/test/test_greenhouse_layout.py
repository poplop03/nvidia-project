from pathlib import Path

import pytest

from conceptgraphs_bringup.greenhouse_layout import LayoutValidationError, load_layout

CONFIG = Path(__file__).parents[1] / "config/greenhouse_layout.yaml"
SINGLE_DATASET_LAYOUT = (
    Path(__file__).parents[1] / "config/layouts/greenhouse_20261029_single_aisle.yaml"
)
THREE_AISLE_LAYOUT = (
    Path(__file__).parents[1] / "config/layouts/greenhouse_20261029_three_aisles.yaml"
)


def test_load_pilot_layout():
    layout = load_layout(CONFIG)
    assert layout.mode == "single_aisle"
    assert layout.greenhouse_id == "greenhouse_01"
    assert layout.frame_id == "camera_init"
    assert len(layout.aisles) == 1
    assert layout.aisles[0].id == "aisle_01"
    assert layout.aisles[0].polygon_xy == (
        (1.0, 2.0), (1.0, -1.4), (30.0, -1.4), (30.0, 2.0)
    )
    assert layout.aisles[0].centerline == ((1.0, 0.3), (30.0, 0.3))
    assert layout.aisles[0].head_point == (1.0, 0.3)
    assert layout.aisles[0].tail_point == (30.0, 0.3)


def test_frozen_single_dataset_layout_matches_legacy_default():
    assert load_layout(SINGLE_DATASET_LAYOUT) == load_layout(CONFIG)


def test_three_aisle_provisional_layout_loads_without_overlap_at_centerlines():
    layout = load_layout(THREE_AISLE_LAYOUT)
    assert layout.mode == "polygon"
    assert [aisle.id for aisle in layout.aisles] == ["aisle_01", "aisle_02", "aisle_03"]
    assert layout.assign_aisle(10.0, 0.6) == "aisle_01"
    assert layout.assign_aisle(10.0, 5.8) == "aisle_02"
    assert layout.assign_aisle(10.0, 10.9) == "aisle_03"
    assert layout.assign_aisle(24.5, 6.5) is None
    assert layout.assign_nearest_aisle(24.5, 6.5, 4.0) == (
        "aisle_02", pytest.approx(1.5), "nearest_polygon"
    )


def test_nearest_polygon_assignment_respects_maximum_distance_and_tie_breaks(tmp_path):
    path = tmp_path / "nearest.yaml"
    path.write_text("""
mode: polygon
greenhouse: {id: greenhouse_01, frame_id: camera_init}
aisles:
  - {id: aisle_01, polygon_xy: [[0, 0], [2, 0], [2, 2], [0, 2]], centerline: [], head_point: [], tail_point: []}
  - {id: aisle_02, polygon_xy: [[4, 0], [6, 0], [6, 2], [4, 2]], centerline: [], head_point: [], tail_point: []}
""", encoding="utf-8")
    layout = load_layout(path)
    assert layout.assign_nearest_aisle(3, 1, 2.0) == (
        "aisle_01", pytest.approx(1.0), "nearest_polygon"
    )
    assert layout.assign_nearest_aisle(20, 20, 2.0) == (
        None, pytest.approx((14 ** 2 + 18 ** 2) ** 0.5), "unassigned"
    )


@pytest.mark.parametrize("point", [(0.0, 0.0), (-999.0, 42.5), (1e6, -1e6)])
def test_single_aisle_assignment_always_returns_pilot_aisle(point):
    assert load_layout(CONFIG).assign_aisle(*point) == "aisle_01"


def test_single_aisle_rejects_multiple_aisles(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("""
mode: single_aisle
greenhouse: {id: greenhouse_01, frame_id: camera_init}
aisles:
  - {id: aisle_01, polygon_xy: [], centerline: [], head_point: [], tail_point: []}
  - {id: aisle_02, polygon_xy: [], centerline: [], head_point: [], tail_point: []}
""", encoding="utf-8")
    with pytest.raises(LayoutValidationError, match="exactly one aisle"):
        load_layout(path)


def test_polygon_mode_is_ready_for_future_assignment(tmp_path):
    path = tmp_path / "polygon.yaml"
    path.write_text("""
mode: polygon
greenhouse: {id: greenhouse_01, frame_id: camera_init}
aisles:
  - id: aisle_01
    polygon_xy: [[0, 0], [2, 0], [2, 2], [0, 2]]
    centerline: [[1, 0], [1, 2]]
    head_point: [1, 0]
    tail_point: [1, 2]
""", encoding="utf-8")
    layout = load_layout(path)
    assert layout.assign_aisle(1, 1) == "aisle_01"
    assert layout.assign_aisle(3, 3) is None
