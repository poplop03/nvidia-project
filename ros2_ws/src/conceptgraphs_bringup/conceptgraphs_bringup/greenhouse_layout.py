"""Load and query the greenhouse aisle layout."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import yaml

Point2D = Tuple[float, float]


class LayoutValidationError(ValueError):
    """Raised when a greenhouse layout does not match the expected schema."""


@dataclass(frozen=True)
class Aisle:
    id: str
    polygon_xy: Tuple[Point2D, ...]
    centerline: Tuple[Point2D, ...]
    head_point: Optional[Point2D]
    tail_point: Optional[Point2D]


@dataclass(frozen=True)
class GreenhouseLayout:
    mode: str
    greenhouse_id: str
    frame_id: str
    aisles: Tuple[Aisle, ...]

    def assign_aisle(self, x: float, y: float) -> Optional[str]:
        """Return the aisle containing an XY point, or None when unassigned."""
        point = (float(x), float(y))
        if self.mode == "single_aisle":
            return self.aisles[0].id
        for aisle in self.aisles:
            if aisle.polygon_xy and _point_in_polygon(point, aisle.polygon_xy):
                return aisle.id
        return None

    def assign_nearest_aisle(
        self, x: float, y: float, maximum_distance_m: float
    ) -> Tuple[Optional[str], Optional[float], str]:
        """Assign by containment, then by distance to the nearest polygon.

        Returns ``(aisle_id, distance_m, method)``.  Contained points have a
        zero distance.  Points farther than ``maximum_distance_m`` remain
        unassigned so bad poses cannot silently be attached to an aisle.
        """
        assigned = self.assign_aisle(x, y)
        if assigned is not None:
            return assigned, 0.0, "polygon"
        if self.mode == "single_aisle":
            return self.aisles[0].id, 0.0, "single_aisle"
        if maximum_distance_m < 0:
            raise ValueError("maximum_distance_m must be non-negative")
        point = (float(x), float(y))
        candidates = [
            (_distance_to_polygon(point, aisle.polygon_xy), aisle.id)
            for aisle in self.aisles
            if aisle.polygon_xy
        ]
        if not candidates:
            return None, None, "unassigned"
        distance, aisle_id = min(candidates, key=lambda item: (item[0], item[1]))
        if distance <= maximum_distance_m:
            return aisle_id, distance, "nearest_polygon"
        return None, distance, "unassigned"


def _require_mapping(value: Any, field: str) -> dict:
    if not isinstance(value, dict):
        raise LayoutValidationError(f"'{field}' must be a mapping")
    return value


def _require_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LayoutValidationError(f"'{field}' must be a non-empty string")
    return value


def _points(value: Any, field: str, minimum: int = 0) -> Tuple[Point2D, ...]:
    if not isinstance(value, list):
        raise LayoutValidationError(f"'{field}' must be a list")
    parsed: List[Point2D] = []
    for index, point in enumerate(value):
        if (not isinstance(point, (list, tuple)) or len(point) != 2 or
                isinstance(point[0], bool) or isinstance(point[1], bool) or
                not isinstance(point[0], (int, float)) or
                not isinstance(point[1], (int, float))):
            raise LayoutValidationError(f"'{field}[{index}]' must be [x, y]")
        parsed.append((float(point[0]), float(point[1])))
    if parsed and len(parsed) < minimum:
        raise LayoutValidationError(
            f"'{field}' must be empty or contain at least {minimum} points"
        )
    return tuple(parsed)


def _optional_point(value: Any, field: str) -> Optional[Point2D]:
    points = _points([value] if value else [], field)
    return points[0] if points else None


def load_layout(path: Path | str) -> GreenhouseLayout:
    """Read and validate a greenhouse layout YAML file."""
    layout_path = Path(path)
    try:
        with layout_path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise LayoutValidationError(f"Invalid YAML in {layout_path}: {exc}") from exc
    root = _require_mapping(data, "root")
    mode = _require_id(root.get("mode"), "mode")
    if mode not in {"single_aisle", "polygon"}:
        raise LayoutValidationError("'mode' must be 'single_aisle' or 'polygon'")
    greenhouse = _require_mapping(root.get("greenhouse"), "greenhouse")
    greenhouse_id = _require_id(greenhouse.get("id"), "greenhouse.id")
    frame_id = _require_id(greenhouse.get("frame_id"), "greenhouse.frame_id")
    aisle_data = root.get("aisles")
    if not isinstance(aisle_data, list) or not aisle_data:
        raise LayoutValidationError("'aisles' must be a non-empty list")
    aisles = []
    seen_ids = set()
    for index, raw_aisle in enumerate(aisle_data):
        raw = _require_mapping(raw_aisle, f"aisles[{index}]")
        aisle_id = _require_id(raw.get("id"), f"aisles[{index}].id")
        if aisle_id in seen_ids:
            raise LayoutValidationError(f"Duplicate aisle id: {aisle_id}")
        seen_ids.add(aisle_id)
        aisles.append(Aisle(
            id=aisle_id,
            polygon_xy=_points(raw.get("polygon_xy"), f"{aisle_id}.polygon_xy", 3),
            centerline=_points(raw.get("centerline"), f"{aisle_id}.centerline", 2),
            head_point=_optional_point(raw.get("head_point"), f"{aisle_id}.head_point"),
            tail_point=_optional_point(raw.get("tail_point"), f"{aisle_id}.tail_point"),
        ))
    if mode == "single_aisle" and len(aisles) != 1:
        raise LayoutValidationError("'single_aisle' mode requires exactly one aisle")
    if mode == "polygon" and any(not aisle.polygon_xy for aisle in aisles):
        raise LayoutValidationError("'polygon' mode requires polygon_xy for every aisle")
    return GreenhouseLayout(mode, greenhouse_id, frame_id, tuple(aisles))


def _point_in_polygon(point: Point2D, polygon: Sequence[Point2D]) -> bool:
    """Ray-casting containment test used by polygon mode."""
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if ((y1 > y) != (y2 > y)) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
        previous = current
    return inside


def _distance_to_segment(point: Point2D, start: Point2D, end: Point2D) -> float:
    """Euclidean distance from a point to a closed line segment."""
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    length_squared = dx * dx + dy * dy
    if length_squared == 0.0:
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_squared))
    nearest_x, nearest_y = x1 + t * dx, y1 + t * dy
    return ((px - nearest_x) ** 2 + (py - nearest_y) ** 2) ** 0.5


def _distance_to_polygon(point: Point2D, polygon: Sequence[Point2D]) -> float:
    """Return zero inside a polygon, otherwise distance to its boundary."""
    if _point_in_polygon(point, polygon):
        return 0.0
    return min(
        _distance_to_segment(point, polygon[index - 1], polygon[index])
        for index in range(len(polygon))
    )
