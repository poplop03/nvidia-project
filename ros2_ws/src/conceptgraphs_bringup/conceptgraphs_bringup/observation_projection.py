#!/usr/bin/env python3
"""Project M3 SAM3 masks through aligned depth into camera_init."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import yaml


def quaternion_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Match the xyzw convention used by the M2 View extractor."""
    x, y, z, w = quaternion
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm == 0.0:
        raise ValueError("Quaternion must not be zero")
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def transform_matrix(translation: Sequence[float], quaternion: Sequence[float]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = quaternion_to_matrix(quaternion)
    result[:3, 3] = translation
    return result


def read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def atomic_jsonl(path: Path, records: Sequence[dict]):
    records = list(records)
    atomic_text(path, "".join(json.dumps(record, ensure_ascii=False) + "\n"
                              for record in records))
    atomic_text(
        path.with_suffix(".json"),
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def projection_fingerprint(config: dict, observations_path: Path, views_path: Path) -> str:
    payload = {
        "projection": config["projection"],
        "observations_sha256": file_sha256(observations_path),
        "views_sha256": file_sha256(views_path),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def rejected_record(observation: dict, reason: str, point_count: int = 0,
                    median_depth_m: Optional[float] = None,
                    valid_depth_ratio: float = 0.0, details: Optional[dict] = None) -> dict:
    result = dict(observation)
    result.update({
        "frame_id": "camera_init", "centroid_3d": None,
        "bbox_3d_min": None, "bbox_3d_max": None,
        "median_depth_m": median_depth_m,
        "valid_depth_ratio": float(valid_depth_ratio),
        "point_count": int(point_count),
        "projection_status": "rejected", "rejection_reason": reason,
        "projection_details": details or {},
    })
    return result


def project_arrays(observation: dict, view: dict, mask: np.ndarray, depth: np.ndarray,
                   camera_info: dict, config: dict) -> dict:
    """Project one observation; inputs are separated for deterministic unit tests."""
    projection = config["projection"]
    map_frame = config["frames"]["map"]
    camera_frame = config["frames"]["camera"]
    if view.get("frame_id") != map_frame:
        return rejected_record(observation, "view_map_frame_mismatch",
                               details={"expected": map_frame, "actual": view.get("frame_id")})
    if view.get("camera_frame_id") != camera_frame:
        return rejected_record(observation, "view_camera_frame_mismatch",
                               details={"expected": camera_frame,
                                        "actual": view.get("camera_frame_id")})
    if camera_info.get("frame_id") != camera_frame:
        return rejected_record(observation, "camera_info_frame_mismatch",
                               details={"expected": camera_frame,
                                        "actual": camera_info.get("frame_id")})
    if mask.ndim != 2 or depth.ndim != 2 or mask.shape != depth.shape:
        return rejected_record(observation, "mask_depth_shape_mismatch",
                               details={"mask_shape": list(mask.shape),
                                        "depth_shape": list(depth.shape)})
    if depth.dtype != np.uint16:
        return rejected_record(observation, "unsupported_depth_encoding",
                               details={"dtype": str(depth.dtype)})
    if camera_info.get("width") != depth.shape[1] or camera_info.get("height") != depth.shape[0]:
        return rejected_record(observation, "camera_info_shape_mismatch")
    intrinsic = np.asarray(camera_info.get("k", []), dtype=np.float64)
    if intrinsic.size != 9:
        return rejected_record(observation, "invalid_camera_intrinsics")
    intrinsic = intrinsic.reshape(3, 3)
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        return rejected_record(observation, "invalid_camera_intrinsics")

    binary_mask = mask > 0
    mask_pixels = int(np.count_nonzero(binary_mask))
    if mask_pixels == 0:
        return rejected_record(observation, "empty_mask")
    kernel_size = int(projection["mask_erosion_kernel_px"])
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        projection_mask = cv2.erode(binary_mask.astype(np.uint8), kernel, iterations=1) > 0
    else:
        projection_mask = binary_mask
    eroded_pixels = int(np.count_nonzero(projection_mask))
    if eroded_pixels == 0:
        return rejected_record(observation, "mask_empty_after_erosion",
                               details={"mask_pixels": mask_pixels, "eroded_mask_pixels": 0})

    rows, columns = np.nonzero(projection_mask)
    raw_depth = depth[rows, columns].astype(np.float64)
    z = raw_depth * float(projection["depth_scale_m"])
    valid = ((raw_depth > 0) & np.isfinite(z) &
             (z > float(projection["minimum_depth_m"])) &
             (z < float(projection["maximum_depth_m"])))
    rows, columns, z = rows[valid], columns[valid], z[valid]
    minimum_samples = int(projection["minimum_depth_samples"])
    raw_valid_count = len(z)
    if raw_valid_count < minimum_samples:
        return rejected_record(
            observation, "insufficient_valid_depth", raw_valid_count,
            float(np.median(z)) if raw_valid_count else None,
            raw_valid_count / mask_pixels,
            {"mask_pixels": mask_pixels, "eroded_mask_pixels": eroded_pixels,
             "minimum_depth_samples": minimum_samples})

    initial_median = float(np.median(z))
    outlier_band = max(float(projection["outlier_absolute_m"]),
                       initial_median * float(projection["outlier_relative"]))
    inlier = np.abs(z - initial_median) < outlier_band
    rows, columns, z = rows[inlier], columns[inlier], z[inlier]
    point_count = len(z)
    ratio = point_count / mask_pixels
    if point_count < minimum_samples:
        return rejected_record(
            observation, "insufficient_inlier_depth", point_count,
            float(np.median(z)) if point_count else initial_median, ratio,
            {"mask_pixels": mask_pixels, "eroded_mask_pixels": eroded_pixels,
             "raw_valid_depth_count": raw_valid_count, "outlier_band_m": outlier_band,
             "minimum_depth_samples": minimum_samples})

    median_depth = float(np.median(z))
    camera_points = np.column_stack((
        (columns - intrinsic[0, 2]) * z / intrinsic[0, 0],
        (rows - intrinsic[1, 2]) * z / intrinsic[1, 1], z,
    ))
    camera_centroid = np.median(camera_points, axis=0)
    distance = float(np.linalg.norm(camera_centroid))
    if distance > float(projection["maximum_object_distance_m"]):
        return rejected_record(
            observation, "object_beyond_maximum_distance", point_count,
            median_depth, ratio,
            {"camera_centroid_xyz": camera_centroid.tolist(),
             "camera_distance_m": distance, "mask_pixels": mask_pixels,
             "eroded_mask_pixels": eroded_pixels,
             "raw_valid_depth_count": raw_valid_count,
             "outlier_band_m": outlier_band})

    pose = view.get("pose", {})
    translation, quaternion = pose.get("translation_xyz"), pose.get("quaternion_xyzw")
    if not (isinstance(translation, list) and len(translation) == 3 and
            isinstance(quaternion, list) and len(quaternion) == 4):
        return rejected_record(observation, "invalid_view_pose", point_count,
                               median_depth, ratio)
    world_from_camera = transform_matrix(translation, quaternion)
    homogeneous = np.column_stack((camera_points, np.ones(point_count)))
    world_points = (world_from_camera @ homogeneous.T).T[:, :3]
    centroid = np.median(world_points, axis=0)
    result = dict(observation)
    result.update({
        "frame_id": map_frame, "centroid_3d": centroid.tolist(),
        "bbox_3d_min": np.min(world_points, axis=0).tolist(),
        "bbox_3d_max": np.max(world_points, axis=0).tolist(),
        "median_depth_m": median_depth, "valid_depth_ratio": float(ratio),
        "point_count": int(point_count), "aisle_id": view["aisle_id"],
        "projection_status": "valid", "rejection_reason": None,
        "projection_details": {
            "camera_frame_id": camera_frame,
            "camera_centroid_xyz": camera_centroid.tolist(),
            "camera_distance_m": distance, "mask_pixels": mask_pixels,
            "eroded_mask_pixels": eroded_pixels,
            "raw_valid_depth_count": raw_valid_count,
            "outlier_band_m": outlier_band,
            "projection_version": projection["version"],
        },
    })
    return result


def project_observation(dataset: Path, observation: dict, view: dict, config: dict) -> dict:
    mask_path = dataset / observation["mask_path"]
    depth_path = dataset / view["depth_path"]
    info_path = dataset / view["camera_info_path"]
    try:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            return rejected_record(observation, "missing_or_unreadable_mask")
        if depth is None:
            return rejected_record(observation, "missing_or_unreadable_depth")
        camera_info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return rejected_record(observation, "missing_or_invalid_camera_info",
                               details={"error": str(error)})
    return project_arrays(observation, view, mask, depth, camera_info, config)


def numeric_summary(values: Sequence[float]) -> dict:
    if not values:
        return {"count": 0, "minimum": None, "mean": None, "median": None,
                "p90": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "minimum": float(np.min(array)),
            "mean": float(np.mean(array)), "median": float(np.median(array)),
            "p90": float(np.percentile(array, 90)), "maximum": float(np.max(array))}


def cross_pass_qa(records: Sequence[dict], views_by_id: Dict[str, dict]) -> dict:
    valid = [record for record in records if record["projection_status"] == "valid"]
    pairs = []
    for forward in valid:
        if views_by_id[forward["view_id"]].get("travel_pass") != "forward":
            continue
        candidates = [other for other in valid
                      if other["canonical_label"] == forward["canonical_label"]
                      and views_by_id[other["view_id"]].get("travel_pass") == "return"]
        if not candidates:
            continue
        source = np.asarray(forward["centroid_3d"])
        nearest = min(candidates, key=lambda item: np.linalg.norm(
            np.asarray(item["centroid_3d"]) - source))
        delta = np.asarray(nearest["centroid_3d"]) - source
        pairs.append({"forward_observation_id": forward["observation_id"],
                      "return_observation_id": nearest["observation_id"],
                      "canonical_label": forward["canonical_label"],
                      "distance_m": float(np.linalg.norm(delta)),
                      "delta_xyz_m": delta.tolist()})
    distances = [pair["distance_m"] for pair in pairs]
    deltas = np.asarray([pair["delta_xyz_m"] for pair in pairs]) if pairs else np.empty((0, 3))
    return {
        "method": "nearest return-pass centroid with same canonical label; diagnostic only, not association",
        "pair_count": len(pairs), "distance_m": numeric_summary(distances),
        "mean_delta_xyz_m": np.mean(deltas, axis=0).tolist() if pairs else None,
        "within_0_5_m": sum(value <= 0.5 for value in distances),
        "within_1_0_m": sum(value <= 1.0 for value in distances),
        "pairs": sorted(pairs, key=lambda item: item["distance_m"]),
    }


def build_report(dataset_id: str, observations: Sequence[dict], expected_total: int,
                 fingerprint: str, projection: dict, started_at: str,
                 views_by_id: Dict[str, dict]) -> dict:
    statuses = Counter(record["projection_status"] for record in observations)
    reasons = Counter(record["rejection_reason"] for record in observations
                      if record["projection_status"] == "rejected")
    valid = [record for record in observations if record["projection_status"] == "valid"]
    all_depth = [record["median_depth_m"] for record in observations
                 if record["median_depth_m"] is not None]
    all_ratios = [record["valid_depth_ratio"] for record in observations]
    all_points = [record["point_count"] for record in observations]
    return {
        "dataset_id": dataset_id, "stage": "M4_observation_projection",
        "status": "complete" if len(observations) == expected_total else "partial",
        "started_at": started_at, "updated_at": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": fingerprint, "projection_config": projection,
        "observations_total": len(observations),
        "projection_status_counts": {"valid": statuses.get("valid", 0),
                                     "rejected": statuses.get("rejected", 0)},
        "rejection_reasons": dict(reasons),
        "quality": {"median_depth_m": numeric_summary(all_depth),
                    "valid_depth_ratio": numeric_summary(all_ratios),
                    "point_count": numeric_summary(all_points),
                    "valid_only_depth_m": numeric_summary(
                        [record["median_depth_m"] for record in valid]),
                    "valid_only_ratio": numeric_summary(
                        [record["valid_depth_ratio"] for record in valid])},
        "cross_pass_tf_qa": cross_pass_qa(observations, views_by_id),
    }


def validate_output(source: Sequence[dict], projected: Sequence[dict]) -> dict:
    source_ids = [item["observation_id"] for item in source]
    output_ids = [item["observation_id"] for item in projected]
    if len(output_ids) != len(set(output_ids)):
        raise ValueError("Duplicate observation IDs in observations_3d.jsonl")
    if sorted(source_ids) != sorted(output_ids):
        raise ValueError("M4 output does not contain exactly the M3 observations")
    for record in projected:
        if record["projection_status"] not in {"valid", "rejected"}:
            raise ValueError("Every observation must be valid or rejected")
        if record["frame_id"] != "camera_init":
            raise ValueError("Every projection record must use camera_init")
        if record["projection_status"] == "valid":
            for key in ("centroid_3d", "bbox_3d_min", "bbox_3d_max"):
                if not isinstance(record[key], list) or len(record[key]) != 3:
                    raise ValueError(f"Invalid {key} for {record['observation_id']}")
            if not np.all(np.isfinite(record["centroid_3d"])):
                raise ValueError("Non-finite centroid")
        elif not record["rejection_reason"]:
            raise ValueError("Rejected observation has no reason")
    return {"valid": True, "source_records": len(source),
            "output_records": len(projected), "unique_observation_ids": len(set(output_ids))}


def load_config(dataset: Path) -> dict:
    with (dataset / "pipeline_config.yaml").open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if "projection" not in config:
        raise ValueError("pipeline_config.yaml has no projection section")
    return config


def run(dataset: Path, limit: Optional[int] = None, force: bool = False) -> dict:
    dataset = dataset.resolve()
    config = load_config(dataset)
    source_path = dataset / "observations/observations.jsonl"
    views_path = dataset / "keyframes/views.jsonl"
    output_path = dataset / "observations/observations_3d.jsonl"
    report_path = dataset / "reports/projection_report.json"
    source = read_jsonl(source_path)
    views = read_jsonl(views_path)
    views_by_id = {view["view_id"]: view for view in views}
    if len(source) != len({item["observation_id"] for item in source}):
        raise ValueError("Duplicate observation IDs in M3 input")
    if any(item["view_id"] not in views_by_id for item in source):
        raise ValueError("M3 observation references an unknown View")
    fingerprint = projection_fingerprint(config, source_path, views_path)
    if force:
        for path in (output_path, report_path):
            if path.exists():
                path.unlink()
    projected = read_jsonl(output_path) if output_path.exists() else []
    started_at = datetime.now(timezone.utc).isoformat()
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("config_fingerprint") != fingerprint:
            raise ValueError("Projection inputs/config changed; rerun with --force")
        started_at = previous.get("started_at", started_at)
    completed = {item["observation_id"] for item in projected}
    selected = source if limit is None else source[:limit]
    run_started = time.monotonic()
    for index, observation in enumerate(selected, 1):
        if observation["observation_id"] in completed:
            print(f"[{index}/{len(source)}] {observation['observation_id']}: resume skip", flush=True)
            continue
        result = project_observation(dataset, observation,
                                     views_by_id[observation["view_id"]], config)
        projected.append(result)
        projected.sort(key=lambda item: item["observation_id"])
        atomic_jsonl(output_path, projected)
        report = build_report(json.loads((dataset / "manifest.json").read_text())["dataset_id"],
                              projected, len(source), fingerprint, config["projection"],
                              started_at, views_by_id)
        report["last_run_elapsed_sec"] = time.monotonic() - run_started
        atomic_json(report_path, report)
        print(f"[{index}/{len(source)}] {observation['observation_id']}: "
              f"{result['projection_status']}"
              + (f" ({result['rejection_reason']})" if result["rejection_reason"] else ""),
              flush=True)
    validation = validate_output(source, projected) if len(projected) == len(source) else None
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    report = build_report(manifest["dataset_id"], projected, len(source), fingerprint,
                          config["projection"], started_at, views_by_id)
    report["validation"] = validation
    report["last_run_elapsed_sec"] = time.monotonic() - run_started
    atomic_json(report_path, report)
    if validation:
        manifest.setdefault("stages", {})["observation_projection"] = {
            "status": "complete", "observations_3d_file": "observations/observations_3d.jsonl",
            "observations_3d_json_file": "observations/observations_3d.json",
            "report_file": "reports/projection_report.json",
            "valid_count": report["projection_status_counts"]["valid"],
            "rejected_count": report["projection_status_counts"]["rejected"],
            "projection_version": config["projection"]["version"],
        }
        atomic_json(dataset / "manifest.json", manifest)
    return report


def parse_args(argv=None):
    from .paths import default_dataset
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_dataset())
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = run(args.dataset, args.limit, args.force)
    summary_keys = ("status", "observations_total", "projection_status_counts",
                    "rejection_reasons", "quality", "cross_pass_tf_qa",
                    "validation", "last_run_elapsed_sec")
    print(json.dumps({key: report.get(key) for key in summary_keys},
                     indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
