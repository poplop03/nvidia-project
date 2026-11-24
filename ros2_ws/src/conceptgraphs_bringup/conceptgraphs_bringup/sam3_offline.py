#!/usr/bin/env python3
"""Run SAM3 sequentially over an extracted M2 View dataset."""

import argparse
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory

from .paths import default_dataset, resolve_config_path, resolve_sam3_model, resolve_worker_python
from .sam3_prompt_config import load_sam3_prompt_config


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


def atomic_image(path: Path, image: np.ndarray, parameters=None):
    suffix = path.suffix.lower()
    ok, encoded = cv2.imencode(suffix, image, parameters or [])
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded.tobytes())
    os.replace(temporary, path)


def read_exact(stream, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            raise EOFError("SAM3 worker stopped unexpectedly")
        data.extend(chunk)
    return bytes(data)


def canonical_prompt_map(config: dict) -> Dict[str, str]:
    prompts = config["sam3"]["prompts"]
    if not prompts or any(not item.get("text") or not item.get("canonical_label")
                          for item in prompts):
        raise ValueError("sam3.prompts must contain text and canonical_label")
    texts = [item["text"] for item in prompts]
    if len(texts) != len(set(texts)):
        raise ValueError("SAM3 prompt texts must be unique")
    return {item["text"]: item["canonical_label"] for item in prompts}


def config_fingerprint(sam3_config: dict) -> str:
    serialized = json.dumps(sam3_config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def observation_id(view_id: str, detection_index: int) -> str:
    if not view_id.startswith("view_"):
        raise ValueError(f"Unexpected View ID: {view_id}")
    return f"obs_{view_id[5:]}_{detection_index:03d}"


def clip_bbox(bbox: Sequence[float], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = bbox
    result = [max(0, min(width - 1, int(math.floor(x1)))),
              max(0, min(height - 1, int(math.floor(y1)))),
              max(1, min(width, int(math.ceil(x2)))),
              max(1, min(height, int(math.ceil(y2))))]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError(f"Invalid bounding box after clipping: {result}")
    return result


def context_bbox(bbox: Sequence[int], width: int, height: int, padding_ratio: float) -> List[int]:
    x1, y1, x2, y2 = bbox
    padding = int(math.ceil(max(x2 - x1, y2 - y1) * padding_ratio))
    return [max(0, x1 - padding), max(0, y1 - padding),
            min(width, x2 + padding), min(height, y2 + padding)]


def masked_crop(image: np.ndarray, mask: np.ndarray, bbox: Sequence[int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]
    crop[crop_mask == 0] = 0
    return crop


def render_overlay(image: np.ndarray, detections: Sequence[dict], prompt_order: Dict[str, int]) -> np.ndarray:
    overlay = image.copy()
    colors = [(0, 215, 255), (50, 50, 255), (80, 220, 80), (255, 120, 30)]
    for detection in detections:
        color = colors[prompt_order[detection["prompt"]] % len(colors)]
        mask = detection["mask"] > 0
        colored = np.zeros_like(image)
        colored[:] = color
        overlay[mask] = cv2.addWeighted(image[mask], 0.45, colored[mask], 0.55, 0)
        x1, y1, x2, y2 = detection["bbox"]
        cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), color, 2)
        label = f"{detection['canonical_label']} {detection['confidence']:.2f}"
        cv2.putText(overlay, label, (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return overlay


class Sam3Worker:
    def __init__(self, config: dict, worker_script: Path):
        sam3 = config["sam3"]
        model_path = resolve_sam3_model(sam3.get("model_path"))
        command = [resolve_worker_python(sam3.get("worker_python")), str(worker_script),
                   "--model", str(model_path),
                   "--conf", str(sam3["confidence_threshold"]),
                   "--device", str(sam3["device"]),
                   "--imgsz", str(sam3["image_size"]),
                   "--mode", "offline_2d"]
        for prompt in sam3["prompts"]:
            command.extend(["--prompt", prompt["text"]])
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE)

    def infer(self, view_id: str, image: np.ndarray) -> List[dict]:
        height, width = image.shape[:2]
        header = json.dumps({"view_id": view_id, "height": height,
                             "width": width}).encode()
        self.process.stdin.write(struct.pack("!I", len(header)) + header + image.tobytes())
        self.process.stdin.flush()
        payload_size = struct.unpack("!I", read_exact(self.process.stdout, 4))[0]
        result = json.loads(read_exact(self.process.stdout, payload_size))
        return result.get("detections", [])

    def close(self):
        if self.process.stdin:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)


def prepare_detection(raw: dict, image: np.ndarray, prompt_map: Dict[str, str],
                      prompt_order: Dict[str, int]) -> dict:
    prompt = raw["label"]
    if prompt not in prompt_map:
        raise ValueError(f"Worker returned unknown prompt: {prompt}")
    encoded = base64.b64decode(raw["mask_png"])
    mask = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != image.shape[:2]:
        raise ValueError("Worker returned an invalid mask")
    mask = (mask > 0).astype(np.uint8) * 255
    bbox = clip_bbox(raw["bbox_xyxy"], image.shape[1], image.shape[0])
    return {"prompt": prompt, "canonical_label": prompt_map[prompt],
            "confidence": float(raw["confidence"]), "bbox": bbox, "mask": mask,
            "mask_area_pixels": int(np.count_nonzero(mask)),
            "prompt_index": prompt_order[prompt]}


def validate_dataset(dataset: Path, views: Sequence[dict], observations: Sequence[dict],
                     statuses: Dict[str, dict], sam3_config: dict) -> dict:
    view_ids = {view["view_id"] for view in views}
    if len(view_ids) != len(views):
        raise ValueError("Duplicate view_id in views.jsonl")
    if set(statuses) != view_ids:
        raise ValueError("Every View must have a SAM3 status")
    allowed_statuses = {"processed", "no_detection", "failed"}
    if any(item["status"] not in allowed_statuses for item in statuses.values()):
        raise ValueError("Invalid View status")
    observation_ids = set()
    expected_prompts = canonical_prompt_map({"sam3": sam3_config})
    for record in observations:
        if record["observation_id"] in observation_ids:
            raise ValueError(f"Duplicate observation_id: {record['observation_id']}")
        observation_ids.add(record["observation_id"])
        if record["view_id"] not in view_ids or record["prompt"] not in expected_prompts:
            raise ValueError("Observation references an unknown View or prompt")
        if record["canonical_label"] != expected_prompts[record["prompt"]]:
            raise ValueError("Canonical label does not match prompt config")
        if record["projection_status"] != "pending":
            raise ValueError("M3 observations must have projection_status=pending")
        if record["sam_confidence"] < float(sam3_config["confidence_threshold"]):
            raise ValueError("Observation confidence is below configured threshold")
        for path_key in ("mask_path", "crop_path", "context_crop_path", "overlay_path"):
            path = dataset / record[path_key]
            if not path.is_file():
                raise FileNotFoundError(path)
        mask = cv2.imread(str(dataset / record["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None or not set(np.unique(mask)).issubset({0, 255}):
            raise ValueError(f"Mask is not binary: {record['mask_path']}")
    return {"valid": True, "view_records": len(views),
            "observation_records": len(observations), "paths_checked": len(observations) * 4}


def build_report(dataset_id: str, views: Sequence[dict], observations: Sequence[dict],
                 statuses: Dict[str, dict], fingerprint: str, sam3_config: dict,
                 started_at: str, complete: bool, validation=None) -> dict:
    raw_status_counts = Counter(item["status"] for item in statuses.values())
    status_counts = {name: raw_status_counts.get(name, 0)
                     for name in ("processed", "no_detection", "failed")}
    label_counts = Counter(item["canonical_label"] for item in observations)
    prompt_counts = Counter(item["prompt"] for item in observations)
    confidence_by_label = defaultdict(list)
    confidence_by_prompt = defaultdict(list)
    for observation in observations:
        confidence_by_label[observation["canonical_label"]].append(observation["sam_confidence"])
        confidence_by_prompt[observation["prompt"]].append(observation["sam_confidence"])
    def confidence_summary(grouped):
        return {name: {"count": len(values), "minimum": min(values),
                       "mean": sum(values) / len(values), "maximum": max(values)}
                for name, values in grouped.items()}
    return {
        "dataset_id": dataset_id, "stage": "M3_sam3_offline",
        "status": "complete" if complete else "partial",
        "started_at": started_at, "updated_at": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": fingerprint,
        "model": {key: sam3_config[key] for key in
                  ("model_name", "model_version", "model_sha256")},
        "prompt_version": sam3_config["prompt_version"],
        "confidence_threshold": sam3_config["confidence_threshold"],
        "views_total": len(views), "view_status_counts": status_counts,
        "observations_total": len(observations),
        "detections_by_prompt": dict(prompt_counts),
        "detections_by_label": dict(label_counts),
        "confidence_by_prompt": confidence_summary(confidence_by_prompt),
        "confidence_by_label": confidence_summary(confidence_by_label),
        "views": [statuses[view["view_id"]] for view in views if view["view_id"] in statuses],
        "validation": validation,
    }


def load_pipeline_config(dataset: Path, prompt_config: Path = None) -> dict:
    path = dataset / "pipeline_config.yaml"
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if "sam3" not in config:
        raise ValueError("pipeline_config.yaml has no sam3 section")
    configured_path = prompt_config or config["sam3"].get("prompt_config")
    if configured_path is None:
        configured_path = (Path(get_package_share_directory("conceptgraphs_bringup"))
                           / "config/sam3_prompts.yaml")
    else:
        configured_path = resolve_config_path(configured_path, dataset)
    shared = load_sam3_prompt_config(configured_path)
    config["sam3"]["prompts"] = shared["prompts"]
    config["sam3"]["prompt_version"] = shared["prompt_version"]
    config["sam3"]["prompt_config_path"] = shared["prompt_config_path"]
    canonical_prompt_map(config)
    return config


def clean_m3_outputs(dataset: Path):
    for path in (dataset / "observations", dataset / "previews/sam3_overlays"):
        if path.exists():
            shutil.rmtree(path)
    report = dataset / "reports/sam3_report.json"
    if report.exists():
        report.unlink()


def run(dataset: Path, limit: int = None, start_index: int = 0, force: bool = False,
        prompt_config: Path = None) -> dict:
    dataset = dataset.resolve()
    config = load_pipeline_config(dataset, prompt_config)
    sam3 = config["sam3"]
    views_path = dataset / "keyframes/views.jsonl"
    views = read_jsonl(views_path)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest["view_count"] != len(views):
        raise ValueError("manifest view_count does not match views.jsonl")
    for view in views:
        rgb_path = dataset / view["rgb_path"]
        if not rgb_path.is_file():
            raise FileNotFoundError(rgb_path)

    observations_path = dataset / "observations/observations.jsonl"
    report_path = dataset / "reports/sam3_report.json"
    if force:
        clean_m3_outputs(dataset)
    fingerprint = config_fingerprint(sam3)
    observations = read_jsonl(observations_path) if observations_path.exists() else []
    statuses = {}
    started_at = datetime.now(timezone.utc).isoformat()
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("config_fingerprint") != fingerprint:
            raise ValueError("SAM3 config changed; rerun with --force")
        statuses = {item["view_id"]: item for item in previous.get("views", [])}
        started_at = previous.get("started_at", started_at)

    output_dirs = [dataset / "observations/masks", dataset / "observations/crops",
                   dataset / "observations/context_crops", dataset / "previews/sam3_overlays"]
    for directory in output_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    worker_script = Path(get_package_share_directory("conceptgraphs_bringup")) / "scripts/sam3_inference_worker.py"
    prompt_map = canonical_prompt_map(config)
    prompt_order = {item["text"]: index for index, item in enumerate(sam3["prompts"])}
    selected_views = views[start_index:(None if limit is None else start_index + limit)]
    worker = None
    run_started = time.monotonic()
    try:
        for ordinal, view in enumerate(selected_views, start_index + 1):
            previous_status = statuses.get(view["view_id"], {}).get("status")
            if previous_status in {"processed", "no_detection"}:
                print(f"[{ordinal}/{len(views)}] {view['view_id']}: resume skip ({previous_status})", flush=True)
                continue
            view_started = time.monotonic()
            try:
                image = cv2.imread(str(dataset / view["rgb_path"]), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Cannot read RGB: {view['rgb_path']}")
                if worker is None:
                    worker = Sam3Worker(config, worker_script)
                raw_detections = worker.infer(view["view_id"], image)
                prepared = [prepare_detection(item, image, prompt_map, prompt_order)
                            for item in raw_detections]
                prepared.sort(key=lambda item: (item["prompt_index"], item["bbox"],
                                                -item["confidence"]))
                overlay_rel = f"previews/sam3_overlays/{view['view_id']}.jpg"
                overlay = render_overlay(image, prepared, prompt_order)
                atomic_image(dataset / overlay_rel, overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
                new_records = []
                for detection_index, detection in enumerate(prepared, 1):
                    obs_id = observation_id(view["view_id"], detection_index)
                    mask_rel = f"observations/masks/{obs_id}.png"
                    crop_rel = f"observations/crops/{obs_id}.jpg"
                    context_rel = f"observations/context_crops/{obs_id}.jpg"
                    bbox = detection["bbox"]
                    context = context_bbox(bbox, image.shape[1], image.shape[0],
                                           float(sam3["context_padding_ratio"]))
                    atomic_image(dataset / mask_rel, detection["mask"])
                    atomic_image(dataset / crop_rel,
                                 masked_crop(image, detection["mask"], bbox),
                                 [cv2.IMWRITE_JPEG_QUALITY, 95])
                    x1, y1, x2, y2 = context
                    atomic_image(dataset / context_rel, image[y1:y2, x1:x2],
                                 [cv2.IMWRITE_JPEG_QUALITY, 95])
                    new_records.append({
                        "observation_id": obs_id, "view_id": view["view_id"],
                        "dataset_id": view["dataset_id"], "greenhouse_id": view["greenhouse_id"],
                        "aisle_id": view["aisle_id"], "timestamp_ns": view["timestamp_ns"],
                        "prompt": detection["prompt"],
                        "canonical_label": detection["canonical_label"],
                        "sam_confidence": detection["confidence"],
                        "bbox_2d_xyxy": bbox, "mask_area_pixels": detection["mask_area_pixels"],
                        "mask_path": mask_rel, "crop_path": crop_rel,
                        "context_crop_path": context_rel, "overlay_path": overlay_rel,
                        "model_name": sam3["model_name"], "model_version": sam3["model_version"],
                        "model_sha256": sam3["model_sha256"],
                        "prompt_version": sam3["prompt_version"],
                        "projection_status": "pending",
                    })
                # Replace this View in the checkpoint only after every artifact
                # for the View has been written successfully.
                observations = [item for item in observations if item["view_id"] != view["view_id"]]
                observations.extend(new_records)
                observations.sort(key=lambda item: item["observation_id"])
                status = "processed" if prepared else "no_detection"
                statuses[view["view_id"]] = {
                    "view_id": view["view_id"], "status": status,
                    "observation_count": len(prepared), "error": None,
                    "duration_sec": time.monotonic() - view_started,
                }
                print(f"[{ordinal}/{len(views)}] {view['view_id']}: {status}, detections={len(prepared)}", flush=True)
            except Exception as error:
                statuses[view["view_id"]] = {
                    "view_id": view["view_id"], "status": "failed",
                    "observation_count": 0, "error": f"{type(error).__name__}: {error}",
                    "duration_sec": time.monotonic() - view_started,
                }
                print(f"[{ordinal}/{len(views)}] {view['view_id']}: FAILED: {error}", flush=True)
                if worker is not None:
                    worker.close()
                    worker = None
            atomic_jsonl(observations_path, observations)
            complete = len(statuses) == len(views) and all(
                item["status"] in {"processed", "no_detection"} for item in statuses.values())
            report = build_report(manifest["dataset_id"], views, observations, statuses,
                                  fingerprint, sam3, started_at, complete)
            report["last_run_elapsed_sec"] = time.monotonic() - run_started
            atomic_json(report_path, report)
    finally:
        if worker is not None:
            worker.close()

    complete = len(statuses) == len(views) and all(
        item["status"] in {"processed", "no_detection"} for item in statuses.values())
    validation = validate_dataset(dataset, views, observations, statuses, sam3) if complete else None
    report = build_report(manifest["dataset_id"], views, observations, statuses,
                          fingerprint, sam3, started_at, complete, validation)
    report["last_run_elapsed_sec"] = time.monotonic() - run_started
    atomic_json(report_path, report)
    if complete:
        manifest.setdefault("stages", {})["sam3_offline"] = {
            "status": "complete",
            "observations_file": "observations/observations.jsonl",
            "observations_json_file": "observations/observations.json",
            "report_file": "reports/sam3_report.json",
            "observation_count": len(observations),
            "prompt_version": sam3["prompt_version"],
            "model_version": sam3["model_version"],
        }
        atomic_json(dataset / "manifest.json", manifest)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_dataset())
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prompt-config", type=Path, help=(
        "Shared SAM3 prompt YAML; defaults to config/sam3_prompts.yaml"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = run(args.dataset, args.limit, args.start_index, args.force,
                 args.prompt_config)
    summary = {key: report[key] for key in (
        "status", "views_total", "view_status_counts", "observations_total",
        "detections_by_prompt", "detections_by_label", "confidence_by_prompt",
        "confidence_by_label",
        "validation", "last_run_elapsed_sec")}
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
