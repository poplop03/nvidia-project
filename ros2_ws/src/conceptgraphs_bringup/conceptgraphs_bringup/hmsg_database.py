"""Build and validate the M7 file-backed HMSG SQLite database."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
TABLES = (
    "datasets", "greenhouses", "aisles", "views", "observations", "objects",
    "object_observations", "view_object_edges", "captions",
    "traversability_assessments", "embedding_indexes",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_records(path: Path):
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON array: {path}")
    return value


def records(dataset: Path, stem: str):
    jsonl = dataset / f"{stem}.jsonl"
    plain = dataset / f"{stem}.json"
    path = jsonl if jsonl.exists() else plain
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset artifact: {jsonl} or {plain}")
    return read_records(path)


def relative_path(dataset: Path, value: str | None, *, require=True):
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(dataset.resolve())
        except ValueError as exc:
            raise ValueError(f"Path is outside dataset root: {value}") from exc
    normalized = Path(*[part for part in path.parts if part not in ("", ".")])
    if ".." in normalized.parts:
        raise ValueError(f"Unsafe relative path: {value}")
    if require and not (dataset / normalized).is_file():
        raise FileNotFoundError(f"Missing referenced file: {normalized}")
    return normalized.as_posix()


def schema_source():
    source_tree = Path(__file__).resolve().parents[1] / "sql" / "001_initial_hmsg.sql"
    if source_tree.exists():
        return source_tree
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory("conceptgraphs_bringup")) / "sql" / "001_initial_hmsg.sql"


def prepare_database_files(dataset: Path):
    out = dataset / "database"
    migrations = out / "migrations"
    migrations.mkdir(parents=True, exist_ok=True)
    source = schema_source()
    if not source.exists():
        raise FileNotFoundError(f"Installed SQL migration is missing: {source}")
    sql = source.read_text()
    (out / "schema.sql").write_text(sql)
    shutil.copyfile(source, migrations / source.name)
    return out / "greenhouse_hmsg.sqlite", sql


def connect(path: Path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    return db


def upsert(db, table, key_columns, row, stats):
    where = " AND ".join(f"{x}=?" for x in key_columns)
    keys = tuple(row[x] for x in key_columns)
    old = db.execute(f"SELECT * FROM {table} WHERE {where}", keys).fetchone()
    if old is not None and all(old[k] == v for k, v in row.items()):
        stats["unchanged"] += 1
        return
    columns = list(row)
    placeholders = ",".join("?" for _ in columns)
    conflict = ",".join(key_columns)
    updates = ",".join(f"{x}=excluded.{x}" for x in columns if x not in key_columns)
    action = f"DO UPDATE SET {updates}" if updates else "DO NOTHING"
    sql = (f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders}) "
           f"ON CONFLICT({conflict}) {action}")
    db.execute(sql, tuple(row[x] for x in columns))
    stats["updated" if old is not None else "inserted"] += 1


def caption_fields(record):
    parsed = record.get("parsed") or {}
    objects = parsed.get("physical_objects") or []
    item = objects[0] if objects else {}
    traversal = item.get("object_traversability") or {}
    cues = parsed.get("traversability_cues") or {}
    return {
        "category": item.get("category"), "color": item.get("color"),
        "material": item.get("material"),
        "semantic_class": traversal.get("semantic_class"),
        "path_assessment": cues.get("vlm_path_assessment"),
        "confidence": traversal.get("confidence", cues.get("confidence")),
        "scene_summary": parsed.get("scene_summary"),
    }


def validate_sources(dataset, views, observations, objects, edges, captions, assessments, embedding_meta):
    errors = []
    view_ids = {x["view_id"] for x in views}
    observation_ids = {x["observation_id"] for x in observations}
    object_ids = {x["object_id"] for x in objects}
    if len(view_ids) != len(views) or len(observation_ids) != len(observations) or len(object_ids) != len(objects):
        errors.append("duplicate stable IDs in source artifacts")
    for view in views:
        for key in ("rgb_path", "depth_path", "camera_info_path"):
            try: relative_path(dataset, view.get(key))
            except Exception as exc: errors.append(str(exc))
    for obs in observations:
        if obs["view_id"] not in view_ids: errors.append(f"unknown View: {obs['view_id']}")
        for key in ("mask_path", "crop_path", "context_crop_path", "overlay_path"):
            try: relative_path(dataset, obs.get(key))
            except Exception as exc: errors.append(str(exc))
        if obs.get("projection_status") == "valid" and obs.get("frame_id") != "camera_init":
            errors.append(f"valid Observation not in camera_init: {obs['observation_id']}")
        if obs.get("projection_status") == "rejected" and not obs.get("rejection_reason"):
            errors.append(f"rejected Observation lacks reason: {obs['observation_id']}")
    for obj in objects:
        if obj.get("representative_view_id") not in view_ids: errors.append(f"unknown representative View: {obj['object_id']}")
        if obj.get("representative_observation_id") not in observation_ids: errors.append(f"unknown representative Observation: {obj['object_id']}")
        for oid in obj.get("observation_ids", []):
            if oid not in observation_ids: errors.append(f"unknown Object Observation: {oid}")
    for edge in edges:
        if edge["source_id"] not in view_ids or edge["target_id"] not in object_ids or edge["observation_id"] not in observation_ids:
            errors.append(f"orphan graph edge: {edge['edge_id']}")
    for owner_type, rows in captions.items():
        key = f"{owner_type}_id"; allowed = object_ids if owner_type == "object" else view_ids
        for row in rows:
            if row[key] not in allowed: errors.append(f"orphan {owner_type} caption: {row[key]}")
            for value in row.get("input_image_paths", []):
                try: relative_path(dataset, value)
                except Exception as exc: errors.append(str(exc))
    for row in assessments:
        if row["view_id"] not in view_ids: errors.append(f"orphan assessment: {row['view_id']}")
    dimension = embedding_meta["dimension"]
    seen_vectors = set()
    for filename, info in embedding_meta["files"].items():
        npy = dataset / "embeddings" / filename
        if not npy.is_file(): errors.append(f"missing embedding array: {npy}"); continue
        array = np.load(npy, mmap_mode="r")
        if list(array.shape) != info["shape"] or array.shape[1] != dimension:
            errors.append(f"embedding shape mismatch: {filename}")
        for entry in info["entries"]:
            idx = entry["vector_index"]
            if idx < 0 or idx >= array.shape[0] or (filename, idx) in seen_vectors:
                errors.append(f"invalid/duplicate vector index: {filename}:{idx}")
            seen_vectors.add((filename, idx))
            owner = entry.get("view_id") if filename.startswith("view_") else entry.get("object_id")
            allowed = view_ids if filename.startswith("view_") else object_ids
            if owner not in allowed: errors.append(f"unknown embedding owner: {filename}:{idx}")
            for value in entry.get("input_paths", []):
                try: relative_path(dataset, value)
                except Exception as exc: errors.append(str(exc))
    if errors:
        raise ValueError("Source validation failed:\n- " + "\n- ".join(errors[:50]))


def load_sources(dataset):
    manifest = json.loads((dataset / "manifest.json").read_text())
    views = records(dataset, "keyframes/views")
    observations = records(dataset, "observations/observations_3d")
    objects = records(dataset, "objects/objects")
    edges = records(dataset, "graph/edges")
    captions = {kind: records(dataset, f"semantics/{kind}_captions") for kind in ("object", "view")}
    assessments = records(dataset, "semantics/traversability_assessments")
    embedding_meta = json.loads((dataset / "embeddings/index_metadata.json").read_text())
    validate_sources(dataset, views, observations, objects, edges, captions, assessments, embedding_meta)
    return manifest, views, observations, objects, edges, captions, assessments, embedding_meta


def database_counts(db):
    return {table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in TABLES}


def validate_database(db, dataset):
    foreign = [dict(x) for x in db.execute("PRAGMA foreign_key_check")]
    integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    errors = []
    for table, columns in {
        "views": ("rgb_path", "depth_path", "camera_info_path"),
        "observations": ("mask_path", "crop_path", "context_crop_path", "overlay_path"),
        "captions": ("raw_response_path",), "embedding_indexes": ("npy_path",),
    }.items():
        for row in db.execute(f"SELECT * FROM {table}"):
            for column in columns:
                if row[column] and not (dataset / row[column]).is_file():
                    errors.append(f"missing {table}.{column}: {row[column]}")
    for row in db.execute("SELECT owner_type, owner_id FROM captions"):
        table, key = ("objects", "object_id") if row["owner_type"] == "object" else ("views", "view_id")
        if not db.execute(f"SELECT 1 FROM {table} WHERE {key}=?", (row["owner_id"],)).fetchone():
            errors.append(f"orphan caption owner: {row['owner_type']}:{row['owner_id']}")
    for row in db.execute("SELECT * FROM embedding_indexes"):
        table, key = ("views", "view_id") if row["owner_type"] == "view" else ("objects", "object_id")
        if not db.execute(f"SELECT 1 FROM {table} WHERE {key}=?", (row["owner_id"],)).fetchone():
            errors.append(f"orphan embedding owner: {row['embedding_id']}")
        else:
            arr = np.load(dataset / row["npy_path"], mmap_mode="r")
            if row["vector_index"] >= arr.shape[0] or row["dimension"] != arr.shape[1]:
                errors.append(f"invalid embedding index: {row['embedding_id']}")
    return {"valid": integrity == "ok" and not foreign and not errors,
            "integrity_check": integrity, "foreign_key_errors": foreign,
            "reference_or_path_errors": errors, "row_counts": database_counts(db)}


def build_database(dataset: Path, rebuild=False):
    dataset = dataset.resolve()
    started = time.monotonic()
    sources = load_sources(dataset)
    manifest, views, observations, objects, edges, captions, assessments, embedding_meta = sources
    db_path, sql = prepare_database_files(dataset)
    if rebuild and db_path.exists(): db_path.unlink()
    db = connect(db_path)
    stats = {"inserted": 0, "updated": 0, "unchanged": 0}
    try:
        db.executescript(sql)
        with db:
            existing_schema = db.execute("SELECT applied_at FROM schema_versions WHERE version=?", (SCHEMA_VERSION,)).fetchone()
            applied_at = existing_schema[0] if existing_schema else utc_now()
            upsert(db, "schema_versions", ("version",), {"version": SCHEMA_VERSION, "applied_at": applied_at}, stats)
            upsert(db, "datasets", ("dataset_id",), {
                "dataset_id": manifest["dataset_id"], "schema_version": manifest["schema_version"],
                "frame_id": manifest["frame_id"], "source_bag": manifest.get("source_bag"),
                "manifest_json": json_text(manifest), "dataset_root": "."}, stats)
            upsert(db, "greenhouses", ("greenhouse_id",), {"greenhouse_id": manifest["greenhouse_id"], "dataset_id": manifest["dataset_id"]}, stats)
            for aisle_id in manifest["aisle_ids"]:
                upsert(db, "aisles", ("aisle_id",), {"aisle_id": aisle_id, "greenhouse_id": manifest["greenhouse_id"]}, stats)
            for x in views:
                upsert(db, "views", ("view_id",), {
                    "view_id": x["view_id"], "dataset_id": x["dataset_id"], "greenhouse_id": x["greenhouse_id"], "aisle_id": x["aisle_id"],
                    "timestamp_ns": x["timestamp_ns"], "frame_id": x["frame_id"], "camera_frame_id": x.get("camera_frame_id"),
                    "rgb_path": relative_path(dataset, x["rgb_path"]), "depth_path": relative_path(dataset, x["depth_path"]),
                    "camera_info_path": relative_path(dataset, x["camera_info_path"]), "pose_json": json_text(x["pose"]),
                    "body_pose_json": json_text(x.get("body_pose")), "aisle_coordinate_s": x.get("aisle_coordinate_s"),
                    "aisle_section": x.get("aisle_section"), "travel_pass": x.get("travel_pass"),
                    "processing_status": x.get("processing_status"), "source_json": json_text(x)}, stats)
            for x in observations:
                upsert(db, "observations", ("observation_id",), {
                    "observation_id": x["observation_id"], "view_id": x["view_id"], "dataset_id": x["dataset_id"], "aisle_id": x["aisle_id"],
                    "timestamp_ns": x.get("timestamp_ns"), "prompt": x.get("prompt"), "canonical_label": x["canonical_label"],
                    "sam_confidence": x.get("sam_confidence"), "bbox_2d_json": json_text(x.get("bbox_2d_xyxy")),
                    "mask_path": relative_path(dataset, x.get("mask_path")), "crop_path": relative_path(dataset, x.get("crop_path")),
                    "context_crop_path": relative_path(dataset, x.get("context_crop_path")), "overlay_path": relative_path(dataset, x.get("overlay_path")),
                    "projection_status": x["projection_status"], "frame_id": x.get("frame_id"), "centroid_3d_json": json_text(x.get("centroid_3d")),
                    "bbox_3d_min_json": json_text(x.get("bbox_3d_min")), "bbox_3d_max_json": json_text(x.get("bbox_3d_max")),
                    "median_depth_m": x.get("median_depth_m"), "valid_depth_ratio": x.get("valid_depth_ratio"), "point_count": x.get("point_count"),
                    "rejection_reason": x.get("rejection_reason"), "model_version": x.get("model_version"), "prompt_version": x.get("prompt_version"),
                    "source_json": json_text(x)}, stats)
            for x in objects:
                upsert(db, "objects", ("object_id",), {
                    "object_id": x["object_id"], "dataset_id": x["dataset_id"], "greenhouse_id": x["greenhouse_id"], "aisle_id": x["aisle_id"],
                    "canonical_label": x["canonical_label"], "frame_id": x["frame_id"], "centroid_3d_json": json_text(x["centroid_3d"]),
                    "bbox_3d_min_json": json_text(x.get("bbox_3d_min")), "bbox_3d_max_json": json_text(x.get("bbox_3d_max")),
                    "position_quality": x.get("position_quality"), "association_status": x.get("association_status"),
                    "representative_view_id": x.get("representative_view_id"), "representative_observation_id": x.get("representative_observation_id"),
                    "observation_count": x.get("observation_count"), "cluster_diameter_m": x.get("cluster_diameter_m"), "source_json": json_text(x)}, stats)
                for oid in x.get("observation_ids", []):
                    upsert(db, "object_observations", ("object_id", "observation_id"), {"object_id": x["object_id"], "observation_id": oid}, stats)
            for x in edges:
                upsert(db, "view_object_edges", ("edge_id",), {"edge_id": x["edge_id"], "view_id": x["source_id"], "object_id": x["target_id"],
                    "observation_id": x["observation_id"], "edge_type": x["edge_type"], "source_json": json_text(x)}, stats)
            for kind, rows in captions.items():
                key = f"{kind}_id"
                for x in rows:
                    fields = caption_fields(x); owner = x[key]
                    upsert(db, "captions", ("caption_id",), {"caption_id": f"{kind}:{owner}", "owner_type": kind, "owner_id": owner,
                        "status": x["status"], "model_tag": x.get("model_tag"), "model_digest": x.get("model_digest"), "prompt_version": x.get("prompt_version"),
                        **fields, "parsed_json": json_text(x.get("parsed")), "raw_response_path": relative_path(dataset, x.get("raw_response_path")),
                        "input_image_paths_json": json_text([relative_path(dataset, p) for p in x.get("input_image_paths", [])]),
                        "input_ids_json": json_text(x.get("input_ids")), "source_json": json_text(x)}, stats)
            for x in assessments:
                upsert(db, "traversability_assessments", ("view_id",), {"view_id": x["view_id"], "status": x["status"], "passability": x.get("passability"),
                    "can_robot_pass": x.get("can_robot_pass"), "blocking_object_ids_json": json_text(x.get("blocking_object_ids", [])),
                    "traversable_object_ids_json": json_text(x.get("traversable_object_ids", [])), "confidence": x.get("confidence"),
                    "model_tag": x.get("model_tag"), "model_digest": x.get("model_digest"), "prompt_version": x.get("prompt_version"), "source_json": json_text(x)}, stats)
            for filename, info in embedding_meta["files"].items():
                npy_path = f"embeddings/{filename}"
                for x in info["entries"]:
                    owner_type = "view" if "view_id" in x and filename.startswith("view_") else "object"
                    owner_id = x["view_id"] if owner_type == "view" else x["object_id"]
                    eid = f"{filename}:{x['vector_index']}"
                    upsert(db, "embedding_indexes", ("embedding_id",), {"embedding_id": eid, "owner_type": owner_type, "owner_id": owner_id,
                        "entry_type": x["entry_type"], "npy_path": npy_path, "vector_index": x["vector_index"], "dimension": embedding_meta["dimension"],
                        "dtype": info["dtype"], "model_name": embedding_meta["model_name"], "model_version": embedding_meta["model_version"],
                        "model_sha256": embedding_meta.get("model_sha256"),
                        "input_paths_json": json_text([relative_path(dataset, p) for p in x.get("input_paths", [])])}, stats)
        validation = validate_database(db, dataset)
        if not validation["valid"]: raise ValueError(f"Database validation failed: {validation}")
        report = {"stage": "hmsg_database", "schema_version": SCHEMA_VERSION, "database_path": "database/greenhouse_hmsg.sqlite",
                  "database_size_bytes": db_path.stat().st_size, "duration_sec": time.monotonic()-started,
                  "changes": stats, "validation": validation, "completed_at": utc_now()}
        report_path = dataset / "reports/database_import_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        return report
    finally:
        db.close()


def validate_only(dataset: Path):
    dataset = dataset.resolve(); db_path = dataset / "database/greenhouse_hmsg.sqlite"
    if not db_path.exists(): raise FileNotFoundError(db_path)
    db = connect(db_path)
    try: return validate_database(db, dataset)
    finally: db.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.rebuild and args.validate_only: parser.error("--rebuild and --validate-only are mutually exclusive")
    try:
        result = validate_only(args.dataset) if args.validate_only else build_database(args.dataset, args.rebuild)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("valid", result.get("validation", {}).get("valid")) else 1
    except Exception as exc:
        print(f"HMSG database error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
