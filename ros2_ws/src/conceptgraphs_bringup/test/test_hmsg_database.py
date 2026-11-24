import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from conceptgraphs_bringup.hmsg_database import build_database, validate_only
from conceptgraphs_bringup.hmsg_query import open_database, run_query


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_dataset(tmp_path):
    root = tmp_path / "dataset"
    files = [
        "keyframes/rgb/v1.jpg", "keyframes/depth/v1.png", "keyframes/camera_info/v1.json",
        "observations/masks/o1.png", "observations/crops/o1.jpg",
        "observations/context_crops/o1.jpg", "previews/sam3_overlays/v1.jpg",
        "semantics/raw_responses/object_obj1.json", "semantics/raw_responses/view_v1.json",
    ]
    for name in files:
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"x")
    manifest = {"schema_version": "1.0", "dataset_id": "d1", "greenhouse_id": "g1",
                "aisle_ids": ["a1"], "frame_id": "camera_init", "source_bag": "bag"}
    view = {"view_id": "v1", "dataset_id": "d1", "greenhouse_id": "g1", "aisle_id": "a1",
            "timestamp_ns": 1, "frame_id": "camera_init", "camera_frame_id": "camera",
            "rgb_path": "keyframes/rgb/v1.jpg", "depth_path": "keyframes/depth/v1.png",
            "camera_info_path": "keyframes/camera_info/v1.json", "pose": {"translation_xyz": [0,0,0]},
            "travel_pass": "forward", "processing_status": "extracted"}
    observation = {"observation_id": "o1", "view_id": "v1", "dataset_id": "d1", "aisle_id": "a1",
                   "canonical_label": "box", "projection_status": "valid", "frame_id": "camera_init",
                   "centroid_3d": [1,2,3], "bbox_3d_min": [0,1,2], "bbox_3d_max": [2,3,4],
                   "mask_path": "observations/masks/o1.png", "crop_path": "observations/crops/o1.jpg",
                   "context_crop_path": "observations/context_crops/o1.jpg", "overlay_path": "previews/sam3_overlays/v1.jpg"}
    obj = {"object_id": "obj1", "dataset_id": "d1", "greenhouse_id": "g1", "aisle_id": "a1",
           "canonical_label": "box", "frame_id": "camera_init", "centroid_3d": [1,2,3],
           "observation_ids": ["o1"], "representative_view_id": "v1",
           "representative_observation_id": "o1", "observation_count": 1}
    edge = {"edge_id": "e1", "edge_type": "OBSERVES", "source_id": "v1", "target_id": "obj1", "observation_id": "o1"}
    parsed = {"scene_summary": "box", "physical_objects": [{"category": "box", "color": "red", "material": "plastic",
              "object_traversability": {"semantic_class": "non_traversable_obstacle", "confidence": .9}}],
              "traversability_cues": {"vlm_path_assessment": "passable", "confidence": .8}}
    object_caption = {"object_id": "obj1", "status": "complete", "model_tag": "m", "prompt_version": "p",
                      "parsed": parsed, "raw_response_path": "semantics/raw_responses/object_obj1.json",
                      "input_image_paths": ["observations/crops/o1.jpg"], "input_ids": {"object_id": "obj1"}}
    view_caption = {"view_id": "v1", "status": "complete", "model_tag": "m", "prompt_version": "p",
                    "parsed": parsed, "raw_response_path": "semantics/raw_responses/view_v1.json",
                    "input_image_paths": ["keyframes/rgb/v1.jpg"], "input_ids": {"view_id": "v1"}}
    assessment = {"view_id": "v1", "status": "complete", "passability": "passable", "can_robot_pass": True,
                  "blocking_object_ids": [], "traversable_object_ids": [], "confidence": .8}
    write_json(root / "manifest.json", manifest)
    for name, value in [("keyframes/views", [view]), ("observations/observations_3d", [observation]),
                        ("objects/objects", [obj]), ("graph/edges", [edge]),
                        ("semantics/object_captions", [object_caption]), ("semantics/view_captions", [view_caption]),
                        ("semantics/traversability_assessments", [assessment])]:
        write_json(root / f"{name}.json", value)
    emb = root / "embeddings"; emb.mkdir()
    np.save(emb / "view_embeddings.npy", np.ones((1, 512), dtype=np.float32))
    np.save(emb / "object_embeddings.npy", np.ones((1, 512), dtype=np.float32))
    meta = {"model_name": "ViT-B-16", "model_version": "test", "model_sha256": "abc", "dimension": 512,
            "files": {"view_embeddings.npy": {"shape": [1,512], "dtype": "float32", "entries": [
                {"entry_type": "view_rgb", "view_id": "v1", "input_paths": [str(root / "keyframes/rgb/v1.jpg")], "vector_index": 0}]},
                "object_embeddings.npy": {"shape": [1,512], "dtype": "float32", "entries": [
                {"entry_type": "object_crop_mean", "object_id": "obj1", "input_paths": [str(root / "observations/crops/o1.jpg")], "vector_index": 0}]}}}
    write_json(emb / "index_metadata.json", meta)
    return root


def test_import_schema_queries_and_idempotency(tmp_path):
    root = fixture_dataset(tmp_path)
    first = build_database(root)
    assert first["validation"]["valid"]
    assert first["validation"]["row_counts"]["views"] == 1
    second = build_database(root)
    assert second["changes"]["inserted"] == 0
    assert second["changes"]["updated"] == 0
    assert validate_only(root)["valid"]
    db = open_database(root)
    try:
        assert run_query(db, "views_for_object", "obj1")[0]["view_id"] == "v1"
        assert run_query(db, "observations_for_object", "obj1")[0]["observation_id"] == "o1"
        assert json.loads(db.execute("SELECT parsed_json FROM captions WHERE caption_id='object:obj1'").fetchone()[0]) == json.loads(json.dumps({
            "scene_summary": "box", "physical_objects": [{"category": "box", "color": "red", "material": "plastic", "object_traversability": {"semantic_class": "non_traversable_obstacle", "confidence": .9}}], "traversability_cues": {"vlm_path_assessment": "passable", "confidence": .8}}))
    finally:
        db.close()


def test_invalid_reference_does_not_create_database(tmp_path):
    root = fixture_dataset(tmp_path)
    objects = json.loads((root / "objects/objects.json").read_text())
    objects[0]["representative_view_id"] = "missing"
    write_json(root / "objects/objects.json", objects)
    with pytest.raises(ValueError):
        build_database(root)
    assert not (root / "database/greenhouse_hmsg.sqlite").exists()


def test_foreign_keys_are_enabled(tmp_path):
    root = fixture_dataset(tmp_path); build_database(root)
    db = open_database(root)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO object_observations VALUES ('missing','o1')")
    finally:
        db.close()
