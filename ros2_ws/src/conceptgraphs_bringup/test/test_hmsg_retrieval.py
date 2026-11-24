import json
import sqlite3

import numpy as np

from conceptgraphs_bringup.hmsg_database import build_database
from conceptgraphs_bringup.hmsg_retrieval import (
    ambiguity, candidate_rows, parse_intent, resolve_canonical_category, retrieve, rule_parse,
)
from test_hmsg_database import fixture_dataset


def config():
    return {
        "version": "test_retrieval_v1",
        "clip": {"python": "python", "model_name": "ViT-B-16", "device": "cpu"},
        "ambiguity": {"minimum_similarity": .55, "minimum_top1_margin": .08,
                      "minimum_parser_confidence": .75, "verification_top_k": 3},
        "ollama": {"base_url": "http://127.0.0.1:11434", "model": "unused",
                   "temperature": 0},
    }


def test_rule_parser_extracts_filters_and_safety_flag():
    value = rule_parse("Can the robot avoid the red plastic container in aisle_01?")
    assert value["color"] == "red"
    assert value["material"] == "plastic"
    assert value["aisle_id"] == "aisle_01"
    assert value["requires_geometric_verification"] is True


def test_rule_parser_normalizes_unpadded_aisle_number():
    value = rule_parse("Find the wheelbarrow in aisle 3")
    assert value["aisle_id"] == "aisle_03"


def test_distance_from_head_is_not_a_spatial_filter():
    value = rule_parse("How far is the ladder from the aisle head?")
    assert value["spatial_filter"] is None


def test_exact_canonical_category_does_not_match_incidental_caption_text():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript("""
      CREATE TABLE objects (object_id TEXT, canonical_label TEXT, representative_view_id TEXT,
                            representative_observation_id TEXT);
      CREATE TABLE captions (owner_type TEXT, owner_id TEXT, category TEXT, color TEXT,
                             material TEXT, scene_summary TEXT, parsed_json TEXT,
                             semantic_class TEXT, path_assessment TEXT, confidence REAL);
      CREATE TABLE views (view_id TEXT, rgb_path TEXT);
      CREATE TABLE observations (observation_id TEXT, crop_path TEXT);
    """)
    db.executemany("INSERT INTO objects VALUES (?,?,?,?)", [
        ("ladder", "ladder", None, None),
        ("yellow", "yellow_plastic_container", None, None),
    ])
    db.executemany("INSERT INTO captions VALUES ('object',?,?,?,?,?,?,?,?,?)", [
        ("ladder", "ladder", "silver", "metal", "A ladder.", "{}", None, None, .9),
        ("yellow", "container", "yellow", "plastic",
         "A yellow container; a ladder stands nearby under overhead shade.", "{}", None, None, .9),
    ])
    intent = {"target_type": "object", "category": "ladder",
              "category_resolution": "canonical_label_query_match"}
    rows, _ = candidate_rows(db, intent)
    assert [row["object_id"] for row in rows] == ["ladder"]


def test_ambiguity_uses_similarity_and_margin():
    intent = {"confidence": .9, "object_id": None, "view_id": None}
    assert ambiguity([{"similarity": .8}, {"similarity": .79}], intent, config()) == (True, ["small_top1_margin"])
    assert ambiguity([{"similarity": .8}, {"similarity": .6}], intent, config()) == (False, [])
    exact = {"confidence": .9, "object_id": None, "view_id": None,
             "category_resolution": "canonical_label_query_match"}
    assert ambiguity([{"similarity": .1}], exact, config()) == (False, [])


def test_complex_query_uses_gemma_parser(monkeypatch):
    parsed = {"intent": "find", "target_type": "object", "greenhouse_id": None,
              "aisle_id": None, "object_id": None, "view_id": None, "category": None,
              "color": None, "material": None, "spatial_filter": None,
              "traversability_filter": None, "requested_fields": ["id"],
              "confidence": .8, "requires_geometric_verification": False}
    monkeypatch.setattr("conceptgraphs_bringup.hmsg_retrieval.available_ollama",
                        lambda cfg: ("http://local", "gemma", "digest"))
    monkeypatch.setattr("conceptgraphs_bringup.hmsg_retrieval.ollama_json",
                        lambda *args, **kwargs: dict(parsed))
    result = parse_intent("What should I inspect?", None, config(), {"intent_prompt": "parse"})
    assert result["parser"] == "gemma"
    assert result["parser_model_digest"] == "digest"


def test_explicit_database_label_overrides_llm_superclass():
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE objects (canonical_label TEXT)")
    db.executemany("INSERT INTO objects VALUES (?)", [("ladder",), ("wheelbarrow",)])
    intent = {"target_type": "object", "category": "tool"}
    result = resolve_canonical_category("Find the ladder", db, intent)
    assert result["category"] == "ladder"
    assert result["category_resolution"] == "canonical_label_query_match"


def test_retrieval_is_idempotent_and_preserves_geometry(tmp_path, monkeypatch):
    root = fixture_dataset(tmp_path)
    (root / "pipeline_config.yaml").write_text("{}\n")
    build_database(root)
    monkeypatch.setattr("conceptgraphs_bringup.hmsg_retrieval.load_config", lambda dataset: config())
    monkeypatch.setattr("conceptgraphs_bringup.hmsg_retrieval.prompt_config", lambda dataset, cfg: {})
    vector = np.ones(512, dtype=np.float32); vector /= np.linalg.norm(vector)
    one = retrieve(root, "Find red plastic", top_k=5, text_vector=vector,
                   allow_llm=False, allow_verification=False)
    two = retrieve(root, "Find red plastic", top_k=5, text_vector=vector,
                   allow_llm=False, allow_verification=False)
    assert one["query_id"] == two["query_id"]
    assert one["results"][0]["centroid_3d"] == [1, 2, 3]
    records = [json.loads(x) for x in (root / "retrieval/query_results.jsonl").read_text().splitlines()]
    assert len(records) == 1
