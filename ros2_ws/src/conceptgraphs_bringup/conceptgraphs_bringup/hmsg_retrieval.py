"""M8 fast-to-slow retrieval over the M7 HMSG database and M6 CLIP vectors."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import yaml

from .paths import resolve_config_path, resolve_worker_python


INTENT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["intent", "target_type", "greenhouse_id", "aisle_id", "object_id",
                 "view_id", "category", "color", "material", "spatial_filter",
                 "traversability_filter", "requested_fields", "confidence",
                 "requires_geometric_verification"],
    "properties": {
        "intent": {"type": "string"},
        "target_type": {"enum": ["object", "view"]},
        "greenhouse_id": {"type": ["string", "null"]},
        "aisle_id": {"type": ["string", "null"]},
        "object_id": {"type": ["string", "null"]},
        "view_id": {"type": ["string", "null"]},
        "category": {"type": ["string", "null"]},
        "color": {"type": ["string", "null"]},
        "material": {"type": ["string", "null"]},
        "spatial_filter": {"type": ["string", "null"]},
        "traversability_filter": {"type": ["string", "null"]},
        "requested_fields": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "requires_geometric_verification": {"type": "boolean"},
    },
}

VERIFY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["selected_ids", "confidence", "status", "reasoning_summary",
                 "requires_geometric_verification"],
    "properties": {
        "selected_ids": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "status": {"enum": ["verified", "uncertain", "not_found"]},
        "reasoning_summary": {"type": "string"},
        "requires_geometric_verification": {"type": "boolean"},
    },
}


def now():
    return datetime.now(timezone.utc).isoformat()


def stable_id(query, version, top_k):
    value = json.dumps({"query": query.strip(), "version": version, "top_k": top_k}, sort_keys=True)
    return "query_" + hashlib.sha256(value.encode()).hexdigest()[:16]


def load_config(dataset):
    config = yaml.safe_load((dataset / "pipeline_config.yaml").read_text())
    if "retrieval" not in config:
        source = Path(__file__).resolve().parents[1] / "config" / "keyframe_extractor.yaml"
        if not source.exists():
            from ament_index_python.packages import get_package_share_directory
            source = Path(get_package_share_directory("conceptgraphs_bringup")) / "config" / "keyframe_extractor.yaml"
        config = yaml.safe_load(source.read_text())
    return config["retrieval"]


def prompt_config(dataset, cfg):
    path = resolve_config_path(cfg["prompt_config"], dataset)
    data = yaml.safe_load(path.read_text())
    if data["version"] != cfg["version"]:
        raise ValueError("retrieval prompt/config version mismatch")
    return data


def normalize_numbered_id(value, prefix):
    """Normalize human forms such as ``aisle 3`` to ``aisle_03``."""
    if value is None:
        return None
    match = re.fullmatch(rf"{re.escape(prefix)}[_ -]?(\d+)", value.lower().strip())
    if not match:
        return value
    return f"{prefix}_{int(match.group(1)):02d}"


def normalize_intent_ids(intent):
    intent["aisle_id"] = normalize_numbered_id(intent.get("aisle_id"), "aisle")
    intent["greenhouse_id"] = normalize_numbered_id(
        intent.get("greenhouse_id"), "greenhouse"
    )
    return intent


def rule_parse(query):
    q = query.lower().strip()
    object_match = re.search(r"object_\d+", q)
    view_match = re.search(r"view_\d+", q)
    aisle_match = re.search(r"aisle[_ -]?\d+", q)
    greenhouse_match = re.search(r"greenhouse[_ -]?\d+", q)
    color_aliases = {"red": ("red", "đỏ"), "yellow": ("yellow", "vàng"), "blue": ("blue", "xanh dương"),
                     "green": ("green", "xanh lá"), "orange": ("orange", "cam"), "black": ("black", "đen"), "white": ("white", "trắng")}
    colors = next((color for color, aliases in color_aliases.items() if any(x in q for x in aliases)), None)
    materials = next((x for x in ("plastic", "metal", "wood", "fabric", "glass") if x in q), None)
    category = None
    if any(x in q for x in ("container", "crate", "box", "thùng", "sọt")):
        category = "plastic_container" if materials == "plastic" or "plastic" in q else "container"
    target = "view" if view_match or any(x in q for x in ("view", "image", "frame", "ảnh")) else "object"
    navigation = any(x in q for x in ("pass", "avoid", "blocked", "travers", "né", "đi tiếp", "chặn", "vượt"))
    distance_reference = bool(
        re.search(r"\b(how far|distance)\b.*\b(head|tail)\b", q)
        or re.search(r"\bfrom\s+(?:the\s+)?(?:aisle\s+)?(head|tail)\b", q)
    )
    spatial_terms = ("adjacent", "beside", "center", "left", "right", "head", "tail", "middle")
    spatial = next((x for x in spatial_terms if re.search(rf"\b{re.escape(x)}\b", q)), None)
    if distance_reference and spatial in {"head", "tail"}:
        spatial = None
    recognized = sum(bool(x) for x in (object_match, view_match, aisle_match, greenhouse_match, colors, materials, category, spatial, navigation))
    simple = recognized > 0
    return normalize_intent_ids({
        "intent": "assess_traversability" if navigation else "find",
        "target_type": target,
        "greenhouse_id": greenhouse_match.group(0).replace("-", "_").replace(" ", "_") if greenhouse_match else None,
        "aisle_id": aisle_match.group(0).replace("-", "_").replace(" ", "_") if aisle_match else None,
        "object_id": object_match.group(0) if object_match else None,
        "view_id": view_match.group(0) if view_match else None,
        "category": category, "color": colors, "material": materials,
        "spatial_filter": spatial,
        "traversability_filter": "passability" if navigation else None,
        "requested_fields": ["id", "image", "caption", "geometry", "traversability"],
        "confidence": 1.0 if object_match or view_match else (0.9 if simple else 0.25),
        "requires_geometric_verification": navigation,
        "parser": "rule", "needs_llm_parser": not simple,
    })


def available_ollama(cfg):
    base = cfg["ollama"]["base_url"].rstrip("/")
    tags = requests.get(base + "/api/tags", timeout=10).json().get("models", [])
    names = {x["name"]: x for x in tags}
    requested = cfg["ollama"]["model"]
    actual = requested
    if actual not in names:
        fallback = cfg["ollama"].get("local_fallback_model")
        if not cfg["ollama"].get("allow_local_fallback") or fallback not in names:
            raise RuntimeError(f"Ollama model unavailable: {requested}")
        actual = fallback
    item = names[actual]
    return base, actual, item.get("digest")


def ollama_json(base, model, prompt, schema, images=None, temperature=0, num_predict=None):
    payload = {"model": model, "prompt": prompt, "format": schema, "stream": False,
               "think": False, "keep_alive": "30m", "options": {"temperature": temperature}}
    if num_predict is not None:
        payload["options"]["num_predict"] = num_predict
    if images:
        payload["images"] = [base64.b64encode(Path(x).read_bytes()).decode() for x in images]
    response = requests.post(base + "/api/generate", json=payload, timeout=900)
    response.raise_for_status()
    return json.loads(response.json()["response"])


def parse_intent(query, dataset, cfg, prompts, allow_llm=True):
    intent = rule_parse(query)
    if intent.pop("needs_llm_parser") and allow_llm:
        try:
            base, model, digest = available_ollama(cfg)
            parsed = ollama_json(base, model, prompts["intent_prompt"] + "\n\nUser query: " + query,
                                 INTENT_SCHEMA, temperature=cfg["ollama"]["temperature"])
            parsed["parser"] = "gemma"; parsed["parser_model"] = model; parsed["parser_model_digest"] = digest
            return normalize_intent_ids(parsed)
        except Exception as exc:
            intent["parser"] = "rule_fallback"
            intent["parser_error"] = str(exc)
    return normalize_intent_ids(intent)


def worker_path():
    source = Path(__file__).resolve().parents[1] / "scripts" / "clip_text_worker.py"
    if source.exists(): return source
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory("conceptgraphs_bringup")) / "scripts" / "clip_text_worker.py"


def encode_text(query, cfg):
    with tempfile.TemporaryDirectory(prefix="hmsg_clip_") as temp:
        output = Path(temp) / "query.npy"
        command = [resolve_worker_python(cfg["clip"].get("python")), str(worker_path()), "--text", query, "--output", str(output),
                   "--model", cfg["clip"]["model_name"], "--device", cfg["clip"]["device"]]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
        value = np.load(output).astype(np.float32)
    return value / max(float(np.linalg.norm(value)), 1e-12)


def resolve_canonical_category(query, db, intent):
    """Prefer an explicit database label in the query over an LLM superclass."""
    if intent.get("target_type") != "object":
        return intent
    normalized_query = " " + re.sub(r"[^a-z0-9]+", " ", query.lower()).strip() + " "
    labels = [row[0] for row in db.execute(
        "SELECT DISTINCT canonical_label FROM objects WHERE canonical_label IS NOT NULL")]
    matches = []
    for label in labels:
        phrase = re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()
        if phrase and f" {phrase} " in normalized_query:
            matches.append((len(phrase), label))
    if matches:
        intent["category"] = max(matches)[1]
        intent["category_resolution"] = "canonical_label_query_match"
    return intent


def candidate_rows(db, intent):
    if intent["target_type"] == "object":
        rows = [dict(x) for x in db.execute(
            "SELECT o.*,c.category,c.color,c.material,c.scene_summary,c.parsed_json,c.semantic_class,c.path_assessment,c.confidence AS semantic_confidence,"
            "v.rgb_path AS representative_rgb_path,ob.crop_path AS representative_crop_path "
            "FROM objects o LEFT JOIN captions c ON c.owner_type='object' AND c.owner_id=o.object_id "
            "LEFT JOIN views v ON v.view_id=o.representative_view_id LEFT JOIN observations ob ON ob.observation_id=o.representative_observation_id")]
        key = "object_id"
    else:
        rows = [dict(x) for x in db.execute(
            "SELECT v.*,c.category,c.color,c.material,c.scene_summary,c.parsed_json,c.path_assessment,c.confidence AS semantic_confidence,"
            "t.passability,t.can_robot_pass,t.blocking_object_ids_json,t.confidence AS traversability_confidence "
            "FROM views v LEFT JOIN captions c ON c.owner_type='view' AND c.owner_id=v.view_id "
            "LEFT JOIN traversability_assessments t ON t.view_id=v.view_id")]
        key = "view_id"
    def matches(row):
        checks = [(intent.get("aisle_id"), row.get("aisle_id")),
                  (intent.get("greenhouse_id"), row.get("greenhouse_id")),
                  (intent.get("color"), row.get("color")), (intent.get("material"), row.get("material"))]
        exact = intent.get(key)
        if exact and row.get(key) != exact: return False
        if any(want and (not have or want.lower() not in str(have).lower()) for want, have in checks): return False
        category = intent.get("category")
        if category:
            if intent.get("category_resolution") == "canonical_label_query_match":
                if str(row.get("canonical_label") or "").lower() != category.lower():
                    return False
            else:
                haystack = " ".join(str(row.get(x) or "") for x in ("canonical_label", "category")).replace("_", " ").lower()
                words = category.replace("_", " ").split()
                if not all(re.search(rf"\b{re.escape(word)}\b", haystack) for word in words):
                    return False
        spatial = intent.get("spatial_filter")
        if spatial and not re.search(
            rf"\b{re.escape(spatial.lower())}\b",
            str(row.get("parsed_json") or "").lower(),
        ):
            return False
        return True
    return [x for x in rows if matches(x)], key


def rank_candidates(dataset, db, candidates, key, target_type, vector, top_k):
    if not candidates: return []
    allowed = {x[key] for x in candidates}
    rows = [dict(x) for x in db.execute("SELECT * FROM embedding_indexes WHERE owner_type=?", (target_type,))]
    arrays = {}
    scores = {}
    for row in rows:
        if row["owner_id"] not in allowed: continue
        path = dataset / row["npy_path"]
        arrays.setdefault(path, np.load(path, mmap_mode="r"))
        score = float(np.dot(arrays[path][row["vector_index"]], vector))
        scores[row["owner_id"]] = max(score, scores.get(row["owner_id"], -1.0))
    output = []
    for row in candidates:
        item = dict(row); item["similarity"] = scores.get(row[key], -1.0)
        item["result_id"] = row[key]
        for name in ("centroid_3d_json", "bbox_3d_min_json", "bbox_3d_max_json", "parsed_json", "blocking_object_ids_json"):
            if item.get(name): item[name[:-5] if name.endswith("_json") else name] = json.loads(item[name])
        output.append(item)
    return sorted(output, key=lambda x: (-x["similarity"], x["result_id"]))[:top_k]


def ambiguity(ranked, intent, cfg):
    gates = cfg["ambiguity"]
    reasons = []
    if not ranked: return True, ["no_candidates"]
    if intent.get("object_id") or intent.get("view_id"):
        return False, []
    if intent.get("category_resolution") == "canonical_label_query_match" and len(ranked) == 1:
        return False, []
    deterministic_filters = sum(bool(intent.get(key)) for key in
                                ("greenhouse_id", "aisle_id", "category", "color", "material", "spatial_filter"))
    if len(ranked) == 1 and deterministic_filters >= 2:
        return False, []
    if intent["confidence"] < gates["minimum_parser_confidence"]: reasons.append("low_parser_confidence")
    if ranked[0]["similarity"] < gates["minimum_similarity"]: reasons.append("low_top1_similarity")
    if len(ranked) > 1 and ranked[0]["similarity"] - ranked[1]["similarity"] < gates["minimum_top1_margin"]:
        reasons.append("small_top1_margin")
    return bool(reasons), reasons


def verify(query, ranked, dataset, cfg, prompts):
    base, model, digest = available_ollama(cfg)
    limit = cfg["ambiguity"]["verification_top_k"]
    selected = ranked[:limit]
    summary = [{"id": x["result_id"], "similarity": x["similarity"], "caption": x.get("scene_summary"),
                "label": x.get("canonical_label"), "aisle_id": x.get("aisle_id")} for x in selected]
    paths = []
    for row in selected:
        value = row.get("representative_crop_path") or row.get("representative_rgb_path") or row.get("rgb_path")
        if value: paths.append(dataset / value)
    prompt = prompts["verification_prompt"] + "\n\nQuery: " + query + "\nCandidates: " + json.dumps(summary, ensure_ascii=False)
    parsed = ollama_json(base, model, prompt, VERIFY_SCHEMA, paths, cfg["ollama"]["temperature"])
    allowed = {x["result_id"] for x in selected}
    if not set(parsed["selected_ids"]) <= allowed: raise ValueError("Verifier returned an unknown candidate ID")
    parsed.update({"model_tag": model, "model_digest": digest, "candidate_ids": sorted(allowed)})
    return parsed


def compact_results(ranked, target_type):
    output = []
    for row in ranked:
        parsed = row.get("parsed") or {}
        cues = parsed.get("traversability_cues") or {}
        if target_type == "object":
            described = (parsed.get("physical_objects") or [{}])[0]
            item = {
                "object_id": row["object_id"], "canonical_label": row.get("canonical_label"),
                "category": described.get("category") or row.get("category"),
                "color": described.get("color") or row.get("color"),
                "material": described.get("material") or row.get("material"),
                "shape": described.get("shape"), "state": described.get("state"),
                "physical_properties": described.get("physical_properties") or {},
                "aisle_id": row.get("aisle_id"), "frame_id": row.get("frame_id"),
                "similarity": row["similarity"], "semantic_confidence": row.get("semantic_confidence"),
                "centroid_3d": row.get("centroid_3d"), "bbox_3d_min": row.get("bbox_3d_min"),
                "bbox_3d_max": row.get("bbox_3d_max"), "representative_view_id": row.get("representative_view_id"),
                "representative_rgb_path": row.get("representative_rgb_path"),
                "representative_crop_path": row.get("representative_crop_path"),
                "caption": row.get("scene_summary"),
                "traversability": {
                    "object_semantic_class": row.get("semantic_class"),
                    "path_assessment": row.get("path_assessment"),
                    "can_robot_pass": cues.get("can_robot_pass"),
                    "blocking_object_ids": cues.get("blocking_object_ids", []),
                    "traversable_object_ids": cues.get("traversable_object_ids", []),
                    "visible_obstacles": cues.get("visible_obstacles", []),
                    "reasoning_summary": cues.get("reasoning_summary"),
                    "requires_geometric_verification": True,
                },
            }
        else:
            item = {
                "view_id": row["view_id"], "aisle_id": row.get("aisle_id"),
                "frame_id": row.get("frame_id"), "similarity": row["similarity"],
                "rgb_path": row.get("rgb_path"), "caption": row.get("scene_summary"),
                "passability": row.get("passability"), "can_robot_pass": row.get("can_robot_pass"),
                "blocking_object_ids": row.get("blocking_object_ids", []),
                "traversability_confidence": row.get("traversability_confidence"),
                "requires_geometric_verification": True,
            }
        output.append(item)
    return output


def save_result(dataset, record):
    out = dataset / "retrieval"; out.mkdir(parents=True, exist_ok=True)
    path = out / "query_results.jsonl"
    existing = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                item = json.loads(line); existing[item["query_id"]] = item
    existing[record["query_id"]] = record
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in sorted(existing.values(), key=lambda x: x["query_id"])))
    values = list(existing.values())
    report = {
        "stage": "fast_to_slow_retrieval", "version": record["provenance"]["retrieval_version"],
        "query_count": len(values), "fast_path_count": sum(x["provenance"]["fast_path"] for x in values),
        "verification_count": sum(not x["provenance"]["fast_path"] for x in values),
        "not_found_count": sum(x["status"] == "not_found" for x in values),
        "ambiguous_count": sum(x["ambiguous"] for x in values),
        "average_latency_sec": sum(x["latency_sec"] for x in values) / len(values),
        "clip_model": record["provenance"]["clip_model"], "updated_at": now(),
    }
    reports = dataset / "reports"; reports.mkdir(exist_ok=True)
    (reports / "retrieval_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    previews = out / "previews"; previews.mkdir(exist_ok=True)
    cards = []
    for query_record in sorted(values, key=lambda x: x["query_id"]):
        result = query_record["results"][0] if query_record["results"] else {}
        image_path = result.get("representative_crop_path") or result.get("representative_rgb_path") or result.get("rgb_path")
        image_tag = f'<img src="../../{html.escape(image_path)}" style="max-width:360px">' if image_path else ""
        cards.append(f"<section><h2>{html.escape(query_record['query'])}</h2>{image_tag}<pre>{html.escape(json.dumps(query_record, indent=2, ensure_ascii=False))}</pre></section>")
    (previews / "index.html").write_text("<!doctype html><meta charset='utf-8'><title>HMSG retrieval</title>" + "".join(cards))
    return path


def retrieve(dataset, query, top_k=5, text_vector=None, allow_llm=True, allow_verification=True):
    dataset = Path(dataset).resolve(); cfg = load_config(dataset); prompts = prompt_config(dataset, cfg)
    started = time.monotonic(); intent = parse_intent(query, dataset, cfg, prompts, allow_llm)
    db_path = dataset / "database/greenhouse_hmsg.sqlite"
    db = sqlite3.connect(db_path); db.row_factory = sqlite3.Row
    try:
        intent = resolve_canonical_category(query, db, intent)
        candidates, key = candidate_rows(db, intent)
        if text_vector is None: text_vector = encode_text(query, cfg)
        ranked = rank_candidates(dataset, db, candidates, key, intent["target_type"], text_vector, top_k)
    finally: db.close()
    is_ambiguous, reasons = ambiguity(ranked, intent, cfg)
    verification = None
    if is_ambiguous and ranked and allow_verification:
        try:
            verification = verify(query, ranked, dataset, cfg, prompts)
            if intent["requires_geometric_verification"]:
                verification["requires_geometric_verification"] = True
            if verification.get("selected_ids"):
                order = {value: index for index, value in enumerate(verification["selected_ids"])}
                ranked = sorted((x for x in ranked if x["result_id"] in order), key=lambda x: order[x["result_id"]])
        except Exception as exc:
            verification = {"status": "failed", "error": str(exc), "candidate_ids": [x["result_id"] for x in ranked[:cfg["ambiguity"]["verification_top_k"]]]}
    status = "not_found" if not ranked else (verification.get("status") if verification else "complete")
    record = {"query_id": stable_id(query, cfg["version"], top_k), "query": query, "status": status,
              "intent": intent, "results": compact_results(ranked, intent["target_type"]), "ambiguous": is_ambiguous, "ambiguity_reasons": reasons,
              "verification": verification, "provenance": {"retrieval_version": cfg["version"],
              "database": "database/greenhouse_hmsg.sqlite", "clip_model": cfg["clip"]["model_name"],
              "fast_path": verification is None, "requires_geometric_verification": bool(intent["requires_geometric_verification"])},
              "latency_sec": time.monotonic()-started, "created_at": now()}
    save_result(dataset, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-llm-parser", action="store_true")
    parser.add_argument("--no-verification", action="store_true")
    args = parser.parse_args(argv)
    if args.top_k < 1: parser.error("--top-k must be positive")
    try:
        result = retrieve(args.dataset, args.query, args.top_k, allow_llm=not args.no_llm_parser,
                          allow_verification=not args.no_verification)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"HMSG retrieval error: {exc}", file=sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
