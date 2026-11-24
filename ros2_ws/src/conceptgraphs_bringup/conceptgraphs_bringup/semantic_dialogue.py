"""M8.2 intent-aware dialogue over verified M8/HMSG results."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .hmsg_retrieval import available_ollama, ollama_json, retrieve


RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "mentioned_object_ids", "confidence", "requires_geometric_verification"],
    "properties": {
        "answer": {"type": "string"},
        "mentioned_object_ids": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "requires_geometric_verification": {"type": "boolean"},
    },
}

CLASSIFIER_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["intent", "confidence", "reason"],
    "properties": {
        "intent": {"enum": ["general_chat", "map_query", "clarification"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
}

CHAT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["answer", "confidence"],
    "properties": {"answer": {"type": "string"},
                   "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
}


def route_intent(query):
    q = query.lower().strip()
    list_words = ("what objects", "which objects", "list objects", "objects are", "có object", "có vật", "những object")
    count_words = ("how many objects", "number of objects", "bao nhiêu object", "bao nhiêu vật")
    property_words = ("physical properties", "properties", "property", "rigidity", "rigid",
                      "deformability", "fragility", "material", "shape", "mass", "compressibility")
    location_words = ("where", "location", "position", "located", "coordinate", "coordinates",
                      "centroid", "xyz", "x,y,z", "which aisle", "what aisle", "how far",
                      "head", "tail", "middle", "đâu", "vị trí", "đầu", "cuối", "giữa")
    traversal_words = ("pass", "avoid", "blocked", "travers", "né", "đi qua", "đi tiếp", "chặn", "vượt")
    if any(x in q for x in count_words): return "count_objects"
    if any(x in q for x in list_words): return "list_objects"
    if any(x in q for x in traversal_words): return "assess_traversability"
    if any(x in q for x in property_words): return "describe_object_properties"
    if any(x in q for x in location_words): return "locate_object"
    return "find_object"


def response_spec(query, intent):
    """Deterministic contract for the fields and style the answer must contain."""
    q = query.lower().strip()
    spec = {"intent": intent, "format": "summary", "requested_fields": []}
    if intent == "locate_object":
        if any(x in q for x in ("xyz", "x,y,z", "coordinate", "coordinates", "centroid")):
            spec.update(format="xyz", requested_fields=["centroid_3d", "frame_id", "aisle_id"])
        elif any(x in q for x in ("which aisle", "what aisle", "aisle contains", "located in which aisle")):
            spec.update(format="aisle", requested_fields=["aisle_id"])
        elif ("how far" in q or "distance" in q) and "head" in q:
            spec.update(format="distance_from_head", requested_fields=["distance_from_head_m", "aisle_id"])
        elif ("how far" in q or "distance" in q) and "tail" in q:
            spec.update(format="distance_to_tail", requested_fields=["distance_to_tail_m", "aisle_id"])
        else:
            spec.update(format="topological", requested_fields=["aisle_id", "section", "distance_from_head_m"])
    elif intent == "describe_object_properties":
        spec.update(format="physical_properties", requested_fields=[
            "category", "color", "material", "shape", "state", "rigidity",
            "deformability", "compressibility", "fragility", "estimated_mass_class",
            "anchored_to_ground",
        ])
    elif intent == "assess_traversability":
        spec.update(format="traversability", requested_fields=[
            "object_traversability", "path_assessment", "available_clearance_m",
            "requires_geometric_verification",
        ])
    return spec


def route_channel(query):
    """Fast channel router; returns None when Gemma classification is needed."""
    q = query.lower().strip()
    chat = ("xin chào", "chào bạn", "hello", "hi ", "cảm ơn", "thank you", "thanks",
            "bạn có thể làm", "bạn làm được", "what can you do", "who are you", "bạn là ai")
    map_terms = ("object", "container", "crate", "box", "ladder", "wheelbarrow", "thùng", "sọt",
                 "map", "bản đồ", "aisle", "greenhouse", "view", "centroid", "coordinate",
                 "xyz", "property", "properties", "material", "rigidity", "tọa độ", "vị trí",
                 "travers", "pass", "chặn")
    if any(x in q for x in chat): return "general_chat"
    if any(x in q for x in map_terms): return "map_query"
    return None


def clarification_for(query):
    q = query.lower()
    watermelon = "watermelon" in q or "dưa hấu" in q
    specific = any(x in q for x in ("flesh", "inside", "ruột", "rind", "skin", "vỏ"))
    if watermelon and not specific:
        return "Do you mean the green rind or the red flesh of a watermelon?"
    return None


def _load_prompts():
    path = Path(__file__).resolve().parents[1] / "config" / "dialogue_prompts_v1.yaml"
    if not path.exists():
        from ament_index_python.packages import get_package_share_directory
        path = Path(get_package_share_directory("conceptgraphs_bringup")) / "config" / "dialogue_prompts_v1.yaml"
    return yaml.safe_load(path.read_text())


def _load_chat_prompts():
    path = Path(__file__).resolve().parents[1] / "config" / "general_chat_prompts_v1.yaml"
    if not path.exists():
        from ament_index_python.packages import get_package_share_directory
        path = Path(get_package_share_directory("conceptgraphs_bringup")) / "config" / "general_chat_prompts_v1.yaml"
    return yaml.safe_load(path.read_text())


def classify_channel(query, cfg, allow_llm=True):
    routed = route_channel(query)
    if routed:
        return {"intent": routed, "confidence": 1.0, "reason": "rule_router", "source": "rule"}
    if not allow_llm:
        return {"intent": "general_chat", "confidence": 0.5, "reason": "llm_disabled", "source": "fallback"}
    try:
        prompts = _load_chat_prompts(); base, model, digest = available_ollama(cfg)
        parsed = ollama_json(base, model, prompts["intent_classifier_prompt"] + "\n\nUser message: " + query,
                             CLASSIFIER_SCHEMA, temperature=0, num_predict=80)
        parsed.update({"source": "gemma_classifier", "model_tag": model, "model_digest": digest})
        return parsed
    except Exception as exc:
        return {"intent": "general_chat", "confidence": 0.0, "reason": str(exc), "source": "classifier_fallback"}


def general_chat(query, history, cfg, allow_llm=True):
    prompts = _load_chat_prompts()
    fallback = "I can chat with you and help you formulate questions about the greenhouse map."
    if not allow_llm:
        return {"answer": fallback, "source": "chat_fallback", "confidence": 0.0}
    try:
        base, model, digest = available_ollama(cfg)
        safe_history = [{"role": x.get("role"), "content": str(x.get("content", ""))[:1000]}
                        for x in (history or [])[-10:] if x.get("role") in ("user", "assistant")]
        prompt = (prompts["system_prompt"] + "\n\nRecent conversation: " +
                  json.dumps(safe_history, ensure_ascii=False) + "\n\nUser: " + query)
        parsed = ollama_json(base, model, prompt, CHAT_SCHEMA, temperature=0, num_predict=140)
        return {"answer": parsed["answer"], "source": "gemma_general_chat", "confidence": parsed["confidence"],
                "model_tag": model, "model_digest": digest}
    except Exception as exc:
        return {"answer": fallback, "source": "chat_fallback", "confidence": 0.0, "error": str(exc)}


def _all_objects(dataset):
    db = sqlite3.connect(dataset / "database/greenhouse_hmsg.sqlite")
    db.row_factory = sqlite3.Row
    try:
        rows = db.execute(
            "SELECT o.object_id,o.canonical_label,o.aisle_id,o.frame_id,o.centroid_3d_json,"
            "o.bbox_3d_min_json,o.bbox_3d_max_json,o.representative_view_id,"
            "v.rgb_path AS representative_rgb_path,ob.crop_path AS representative_crop_path,"
            "c.category,c.color,c.material,c.scene_summary,c.parsed_json,c.semantic_class,c.path_assessment "
            "FROM objects o LEFT JOIN views v ON v.view_id=o.representative_view_id "
            "LEFT JOIN observations ob ON ob.observation_id=o.representative_observation_id "
            "LEFT JOIN captions c ON c.owner_type='object' AND c.owner_id=o.object_id ORDER BY o.object_id"
        ).fetchall()
    finally:
        db.close()
    output = []
    for row in rows:
        item = dict(row)
        for source, target in (("centroid_3d_json", "centroid_3d"), ("bbox_3d_min_json", "bbox_3d_min"),
                               ("bbox_3d_max_json", "bbox_3d_max"), ("parsed_json", "parsed")):
            item[target] = json.loads(item.pop(source)) if item.get(source) else None
        parsed = item.pop("parsed", {}) or {}; cues = parsed.get("traversability_cues") or {}
        item["caption"] = item.pop("scene_summary", None)
        item["traversability"] = {"object_semantic_class": item.pop("semantic_class", None),
                                  "path_assessment": item.pop("path_assessment", None),
                                  "can_robot_pass": cues.get("can_robot_pass"),
                                  "reasoning_summary": cues.get("reasoning_summary"),
                                  "requires_geometric_verification": True}
        item["similarity"] = None
        output.append(item)
    return output


def _layout(dataset):
    from .paths import resolve_config_path
    pipeline = yaml.safe_load((dataset / "pipeline_config.yaml").read_text())
    path = resolve_config_path(pipeline["layout_path"], dataset)
    return yaml.safe_load(path.read_text())


def _project_polyline(point, line):
    best = None; walked = 0.0; total = sum(math.dist(a, b) for a, b in zip(line, line[1:]))
    for a, b in zip(line, line[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]; length2 = dx * dx + dy * dy
        t = max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length2)) if length2 else 0.0
        projected = [a[0] + t * dx, a[1] + t * dy]
        candidate = (math.dist(point[:2], projected), walked + math.sqrt(length2) * t, projected)
        if best is None or candidate[0] < best[0]: best = candidate
        walked += math.sqrt(length2)
    return best, total


def spatial_facts(dataset, result):
    centroid = result.get("centroid_3d")
    if not centroid: return None
    aisle = next((x for x in _layout(dataset).get("aisles", []) if x["id"] == result.get("aisle_id")), None)
    if not aisle or len(aisle.get("centerline") or []) < 2: return None
    projected, total = _project_polyline(centroid, aisle["centerline"])
    along = projected[1]; ratio = along / total if total else 0
    section = "head" if ratio < 1 / 3 else ("tail" if ratio > 2 / 3 else "middle")
    tendency = "toward_head" if along < total - along else "toward_tail"
    return {"section": section, "tendency": tendency, "distance_from_head_m": round(along, 2),
            "distance_to_tail_m": round(total - along, 2), "aisle_length_m": round(total, 2),
            "centerline_offset_m": round(projected[0], 2), "source": "layout_geometry"}


def geometric_path_facts(dataset, result):
    """Estimate side clearance from the aisle polygon and the object's 3-D AABB."""
    centroid = result.get("centroid_3d")
    lower, upper = result.get("bbox_3d_min"), result.get("bbox_3d_max")
    if not centroid or not lower or not upper:
        return None
    aisle = next((x for x in _layout(dataset).get("aisles", [])
                  if x["id"] == result.get("aisle_id")), None)
    if not aisle or len(aisle.get("centerline") or []) < 2 or len(aisle.get("polygon_xy") or []) < 3:
        return None
    line = aisle["centerline"]
    best = None
    for start, end in zip(line, line[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length2 = dx * dx + dy * dy
        if not length2:
            continue
        t = max(0.0, min(1.0, ((centroid[0] - start[0]) * dx +
                               (centroid[1] - start[1]) * dy) / length2))
        point = (start[0] + t * dx, start[1] + t * dy)
        distance = math.dist(centroid[:2], point)
        if best is None or distance < best[0]:
            best = (distance, start, end, math.sqrt(length2))
    if best is None:
        return None
    _, start, end, length = best
    normal = (-(end[1] - start[1]) / length, (end[0] - start[0]) / length)
    project = lambda point: point[0] * normal[0] + point[1] * normal[1]
    aisle_values = [project(point) for point in aisle["polygon_xy"]]
    object_values = [project(point) for point in (
        (lower[0], lower[1]), (lower[0], upper[1]),
        (upper[0], lower[1]), (upper[0], upper[1]),
    )]
    aisle_min, aisle_max = min(aisle_values), max(aisle_values)
    object_min, object_max = min(object_values), max(object_values)
    clearances = [max(0.0, object_min - aisle_min), max(0.0, aisle_max - object_max)]
    available = max(clearances)
    # Conservative pilot defaults; geometry remains advisory until the actual
    # robot footprint and safety margin are calibrated.
    robot_width_m, safety_margin_m = 0.8, 0.2
    required = robot_width_m + 2 * safety_margin_m
    passable = available >= required
    return {
        "path_assessment": "passable" if passable else "blocked",
        "can_robot_pass": passable,
        "available_clearance_m": round(available, 2),
        "left_clearance_m": round(clearances[0], 2),
        "right_clearance_m": round(clearances[1], 2),
        "required_clearance_m": round(required, 2),
        "robot_width_m": robot_width_m,
        "safety_margin_m": safety_margin_m,
        "source": "aisle_polygon_and_object_bbox",
        "requires_geometric_verification": True,
    }


def verified_facts(dataset, intent, results, spec=None, retrieval=None):
    objects = []
    for row in results:
        physical = row.get("physical_properties") or {}
        semantic = row.get("traversability") or {}
        path_geometry = geometric_path_facts(dataset, row)
        objects.append({"object_id": row["object_id"], "label": row.get("canonical_label"),
                        "category": row.get("category"), "color": row.get("color"),
                        "material": row.get("material"), "aisle_id": row.get("aisle_id"),
                        "shape": row.get("shape"), "state": row.get("state"),
                        "physical_properties": physical, "frame_id": row.get("frame_id"),
                        "centroid_3d": row.get("centroid_3d"),
                        "bbox_3d_min": row.get("bbox_3d_min"), "bbox_3d_max": row.get("bbox_3d_max"),
                        "spatial": spatial_facts(dataset, row),
                        "object_traversability": semantic.get("object_semantic_class"),
                        "semantic_path_assessment": semantic.get("path_assessment"),
                        "path_geometry": path_geometry,
                        "path_assessment": (path_geometry or {}).get("path_assessment", semantic.get("path_assessment")),
                        "can_robot_pass": (path_geometry or {}).get("can_robot_pass", semantic.get("can_robot_pass")),
                        "semantic_reason": semantic.get("reasoning_summary")})
    verification = (retrieval or {}).get("verification") or {}
    evidence = {
        "retrieval_status": (retrieval or {}).get("status", "complete"),
        "ambiguous": bool((retrieval or {}).get("ambiguous", False)),
        "ambiguity_reasons": (retrieval or {}).get("ambiguity_reasons", []),
        "verification_status": verification.get("status"),
        "verification_confidence": verification.get("confidence"),
        "verification_reason": verification.get("reasoning_summary"),
    }
    return {"intent": intent, "response_spec": spec or response_spec("", intent),
            "object_count": len(objects), "objects": objects,
            "geometry_frame": "camera_init", "retrieval_evidence": evidence,
            "facts_are_authoritative": True}


def fallback_answer(query, facts):
    objects = facts["objects"]; intent = facts["intent"]
    if not objects:
        if (facts.get("retrieval_evidence") or {}).get("ambiguous"):
            return ("I could not identify the requested object with sufficient confidence. "
                    "Please specify its category or aisle.")
        return f"I could not find a matching object for ‘{query}’ in the current dataset."
    if intent in ("list_objects", "count_objects"):
        labels = [x["label"].replace("_", " ") for x in objects]
        return f"I found {len(objects)} mapped objects: " + ", ".join(labels) + "."
    obj = objects[0]; label = obj["label"].replace("_", " ")
    aisle = str(obj.get("aisle_id") or "the current aisle").replace("_", " ")
    spec = facts.get("response_spec") or response_spec(query, intent)
    if intent == "locate_object":
        if spec["format"] == "xyz" and obj.get("centroid_3d"):
            x, y, z = obj["centroid_3d"]
            frame = obj.get("frame_id") or facts.get("geometry_frame", "camera_init")
            return (f"The {label} is at (x, y, z) = ({x:.2f}, {y:.2f}, {z:.2f}) metres "
                    f"in the {frame} map frame, within {aisle}.")
        if spec["format"] == "aisle":
            return f"The {label} is located in {aisle}."
        if obj.get("spatial"):
            s = obj["spatial"]
            if spec["format"] == "distance_from_head":
                return f"The {label} is approximately {s['distance_from_head_m']:.1f} metres from the head of {aisle}."
            if spec["format"] == "distance_to_tail":
                return f"The {label} is approximately {s['distance_to_tail_m']:.1f} metres from the tail of {aisle}."
            return (f"The {label} is located near the {s['section']} end of {aisle}, approximately "
                    f"{s['distance_from_head_m']:.1f} metres from the aisle head.")
    if intent == "describe_object_properties":
        physical = obj.get("physical_properties") or {}
        attributes = [x for x in (obj.get("color"), obj.get("material"), obj.get("shape")) if x]
        properties = [
            physical.get("rigidity"),
            f"{physical['deformability']} deformability" if physical.get("deformability") else None,
            f"{physical['compressibility']} compressibility" if physical.get("compressibility") else None,
            f"{physical['fragility']} fragility" if physical.get("fragility") else None,
            f"{physical['estimated_mass_class']} estimated mass" if physical.get("estimated_mass_class") else None,
        ]
        description = ", ".join(str(x).replace("_", " ") for x in attributes) or "mapped object"
        details = ", ".join(str(x).replace("_", " ") for x in properties if x)
        return f"The {label} is a {description}. Its physical properties are {details}."
    if intent == "assess_traversability":
        object_class = str(obj.get("object_traversability") or "unknown").replace("_", " ")
        geometry = obj.get("path_geometry") or {}
        if geometry.get("can_robot_pass"):
            return (f"The {label} itself is a {object_class}, so the robot should not drive over or through it. "
                    f"However, the route around it is estimated to be passable, with approximately "
                    f"{geometry['available_clearance_m']:.2f} metres of side clearance. Geometric verification "
                    "with the calibrated robot footprint is required before motion.")
        return (f"The {label} itself is a {object_class}. The current geometry estimate does not show enough "
                "side clearance to pass safely. Geometric verification with the calibrated robot footprint "
                "is required before motion.")
    indirect = any(x in query.lower() for x in ("like", "similar", "giống", "watermelon", "dưa hấu"))
    if indirect:
        return f"I interpreted the description semantically. The best verified match is the {label} in {aisle}."
    return f"I found the {label} in {aisle}."


def answer_contains_number(answer, target, tolerance=0.02):
    """Accept natural numeric formatting while preserving factual accuracy."""
    values = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", answer)]
    return any(abs(value - float(target)) <= tolerance for value in values)


def compose_answer(dataset, query, facts, cfg, allow_llm=True):
    fallback = fallback_answer(query, facts)
    if not facts["objects"]:
        return {"answer": fallback, "source": "verified_not_found", "error": None}
    if not allow_llm: return {"answer": fallback, "source": "template", "error": None}
    try:
        prompts = _load_prompts(); base, model, digest = available_ollama(cfg)
        prompt = prompts["response_prompt"] + "\n\nUser query: " + query + "\nVerified facts: " + json.dumps(facts, ensure_ascii=False)
        parsed = ollama_json(base, model, prompt, RESPONSE_SCHEMA, temperature=0, num_predict=300)
        allowed = {x["object_id"] for x in facts["objects"]}
        if not set(parsed["mentioned_object_ids"]) <= allowed:
            raise ValueError("response composer mentioned an unverified Object ID")
        expected_geometry_flag = facts["intent"] == "assess_traversability"
        safety_corrected = bool(parsed["requires_geometric_verification"]) != expected_geometry_flag
        parsed["requires_geometric_verification"] = expected_geometry_flag
        denial = any(x in parsed["answer"].lower() for x in
                     ("could not find", "couldn't find", "no matching", "not found", "không tìm thấy"))
        if facts["objects"] and denial:
            raise ValueError("response composer contradicted verified retrieval results")
        spec = facts.get("response_spec") or {}
        obj = facts["objects"][0]
        if spec.get("format") == "xyz" and obj.get("centroid_3d"):
            if not all(answer_contains_number(parsed["answer"], value)
                       for value in obj["centroid_3d"]):
                raise ValueError("response composer omitted requested xyz coordinates")
        if spec.get("format") == "distance_from_head" and obj.get("spatial"):
            if not answer_contains_number(
                parsed["answer"], obj["spatial"]["distance_from_head_m"], tolerance=0.11
            ):
                raise ValueError("response composer omitted requested head distance")
        if expected_geometry_flag and "verif" not in parsed["answer"].lower():
            parsed["answer"] = parsed["answer"].rstrip() + (
                " Geometric verification with the calibrated robot footprint is required "
                "before motion."
            )
        evidence = facts.get("retrieval_evidence") or {}
        confidence = parsed["confidence"]
        if evidence.get("verification_status") == "uncertain" and evidence.get("verification_confidence") is not None:
            confidence = min(confidence, evidence["verification_confidence"])
        if expected_geometry_flag:
            confidence = min(confidence, 0.8)
        return {"answer": parsed["answer"], "source": "gemma", "confidence": confidence,
                "model_tag": model, "model_digest": digest, "error": None,
                "safety_flag_corrected": safety_corrected}
    except Exception as exc:
        return {"answer": fallback, "source": "template_fallback", "error": str(exc)}


def dialogue(dataset, query, top_k=5, context_object_id=None, allow_llm=True):
    dataset = Path(dataset).resolve(); intent = route_intent(query); spec = response_spec(query, intent)
    clarification = clarification_for(query)
    if clarification:
        return {"dialogue_id": _stable_dialogue_id(query, context_object_id), "status": "clarification_needed",
                "query": query, "intent": intent, "answer": clarification, "answer_source": "rule",
                "results": [], "verified_facts": {"intent": intent, "response_spec": spec, "objects": []}, "retrieval": None}
    effective = query
    if context_object_id and re.search(r"\b(this|that|it)\b|vật này|\bnó\b", query.lower()) and "object_" not in query.lower():
        effective += " " + context_object_id
    if intent in ("list_objects", "count_objects"):
        results = _all_objects(dataset); retrieval_record = None
    else:
        retrieval_record = retrieve(dataset, effective, top_k, allow_llm=allow_llm)
        results = retrieval_record["results"][:1]
    facts = verified_facts(dataset, intent, results, spec, retrieval_record)
    cfg = yaml.safe_load((dataset / "pipeline_config.yaml").read_text()).get("retrieval")
    if not cfg:
        from .hmsg_retrieval import load_config
        cfg = load_config(dataset)
    composed = compose_answer(dataset, query, facts, cfg, allow_llm)
    record = {"dialogue_id": _stable_dialogue_id(query, context_object_id), "status": "success" if results else "not_found",
              "query": query, "effective_query": effective, "intent": intent, "response_spec": spec,
              "answer": composed["answer"],
              "answer_source": composed["source"], "composer": {k: v for k, v in composed.items() if k != "answer"},
              "results": results, "verified_facts": facts, "retrieval": retrieval_record}
    _save_dialogue(dataset, record)
    return record


def interact(dataset, query, mode="auto", history=None, top_k=5, context_object_id=None, allow_llm=True):
    """Route one turn without allowing general chat to touch retrieval."""
    dataset = Path(dataset).resolve()
    if mode not in ("auto", "chat", "map"):
        raise ValueError("mode must be auto, chat, or map")
    from .hmsg_retrieval import load_config
    cfg = load_config(dataset)
    if mode == "chat":
        routing = {"intent": "general_chat", "confidence": 1.0, "reason": "manual_chat", "source": "manual"}
    elif mode == "map":
        routing = {"intent": "map_query", "confidence": 1.0, "reason": "manual_map", "source": "manual"}
    else:
        routing = classify_channel(query, cfg, allow_llm)
    if routing["intent"] == "clarification":
        return {"status": "clarification_needed", "response_type": "clarification", "intent": "clarification",
                "answer": "Could you clarify whether you want normal conversation or information from the current map?",
                "answer_source": routing["source"], "results": [], "routing": routing}
    if routing["intent"] == "general_chat":
        result = general_chat(query, history, cfg, allow_llm)
        return {"status": "success", "response_type": "general_chat", "intent": "general_chat",
                "answer": result["answer"], "answer_source": result["source"], "results": [],
                "routing": routing, "chat": result, "retrieval": None}
    result = dialogue(dataset, query, top_k, context_object_id, allow_llm)
    result["response_type"] = "map_query"; result["routing"] = routing
    return result


def _stable_dialogue_id(query, context):
    raw = json.dumps({"query": query.strip(), "context": context, "version": "greenhouse_dialogue_v1"}, sort_keys=True)
    return "dialogue_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def _save_dialogue(dataset, record):
    path = dataset / "retrieval" / "dialogue_results.jsonl"; path.parent.mkdir(exist_ok=True)
    existing = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                item = json.loads(line); existing[item["dialogue_id"]] = item
    stored = dict(record); stored["created_at"] = datetime.now(timezone.utc).isoformat()
    existing[stored["dialogue_id"]] = stored
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in sorted(existing.values(), key=lambda x: x["dialogue_id"])))
