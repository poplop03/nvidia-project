"""Local FastAPI interface for M8.2 intent-aware semantic dialogue."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .semantic_dialogue import interact


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    mode: str = Field(default="auto", pattern="^(auto|chat|map)$")
    history: list[dict[str, str]] = Field(default_factory=list, max_length=10)
    context_object_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)


def ensure_ollama(dataset):
    """Start the dataset's local Ollama server when it is not available."""
    pipeline = yaml.safe_load((dataset / "pipeline_config.yaml").read_text())
    config = pipeline["retrieval"]["ollama"]
    base_url = config["base_url"].rstrip("/")
    try:
        version = requests.get(base_url + "/api/version", timeout=2).json()["version"]
        print(f"Ollama already available at {base_url} (version {version})")
        return None
    except (requests.RequestException, KeyError, ValueError):
        pass
    if not config.get("auto_start", False):
        print(f"Ollama is unavailable at {base_url}; auto_start is disabled")
        return None

    parsed = urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise RuntimeError("Web auto-start only supports a local HTTP Ollama endpoint")
    if not parsed.port:
        raise RuntimeError("Ollama base_url must include an explicit port")

    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"{parsed.hostname}:{parsed.port}"
    env["OLLAMA_MODELS"] = config["models_path"]
    env["OLLAMA_LLM_LIBRARY"] = config.get("llm_library", "cuda_v13")
    env["OLLAMA_CONTEXT_LENGTH"] = str(config.get("context_length", 8192))
    reports = dataset / "reports"
    reports.mkdir(exist_ok=True)
    log = (reports / "ollama_web.log").open("a", encoding="utf-8")
    process = subprocess.Popen(
        ["ollama", "serve"], env=env, stdout=log, stderr=subprocess.STDOUT
    )
    deadline = time.monotonic() + float(config.get("startup_timeout_sec", 30))
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log.close()
            raise RuntimeError(
                f"Ollama exited during startup; inspect {reports / 'ollama_web.log'}"
            )
        try:
            version = requests.get(base_url + "/api/version", timeout=1).json()["version"]
            print(f"Started Ollama at {base_url} (version {version}, PID {process.pid})")
            return process, log
        except (requests.RequestException, KeyError, ValueError):
            time.sleep(0.25)
    process.terminate()
    process.wait(timeout=5)
    log.close()
    raise RuntimeError(f"Timed out starting Ollama at {base_url}")


def stop_managed_ollama(runtime):
    """Stop only an Ollama process that this Web invocation started."""
    if runtime is None:
        return
    process, log = runtime
    try:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    finally:
        log.close()


def web_root():
    source = Path(__file__).resolve().parents[1] / "web"
    if source.exists():
        return source
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory("conceptgraphs_bringup")) / "web"


def safe_dataset_file(dataset, relative):
    root = dataset.resolve()
    candidate = (root / relative).resolve()
    if root not in candidate.parents or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Dataset file not found")
    return candidate


def graph_for_object(dataset, object_id, *, maximum_views=5):
    """Return a compact HMSG neighborhood around one focused Object.

    The browser needs the hierarchy as well as observation evidence.  Keep the
    graph deliberately local: one Greenhouse, the current Aisle, every sibling
    Object in that Aisle, and a bounded number of Views for the focused Object.
    """
    db = sqlite3.connect(dataset / "database/greenhouse_hmsg.sqlite")
    db.row_factory = sqlite3.Row
    try:
        obj = db.execute(
            "SELECT object_id,canonical_label,greenhouse_id,aisle_id "
            "FROM objects WHERE object_id=?", (object_id,)
        ).fetchone()
        if not obj:
            return {"nodes": [], "edges": []}
        observations = [dict(x) for x in db.execute(
            "SELECT e.edge_id,e.edge_type,e.view_id,e.object_id,e.observation_id,v.rgb_path "
            "FROM view_object_edges e JOIN views v ON v.view_id=e.view_id "
            "WHERE e.object_id=? ORDER BY v.timestamp_ns LIMIT ?",
            (object_id, maximum_views))]
        siblings = [dict(x) for x in db.execute(
            "SELECT object_id,canonical_label,greenhouse_id,aisle_id,centroid_3d_json "
            "FROM objects WHERE aisle_id=? ORDER BY object_id", (obj["aisle_id"],))]

        nodes = [
            {"id": obj["greenhouse_id"], "type": "Greenhouse",
             "label": obj["greenhouse_id"], "role": "ancestor"},
            {"id": obj["aisle_id"], "type": "Aisle",
             "label": obj["aisle_id"], "role": "ancestor"},
        ]
        for sibling in siblings:
            nodes.append({
                "id": sibling["object_id"], "type": "Object",
                "label": sibling["canonical_label"],
                "role": "focus" if sibling["object_id"] == object_id else "sibling",
                "focus": sibling["object_id"] == object_id,
                "centroid_3d": json.loads(sibling["centroid_3d_json"]),
            })
        nodes.extend({"id": x["view_id"], "type": "View", "label": x["view_id"],
                      "role": "evidence"} for x in observations)

        edges = [
            {"id": f"contains:{obj['greenhouse_id']}:{obj['aisle_id']}",
             "source": obj["greenhouse_id"], "target": obj["aisle_id"],
             "relation": "CONTAINS", "role": "hierarchy"},
        ]
        edges.extend({
            "id": f"contains:{obj['aisle_id']}:{sibling['object_id']}",
            "source": obj["aisle_id"], "target": sibling["object_id"],
            "relation": "CONTAINS",
            "role": "focus" if sibling["object_id"] == object_id else "context",
        } for sibling in siblings)
        edges.extend({"id": x["edge_id"], "source": x["object_id"],
                      "target": x["view_id"], "relation": x["edge_type"],
                      "observation_id": x["observation_id"], "role": "evidence"}
                     for x in observations)
        return {"focus_node_id": object_id, "nodes": nodes, "edges": edges}
    finally:
        db.close()


def related_images(dataset, result):
    images = []

    def add(kind, label, path, view_id=None):
        if not path or not (dataset / path).is_file() or any(x["path"] == path for x in images):
            return
        images.append({"kind": kind, "label": label, "path": path,
                       "url": "/dataset/" + path, "view_id": view_id})

    add("crop", "Object crop", result.get("representative_crop_path"),
        result.get("representative_view_id"))
    add("view", "Representative view", result.get("representative_rgb_path"),
        result.get("representative_view_id"))
    db = sqlite3.connect(dataset / "database/greenhouse_hmsg.sqlite")
    db.row_factory = sqlite3.Row
    try:
        rows = db.execute(
            "SELECT v.view_id,v.rgb_path,o.overlay_path,o.crop_path "
            "FROM object_observations oo "
            "JOIN observations o ON o.observation_id=oo.observation_id "
            "JOIN views v ON v.view_id=o.view_id "
            "WHERE oo.object_id=? ORDER BY v.timestamp_ns", (result["object_id"],)
        ).fetchall()
        for row in rows:
            add("view", row["view_id"], row["rgb_path"], row["view_id"])
            add("overlay", f"{row['view_id']} segmentation", row["overlay_path"], row["view_id"])
            if len(images) >= 14:
                break
    finally:
        db.close()
    return images


def semantic_details(result):
    db = sqlite3.connect(result["_database"])
    try:
        row = db.execute("SELECT parsed_json FROM captions WHERE owner_type='object' AND owner_id=?",
                         (result["object_id"],)).fetchone()
    finally:
        db.close()
    if not row or not row[0]:
        return {"physical_properties": {}, "attributes": {}}
    parsed = json.loads(row[0]); objects = parsed.get("physical_objects") or []
    if not objects:
        return {"physical_properties": {}, "attributes": {}}
    obj = objects[0]
    return {
        "physical_properties": obj.get("physical_properties", {}),
        "attributes": {key: obj.get(key) for key in ("category", "color", "material", "shape", "state")},
    }


def answer_text(result):
    traversal = result.get("traversability") or {}
    label = (result.get("canonical_label") or "object").replace("_", " ")
    aisle = (result.get("aisle_id") or "the current aisle").replace("_", " ")
    object_class = (traversal.get("object_semantic_class") or "unknown").replace("_", " ")
    path = (traversal.get("path_assessment") or "uncertain").replace("_", " ")
    reason = traversal.get("reasoning_summary") or "No semantic explanation is available."
    return (f"I found the {label} in {aisle}. The object is {object_class}, while the observed path is "
            f"{path}. {reason} Geometric verification is still required before robot motion.")


def api_payload(dataset, dialogue_record):
    if not dialogue_record["results"]:
        return {"status": dialogue_record["status"], "answer": dialogue_record["answer"],
                "answer_source": dialogue_record["answer_source"], "intent": dialogue_record["intent"],
                "response_type": dialogue_record.get("response_type", "map_query"),
                "routing": dialogue_record.get("routing"),
                "matched_object": None, "best_view": None, "related_images": [],
                "matched_objects": [], "scene_graph": {"nodes": [], "edges": []}, "retrieval": None}
    enriched = []
    verified_by_id = {
        item["object_id"]: item
        for item in (dialogue_record.get("verified_facts") or {}).get("objects", [])
    }
    for value in dialogue_record["results"]:
        result = dict(value)
        result["_database"] = str(dataset / "database/greenhouse_hmsg.sqlite")
        details = semantic_details(result); result.pop("_database")
        result.update(details)
        verified = verified_by_id.get(result.get("object_id")) or {}
        geometry = verified.get("path_geometry") or {}
        if geometry:
            traversal = dict(result.get("traversability") or {})
            traversal.update({
                "path_assessment": geometry.get("path_assessment"),
                "can_robot_pass": geometry.get("can_robot_pass"),
                "available_clearance_m": geometry.get("available_clearance_m"),
                "required_clearance_m": geometry.get("required_clearance_m"),
                "reasoning_summary": (
                    f"Geometry estimates {geometry.get('available_clearance_m'):.2f} m of side "
                    f"clearance versus {geometry.get('required_clearance_m'):.2f} m required. "
                    "The estimate must be verified with the calibrated robot footprint."
                ),
                "requires_geometric_verification": True,
                "assessment_source": geometry.get("source"),
            })
            result["traversability"] = traversal
        enriched.append(result)
    result = enriched[0]
    images = related_images(dataset, result)
    graph = graph_for_object(dataset, result["object_id"])
    retrieval = dialogue_record.get("retrieval") or {}
    return {"status": "success", "answer": dialogue_record["answer"],
            "answer_source": dialogue_record["answer_source"], "intent": dialogue_record["intent"],
            "response_type": dialogue_record.get("response_type", "map_query"),
            "routing": dialogue_record.get("routing"),
            "verified_facts": dialogue_record["verified_facts"], "matched_object": result,
            "matched_objects": enriched,
            "best_view": {"view_id": result.get("representative_view_id"),
                          "image_url": "/dataset/" + result["representative_rgb_path"] if result.get("representative_rgb_path") else None},
            "related_images": images, "scene_graph": graph,
            "retrieval": {"query_id": retrieval.get("query_id"), "similarity": result.get("similarity"),
                          "ambiguous": retrieval.get("ambiguous", False), "verification": retrieval.get("verification"),
                          "latency_sec": retrieval.get("latency_sec"),
                          "requires_geometric_verification": dialogue_record["intent"] == "assess_traversability"}}


def create_app(dataset_path):
    dataset = Path(dataset_path).resolve()
    if not (dataset / "database/greenhouse_hmsg.sqlite").is_file():
        raise FileNotFoundError("M7 database is missing")
    app = FastAPI(title="Greenhouse Semantic Query", version="8.2.1")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (web_root() / "index.html").read_text()

    @app.get("/api/status")
    def status():
        manifest = json.loads((dataset / "manifest.json").read_text())
        db = sqlite3.connect(dataset / "database/greenhouse_hmsg.sqlite")
        try:
            object_count = db.execute("SELECT COUNT(*) FROM objects").fetchone()[0]
            aisle_count = db.execute("SELECT COUNT(*) FROM aisles").fetchone()[0]
            view_count = db.execute("SELECT COUNT(*) FROM views").fetchone()[0]
        finally:
            db.close()
        return {
            "dataset_id": manifest.get("dataset_id", dataset.name),
            "greenhouse_id": manifest.get("greenhouse_id"),
            "frame_id": manifest.get("frame_id"),
            "object_count": object_count,
            "aisle_count": aisle_count,
            "view_count": view_count,
            "pipeline": "local",
        }

    @app.get("/assets/{name}")
    def asset(name: str):
        if name not in {"style.css", "dialogue.css", "app.js"}:
            raise HTTPException(status_code=404)
        return FileResponse(web_root() / name)

    @app.get("/dataset/{relative:path}")
    def dataset_file(relative: str):
        return FileResponse(safe_dataset_file(dataset, relative))

    @app.get("/api/health")
    def health():
        return {"status": "ok", "dataset": dataset.name}

    @app.post("/api/query")
    async def query(request: QueryRequest):
        try:
            value = await asyncio.to_thread(interact, dataset, request.query.strip(), request.mode,
                                            request.history, request.top_k, request.context_object_id)
            return api_payload(dataset, value)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def reuses_context(text):
    lowered = text.lower()
    return any(x in lowered for x in ("this object", "that object", "it", "vật này", "nó")) and "object_" not in lowered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    dataset = args.dataset.resolve()
    runtime = ensure_ollama(dataset)
    try:
        uvicorn.run(create_app(dataset), host=args.host, port=args.port, log_level="info")
    finally:
        stop_managed_ollama(runtime)


if __name__ == "__main__":
    main()
