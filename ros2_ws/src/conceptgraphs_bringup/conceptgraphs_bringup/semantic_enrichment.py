#!/usr/bin/env python3
"""M6 offline CLIP embeddings and structured Ollama semantic enrichment."""

import argparse
import base64
import hashlib
import html
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import yaml

from .paths import default_dataset, resolve_config_path, resolve_worker_python

DEFAULT = default_dataset()


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def jsonl(path, records):
    records = list(records)
    atomic(path, "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records))
    atomic(
        path.with_suffix(".json"),
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
    )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def b64(path):
    return base64.b64encode(path.read_bytes()).decode()


def schema(python, worker, kind):
    return json.loads(
        subprocess.check_output([python, str(worker), "schema", kind], text=True)
    )


def pydantic_validate(python, worker, kind, value):
    result = subprocess.run(
        [python, str(worker), "validate", kind],
        input=json.dumps(value),
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise ValueError(result.stderr.strip())
    return json.loads(result.stdout)


def preflight(config, override=None):
    ollama = config["semantic_enrichment"]["ollama"]
    base = ollama["base_url"].rstrip("/")
    version = requests.get(base + "/api/version", timeout=5).json()["version"]
    tags = requests.get(base + "/api/tags", timeout=10).json()["models"]
    by_name = {x["name"]: x for x in tags}
    requested = override or ollama["model"]
    actual = requested
    if actual not in by_name:
        fallback = ollama.get("local_fallback_model")
        if not ollama.get("allow_local_fallback") or fallback not in by_name:
            raise RuntimeError(f"Ollama model not installed: {requested}")
        actual = fallback
    show = requests.post(base + "/api/show", json={"model": actual}, timeout=30).json()
    caps = show.get("capabilities", [])
    if "vision" not in caps:
        raise RuntimeError(f"{actual} has no vision capability")
    gpu = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .strip()
        .split(", ")
    )
    return {
        "base_url": base,
        "requested_model": requested,
        "actual_model": actual,
        "fallback_used": actual != requested,
        "ollama_version": version,
        "model_digest": by_name[actual].get("digest"),
        "model_details": show.get("details", {}),
        "capabilities": caps,
        "gpu": {
            "name": gpu[0],
            "memory_total_mib": int(gpu[1]),
            "memory_free_mib": int(gpu[2]),
            "driver_version": gpu[3],
        },
    }


def embedding_manifest(dataset, objects, observations, views):
    by_obs = {x["observation_id"]: x for x in observations}
    view_entries = [
        {
            "entry_type": "view_rgb",
            "view_id": x["view_id"],
            "paths": [str(dataset / x["rgb_path"])],
        }
        for x in views
    ]
    views = {x["view_id"]: x for x in views} if not isinstance(views, dict) else views
    object_entries = []
    for obj in objects:
        crops = [str(dataset / by_obs[x]["crop_path"]) for x in obj["observation_ids"]]
        rep = views[obj["representative_view_id"]]
        object_entries += [
            {
                "entry_type": "object_crop_mean",
                "object_id": obj["object_id"],
                "paths": crops,
            },
            {
                "entry_type": "object_representative_view",
                "object_id": obj["object_id"],
                "view_id": rep["view_id"],
                "paths": [str(dataset / rep["rgb_path"])],
            },
        ]
    return {"views": view_entries, "objects": object_entries}


def quality(o):
    return (
        0.35 * o["sam_confidence"]
        + 0.2 / max(o["median_depth_m"], 0.1)
        + 0.3 * o["valid_depth_ratio"]
        + 0.15
        * min(
            1,
            o["mask_area_pixels"]
            / max(
                1,
                (o["bbox_2d_xyxy"][2] - o["bbox_2d_xyxy"][0])
                * (o["bbox_2d_xyxy"][3] - o["bbox_2d_xyxy"][1]),
            ),
        )
    )


def inputs_for_object(dataset, obj, by_obs, by_view, max_views):
    obs = sorted(
        (by_obs[x] for x in obj["observation_ids"]),
        key=lambda x: (quality(x), x["observation_id"]),
        reverse=True,
    )
    rep = by_obs[obj["representative_observation_id"]]
    view_ids = []
    for o in obs:
        if o["view_id"] not in view_ids:
            view_ids.append(o["view_id"])
    paths = [dataset / rep["crop_path"], dataset / rep["context_crop_path"]] + [
        dataset / by_view[x]["rgb_path"] for x in view_ids[:max_views]
    ]
    return view_ids[:max_views], paths


def call_ollama(pre, prompt, paths, json_schema, temperature):
    payload = {
        "model": pre["actual_model"],
        "prompt": prompt,
        "images": [b64(x) for x in paths],
        "stream": False,
        "format": json_schema,
        "think": False,
        # Some reasoning-capable Ollama models do not reliably stop after
        # completing structured JSON. Bound generation so one malformed
        # response cannot run for tens of thousands of tokens.
        "options": {
            "temperature": temperature,
            "num_predict": 2048,
            "num_ctx": 8192,
        },
    }
    started = time.monotonic()
    response = requests.post(
        pre["base_url"] + "/api/generate", json=payload, timeout=900
    )
    response.raise_for_status()
    value = response.json()
    value["_client_elapsed_sec"] = time.monotonic() - started
    return value


def process_record(
    kind, record_id, prompt, paths, input_ids, pre, cfg, python, worker, outdir
):
    rawpath = outdir / "raw_responses" / f"{kind}_{record_id}.json"
    retries = cfg["maximum_retries"]
    errors = []
    total = 0
    current_prompt = prompt
    json_schema = schema(python, worker, kind)
    if kind == "view":
        expected_objects = len(input_ids["object_ids"])
        physical_objects = json_schema["properties"]["physical_objects"]
        physical_objects["minItems"] = expected_objects
        physical_objects["maxItems"] = expected_objects
    for attempt in range(retries + 1):
        try:
            raw = call_ollama(
                pre,
                current_prompt,
                paths,
                json_schema,
                cfg["temperature"],
            )
            total += raw.get("_client_elapsed_sec", 0)
            atomic(rawpath, json.dumps(raw, indent=2, ensure_ascii=False) + "\n")
            parsed = pydantic_validate(
                python, worker, kind, json.loads(raw["response"])
            )
            if kind == "view":
                expected_objects = len(input_ids["object_ids"])
                actual_objects = len(parsed["physical_objects"])
                if actual_objects != expected_objects:
                    raise ValueError(
                        f"physical_objects must contain exactly {expected_objects} linked "
                        f"Object entries; received {actual_objects}. Remove every extra "
                        "pathway, plant, greenhouse structure, background, and duplicate "
                        "entry; retain only the supplied linked Object."
                    )
            return {
                "status": "complete",
                "attempts": attempt + 1,
                "validation_retries": attempt,
                "error": None,
                "parsed": parsed,
                "raw_response_path": str(rawpath.relative_to(outdir.parent)),
                "input_image_paths": [str(x.relative_to(outdir.parent)) for x in paths],
                "input_ids": input_ids,
                "inference_time_sec": total,
            }
        except Exception as e:
            errors.append(str(e))
            current_prompt = (
                prompt
                + " Previous response failed validation: "
                + str(e)[-800:]
                + " Correct it and return only valid JSON."
            )
    return {
        "status": "failed",
        "attempts": retries + 1,
        "validation_retries": retries,
        "error": errors[-1] if errors else "unknown",
        "errors": errors,
        "parsed": None,
        "raw_response_path": (
            str(rawpath.relative_to(outdir.parent)) if rawpath.exists() else None
        ),
        "input_image_paths": [str(x.relative_to(outdir.parent)) for x in paths],
        "input_ids": input_ids,
        "inference_time_sec": total,
    }


def validate_outputs(
    dataset, object_records, view_records, assessment_records, objects, views
):
    object_ids = {x["object_id"] for x in objects}
    view_ids = {x["view_id"] for x in views}
    errors = []
    for records, key, allowed in (
        (object_records, "object_id", object_ids),
        (view_records, "view_id", view_ids),
        (assessment_records, "view_id", view_ids),
    ):
        ids = [x[key] for x in records]
        if len(ids) != len(set(ids)):
            errors.append(f"duplicate {key}")
        if not set(ids) <= allowed:
            errors.append(f"unknown {key}")
        for x in records:
            for path in x.get("input_image_paths", []):
                if not (dataset / path).is_file():
                    errors.append(f"missing {path}")
    for record in view_records:
        allowed = set(record["input_ids"]["object_ids"])
        if record.get("parsed"):
            cues = record["parsed"]["traversability_cues"]
            referenced = (
                set(cues["blocking_object_ids"])
                | set(cues["traversable_object_ids"])
                | {item["object_id"] for item in cues["visible_obstacles"]}
            )
            if not referenced <= allowed:
                errors.append(f"unknown Object ID in {record['view_id']}")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "valid": True,
        "object_records": len(object_records),
        "view_records": len(view_records),
        "assessment_records": len(assessment_records),
        "path_and_reference_integrity": True,
    }


def preview(dataset, objects, object_records, view_records):
    rows = []
    by = {x["object_id"]: x for x in object_records}
    for obj in objects:
        if obj["object_id"] not in by:
            continue
        r = by[obj["object_id"]]
        image = r["input_image_paths"][0] if r.get("input_image_paths") else ""
        rows.append(
            f"<section><h2>{html.escape(obj['object_id'])}</h2><img src='../{html.escape(image)}' width='360'><pre>{html.escape(json.dumps(r.get('parsed'),indent=2,ensure_ascii=False))}</pre></section>"
        )
    atomic(
        dataset / "previews/semantic_review.html",
        "<html><meta charset='utf-8'><h1>M6 semantic review</h1>"
        + "".join(rows)
        + f"<h2>View records: {len(view_records)}</h2></html>",
    )


def run(
    dataset,
    object_limit=None,
    view_limit=None,
    force=False,
    model_override=None,
    skip_embeddings=False,
):
    dataset = dataset.resolve()
    cfg = yaml.safe_load((dataset / "pipeline_config.yaml").read_text())
    sec = cfg["semantic_enrichment"]
    objects = lines(dataset / "objects/objects.jsonl")
    observations = lines(dataset / "observations/observations_3d.jsonl")
    views = lines(dataset / "keyframes/views.jsonl")
    edges = lines(dataset / "graph/edges.jsonl")
    by_obs = {x["observation_id"]: x for x in observations}
    by_view = {x["view_id"]: x for x in views}
    pre = preflight(cfg, model_override)
    source = Path(__file__).resolve().parents[1]
    python = resolve_worker_python(sec.get("python"))
    schema_worker = source / "scripts/semantic_schema_worker.py"
    clip_worker = source / "scripts/clip_embedding_worker.py"
    out = dataset / "semantics"
    (out / "raw_responses").mkdir(parents=True, exist_ok=True)
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    report_path = dataset / "reports/semantic_enrichment_report.json"
    prompt_config_path = resolve_config_path(sec["prompt_config"], dataset)
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "semantic_enrichment": sec,
                "actual_model": pre["actual_model"],
                "model_digest": pre["model_digest"],
                "objects": digest(dataset / "objects/objects.jsonl"),
                "views": digest(dataset / "keyframes/views.jsonl"),
                "edges": digest(dataset / "graph/edges.jsonl"),
                "prompt_config": digest(prompt_config_path),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if report_path.exists() and not force:
        previous = json.loads(report_path.read_text())
        if (
            previous.get("config_fingerprint")
            and previous["config_fingerprint"] != fingerprint
        ):
            raise ValueError("Semantic inputs/config changed; rerun with --force")
    prompt_config = yaml.safe_load(prompt_config_path.read_text())
    if prompt_config["version"] != sec["prompt_version"]:
        raise ValueError(
            "Prompt config version does not match semantic_enrichment.prompt_version"
        )
    common_rules = prompt_config["common_rules"].strip()
    prompts = {
        "object": common_rules + "\n\n" + prompt_config["object_prompt"].strip(),
        "view": common_rules + "\n\n" + prompt_config["view_prompt"].strip(),
    }
    atomic(
        out / "prompts" / f'{sec["prompt_version"]}.yaml',
        prompt_config_path.read_text(),
    )
    for k, v in prompts.items():
        atomic(out / "prompts" / f'{k}_{sec["prompt_version"]}.txt', v + "\n")
        atomic(
            out / "prompts" / f"{k}_schema.json",
            json.dumps(schema(python, schema_worker, k), indent=2) + "\n",
        )
    if not skip_embeddings:
        manifest = embedding_manifest(dataset, objects, observations, views)
        mp = dataset / "embeddings/.clip_manifest.json"
        atomic(mp, json.dumps(manifest))
        subprocess.run(
            [
                python,
                str(clip_worker),
                "--manifest",
                str(mp),
                "--output-dir",
                str(dataset / "embeddings"),
                "--model",
                sec["clip"]["model_name"],
                "--device",
                sec["clip"]["device"],
            ],
            check=True,
        )
        mp.unlink()
    selected_objects = objects[:object_limit] if object_limit else objects
    linked = sorted({e["source_id"] for e in edges if e["edge_type"] == "OBSERVES"})
    selected_views = linked[:view_limit] if view_limit else linked
    paths = {
        "object": out / "object_captions.jsonl",
        "view": out / "view_captions.jsonl",
        "assessment": out / "traversability_assessments.jsonl",
    }
    if force:
        for p in list(paths.values()) + [
            dataset / "reports/semantic_enrichment_report.json"
        ]:
            if p.exists():
                p.unlink()
    existing = {
        k: (
            {x[k + "_id"] if k != "assessment" else x["view_id"]: x for x in lines(p)}
            if p.exists()
            else {}
        )
        for k, p in paths.items()
    }
    started = time.monotonic()
    for obj in selected_objects:
        oid = obj["object_id"]
        if oid in existing["object"]:
            continue
        vids, images = inputs_for_object(
            dataset, obj, by_obs, by_view, sec["maximum_object_views"]
        )
        object_prompt = (
            prompts["object"]
            + f"\n\nTarget object ID: {oid}."
            + f"\nKnown canonical label: {obj['canonical_label']}."
            + "\nDescribe exactly this target object and no other object in physical_objects."
        )
        result = process_record(
            "object",
            oid,
            object_prompt,
            images,
            {
                "object_id": oid,
                "view_ids": vids,
                "observation_ids": obj["observation_ids"],
            },
            pre,
            sec,
            python,
            schema_worker,
            out,
        )
        existing["object"][oid] = {
            "object_id": oid,
            "model_tag": pre["actual_model"],
            "model_digest": pre["model_digest"],
            "prompt_version": sec["prompt_version"],
        } | result
        jsonl(
            paths["object"],
            sorted(existing["object"].values(), key=lambda x: x["object_id"]),
        )
    linked_objects = defaultdict(list)
    for e in edges:
        linked_objects[e["source_id"]].append(e["target_id"])
    for vid in selected_views:
        if vid in existing["view"]:
            continue
        images = [dataset / by_view[vid]["rgb_path"]]
        ids = sorted(set(linked_objects[vid]))
        labels = {o["object_id"]: o["canonical_label"] for o in objects}
        prompt = (
            prompts["view"]
            + "\n\nLinked objects: "
            + ", ".join(f"{x} ({labels[x]})" for x in ids)
            + f"\nReturn exactly {len(ids)} physical_objects entries, one for each linked "
            + "object in the listed order. Do not include the pathway, plants, greenhouse "
            + "structure, or any unlinked item in physical_objects. A rigid container beside "
            + "the path belongs in neither blocking_object_ids nor traversable_object_ids; "
            + "it may still appear in visible_obstacles as non_traversable while "
            + "occupies_drivable_space is false."
        )
        result = process_record(
            "view",
            vid,
            prompt,
            images,
            {"view_id": vid, "object_ids": ids},
            pre,
            sec,
            python,
            schema_worker,
            out,
        )
        base = {
            "view_id": vid,
            "model_tag": pre["actual_model"],
            "model_digest": pre["model_digest"],
            "prompt_version": sec["prompt_version"],
        } | result
        existing["view"][vid] = base
        cues = base["parsed"]["traversability_cues"] if base["parsed"] else None
        existing["assessment"][vid] = {
            "view_id": vid,
            "status": base["status"],
            "model_tag": pre["actual_model"],
            "model_digest": pre["model_digest"],
            "prompt_version": sec["prompt_version"],
            "passability": cues["vlm_path_assessment"] if cues else None,
            "can_robot_pass": cues["can_robot_pass"] if cues else None,
            "blocking_object_ids": cues["blocking_object_ids"] if cues else [],
            "traversable_object_ids": cues["traversable_object_ids"] if cues else [],
            "visible_obstacles": cues["visible_obstacles"] if cues else [],
            "reasoning_summary": cues["reasoning_summary"] if cues else None,
            "confidence": cues["confidence"] if cues else None,
            "uncertain_attributes": (
                base["parsed"]["uncertain_attributes"] if base["parsed"] else []
            ),
            "requires_geometric_verification": True,
            "source_view_caption_status": base["status"],
            "error": base.get("error"),
        }
        jsonl(
            paths["view"], sorted(existing["view"].values(), key=lambda x: x["view_id"])
        )
        jsonl(
            paths["assessment"],
            sorted(existing["assessment"].values(), key=lambda x: x["view_id"]),
        )
    object_records = sorted(existing["object"].values(), key=lambda x: x["object_id"])
    view_records = sorted(existing["view"].values(), key=lambda x: x["view_id"])
    assessments = sorted(existing["assessment"].values(), key=lambda x: x["view_id"])
    validation = validate_outputs(
        dataset, object_records, view_records, assessments, objects, views
    )
    preview(dataset, objects, object_records, view_records)
    records = object_records + view_records
    statuses = Counter(x["status"] for x in records)
    uncertain = sum(
        bool(x.get("parsed", {}).get("uncertain_attributes"))
        for x in records
        if x.get("parsed")
    )
    report = {
        "dataset_id": cfg["dataset_id"],
        "stage": "M6_multimodal_enrichment",
        "status": (
            "complete"
            if len(object_records) == len(objects) and len(view_records) == len(linked)
            else "partial"
        ),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "preflight": pre,
        "clip": (
            json.loads((dataset / "embeddings/index_metadata.json").read_text())
            if (dataset / "embeddings/index_metadata.json").exists()
            else None
        ),
        "counts": {
            "object_total": len(objects),
            "object_records": len(object_records),
            "linked_view_total": len(linked),
            "view_records": len(view_records),
            "complete": statuses["complete"],
            "failed": statuses["failed"],
            "uncertain": uncertain,
        },
        "json_validation": {
            "successful": statuses["complete"],
            "failed": statuses["failed"],
            "retry_count": sum(x.get("validation_retries", 0) for x in records),
            "attempt_count": sum(x.get("attempts", 0) for x in records),
        },
        "inference": {
            "total_model_time_sec": sum(
                x.get("inference_time_sec", 0) for x in records
            ),
            "last_run_wall_time_sec": time.monotonic() - started,
        },
        "validation": validation,
        "prompt": {
            "version": sec["prompt_version"],
            "config_source": str(prompt_config_path),
            "snapshot": f'semantics/prompts/{sec["prompt_version"]}.yaml',
            "object_schema": "semantics/prompts/object_schema.json",
            "view_schema": "semantics/prompts/view_schema.json",
        },
        "preview": "previews/semantic_review.html",
        "config_fingerprint": fingerprint,
    }
    atomic(report_path, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("stages", {})["multimodal_enrichment"] = {
        "status": report["status"],
        "view_embeddings": "embeddings/view_embeddings.npy",
        "object_embeddings": "embeddings/object_embeddings.npy",
        "object_captions": "semantics/object_captions.jsonl",
        "object_captions_json": "semantics/object_captions.json",
        "view_captions": "semantics/view_captions.jsonl",
        "view_captions_json": "semantics/view_captions.json",
        "traversability_assessments": "semantics/traversability_assessments.jsonl",
        "traversability_assessments_json": "semantics/traversability_assessments.json",
        "report": "reports/semantic_enrichment_report.json",
        "model_tag": pre["actual_model"],
        "model_digest": pre["model_digest"],
        "prompt_version": sec["prompt_version"],
        "prompt_config": f'semantics/prompts/{sec["prompt_version"]}.yaml',
    }
    atomic(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return report


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=DEFAULT)
    p.add_argument("--object-limit", type=int)
    p.add_argument("--view-limit", type=int)
    p.add_argument("--force", action="store_true")
    p.add_argument("--model")
    p.add_argument("--skip-embeddings", action="store_true")
    a = p.parse_args(argv)
    print(
        json.dumps(
            run(
                a.dataset,
                a.object_limit,
                a.view_limit,
                a.force,
                a.model,
                a.skip_embeddings,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
