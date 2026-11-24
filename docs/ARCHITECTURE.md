# Architecture — Greenhouse HMSG Pipeline

> **Visual Diagrams:** See [DIAGRAMS.md](DIAGRAMS.md) for all 6 rendered SVG/PNG and Mermaid architecture diagrams (pipeline, HMSG schema, query sequence, 3D transforms, safety boundary, and multi-agent repo layout).

## 1. Hierarchy & Core Graph Model

```text
greenhouse_01
├── aisle_01
├── aisle_02
└── aisle_03
    ├── View (RGB + aligned depth + CameraInfo + pose in camera_init)
    └── Object (persistent 3D centroid + bbox + representative View)

Edges: View ──OBSERVES──> Object
```

## 2. Offline Pipeline Stages (M1–M8.2.1)

| Stage | Executable / Entry Point | Primary Module | Inputs → Outputs |
|---|---|---|---|
| **M1 Layout** | `greenhouse_layout_markers` | `greenhouse_layout.py` | Layout YAML (`config/layouts/*.yaml`) → aisle polygons, centerlines, head/tail |
| **M2 Keyframes** | `extract_keyframes` | `keyframe_extractor.py` | MCAP bag → `keyframes/{rgb,depth,camera_info,views.jsonl}` |
| **M3 SAM3 2D** | `run_sam3_offline` | `sam3_offline.py` + `scripts/sam3_inference_worker.py` | Views + `models/sam3.pt` + `config/sam3_prompts.yaml` → `observations/{masks,crops,context_crops,observations.jsonl}` |
| **M4 3D Projection** | `project_observations_3d` | `observation_projection.py` | 2D masks + aligned depth + poses → `observations/observations_3d.jsonl` |
| **M5 Association** | `associate_objects` | `object_association.py` | Valid M4 observations → `objects/objects.jsonl`, `graph/{nodes,edges}.jsonl` |
| **M6 Enrichment** | `enrich_semantics` | `semantic_enrichment.py` + `scripts/clip_embedding_worker.py` + `scripts/semantic_schema_worker.py` | CLIP `ViT-B-16` → `embeddings/*.npy`; Ollama Gemma 4 → `semantics/{object_captions,view_captions,traversability_assessments}.jsonl` |
| **M7 Database** | `build_hmsg_database` / `query_hmsg_database` | `hmsg_database.py`, `hmsg_query.py`, `sql/001_initial_hmsg.sql` | JSONL artifacts → `database/greenhouse_hmsg.sqlite` (relative paths only) |
| **M8 Retrieval** | `query_hmsg` | `hmsg_retrieval.py` + `scripts/clip_text_worker.py` | NL query → rule/Gemma parser → SQLite filter → CLIP ranking → ambiguity-gated Gemma top-k verification |
| **M8.1–8.2.1 Web & Dialogue** | `query_hmsg_web` | `query_web.py`, `semantic_dialogue.py`, `web/*` | FastAPI + Auto/Chat/Map router + centerline spatial reasoner + grounded Gemma composer |

## 3. Auxiliary / Real-Time ROS 2 Modules
- **Path resolution:** `paths.py` (portable env/relative path helpers used across nodes, launch files, and pipeline stages).
- **Online SAM3 mapper & RViz QA:** `sam3_crate_mapper.py`, `keyframe_marker_publisher.py`, `observation_3d_marker_publisher.py`, `object_marker_publisher.py`, `greenhouse_layout_markers.py`, `crate_marker_publisher.py`.
- **Point cloud / bag prep:** `z_passthrough_filter.py`, `registered_cloud_dynamic_filter.py`, `depth_to_color_align.py`, `rear_person_filter.py`, `convert_kilted_bag.py`, `rgb_frame_extractor.py`.

## 4. Datasets Inventory (`data/datasets/`)
- `greenhouse_20261029_three_aisles` (**active**): 3 aisles, 209 views, 190 obs, 25 valid 3D, 6 objects (`blue/red/yellow_plastic_container`, `ladder`, `wheelbarrow`), enriched with `gemma4:12b-it-q4_K_M`.
- `greenhouse_20261029_aisle01`: 1 aisle, 108 views, 116 obs.
- `greenhouse_20260910_aisle01`: Original single-aisle pilot (108 views, 83 obs, 2 objects, `gemma4:e4b`).
- `archive/greenhouse_20261029_three_aisles_backup_12b`: Backup snapshot.
