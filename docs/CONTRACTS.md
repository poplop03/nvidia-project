# Data, Schema & Safety Contracts

## 1. Coordinate Frames & Spatial Truth
- Map frame: `camera_init`.
- Camera optical frame: `camera_color_optical_frame`.
- All 3D centroids, 3D bounding boxes, aisle assignments, and centerline projections (`head` / `middle` / `tail`, distance from head/tail, centerline offset) are computed deterministically in Python from LiDAR/FAST-LIO poses + aligned depth + layout YAML.
- **VLM restriction:** Gemma 4 must never invent or overwrite `object_id`, `view_id`, `aisle_id`, or metric geometry.

## 2. Safety & Traversability Separation
- Object traversability (`physical_objects[].object_traversability`) and route passability (`traversability_cues.vlm_path_assessment` / `can_robot_pass`) are distinct concepts and must never be conflated.
- Every VLM-derived semantic or traversability output **must** enforce `requires_geometric_verification: true`.
- Neither M8 retrieval nor M8.2 dialogue commands robot motion or computes Nav2 local paths.

## 3. Dataset Directory Contract (`data/datasets/<dataset_id>/`)
- `manifest.json` — Stage completion metadata & file index.
- `pipeline_config.yaml` — Frozen config snapshot for the run (do not mutate in-place on completed datasets; stages check config fingerprints on resume).
- `keyframes/{rgb,depth,camera_info,views.jsonl}`
- `observations/{masks,crops,context_crops,observations.jsonl,observations_3d.jsonl}`
- `objects/objects.jsonl` & `graph/{nodes,edges}.jsonl`
- `embeddings/{view_embeddings.npy,object_embeddings.npy,index_metadata.json}` — L2-normalized float32 CLIP `ViT-B-16` (512-D) vectors.
- `semantics/{object_captions,view_captions,traversability_assessments}.jsonl`
- `database/greenhouse_hmsg.sqlite` — Built from `sql/001_initial_hmsg.sql`; stores `dataset_root = '.'` and **relative paths only** (no binary images in SQLite).

## 4. Prompt & Config Externalization
Prompts live in `ros2_ws/src/conceptgraphs_bringup/config/` (never hardcode prompts in Python):
- `sam3_prompts.yaml` (`greenhouse_objects_v2`)
- `semantic_prompts_v2.yaml` (`greenhouse_semantics_v2`)
- `retrieval_prompts_v1.yaml` (`greenhouse_retrieval_v1`)
- `dialogue_prompts_v1.yaml`
- `general_chat_prompts_v1.yaml`

## 5. Worker Subprocess Contract
Heavy ML dependencies (`ultralytics` SAM3, `clip`, `pydantic` schema validation) run in isolated worker scripts under `ros2_ws/src/conceptgraphs_bringup/scripts/`. Python executables resolve via `paths.resolve_worker_python()` (`VLM_WORKER_PYTHON` → configured executable if present → `sys.executable`).
