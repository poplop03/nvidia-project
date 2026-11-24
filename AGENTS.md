# AGENTS.md — Canonical Agent Instructions

This is the single source of truth for **all AI coding agents** (Claude Code, OpenAI Codex, Gemini/Antigravity, Cursor, Copilot, etc.) working in `/home/duc/VLM`.
Provider files (`CLAUDE.md`, `GEMINI.md`, `.github/copilot-instructions.md`) point here.

## 1. Project Summary

**Greenhouse HMSG (Hierarchical Multi-modal Scene Graph)** on **ROS 2 Humble**:
Offline pipeline transforming Livox Mid-360 LiDAR + RealSense D455 RGB-D + FAST-LIO bags into a queryable 3D semantic map (`Greenhouse → Aisle → View / Object`) using **SAM3**, **OpenAI CLIP (ViT-B-16)**, **Gemma 4 (via Ollama)**, **SQLite**, and a **FastAPI Web UI**.

## 2. Repository Map & Edit Boundaries

| Path | Role | Agent Rule |
|---|---|---|
| `ros2_ws/src/conceptgraphs_bringup/` | Active ROS 2 Python package (code, launch, config, SQL, web, tests) | **EDIT HERE ONLY** |
| `docs/` | Active engineering docs (`HANDOFF.md`, `ARCHITECTURE.md`, `CONTRACTS.md`, `RUNBOOK.md`, `playbooks/`) | Update when architecture, contracts, or handoff state changes |
| `docs/archive/` | Historical milestone logs (`PLAN.md`, `PROJECT_CONTEXT.md`, checkpoints, `.docx` reports) | **Read-only reference** — do not follow outdated `/home/orc/SAM3` paths from here |
| `data/datasets/` | Generated HMSG datasets (images, masks, embeddings, SQLite DBs) | **DO NOT MODIFY** frozen datasets or their `pipeline_config.yaml` unless explicitly requested |
| `data/demo_videos/` & `data/bags/` | Media outputs & external MCAP bags | **DO NOT MODIFY / DO NOT INDEX** |
| `models/sam3.pt` | 3.3 GB SAM3 checkpoint | **DO NOT MODIFY / DO NOT COMMIT** |
| `ros2_ws/build/`, `ros2_ws/install/`, `ros2_ws/log/` | Generated `colcon` outputs | **NEVER EDIT DIRECTLY** — edit `src/` and run `make build` |
| `scratch/` | Gitignored scratchpad for temporary scripts or experiment logs | Safe for disposable experiments; keep root clean |

## 3. Standard Commands (Always Prefer `Makefile`)

```bash
make build          # Source ROS 2 Humble + colcon build --symlink-install
make test           # Run pytest suites (auto-handles ROS PYTHONPATH & -p no:anyio)
make validate-db    # Validate active SQLite HMSG DB integrity, FKs & file paths
make query-db       # List objects in the active dataset via query_hmsg_database
make web            # Start FastAPI Web UI (requires fastapi/uvicorn + Ollama)
```

Active default dataset: `data/datasets/greenhouse_20261029_three_aisles` (override with `DATASET=...`).

## 4. Non-Negotiable Pipeline Invariants

1. **No VLM-invented geometry or IDs:** Centroids, 3D bboxes, aisle assignments, and `object_id`/`view_id` come exclusively from Python/SQLite (M1–M5). Gemma 4 may only describe/verify candidates.
2. **Navigation safety gate:** Every semantic/traversability/dialogue output must preserve `requires_geometric_verification: true`.
3. **Keep ROS package layout intact:** `launch/`, `config/`, `maps/`, `scripts/`, `sql/`, and `web/` must remain inside `ros2_ws/src/conceptgraphs_bringup/` and registered in `setup.py`.
4. **Portable paths only:** Never hardcode `/home/<user>/...` in code or `config/*.yaml`. Use `conceptgraphs_bringup.paths` (`VLM_ROOT`, `VLM_DATASETS_DIR`, `VLM_SAM3_MODEL`, `VLM_WORKER_PYTHON`).
5. **Preserve dataset resume fingerprints:** Frozen datasets in `data/datasets/` keep their original `pipeline_config.yaml`. Runtime path helpers resolve stale `/config/` or python paths automatically without mutating dataset files.

## 5. Multi-Agent Workflow Protocol

- **Start of session:** Read `docs/HANDOFF.md` for current task status, active dataset, and known environment constraints.
- **On-demand context (read only when relevant):**
  - Pipeline stages & module map → `docs/ARCHITECTURE.md` (visuals in `docs/DIAGRAMS.md`)
  - JSONL / SQLite / prompt schemas → `docs/CONTRACTS.md`
  - Full ROS / Web / CLI commands → `docs/RUNBOOK.md`
  - Task playbooks → `docs/playbooks/`
- **End of session:** Run `make test` (and `make validate-db` if touching DB/schema/paths), then update `docs/HANDOFF.md` with what changed, what was verified, and any open items for the next agent.
