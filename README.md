# Greenhouse HMSG — Semantic Mapping & Multimodal Retrieval

ROS 2 Humble pipeline and local interactive workspace for building and querying a **Hierarchical Multi-modal Scene Graph (HMSG)** of agricultural greenhouses using **FAST-LIO**, **SAM3**, **OpenAI CLIP (ViT-B-16)**, and **Gemma 4 (Ollama)**.

## Quickstart

```bash
# 1. Build the ROS 2 package
make build

# 2. Run unit tests
make test

# 3. Validate the latest 3-aisle SQLite HMSG database
make validate-db

# 4. List mapped objects in the active dataset
make query-db

# 5. Start the local Web UI (requires requirements-web.txt + Ollama)
make web
```

## Repository Layout

- `AGENTS.md` — Canonical instructions for AI coding agents across all providers (`CLAUDE.md`, `GEMINI.md`, and `.github/copilot-instructions.md` point here).
- `ros2_ws/src/conceptgraphs_bringup/` — ROS 2 Humble package implementing M1–M8.2 (keyframe extraction, SAM3 segmentation, 3D projection, object association, CLIP/Gemma enrichment, SQLite HMSG DB, fast-to-slow retrieval, and FastAPI Web UI).
- `docs/` — Living engineering docs (`HANDOFF.md`, `ARCHITECTURE.md`, `DIAGRAMS.md`, `CONTRACTS.md`, `RUNBOOK.md`, `playbooks/`) and `docs/archive/` for historical milestone records and `.docx` reports.
- `data/` — Extracted HMSG datasets (`data/datasets/`), external MCAP bags (`data/bags/`), and demo videos (`data/demo_videos/`).
- `models/` — Large model checkpoints (`models/sam3.pt`).
- `scratch/` — Ignored scratchpad for temporary experiments.

## Environment Overrides

All default paths resolve relative to the repository root and can be overridden via environment variables:

- `VLM_ROOT` — Repository root directory
- `VLM_DATASETS_DIR` — Directory containing HMSG datasets (default: `$VLM_ROOT/data/datasets`)
- `VLM_BAGS_DIR` — Directory containing MCAP bags (default: `$VLM_ROOT/data/bags`)
- `VLM_SAM3_MODEL` — Path to `sam3.pt` (default: `$VLM_ROOT/models/sam3.pt`)
- `VLM_WORKER_PYTHON` — Python executable for isolated SAM3/CLIP/Pydantic workers (default: active `sys.executable`)
