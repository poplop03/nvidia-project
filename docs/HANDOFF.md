# Multi-Agent Handoff State

> **Read this at the start of every session and update it when completing significant work.**

## Current Status
- **Active dataset:** `data/datasets/greenhouse_20261029_three_aisles` (3 aisles, 209 views, 190 observations, 6 objects, Gemma 4 12B semantic enrichment, verified HMSG SQLite DB).
- **Pipeline milestones completed:** M1 (Aisle layout) through M8.2.1 (Intent-aware semantic dialogue + Auto/Chat/Map routing Web UI).
- **Last structural change (2026-11-23):** Restructured `/home/duc/VLM` into a multi-provider agent-friendly layout:
  - Created canonical `AGENTS.md` with provider symlinks (`CLAUDE.md`, `GEMINI.md`) and `.github/copilot-instructions.md`.
  - Moved datasets → `data/datasets/`, demo videos → `data/demo_videos/`, model weights → `models/sam3.pt`, historical docs & `.docx` reports → `docs/archive/`.
  - Removed legacy duplicate colcon trees (`build_ros/`, `install_ros/`, `log_ros/`) and empty hidden placeholder dirs.
  - Added portable path resolution (`conceptgraphs_bringup/paths.py`) and removed hardcoded `/home/orc/SAM3` & `/home/orc/anaconda3/bin/python` defaults from source, launch files, configs, and tests without modifying frozen dataset artifacts.
  - Added root `Makefile`, `.gitignore`, `.ignore`, and concise docs (`docs/ARCHITECTURE.md`, `docs/DIAGRAMS.md` with 6 SVG/PNG/Mermaid diagrams under `docs/diagrams/`, `docs/CONTRACTS.md`, `docs/RUNBOOK.md`).

## Local Environment Notes (`/home/duc/VLM`)
- **ROS 2:** Humble installed at `/opt/ros/humble`.
- **Web dependencies:** `fastapi` and `uvicorn` are not yet installed in system `python3` (`ros2_ws/src/conceptgraphs_bringup/requirements-web.txt`), so `make test` automatically skips `test_query_web.py` until installed (`make test-all` runs all 13 suites).
- **Pytest plugin note:** System `pytest` conflicts with `~/.local` `anyio` plugin; `make test` passes `-p no:anyio` automatically.
- **Raw MCAP rosbags:** Stored externally (not bundled in `data/bags/`). All offline HMSG datasets in `data/datasets/` are self-contained and queryable.

## Known Open Engineering Items (from Nov 19 Evaluation Report)
- Improve route-level navigation reasoning, ambiguous object references, and conversational pronoun resolution in `semantic_dialogue.py` / `hmsg_retrieval.py` (see `docs/archive/reports/Greenhouse_HMSG_Five_Group_Evaluation_Report.docx`).
- M9 (navigation integration / geometric free-space fusion) has not started.

## Next Recommended Action
- Install `requirements-web.txt` (if running the Web UI locally) and verify `make test-all` + `make web`.
- Or pick up one of the open dialogue/retrieval improvements in `semantic_dialogue.py`.
