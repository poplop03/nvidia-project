# Greenhouse HMSG — Current System Inputs & Outputs

The current system has **two separate stages**, each with its own input and output:

## 1. Offline Map-Building Pipeline (M1–M7)

- **Inputs:**
  - Recorded ROS 2 MCAP rosbags (Livox Mid-360 LiDAR + RealSense D455 RGB-D + FAST-LIO robot poses)
  - Greenhouse aisle layout YAML (`config/layouts/*.yaml`)
  - Detection & caption prompt configs (`config/*_prompts*.yaml`)
- **Outputs (saved under `data/datasets/<dataset_id>/`):**
  - `database/greenhouse_hmsg.sqlite` — structured 3D scene graph (`Greenhouse → Aisle → View / Object` with 3D centroids, bounding boxes, captions, and traversability cues)
  - `embeddings/*.npy` — 512-D CLIP image vectors
  - `keyframes/rgb/` & `observations/crops/` — saved RGB frames and cropped object images

## 2. Interactive Query & Web UI (M8–M8.2.1)

- **Inputs:**
  - A **text question** from the user (English or Vietnamese), e.g. *"Where is the wheelbarrow?"* or *"Can the robot pass the ladder in aisle 2?"*
  - Optional chat history & mode (`auto`, `chat`, or `map`) via CLI or the FastAPI Web UI (`ros2_ws/src/conceptgraphs_bringup/conceptgraphs_bringup/query_web.py`)
- **Outputs:**
  - A **grounded natural-language answer** (composed by Gemma 4 using SQLite facts)
  - **Deterministic 3D map data** from SQLite: `object_id`, `aisle_id`, `(x, y, z)` centroid in `camera_init`, distance from aisle head/tail, physical properties, and traversability assessment
  - **Visual evidence** from the dataset: the matched object’s stored RGB crop/view image and its local scene graph
  - **Safety gate:** `requires_geometric_verification: true` (it answers questions; it does not move the robot)

---

# Expansion Draft — Greenhouse 4D Watch (Option 1 + Option 4)

## The Core Idea

- **Lab mate owns:** Static 3D Scene Graph per run (`M1–M8`: *what is in the greenhouse right now?*).
- **My contribution:** **Greenhouse 4D Watch** — a continuous **Change-Detection, Triage Agent & H100-to-Edge VLM Flywheel** (*what changed between runs, what needs human action, and how do we run this reliably on both H100 servers and edge hardware?*).

The two tracks form a **closed loop**:
1. **Track A (Option 1)** generates real multi-run diffs, agent traces, and human operator approvals/corrections.
2. **Track B (Option 4)** uses those grounded diffs + H100 teacher models to distill a fast, safe, schema-strict Edge VLM.
3. That distilled Edge VLM plugs right back into **Track A** (and M8 dialogue) through the same OpenAI-compatible HTTP gateway.

## End-to-End Architecture Draft

```text
[Lab Mate's HMSG Outputs — Read-Only, Untouched]
  data/datasets/run_t0 (greenhouse_hmsg.sqlite + crops + embeddings)
  data/datasets/run_t1 (greenhouse_hmsg.sqlite + crops + embeddings)
  ...
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│  TRACK A: 4D Change-Detection & Triage Platform (Option 1)           │
│                                                                      │
│  1. Ingest & Diff Engine (Python + Postgres + pgvector)              │
│     • Loads run snapshots into Postgres (runs, object_snapshots)     │
│     • Deterministic 3D + vector matching across runs:                │
│       UNCHANGED | MOVED | APPEARED | DISAPPEARED | STATE_CHANGED     │
│     • Writes structured events to change_events (keeps M1–M5 truth)  │
│                                                                      │
│  2. Paired-Image VLM Diff Verifier (via Model Gateway)               │
│     • Sends crop(t0) + crop(t1) + deterministic 3D delta to VLM      │
│     • Outputs strict JSON: change_summary, hazard_level,             │
│       traversability_impact, requires_geometric_verification: true   │
│                                                                      │
│  3. HMSG-4D MCP Server + Triage Agent (FastAPI + LangGraph)          │
│     • MCP tools: list_changes, get_object_history, aisle_risk_diff,  │
│       draft_inspection_ticket                                        │
│     • Human-in-the-loop (HITL) approval gate before escalating/      │
│       dispatching alerts or robot re-inspection tasks                │
│                                                                      │
│  4. Operator Console (TypeScript + React + Docker + AWS)             │
│     • Side-by-side t0/t1 visual diff viewer + aisle risk map         │
│     • Approve / reject / correct alerts (feeds Track B!)             │
└──────────────────────────┬───────────────────────────▲───────────────┘
                           │ Verified diffs +          │ Drop-in model
                           │ operator corrections      │ via HTTP gateway
                           ▼                           │
┌──────────────────────────────────────────────────────────────────────┐
│  TRACK B: H100-to-Edge VLM Flywheel (Option 4 — 8× H100s)            │
│                                                                      │
│  5. Grounded Dataset & Preference Generator (vLLM / NVIDIA NIM)      │
│     • 70B+ Teacher VLM (e.g. Qwen2.5-VL-72B / Gemma 4 27B) +         │
│       Postgres ground truth → thousands of verified examples         │
│     • Builds DPO pairs:                                              │
│       Chosen  = valid JSON, real object_ids, safety gate = true      │
│       Rejected = hallucinated IDs/coords, prompt-injection bypasses  │
│                                                                      │
│  6. Post-Training on 8× H100s (LoRA SFT → DPO)                       │
│     • Distills teacher + safety rules into a 3B–8B Edge VLM          │
│                                                                      │
│  7. Automated Eval & Serving Benchmark (Langfuse + 100+ cases)       │
│     • Compares: Frontier API vs 70B Teacher vs Base 4B/8B vs         │
│       Distilled 4B/8B (served on vLLM/NIM & TensorRT-LLM/Ollama)     │
│     • Tracks: diff accuracy, 0% ID hallucination, safety-bypass      │
│       rate, p95 latency, $/1k queries                                │
└──────────────────────────────────────────────────────────────────────┘
```

## How Far This Can Go (The Ceiling)

1. **Level 1 — Working 4D Diff Backend + Model Gateway (Core Foundation)**
   - Ingests multiple HMSG datasets (starting with `greenhouse_20260910_aisle01` vs. `greenhouse_20261029_aisle01` and synthetic perturbations of `greenhouse_20261029_three_aisles`) into **Postgres + `pgvector`**.
   - Deterministic 3D + CLIP/DINOv2 diff engine + FastAPI service + OpenAI-compatible gateway talking to **vLLM / NVIDIA NIM** on the 8× H100s.

2. **Level 2 — Agentic Triage + Evals + Cloud/Docker Shipping (Full FDE Portfolio)**
   - Exposes the 4D diff DB via an **MCP Server**, adds a **LangGraph triage agent** with HITL approval, **Langfuse** tracing, and a **100+ case eval & red-team suite**.
   - Packaged in **Docker Compose**, deployed with **AWS + Terraform + GitHub Actions**, with a **TypeScript + React** diff & approval dashboard.

3. **Level 3 — H100 Distillation & Edge Deployment (NVIDIA Sponsor / Paper Showcase)**
   - Uses the 8× H100s to generate grounded SFT + DPO datasets from the 70B teacher + Postgres truth, fine-tunes a **3B–8B greenhouse specialist VLM**, and benchmarks it on **TensorRT-LLM / NVIDIA NIM / Ollama** (edge-ready for Jetson Orin).

## Track B (Steps 5–6) Quick Reference

| Step | Input | Output |
|---|---|---|
| **5. Teacher (70B on H100)** | `crop(t0)` + `crop(t1)` + Postgres deterministic facts + prompt/schema | Verified gold JSON answers + DPO `(chosen, rejected)` pairs |
| **6a. LoRA SFT (3B–8B)** | `(images, prompt + facts, gold_JSON)` JSONL | SFT LoRA adapter (learns schema & domain) |
| **6b. DPO (3B–8B)** | `(images, prompt + facts, chosen, rejected)` JSONL | Aligned adapter → merged **GGUF / vLLM / TensorRT-LLM** edge model |

## Clean Repo Boundary

- **Read-only:** Never modify `data/datasets/*` or M1–M7 perception scripts.
- **New platform location:** Keep in a clean sibling folder (e.g., `services/hmsg_4d/` with its own `pyproject.toml`, `uv.lock`, `docker-compose.yml`, and `eval/`) reading the SQLite DBs + crops as an upstream data source.
