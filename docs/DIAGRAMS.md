# Greenhouse HMSG — Team Architecture Diagrams

Visual documentation of the Greenhouse HMSG system for robotics, perception, VLM/AI, and web teammates.

Each diagram is provided in **three formats**:
- **SVG** (`docs/diagrams/*.svg`) — crisp vector graphics for docs, wikis, and web
- **High-DPI PNG** (`docs/diagrams/*.png`) — ready to drop into slides and `.docx` reports
- **Editable Mermaid source** (below) — renders natively in GitHub, VS Code, Notion, and GitLab

To regenerate all SVG and PNG files after editing layout/text:
```bash
python3 docs/diagrams/generate_diagrams.py
```

| # | Diagram | Target Audience | Assets |
|---|---|---|---|
| 1 | End-to-End Pipeline (M1–M8.2.1) | Whole team / stakeholders | [SVG](diagrams/01_end_to_end_pipeline.svg) · [PNG](diagrams/01_end_to_end_pipeline.png) |
| 2 | HMSG Scene Graph & SQLite Schema | Backend & data engineers | [SVG](diagrams/02_hmsg_scene_graph_and_schema.svg) · [PNG](diagrams/02_hmsg_scene_graph_and_schema.png) |
| 3 | Fast-to-Slow Query & Dialogue Sequence | VLM / AI & Web engineers | [SVG](diagrams/03_fast_to_slow_query_sequence.svg) · [PNG](diagrams/03_fast_to_slow_query_sequence.png) |
| 4 | 3D Coordinate & Frame Transform Chain | SLAM & perception engineers | [SVG](diagrams/04_coordinate_3d_projection_chain.svg) · [PNG](diagrams/04_coordinate_3d_projection_chain.png) |
| 5 | Truth & Navigation Safety Boundary | Robotics safety / Nav2 / KPI leads | [SVG](diagrams/05_truth_and_safety_boundary.svg) · [PNG](diagrams/05_truth_and_safety_boundary.png) |
| 6 | Multi-Provider Agent Workspace & Edit Boundaries | Engineering leads & AI agents | [SVG](diagrams/06_repo_boundaries_and_agents.svg) · [PNG](diagrams/06_repo_boundaries_and_agents.png) |

---

## 1. End-to-End Pipeline (Offline Mapping + Online Query)

![End-to-End Pipeline](diagrams/01_end_to_end_pipeline.svg)

```mermaid
flowchart LR
    subgraph OFFLINE["Offline Map Building (M1-M7)"]
        BAG["MCAP Bag (Mid-360 + D455 + FAST-LIO)"] --> M2["M2: extract_keyframes (209 Views)"]
        LAYOUT["M1: Aisle Layout YAML"] --> M2
        M2 --> M3["M3: run_sam3_offline (sam3.pt)"]
        M3 --> M4["M4: project_observations_3d"]
        M4 --> M5["M5: associate_objects (6 Objects)"]
        M5 --> M6A["M6a: CLIP ViT-B-16 Embeddings"]
        M5 --> M6B["M6b: Gemma 4 Captions & Traversability"]
        M6A --> M7["M7: build_hmsg_database"]
        M6B --> M7
    end

    subgraph STORE["Self-Contained HMSG Dataset"]
        DB[("SQLite DB (greenhouse_hmsg.sqlite)")]
        FS["Filesystem (RGB, Depth, Masks, NPY)"]
    end

    M7 --> DB
    M7 --> FS

    subgraph ONLINE["Interactive Query & Dialogue (M8-M8.2.1)"]
        USER["User / Operator"] --> UI["FastAPI Web UI (Auto / Chat / Map)"]
        UI --> RET["M8: SQLite Filter + CLIP Fast Search"]
        RET --> VER["Gemma 4 Top-k Verification & Grounded Composer"]
        VER --> UI
    end

    DB --> RET
    FS --> RET
    FS --> UI
```

---

## 2. HMSG Scene Graph Hierarchy & SQLite Schema

![HMSG Scene Graph and SQLite Schema](diagrams/02_hmsg_scene_graph_and_schema.svg)

```mermaid
erDiagram
    DATASETS ||--o{ GREENHOUSES : contains
    GREENHOUSES ||--o{ AISLES : contains
    AISLES ||--o{ VIEWS : contains
    AISLES ||--o{ OBJECTS : contains
    VIEWS ||--o{ OBSERVATIONS : yields
    OBJECTS ||--o{ OBJECT_OBSERVATIONS : aggregates
    OBSERVATIONS ||--o{ OBJECT_OBSERVATIONS : member_of
    VIEWS ||--o{ VIEW_OBJECT_EDGES : "OBSERVES"
    OBJECTS ||--o{ VIEW_OBJECT_EDGES : observed_by
    VIEWS ||--o| TRAVERSABILITY_ASSESSMENTS : evaluated_in
    OBJECTS ||--o{ CAPTIONS : described_by
    VIEWS ||--o{ CAPTIONS : described_by
    OBJECTS ||--o{ EMBEDDING_INDEXES : indexed_in
    VIEWS ||--o{ EMBEDDING_INDEXES : indexed_in

    OBJECTS {
        text object_id PK
        text aisle_id FK
        text canonical_label
        text centroid_3d_json
        text representative_view_id FK
    }
    VIEWS {
        text view_id PK
        text aisle_id FK
        text rgb_path
        text depth_path
        real aisle_coordinate_s
        text aisle_section
    }
    EMBEDDING_INDEXES {
        text embedding_id PK
        text owner_type
        text owner_id
        text npy_path
        int vector_index
    }
```

---

## 3. Fast-to-Slow Query & Semantic Dialogue Sequence

![Fast-to-Slow Query Sequence](diagrams/03_fast_to_slow_query_sequence.svg)

```mermaid
sequenceDiagram
    actor User
    participant Web as "Web UI + FastAPI"
    participant Router as "Dialogue Router (M8.2)"
    participant DB as "SQLite + CLIP (M8 Fast)"
    participant Geom as "Python Spatial Reasoner"
    participant VLM as "Local Gemma 4 (Ollama)"

    User->>Web: "Where is the wheelbarrow in aisle 3?"
    Web->>Router: route(query, mode, history)
    alt General Chat mode
        Router->>VLM: general_chat_prompts_v1 (no map facts)
        VLM-->>Web: Conversational reply
    else Map Exploration mode
        Router->>DB: Rule/Gemma intent -> SQLite filter + CLIP rank
        alt Ambiguous (low score or small top-1 margin)
            DB->>VLM: Verify top-k (max 3) candidate images
            VLM-->>DB: Verified candidate IDs
        end
        DB-->>Router: Grounded Object/View records
        Router->>Geom: Project centroid onto aisle centerline
        Geom-->>Router: Verified section + head/tail distances
        Router->>VLM: Compose answer from verified facts only
        VLM-->>Router: Guarded natural-language reply
        Router-->>Web: Grounded answer + images + HMSG graph + safety badge
    end
    Web-->>User: Render Evidence Workspace
```

---

## 4. 3D Coordinate & Frame Transform Chain

![3D Coordinate and Frame Transform Chain](diagrams/04_coordinate_3d_projection_chain.svg)

```mermaid
flowchart LR
    MASK["2D SAM3 Binary Mask (eroded 5px)"] --> UNPROJ["Back-project via Aligned Depth + CameraInfo K"]
    DEPTH["Aligned Depth (0.15-15m, max obj 3.0m)"] --> UNPROJ
    UNPROJ --> CAM["3D Points in camera_color_optical_frame"]
    CAM --> TF1["Static TF: base_link -> camera_link -> optical"]
    TF1 --> TF2["FAST-LIO /Odometry: camera_init -> body"]
    TF2 --> WORLD["Observation 3D in camera_init (centroid + bbox)"]
    WORLD --> ASSOC["M5 Association (Tracklet+Hungarian / Spatial Cluster)"]
    ASSOC --> OBJ["Persistent Object 3D"]
    OBJ --> CENTER["Project onto Aisle Centerline -> s in 0..1 (head/middle/tail)"]
```

---

## 5. Truth & Navigation Safety Boundary

![Truth and Navigation Safety Boundary](diagrams/05_truth_and_safety_boundary.svg)

```mermaid
flowchart TB
    subgraph TRUTH["Deterministic Ground Truth (Python + LiDAR + Depth + SQLite)"]
        G1["Object ID & View ID"]
        G2["3D Centroid & Metric BBox (camera_init)"]
        G3["Aisle Polygon & Head/Tail Distances"]
    end

    subgraph VLM["Advisory Semantic Priors (Gemma 4 - Cannot Edit Geometry/IDs)"]
        S1["Physical Properties (rigidity, fragility, deformability)"]
        S2["Object Traversability Prior (forbidden / avoid / allowed)"]
        S3["View Traversability Cues (passable / blocked / uncertain)"]
    end

    GATE["Mandatory Safety Gate: requires_geometric_verification = true"]
    FUTURE["Future M9: Rule Fusion (Semantic Prior + Costmap + Robot Footprint)"]
    NAV["Robot Motion / Nav2 Execution"]

    TRUTH --> GATE
    VLM --> GATE
    GATE --> FUTURE
    FUTURE --> NAV
```

---

## 6. Multi-Provider Agent Workspace & Repository Edit Boundaries

![Multi-Provider Agent Workspace and Edit Boundaries](diagrams/06_repo_boundaries_and_agents.svg)

```mermaid
flowchart LR
    subgraph ENTRY["Single Source of Truth"]
        P["CLAUDE.md / GEMINI.md / copilot-instructions.md"] --> A["AGENTS.md"]
        A --> H["docs/HANDOFF.md"]
        H --> M["Makefile (build / test / validate-db)"]
    end

    subgraph ZONES["Repository Boundaries"]
        EDIT["EDIT ZONE: ros2_ws/src/conceptgraphs_bringup/ & docs/"]
        FROZEN["FROZEN: data/datasets/, models/sam3.pt, docs/archive/"]
        GEN["NEVER EDIT (.ignore): ros2_ws/build/, install/, log/"]
    end

    M --> EDIT
```
