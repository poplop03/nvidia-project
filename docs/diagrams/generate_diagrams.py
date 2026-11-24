#!/usr/bin/env python3
"""Generate crisp SVG + PNG architecture diagrams for Greenhouse HMSG docs.

Outputs 6 diagrams under docs/diagrams/:
  01_end_to_end_pipeline.{svg,png}
  02_hmsg_scene_graph_and_schema.{svg,png}
  03_fast_to_slow_query_sequence.{svg,png}
  04_coordinate_3d_projection_chain.{svg,png}
  05_truth_and_safety_boundary.{svg,png}
  06_repo_boundaries_and_agents.{svg,png}
"""

from __future__ import annotations

from html import escape
from pathlib import Path
import shutil
import subprocess


OUT_DIR = Path(__file__).resolve().parent

FONT = "DejaVu Sans, Arial, Helvetica, sans-serif"
MONO = "DejaVu Sans Mono, monospace"

PALETTE = {
    "blue":   ("#EFF6FF", "#2563EB", "#1E40AF"),
    "green":  ("#ECFDF5", "#059669", "#065F46"),
    "amber":  ("#FFFBEB", "#D97706", "#92400E"),
    "purple": ("#F5F3FF", "#7C3AED", "#5B21B6"),
    "rose":   ("#FFF1F2", "#E11D48", "#9F1239"),
    "cyan":   ("#F0F9FF", "#0284C7", "#075985"),
    "slate":  ("#F8FAFC", "#64748B", "#334155"),
}


class SvgCanvas:
    def __init__(self, width: int, height: int, title: str, subtitle: str = ""):
        self.w = width
        self.h = height
        self.parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            "<defs>",
            '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#475569"/></marker>',
            '<marker id="arrow-blue" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#2563EB"/></marker>',
            '<marker id="arrow-amber" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#D97706"/></marker>',
            '<marker id="arrow-rose" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#E11D48"/></marker>',
            '<marker id="arrow-purple" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#7C3AED"/></marker>',
            '<marker id="arrow-green" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#059669"/></marker>',
            '<filter id="shadow" x="-5%" y="-5%" width="110%" height="115%">'
            '<feDropShadow dx="0" dy="2" stdDeviation="2.5" flood-color="#0F172A" flood-opacity="0.06"/>'
            "</filter>",
            "</defs>",
            f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
            f'<text x="32" y="38" font-family="{FONT}" font-size="21" font-weight="700" fill="#0F172A">{escape(title)}</text>',
        ]
        if subtitle:
            self.parts.append(
                f'<text x="32" y="60" font-family="{FONT}" font-size="12.5" fill="#475569">{escape(subtitle)}</text>'
            )

    def group_box(self, x: int, y: int, w: int, h: int, label: str, theme: str = "slate", dashed: bool = False):
        fill, stroke, text_c = PALETTE[theme]
        dash = ' stroke-dasharray="6,4"' if dashed else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.5"{dash}/>'
        )
        self.parts.append(
            f'<text x="{x + 14}" y="{y + 23}" font-family="{FONT}" font-size="12" '
            f'font-weight="700" letter-spacing="0.4" fill="{text_c}">{escape(label)}</text>'
        )

    def card(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        title: str,
        lines: list[str] | None = None,
        theme: str = "blue",
        badge: str = "",
        mono_lines: bool = False,
        line_size: float | None = None,
    ):
        fill, stroke, accent = PALETTE[theme]
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="#FFFFFF" '
            f'stroke="{stroke}" stroke-width="1.7" filter="url(#shadow)"/>'
        )
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="6" rx="3" fill="{stroke}"/>'
        )
        ty = y + 25
        if badge:
            bw = max(44, len(badge) * 7 + 14)
            self.parts.append(
                f'<rect x="{x + 10}" y="{y + 12}" width="{bw}" height="18" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
            )
            self.parts.append(
                f'<text x="{x + 10 + bw / 2}" y="{y + 24.5}" text-anchor="middle" '
                f'font-family="{FONT}" font-size="10" font-weight="700" fill="{accent}">{escape(badge)}</text>'
            )
            ty = y + 46
        self.parts.append(
            f'<text x="{x + 12}" y="{ty}" font-family="{FONT}" font-size="12.8" font-weight="700" fill="#0F172A">{escape(title)}</text>'
        )
        line_font = MONO if mono_lines else FONT
        font_sz = line_size if line_size is not None else (10.3 if mono_lines else 10.8)
        for i, line in enumerate(lines or []):
            ly = ty + 18 + i * 16
            self.parts.append(
                f'<text x="{x + 12}" y="{ly}" font-family="{line_font}" font-size="{font_sz}" fill="#475569">{escape(line)}</text>'
            )

    def pill(self, x: int, y: int, w: int, h: int, text: str, theme: str = "blue"):
        fill, stroke, accent = PALETTE[theme]
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h // 2}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        )
        self.parts.append(
            f'<text x="{x + w / 2}" y="{y + h / 2 + 4}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="11.2" font-weight="700" fill="{accent}">{escape(text)}</text>'
        )

    def label_chip(self, cx: float, cy: float, text: str, color: str = "#334155"):
        w = len(text) * 6.4 + 14
        self.parts.append(
            f'<rect x="{cx - w / 2}" y="{cy - 11}" width="{w}" height="16" rx="4" fill="#FFFFFF" fill-opacity="0.95"/>'
        )
        self.parts.append(
            f'<text x="{cx}" y="{cy + 1}" text-anchor="middle" font-family="{FONT}" '
            f'font-size="10.5" font-weight="600" fill="{color}">{escape(text)}</text>'
        )

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        label: str = "",
        color: str = "#475569",
        marker: str = "arrow",
        dashed: bool = False,
        label_dy: float = -7,
        label_dx: float = 0,
        chip: bool = False,
    ):
        dash = ' stroke-dasharray="5,4"' if dashed else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="1.8"{dash} marker-end="url(#{marker})"/>'
        )
        if label:
            mx = (x1 + x2) / 2 + label_dx
            my = (y1 + y2) / 2 + label_dy
            if chip:
                self.label_chip(mx, my, label, color)
            else:
                self.parts.append(
                    f'<text x="{mx}" y="{my}" text-anchor="middle" font-family="{FONT}" '
                    f'font-size="10.5" font-weight="600" fill="{color}">{escape(label)}</text>'
                )

    def path_arrow(
        self,
        d: str,
        label: str = "",
        lx: float = 0,
        ly: float = 0,
        color: str = "#475569",
        marker: str = "arrow",
        dashed: bool = False,
        chip: bool = False,
    ):
        dash = ' stroke-dasharray="5,4"' if dashed else ""
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.8"{dash} marker-end="url(#{marker})"/>'
        )
        if label:
            if chip:
                self.label_chip(lx, ly, label, color)
            else:
                self.parts.append(
                    f'<text x="{lx}" y="{ly}" text-anchor="middle" font-family="{FONT}" '
                    f'font-size="10.5" font-weight="600" fill="{color}">{escape(label)}</text>'
                )

    def legend(self, x: int, y: int, items: list[tuple[str, str]]):
        cx = x
        for label, theme in items:
            fill, stroke, accent = PALETTE[theme]
            self.parts.append(
                f'<rect x="{cx}" y="{y}" width="14" height="14" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            )
            self.parts.append(
                f'<text x="{cx + 20}" y="{y + 11}" font-family="{FONT}" font-size="11" fill="#334155">{escape(label)}</text>'
            )
            cx += 26 + len(label) * 6.8 + 14

    def render(self) -> str:
        return "\n".join(self.parts + ["</svg>"])


def build_01_pipeline() -> str:
    c = SvgCanvas(
        1380,
        710,
        "1. Greenhouse HMSG — End-to-End Architecture (M1–M8.2.1)",
        "Offline ROS 2 mapping & multimodal indexing feeding a fast-to-slow interactive query workspace",
    )
    c.legend(
        32,
        74,
        [
            ("Deterministic Robotics / Geometry", "blue"),
            ("SAM3 / CLIP Perception", "green"),
            ("Gemma 4 VLM (Advisory)", "amber"),
            ("HMSG Storage", "purple"),
            ("Interactive UI / Router", "cyan"),
        ],
    )

    # Offline group
    c.group_box(32, 106, 1316, 282, "OFFLINE MAP BUILDING PIPELINE (M1–M7)  —  No live rosbag playback during inference", "slate", dashed=True)

    c.card(52, 140, 198, 108, "Sensor Bag (MCAP)", ["Livox Mid-360 LiDAR", "RealSense D455 RGB-D", "FAST-LIO /Odometry + TF"], "blue", "INPUT")
    c.card(52, 264, 198, 108, "Aisle Layout YAML", ["greenhouse_01 -> aisle_01..03", "Polygons & centerlines", "Frame: camera_init"], "blue", "M1")

    c.card(284, 192, 192, 128, "Keyframe Extractor", ["extract_keyframes", "~0.5 m or 15 deg steps", "Synced RGB+Depth+Pose", "209 Views (3 aisles)"], "blue", "M2")
    c.arrow(250, 194, 284, 232)
    c.arrow(250, 318, 284, 276)

    c.card(510, 192, 192, 128, "SAM3 Offline 2D", ["run_sam3_offline", "models/sam3.pt (GPU)", "5 prompts (containers,", "ladder, wheelbarrow)"], "green", "M3")
    c.arrow(476, 256, 510, 256)

    c.card(736, 192, 196, 128, "3D Projection & Assoc", ["project_observations_3d", "associate_objects", "25 valid 3D obs -> 6 Objects", "View OBSERVES Object"], "blue", "M4–M5")
    c.arrow(702, 256, 736, 256)

    c.card(966, 140, 214, 108, "CLIP Embeddings", ["OpenAI ViT-B-16 (512-D)", "209 View + 12 Object vecs", "L2-normalized .npy"], "green", "M6a")
    c.card(966, 264, 214, 108, "Gemma 4 Enrichment", ["gemma4:12b-it-q4_K_M", "Strict Pydantic JSON v2", "Physical & traversability"], "amber", "M6b")
    c.arrow(932, 236, 966, 194)
    c.arrow(932, 276, 966, 318)

    c.card(1210, 192, 120, 128, "SQLite", ["M7 Import", "FK + Path", "Validation", "Idempotent"], "purple", "M7")
    c.arrow(1180, 194, 1210, 234)
    c.arrow(1180, 318, 1210, 278)

    # Storage band
    c.group_box(32, 424, 456, 254, "SELF-CONTAINED HMSG DATASET (data/datasets/<id>/)", "purple")
    c.card(52, 464, 416, 92, "SQLite Database (greenhouse_hmsg.sqlite)", ["11 tables: hierarchy, geometry, captions, traversability,", "embedding index pointers, provenance (dataset_root='.')"], "purple")
    c.card(52, 572, 416, 92, "Filesystem Media & Vectors (relative paths only)", ["keyframes/{rgb,depth,camera_info} | observations/{masks,crops}", "embeddings/{view,object}_embeddings.npy | semantics/*.jsonl"], "purple")

    c.path_arrow(
        "M 1270 320 L 1270 406 L 260 406 L 260 424",
        "M7 persists graph metadata + relative media paths",
        765,
        406,
        "#7C3AED",
        "arrow-purple",
        chip=True,
    )

    # Online Query & Dialogue group
    c.group_box(522, 424, 826, 254, "INTERACTIVE QUERY & DIALOGUE WORKSPACE (M8–M8.2.1)", "cyan")
    c.card(544, 462, 186, 166, "Operator / Browser UI", ["FastAPI (query_hmsg_web)", "Modes: Auto / Chat / Map", "Visual gallery + SVG graph", "Verified facts + safety", "warning badge"], "cyan", "M8.1–8.2.1")
    c.card(766, 462, 246, 166, "Fast Retrieval & Spatial", ["semantic_dialogue + retrieval", "1. Rule / Gemma intent parse", "2. SQLite hierarchy filter", "3. CLIP ViT-B-16 cosine rank", "4. Python centerline s, head/tail"], "blue", "M8 FAST")
    c.card(1048, 462, 280, 166, "Gemma 4 Verification & Reply", ["Only invoked when ambiguous (top-3)", "or composing grounded NL reply", "Cannot modify IDs or 3D geometry", "Enforces geometric verification", "Template fallback if Ollama offline"], "amber", "M8 SLOW")

    c.arrow(730, 528, 766, 528, "", "#0284C7", "arrow-blue")
    c.arrow(766, 560, 730, 560, "", "#0284C7", "arrow-blue")
    c.arrow(1012, 528, 1048, 528, "", "#D97706", "arrow-amber")
    c.arrow(1048, 560, 1012, 560, "", "#D97706", "arrow-amber")

    c.arrow(488, 510, 544, 510, "images", "#7C3AED", "arrow-purple", label_dy=-8)
    c.path_arrow("M 488 648 L 889 648 L 889 628", "", 0, 0, "#7C3AED", "arrow-purple")
    c.parts.append(
        f'<text x="900" y="666" text-anchor="middle" font-family="{FONT}" font-size="10.5" font-weight="600" fill="#7C3AED">'
        f'SQLite metadata + precomputed CLIP .npy vectors (read-only at query time)</text>'
    )
    return c.render()


def build_02_hmsg_schema() -> str:
    c = SvgCanvas(
        1420,
        740,
        "2. HMSG Scene Graph Hierarchy & SQLite Relational Schema",
        "Left: Active 3-aisle scene graph (greenhouse_20261029_three_aisles)   |   Right: Normalized SQLite tables (sql/001_initial_hmsg.sql)",
    )

    # Left: Concrete Scene Graph Hierarchy
    c.group_box(32, 86, 616, 622, "HIERARCHICAL SCENE GRAPH INSTANCE (3 AISLES, 209 VIEWS, 6 OBJECTS)", "blue")
    c.pill(232, 126, 216, 34, "greenhouse_01 (camera_init)", "blue")

    aisles = [
        ("aisle_01", 52, ["1 object:", "obj_000001 blue crate", "(18.71, 1.56, -0.43)"]),
        ("aisle_02", 248, ["2 objects:", "obj_000002 red crate", "obj_000003 yellow crate"]),
        ("aisle_03", 444, ["3 objects:", "obj_000004 ladder", "obj_000005 wheelbarrow", "obj_000006 yellow crate"]),
    ]
    for name, ax, lines in aisles:
        c.pill(ax + 31, 196, 122, 30, name, "cyan")
        c.arrow(340, 160, ax + 92, 196, "", "#2563EB", "arrow-blue")
        c.card(ax, 254, 184, 126, f"Objects ({name})", lines, "blue", line_size=10.5)
        c.arrow(ax + 92, 226, ax + 92, 254, "", "#2563EB", "arrow-blue")

    c.card(52, 434, 258, 126, "Views (209 Keyframes)", ["view_000001 .. view_000209", "RGB + aligned 16-bit depth", "CameraInfo + 6-DoF pose", "aisle_coordinate_s + section"], "green", "VIEW NODE")
    c.card(370, 434, 258, 126, "Observations (190 total)", ["obs_000001_001 ..", "25 valid 3D / 165 rejected", "Binary mask + tight crop", "+ context crop + overlay"], "green", "DETECTION")

    c.arrow(181, 434, 181, 380, "OBSERVES (25 edges)", "#059669", "arrow-green", label_dy=4, label_dx=0, chip=True)
    c.arrow(310, 497, 370, 497, "yields", "#059669", "arrow-green", label_dy=-8)
    c.arrow(499, 434, 499, 380, "merged into", "#2563EB", "arrow-blue", label_dy=4, label_dx=0, chip=True)

    c.card(52, 584, 576, 102, "Multimodal Attachments (Linked by owner_type + owner_id)", [
        "• CLIP ViT-B-16 Embeddings: 209 view vectors + 12 object vectors (crop-mean & rep-view)",
        "• Gemma 4 Captions (30) & Traversability Assessments (24) — strict JSON v2 provenance",
        "• All paths stored relative to dataset_root='.' (datasets are self-contained & relocatable)",
    ], "amber", line_size=10.4)

    # Right: SQLite ER Diagram
    c.group_box(676, 86, 712, 622, "SQLITE SCHEMA (greenhouse_hmsg.sqlite — PRAGMA foreign_keys = ON)", "purple")

    c.card(700, 126, 200, 102, "datasets", ["PK dataset_id", "schema_version, frame_id", "dataset_root = '.'", "source_bag, manifest"], "purple", mono_lines=True)
    c.card(936, 126, 195, 102, "greenhouses", ["PK greenhouse_id", "FK dataset_id"], "purple", mono_lines=True)
    c.card(1168, 126, 196, 102, "aisles", ["PK aisle_id", "FK greenhouse_id"], "purple", mono_lines=True)
    c.arrow(900, 177, 936, 177, "1:N", "#7C3AED", "arrow-purple")
    c.arrow(1131, 177, 1168, 177, "1:N", "#7C3AED", "arrow-purple")

    c.card(700, 268, 200, 134, "views", ["PK view_id", "FK aisle_id, dataset_id", "rgb_path, depth_path", "pose_json, camera_info", "aisle_coordinate_s", "travel_pass"], "blue", mono_lines=True)
    c.card(936, 268, 200, 134, "observations", ["PK observation_id", "FK view_id, aisle_id", "canonical_label, conf", "mask_path, crop_path", "projection_status", "centroid_3d_json"], "green", mono_lines=True)
    c.card(1168, 268, 196, 134, "objects", ["PK object_id", "FK aisle_id", "FK rep_view_id", "canonical_label", "centroid_3d_json", "bbox_3d_min/max"], "blue", mono_lines=True)

    c.path_arrow("M 1266 228 L 1266 268", "1:N", 1282, 252, "#7C3AED", "arrow-purple")
    c.path_arrow("M 1216 228 L 1216 248 L 800 248 L 800 268", "", 0, 0, "#7C3AED", "arrow-purple")
    c.arrow(900, 335, 936, 335, "1:N", "#475569", "arrow")

    c.card(700, 444, 314, 96, "view_object_edges & object_obs", ["PK edge_id | edge_type = 'OBSERVES'", "FK view_id, object_id, observation_id", "Deterministic links (no synthetic edges)"], "cyan", mono_lines=True)
    c.card(1046, 444, 318, 96, "traversability_assessments", ["PK/FK view_id | passability, can_pass", "blocking_object_ids_json, model_digest", "requires_geometric_verification = true"], "amber", mono_lines=True)

    c.arrow(800, 402, 800, 444)
    c.arrow(1000, 402, 910, 444)
    c.arrow(1250, 402, 980, 444)
    c.arrow(838, 402, 1140, 444, "", "#D97706", "arrow-amber", dashed=True)

    c.card(700, 568, 314, 116, "captions", ["PK caption_id | owner_type", "owner_id, model_tag, digest", "category, color, material", "semantic_class, parsed_json", "raw_response_path"], "amber", mono_lines=True)
    c.card(1046, 568, 318, 116, "embedding_indexes & runs", ["PK embedding_id | owner_type", "owner_id, npy_path, vector_index", "dim=512, ViT-B-16, model_sha256", "processing_runs: stage audit log"], "green", mono_lines=True)

    return c.render()


def build_03_sequence() -> str:
    c = SvgCanvas(
        1400,
        764,
        "3. Fast-to-Slow Query & Semantic Dialogue Sequence (M8–M8.2.1)",
        "How query_hmsg_web routes between General Chat and Grounded Map Exploration without letting the VLM query SQLite directly",
    )

    actors = [
        ("Operator / UI", "web/app.js", 136, "cyan"),
        ("FastAPI Server", "query_web.py", 358, "cyan"),
        ("Dialogue Router", "semantic_dialogue.py", 585, "blue"),
        ("Fast Retrieval", "SQLite + CLIP (M8)", 820, "green"),
        ("Spatial Reasoner", "layout centerline", 1045, "blue"),
        ("Local Gemma 4", "Ollama :11435", 1265, "amber"),
    ]

    for title, sub, cx, theme in actors:
        c.card(cx - 88, 86, 176, 62, title, [sub], theme)
        c.parts.append(
            f'<line x1="{cx}" y1="148" x2="{cx}" y2="726" stroke="#CBD5E1" stroke-width="2" stroke-dasharray="6,5"/>'
        )

    def msg(x1: int, x2: int, y: int, text: str, color: str = "#334155", marker: str = "arrow", dashed: bool = False):
        c.arrow(x1, y, x2, y, text, color=color, marker=marker, dashed=dashed, label_dy=-7)

    # Turn start
    msg(136, 358, 186, "POST /api/query (query, mode, history)", "#0284C7", "arrow-blue")
    msg(358, 585, 220, "interact(dataset, query, mode)", "#0284C7", "arrow-blue")

    # Alt block 1: Auto / Chat vs Map
    c.group_box(470, 240, 892, 98, "BRANCH A: General Chat Mode (or Auto -> rule/Gemma classifies as general chat)", "amber", dashed=True)
    msg(585, 1265, 290, "general_chat_prompts_v1 + up to 10 turns (NO map facts / NO DB)", "#D97706", "arrow-amber")
    msg(1265, 585, 320, "Conversational reply (retrieval: null)", "#D97706", "arrow-amber", dashed=True)

    # Alt block 2: Map exploration
    c.group_box(470, 354, 892, 310, "BRANCH B: Explore Map Mode (Verified Grounded Retrieval)", "blue", dashed=True)
    msg(585, 820, 398, "1. Rule intent parse -> SQLite filter", "#2563EB", "arrow-blue")
    c.pill(690, 412, 260, 28, "2. CLIP text encode & cosine rank", "green")

    # Ambiguity gate sub-box
    c.group_box(714, 454, 628, 88, "AMBIGUITY GATE: only if parser conf < 0.75, sim < 0.20, or margin < 0.08", "amber", dashed=True)
    msg(820, 1265, 502, "3. Verify top-k (max 3) candidate crops/views", "#D97706", "arrow-amber")
    msg(1265, 820, 528, "Verified IDs + reasoning_summary (cannot alter geometry)", "#D97706", "arrow-amber", dashed=True)

    msg(820, 585, 568, "Grounded Object/View + captions + traversability", "#059669", "arrow-green", dashed=True)
    msg(585, 1045, 598, "4. Project centroid_3d onto aisle centerline", "#2563EB", "arrow-blue")
    msg(1045, 585, 624, "Verified facts: section, head/tail m, offset m", "#2563EB", "arrow-blue", dashed=True)
    msg(585, 1265, 650, "5. Compose reply from verified facts only (template fallback if offline/invalid)", "#D97706", "arrow-amber")

    # Return to UI
    msg(585, 358, 692, "Response + provenance + safety gate", "#0284C7", "arrow-blue", dashed=True)
    msg(358, 136, 720, "Render answer, gallery, SVG & safety", "#0284C7", "arrow-blue", dashed=True)

    return c.render()


def build_04_coordinates() -> str:
    c = SvgCanvas(
        1360,
        470,
        "4. 3D Coordinate & Frame Transform Chain (2D Pixel Mask → Metric Aisle Object)",
        "Deterministic geometric pipeline from RealSense D455 pixel coordinates to persistent camera_init 3D objects",
    )

    c.card(36, 108, 224, 152, "1. SAM3 2D Mask", [
        "Input: Keyframe RGB (M2)",
        "Binary mask eroded 5 px",
        "(reduces border depth bleed)",
        "2D bbox + tight/context crops",
        "Pixel coords: (u, v)",
    ], "green", "IMAGE PLANE")

    c.card(300, 108, 232, 152, "2. Depth Back-Projection", [
        "Aligned 16-bit depth (mm -> m)",
        "Valid depth: 0.15 m .. 15.0 m",
        "Median outlier band: max(0.2m,10%)",
        "Min 10 pts | Max range: 3.0 m",
        "Frame: camera_color_optical_frame",
    ], "blue", "CAMERA 3D")
    c.arrow(260, 184, 300, 184)

    c.card(572, 108, 240, 152, "3. Static & Odometry TF", [
        "Static chain (/tf_static):",
        "base_link -> camera_link ->",
        "camera_color_frame -> optical",
        "Dynamic pose (/Odometry):",
        "camera_init -> body (FAST-LIO)",
    ], "blue", "TF TRANSFORM")
    c.arrow(532, 184, 572, 184)

    c.card(852, 108, 224, 152, "4. M4 Observation 3D", [
        "World frame: camera_init",
        "Robust 3D centroid + bbox",
        "Rejected kept for QA audit",
        "(3-aisle: 25 valid / 165 rej)",
        "Never modified by M5+",
    ], "purple", "WORLD OBS")
    c.arrow(812, 184, 852, 184)

    c.card(1116, 108, 212, 152, "5. M5 Object & Aisle", [
        "Multi-view clustering /",
        "Tracklet + Hungarian merge",
        "Representative View score",
        "Aisle polygon assignment",
        "Centerline s in [0.0, 1.0]",
    ], "cyan", "PERSISTENT OBJ")
    c.arrow(1076, 184, 1116, 184)

    # Bottom visual strip: Aisle centerline coordinate illustration
    c.group_box(36, 294, 1292, 144, "AISLE CENTERLINE PROJECTION (camera_init)  —  Fixed physical head/tail independent of robot travel_pass", "slate")
    c.parts.append('<rect x="110" y="346" width="380" height="34" rx="6" fill="#DBEAFE" stroke="#2563EB" stroke-width="1.5"/>')
    c.parts.append('<rect x="490" y="346" width="380" height="34" rx="6" fill="#FEF3C7" stroke="#D97706" stroke-width="1.5"/>')
    c.parts.append('<rect x="870" y="346" width="380" height="34" rx="6" fill="#FCE7F3" stroke="#DB2777" stroke-width="1.5"/>')

    c.parts.append(f'<text x="300" y="368" text-anchor="middle" font-family="{FONT}" font-size="12.5" font-weight="700" fill="#1E40AF">HEAD  (0.00 ≤ s &lt; 0.33)</text>')
    c.parts.append(f'<text x="680" y="368" text-anchor="middle" font-family="{FONT}" font-size="12.5" font-weight="700" fill="#92400E">MIDDLE  (0.33 ≤ s &lt; 0.67)</text>')
    c.parts.append(f'<text x="1060" y="368" text-anchor="middle" font-family="{FONT}" font-size="12.5" font-weight="700" fill="#9D174D">TAIL  (0.67 ≤ s ≤ 1.00)</text>')

    c.parts.append(f'<text x="110" y="405" font-family="{MONO}" font-size="11" fill="#334155">s = 0.00 (Head point)</text>')
    c.parts.append(f'<text x="680" y="405" text-anchor="middle" font-family="{MONO}" font-size="11" fill="#334155">Verified outputs: section, tendency, distance_from_head_m, distance_to_tail_m, centerline_offset_m</text>')
    c.parts.append(f'<text x="1250" y="405" text-anchor="end" font-family="{MONO}" font-size="11" fill="#334155">s = 1.00 (Tail point)</text>')

    return c.render()


def build_05_safety_boundary() -> str:
    c = SvgCanvas(
        1360,
        626,
        "5. Truth & Navigation Safety Boundary (Deterministic Geometry vs. Advisory VLM)",
        "Core project invariant: Gemma 4 provides semantic priors only and can never override IDs, geometry, or motion safety",
    )

    # Left: Deterministic Ground Truth
    c.group_box(36, 92, 608, 226, "DETERMINISTIC GROUND TRUTH  (Python + LiDAR + Depth + SQLite)", "blue")
    c.card(58, 134, 270, 164, "Identity & Hierarchy", [
        "• Stable object_id & view_id",
        "• Greenhouse & aisle_id assignment",
        "• Verified View OBSERVES Object",
        "• Source: M1–M5 & SQLite FKs",
        "• Immutable to VLM stages",
    ], "blue", "AUTHORITATIVE")
    c.card(352, 134, 270, 164, "Metric Geometry (camera_init)", [
        "• 3D centroid & 3D bounding box",
        "• Depth stats & observation count",
        "• Centerline s, head/tail meters",
        "• Lateral centerline offset (m)",
        "• Never estimated by Gemma 4",
    ], "blue", "AUTHORITATIVE")

    # Right: Advisory VLM Priors
    c.group_box(716, 92, 608, 226, "ADVISORY SEMANTIC PRIORS  (Gemma 4 via Ollama — Strict Pydantic JSON)", "amber")
    c.card(738, 134, 270, 164, "Object Physical Priors", [
        "• rigidity, deformability, fragility",
        "• compressibility, mass class",
        "• object_traversability prior:",
        "  forbidden / avoid / allowed",
        "• potential_risks & visual evidence",
    ], "amber", "SEMANTIC PRIOR")
    c.card(1032, 134, 270, 164, "Scene Route Cues (Separate!)", [
        "• vlm_path_assessment:",
        "  passable / blocked / uncertain",
        "• can_robot_pass (bool / null)",
        "• blocking_object_ids (disjoint)",
        "• Validated against linked IDs",
    ], "amber", "ADVISORY ONLY")

    # Safety Gate Banner
    c.card(
        240,
        372,
        880,
        74,
        "MANDATORY SAFETY GATE:   requires_geometric_verification: true",
        ["Enforced by Pydantic schema (M6), retrieval (M8), dialogue guardrails (M8.2), and Web UI safety banner"],
        "rose",
    )
    c.arrow(340, 318, 520, 372, "Verified IDs & 3D metrics", "#2563EB", "arrow-blue", label_dx=-10, label_dy=4, chip=True)
    c.arrow(1020, 318, 840, 372, "Advisory priors & cues", "#D97706", "arrow-amber", label_dx=10, label_dy=4, chip=True)

    # Bottom: Future M9 Fusion -> Nav2
    c.card(
        130,
        492,
        470,
        102,
        "Future M9: Geometric + Semantic Policy Fusion",
        ["Combine VLM physical prior + LiDAR/depth object height & width", "+ calibrated robot footprint, ground clearance & local costmap"],
        "purple",
        "PLANNED (M9)",
    )
    c.card(
        760,
        492,
        470,
        102,
        "Robot Navigation / Nav2 Execution",
        ["ONLY traversability_decision after M9 geometric fusion", "Current M1–M8.2.1 system performs ZERO robot motion"],
        "rose",
        "GATED OUTPUT",
    )
    c.arrow(560, 446, 400, 492, "", "#E11D48", "arrow-rose")
    c.arrow(600, 543, 760, 543, "final_traversability", "#E11D48", "arrow-rose", label_dy=-8)

    return c.render()


def build_06_repo_boundaries() -> str:
    c = SvgCanvas(
        1360,
        584,
        "6. Multi-Provider Agent Workspace & Repository Edit Boundaries",
        "Single source of truth (AGENTS.md) + search hygiene (.ignore) + 1-command verification (Makefile) for Claude, Codex, Gemini & Copilot",
    )

    # Top row: Agent entry & protocol
    c.group_box(36, 92, 1288, 144, "MULTI-PROVIDER AGENT ENTRY & HANDOFF PROTOCOL", "cyan")
    c.card(58, 128, 255, 90, "Provider Auto-Discovery", ["CLAUDE.md  -> AGENTS.md", "GEMINI.md  -> AGENTS.md", ".github/copilot-instructions.md"], "cyan")
    c.card(352, 128, 290, 90, "Canonical Rules (AGENTS.md)", ["Single source of truth (<120 lines)", "Edit boundaries, invariants & cmds", "No provider instruction drift"], "blue")
    c.card(682, 128, 290, 90, "Session Handoff (docs/)", ["Start: read docs/HANDOFF.md", "On-demand: ARCHITECTURE, CONTRACTS", "End: run make test & update HANDOFF"], "green")
    c.card(1012, 128, 290, 90, "1-Command Loop (Makefile)", ["make build  |  make test", "make validate-db  |  make query-db", "Auto-sources ROS 2 Humble"], "purple")

    c.arrow(313, 173, 352, 173)
    c.arrow(642, 173, 682, 173)
    c.arrow(972, 173, 1012, 173)

    # Bottom: 3 Zones (Edit / Frozen / Generated)
    c.group_box(36, 262, 412, 292, "ZONE 1: SAFE TO EDIT  (Indexed & Git-Tracked)", "green")
    c.card(58, 302, 368, 110, "ros2_ws/src/conceptgraphs_bringup/", [
        "• Python package, launch/, config/, sql/, web/, test/",
        "• Package-local AGENTS.md + portable paths.py",
        "• Always rebuild via: make build",
    ], "green", "ACTIVE CODE")
    c.card(58, 428, 368, 108, "docs/  &  Root Configs", [
        "• HANDOFF.md, ARCHITECTURE.md, CONTRACTS.md",
        "• RUNBOOK.md, DIAGRAMS.md, playbooks/",
        "• Makefile, AGENTS.md, .gitignore, .ignore",
    ], "green", "ACTIVE DOCS")

    c.group_box(474, 262, 412, 292, "ZONE 2: READ-ONLY / FROZEN  (Ignored in .ignore)", "amber")
    c.card(496, 302, 368, 110, "data/datasets/  &  data/demo_videos/", [
        "• Completed HMSG datasets & SQLite DBs",
        "• DO NOT mutate pipeline_config.yaml",
        "  (preserves M3/M5 resume fingerprints)",
    ], "amber", "FROZEN DATA")
    c.card(496, 428, 368, 108, "models/sam3.pt  &  docs/archive/", [
        "• 3.3 GB SAM3 checkpoint (gitignored)",
        "• Historical PLAN.md, PROJECT_CONTEXT.md,",
        "  checkpoints & *.docx reports (reference only)",
    ], "amber", "ARCHIVE / WEIGHTS")

    c.group_box(912, 262, 412, 292, "ZONE 3: GENERATED / SCRATCH  (Never Edit)", "rose")
    c.card(934, 302, 368, 110, "ros2_ws/build/  install/  log/", [
        "• Colcon outputs — NEVER edit files here!",
        "• Excluded by .ignore so agents never grep",
        "  duplicate installed copies of Python files",
    ], "rose", "GENERATED")
    c.card(934, 428, 368, 108, "scratch/  (Gitignored)", [
        "• Safe area for disposable scripts & logs",
        "• Keeps repository root clean across",
        "  parallel or sequential agent sessions",
    ], "slate", "DISPOSABLE")

    return c.render()


def render_png(svg_path: Path, png_path: Path, width: int, height: int):
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome:
        html_wrapper = svg_path.with_suffix(".tmp.html")
        html_wrapper.write_text(
            f'<!doctype html><html><body style="margin:0;padding:0;background:#fff;overflow:hidden;">'
            f'<img src="{svg_path.name}" width="{width}" height="{height}" style="display:block;"/>'
            f"</body></html>",
            encoding="utf-8",
        )
        try:
            subprocess.run(
                [
                    chrome,
                    "--headless=new",
                    "--disable-gpu",
                    "--no-sandbox",
                    "--hide-scrollbars",
                    "--force-device-scale-factor=2",
                    f"--window-size={width},{height}",
                    f"--screenshot={png_path}",
                    html_wrapper.as_uri(),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            return
        finally:
            html_wrapper.unlink(missing_ok=True)
    subprocess.run(
        ["convert", "-background", "white", "-density", "200", str(svg_path), str(png_path)],
        check=True,
        capture_output=True,
        timeout=30,
    )


def main():
    specs = [
        ("01_end_to_end_pipeline", build_01_pipeline, 1380, 710),
        ("02_hmsg_scene_graph_and_schema", build_02_hmsg_schema, 1420, 740),
        ("03_fast_to_slow_query_sequence", build_03_sequence, 1400, 764),
        ("04_coordinate_3d_projection_chain", build_04_coordinates, 1360, 470),
        ("05_truth_and_safety_boundary", build_05_safety_boundary, 1360, 626),
        ("06_repo_boundaries_and_agents", build_06_repo_boundaries, 1360, 584),
    ]
    for stem, builder, w, h in specs:
        svg_path = OUT_DIR / f"{stem}.svg"
        png_path = OUT_DIR / f"{stem}.png"
        svg_path.write_text(builder(), encoding="utf-8")
        render_png(svg_path, png_path, w, h)
        print(f"Generated {svg_path.name} + {png_path.name}")


if __name__ == "__main__":
    main()
