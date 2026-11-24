"""Structured queries for the M7 HMSG SQLite database."""

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def open_database(dataset):
    path = Path(dataset).resolve() / "database" / "greenhouse_hmsg.sqlite"
    if not path.is_file():
        raise FileNotFoundError(path)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    return db


def run_query(db, action, value=None):
    queries = {
        "list_greenhouses": ("SELECT * FROM greenhouses ORDER BY greenhouse_id", ()),
        "list_aisles": ("SELECT * FROM aisles WHERE (? IS NULL OR greenhouse_id=?) ORDER BY aisle_id", (value, value)),
        "list_views": ("SELECT view_id,aisle_id,timestamp_ns,travel_pass,rgb_path FROM views WHERE (? IS NULL OR aisle_id=?) ORDER BY timestamp_ns", (value, value)),
        "list_objects": ("SELECT object_id,aisle_id,canonical_label,centroid_3d_json,representative_view_id FROM objects WHERE (? IS NULL OR aisle_id=?) ORDER BY object_id", (value, value)),
        "get_view": ("SELECT * FROM views WHERE view_id=?", (value,)),
        "get_object": ("SELECT o.*,c.scene_summary,c.category,c.color,c.material,c.semantic_class,c.confidence,v.rgb_path AS representative_rgb_path,ob.crop_path AS representative_crop_path FROM objects o LEFT JOIN captions c ON c.owner_type='object' AND c.owner_id=o.object_id LEFT JOIN views v ON v.view_id=o.representative_view_id LEFT JOIN observations ob ON ob.observation_id=o.representative_observation_id WHERE o.object_id=?", (value,)),
        "views_for_object": ("SELECT v.*,e.edge_id,e.observation_id FROM view_object_edges e JOIN views v ON v.view_id=e.view_id WHERE e.object_id=? ORDER BY v.timestamp_ns", (value,)),
        "observations_for_object": ("SELECT ob.* FROM object_observations oo JOIN observations ob ON ob.observation_id=oo.observation_id WHERE oo.object_id=? ORDER BY ob.timestamp_ns", (value,)),
        "objects_in_view": ("SELECT o.*,e.edge_id,e.observation_id FROM view_object_edges e JOIN objects o ON o.object_id=e.object_id WHERE e.view_id=? ORDER BY o.object_id", (value,)),
    }
    sql, params = queries[action]
    return [dict(row) for row in db.execute(sql, params)]


def format_table(rows):
    if not rows:
        return "No records."
    columns = list(rows[0])
    widths = {c: max([len(c)] + [len(str(r.get(c, ""))) for r in rows]) for c in columns}
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    separator = "-+-".join("-" * widths[c] for c in columns)
    body = [" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns) for r in rows]
    return "\n".join([header, separator] + body)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--list-greenhouses", action="store_true")
    actions.add_argument("--list-aisles", action="store_true")
    actions.add_argument("--list-views", action="store_true")
    actions.add_argument("--list-objects", action="store_true")
    actions.add_argument("--get-view")
    actions.add_argument("--get-object")
    actions.add_argument("--views-for-object")
    actions.add_argument("--observations-for-object")
    actions.add_argument("--objects-in-view")
    parser.add_argument("--greenhouse-id")
    parser.add_argument("--aisle-id")
    parser.add_argument("--format", choices=("table", "json"), default="table")
    args = parser.parse_args(argv)
    mapping = {
        "list_greenhouses": (args.list_greenhouses, None),
        "list_aisles": (args.list_aisles, args.greenhouse_id),
        "list_views": (args.list_views, args.aisle_id),
        "list_objects": (args.list_objects, args.aisle_id),
        "get_view": (args.get_view is not None, args.get_view),
        "get_object": (args.get_object is not None, args.get_object),
        "views_for_object": (args.views_for_object is not None, args.views_for_object),
        "observations_for_object": (args.observations_for_object is not None, args.observations_for_object),
        "objects_in_view": (args.objects_in_view is not None, args.objects_in_view),
    }
    action, (_, value) = next((key, item) for key, item in mapping.items() if item[0])
    try:
        db = open_database(args.dataset)
        try:
            rows = run_query(db, action, value)
        finally:
            db.close()
        print(json.dumps(rows, indent=2, ensure_ascii=False) if args.format == "json" else format_table(rows))
        return 0
    except Exception as exc:
        print(f"HMSG query error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
