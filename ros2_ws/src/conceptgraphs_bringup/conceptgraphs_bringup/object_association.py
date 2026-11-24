#!/usr/bin/env python3
"""M5 deterministic tracklet + Hungarian association."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
import yaml

from .greenhouse_layout import load_layout

INF = 1e9


def sortable_optional(value):
    """Sort nullable identifiers deterministically without changing stored values."""
    return (value is not None, "" if value is None else str(value))


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def atomic_write(path, records, jsonl=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    text = ("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records)
            if jsonl else json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    if jsonl:
        companion = path.with_suffix(".json")
        companion_tmp = companion.with_name(companion.name + ".tmp")
        companion_tmp.write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(companion_tmp, companion)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def point(item):
    return np.asarray(item["centroid_3d"], dtype=float)


def extent(item):
    return np.asarray(item["bbox_3d_max"], dtype=float) - np.asarray(item["bbox_3d_min"], dtype=float)


def robust_center(items):
    return np.median([point(x) for x in items], axis=0)


def robust_extent(items):
    return np.median([extent(x) for x in items], axis=0)


def size_difference(a, b):
    return float(np.mean(np.abs(a - b) / np.maximum(np.maximum(a, b), 1e-6)))


def diameter(items):
    return max((float(np.linalg.norm(point(a) - point(b)))
                for i, a in enumerate(items) for b in items[i + 1:]), default=0.0)


def tracklet_cost(observation, tracklet, config):
    cfg = config["association"]
    items = tracklet["observations"]
    distance = float(np.linalg.norm(point(observation) - robust_center(items)))
    if distance > cfg["tracklet_max_distance_m"]:
        return INF, "tracklet_distance_gate"
    if diameter(items + [observation]) > cfg["tracklet_max_diameter_m"]:
        return INF, "tracklet_diameter_gate"
    size_delta = size_difference(extent(observation), robust_extent(items))
    if size_delta > cfg["maximum_size_difference"]:
        return INF, "tracklet_size_gate"
    return distance + cfg["size_cost_weight"] * size_delta, "accepted"


def make_tracklets(observations, config, rejected):
    groups = defaultdict(list)
    for observation in observations:
        groups[(observation["canonical_label"], observation["aisle_id"],
                observation["travel_pass"])].append(observation)
    all_tracklets = []
    for group_key in sorted(
            groups,
            key=lambda key: tuple(sortable_optional(value) for value in key)):
        views = defaultdict(list)
        for observation in sorted(groups[group_key], key=lambda x: (x["timestamp_ns"], x["observation_id"])):
            views[(observation["timestamp_ns"], observation["view_id"])].append(observation)
        active = []
        for _, detections in sorted(views.items()):
            costs = np.full((len(active), len(detections)), INF)
            reasons = {}
            for row, tracklet in enumerate(active):
                for column, detection in enumerate(detections):
                    costs[row, column], reasons[row, column] = tracklet_cost(detection, tracklet, config)
            assigned = set()
            if active and detections:
                rows, columns = linear_sum_assignment(costs)
                for row, column in zip(rows, columns):
                    if costs[row, column] >= INF:
                        rejected[reasons[row, column]] += 1
                    else:
                        active[row]["observations"].append(detections[column])
                        assigned.add(column)
            for column, detection in enumerate(detections):
                if column not in assigned:
                    tracklet = {"canonical_label": group_key[0], "aisle_id": group_key[1],
                                "travel_pass": group_key[2], "observations": [detection]}
                    active.append(tracklet)
                    all_tracklets.append(tracklet)
    all_tracklets.sort(key=lambda x: min(o["observation_id"] for o in x["observations"]))
    for index, tracklet in enumerate(all_tracklets, 1):
        tracklet.update(tracklet_id=f"tracklet_{index:06d}",
                        centroid_3d=robust_center(tracklet["observations"]).tolist(),
                        size_3d=robust_extent(tracklet["observations"]).tolist(),
                        diameter_m=diameter(tracklet["observations"]))
    return all_tracklets


def make_spatial_label_clusters(observations, config):
    """Cluster physical objects by label and 3D proximity across View metadata."""
    radius = float(config["association"]["spatial_cluster_radius_m"])
    groups = defaultdict(list)
    for observation in observations:
        groups[observation["canonical_label"]].append(observation)
    tracklets = []
    for label in sorted(groups):
        clusters = []
        ordered = sorted(groups[label], key=lambda item: (
            item["timestamp_ns"], item["observation_id"]))
        for observation in ordered:
            nearest = min(
                clusters,
                key=lambda cluster: np.linalg.norm(
                    point(observation) - robust_center(cluster["observations"])),
                default=None,
            )
            if (nearest is None or
                    np.linalg.norm(point(observation) - robust_center(
                        nearest["observations"])) > radius):
                nearest = {"canonical_label": label, "aisle_id": None,
                           "travel_pass": observation["travel_pass"],
                           "observations": []}
                clusters.append(nearest)
                tracklets.append(nearest)
            nearest["observations"].append(observation)
            passes = {item["travel_pass"] for item in nearest["observations"]}
            nearest["travel_pass"] = next(iter(passes)) if len(passes) == 1 else "mixed"
    tracklets.sort(key=lambda item: min(
        observation["observation_id"] for observation in item["observations"]))
    for index, tracklet in enumerate(tracklets, 1):
        tracklet.update(
            tracklet_id=f"tracklet_{index:06d}",
            centroid_3d=robust_center(tracklet["observations"]).tolist(),
            size_3d=robust_extent(tracklet["observations"]).tolist(),
            diameter_m=diameter(tracklet["observations"]),
        )
    return tracklets


def cross_cost(forward, returned, config):
    cfg = config["association"]
    distance = float(np.linalg.norm(np.asarray(forward["centroid_3d"]) - np.asarray(returned["centroid_3d"])))
    size_delta = size_difference(np.asarray(forward["size_3d"]), np.asarray(returned["size_3d"]))
    detail = {"distance_m": distance, "size_difference": size_delta}
    if distance > cfg["cross_pass_candidate_radius_m"]:
        return INF, "cross_pass_distance_gate", detail
    if size_delta > cfg["maximum_size_difference"]:
        return INF, "cross_pass_size_gate", detail
    cost = distance + cfg["size_cost_weight"] * size_delta
    detail["cost"] = cost
    if cost > cfg["cross_pass_maximum_cost"]:
        return INF, "cross_pass_cost_gate", detail
    return cost, "accepted", detail


def match_tracklets(tracklets, config, rejected):
    groups = defaultdict(lambda: {"forward": [], "return": []})
    for tracklet in tracklets:
        groups[(tracklet["canonical_label"], tracklet["aisle_id"])][tracklet["travel_pass"]].append(tracklet)
    matches, unmatched = [], []
    for group in groups.values():
        forward, returned = group["forward"], group["return"]
        if not forward or not returned:
            unmatched.extend(forward + returned)
            continue
        costs = np.full((len(forward), len(returned)), INF)
        details = {}
        for row, left in enumerate(forward):
            for column, right in enumerate(returned):
                costs[row, column], reason, details[row, column] = cross_cost(left, right, config)
                if reason != "accepted":
                    rejected[reason] += 1
        rows, columns = linear_sum_assignment(costs)
        used_left, used_right = set(), set()
        for row, column in zip(rows, columns):
            if costs[row, column] < INF:
                matches.append((forward[row], returned[column], details[row, column]))
                used_left.add(row); used_right.add(column)
        unmatched.extend(x for i, x in enumerate(forward) if i not in used_left)
        unmatched.extend(x for i, x in enumerate(returned) if i not in used_right)
    return matches, unmatched


def representative_score(observation, config):
    weights = config["association"]["representative_weights"]
    bbox = observation["bbox_2d_xyxy"]
    area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    mask_quality = min(1.0, observation["mask_area_pixels"] / area)
    return (weights["confidence"] * observation["sam_confidence"] +
            weights["near_depth"] / max(observation["median_depth_m"], 0.1) +
            weights["valid_depth_ratio"] * observation["valid_depth_ratio"] +
            weights["mask_quality"] * mask_quality)


def make_object(tracklets, match_detail, config):
    observations = sorted((o for t in tracklets for o in t["observations"]), key=lambda x: x["observation_id"])
    by_pass = {p: [o for o in observations if o["travel_pass"] == p] for p in ("forward", "return")}
    centers = {p: robust_center(items) for p, items in by_pass.items() if items}
    multi = len(centers) == 2
    object_center = ((centers["forward"] + centers["return"]) / 2 if multi else next(iter(centers.values())))
    representative = max(observations, key=lambda x: (representative_score(x, config), x["observation_id"]))
    return {
        "dataset_id": observations[0]["dataset_id"], "greenhouse_id": observations[0]["greenhouse_id"],
        "aisle_id": observations[0]["aisle_id"], "canonical_label": observations[0]["canonical_label"],
        "frame_id": "camera_init", "centroid_3d": object_center.tolist(),
        "bbox_3d_min": np.min([o["bbox_3d_min"] for o in observations], axis=0).tolist(),
        "bbox_3d_max": np.max([o["bbox_3d_max"] for o in observations], axis=0).tolist(),
        "center_forward": centers["forward"].tolist() if "forward" in centers else None,
        "center_return": centers["return"].tolist() if "return" in centers else None,
        "position_quality": "multi_pass" if multi else "single_sided",
        "association_status": "matched" if multi else "ambiguous",
        "observation_ids": [o["observation_id"] for o in observations],
        "tracklet_ids": [t["tracklet_id"] for t in tracklets], "travel_passes": sorted(centers),
        "observation_count": len(observations), "cluster_diameter_m": diameter(observations),
        "representative_view_id": representative["view_id"],
        "representative_observation_id": representative["observation_id"],
        "representative_score": representative_score(representative, config),
        "cross_pass_match": match_detail}


def create_graph(objects, observations, views):
    by_observation = {x["observation_id"]: x for x in observations}
    view_ids = sorted({by_observation[oid]["view_id"] for obj in objects for oid in obj["observation_ids"]})
    nodes = [{"node_id": vid, "node_type": "View", "data": views[vid]} for vid in view_ids]
    nodes += [{"node_id": obj["object_id"], "node_type": "Object", "data": obj} for obj in objects]
    edges = []
    for obj in objects:
        for oid in obj["observation_ids"]:
            vid = by_observation[oid]["view_id"]
            digest = hashlib.sha1(f"{vid}|{obj['object_id']}|{oid}".encode()).hexdigest()[:12]
            edges.append({"edge_id": f"edge_observes_{digest}", "edge_type": "OBSERVES",
                          "source_id": vid, "target_id": obj["object_id"], "observation_id": oid})
    return sorted(nodes, key=lambda x: x["node_id"]), sorted(edges, key=lambda x: x["edge_id"])


def validate(objects, nodes, edges, valid_observations, views):
    object_ids, node_ids, edge_ids = ([x["object_id"] for x in objects], [x["node_id"] for x in nodes], [x["edge_id"] for x in edges])
    assigned = [oid for obj in objects for oid in obj["observation_ids"]]
    expected = [x["observation_id"] for x in valid_observations]
    if any(len(x) != len(set(x)) for x in (object_ids, node_ids, edge_ids)):
        raise ValueError("IDs must be unique")
    if sorted(assigned) != sorted(expected) or len(assigned) != len(set(assigned)):
        raise ValueError("Each valid Observation must be assigned exactly once")
    node_set = set(node_ids)
    if any(e["source_id"] not in views or e["source_id"] not in node_set or e["target_id"] not in node_set for e in edges):
        raise ValueError("Broken graph reference")
    return {"valid": True, "unique_object_ids": len(object_ids), "unique_node_ids": len(node_ids),
            "unique_edge_ids": len(edge_ids), "valid_observations_assigned_once": len(assigned),
            "reference_integrity": True}


def associate(valid, views, config):
    observations = []
    for source in valid:
        item = dict(source); view = views[item["view_id"]]
        item.update(travel_pass=view["travel_pass"], timestamp_ns=view["timestamp_ns"])
        observations.append(item)
    rejected = Counter()
    tracklets = make_tracklets(observations, config, rejected)
    matches, unmatched = match_tracklets(tracklets, config, rejected)
    objects = [make_object([a, b], detail, config) for a, b, detail in matches]
    objects += [make_object([tracklet], None, config) for tracklet in unmatched]
    objects.sort(key=lambda x: (
        sortable_optional(x["aisle_id"]), x["canonical_label"],
        min(x["observation_ids"])))
    for index, obj in enumerate(objects, 1): obj["object_id"] = f"object_{index:06d}"
    nodes, edges = create_graph(objects, observations, views)
    return objects, nodes, edges, tracklets, matches, dict(rejected)


def associate_spatial(valid, views, config, layout=None):
    """Associate multi-aisle observations before assigning object aisle IDs."""
    observations = []
    for source in valid:
        item = dict(source)
        view = views[item["view_id"]]
        item.update(travel_pass=view["travel_pass"], timestamp_ns=view["timestamp_ns"])
        observations.append(item)
    tracklets = make_spatial_label_clusters(observations, config)
    objects = [make_object([tracklet], None, config) for tracklet in tracklets]
    for obj in objects:
        obj["aisle_id"] = (layout.assign_aisle(*obj["centroid_3d"][:2])
                           if layout is not None else None)
    objects.sort(key=lambda item: (
        sortable_optional(item["aisle_id"]), item["canonical_label"],
        min(item["observation_ids"])))
    for index, obj in enumerate(objects, 1):
        obj["object_id"] = f"object_{index:06d}"
    nodes, edges = create_graph(objects, observations, views)
    return objects, nodes, edges, tracklets, [], {}


def run(dataset, force=False):
    dataset = dataset.resolve(); op = dataset / "observations/observations_3d.jsonl"; vp = dataset / "keyframes/views.jsonl"
    config = yaml.safe_load((dataset / "pipeline_config.yaml").read_text())
    if "association" not in config: raise ValueError("pipeline_config.yaml has no association section")
    all_observations = read_jsonl(op); valid = [x for x in all_observations if x["projection_status"] == "valid"]
    views = {x["view_id"]: x for x in read_jsonl(vp)}
    fingerprint = hashlib.sha256(json.dumps({"association": config["association"], "observations": sha256(op), "views": sha256(vp)}, sort_keys=True).encode()).hexdigest()
    paths = [dataset / "objects/objects.jsonl", dataset / "graph/nodes.jsonl", dataset / "graph/edges.jsonl", dataset / "reports/association_report.json"]
    if paths[-1].exists() and not force:
        report = json.loads(paths[-1].read_text())
        if report.get("config_fingerprint") != fingerprint: raise ValueError("Inputs/config changed; rerun with --force")
        if all(p.is_file() for p in paths): return report
    association_mode = config["association"].get("mode", "tracklet_hungarian")
    if association_mode == "spatial_label_cluster":
        layout = load_layout(config["layout_path"])
        objects, nodes, edges, tracklets, matches, rejected = associate_spatial(
            valid, views, config, layout)
        algorithm = "same-label spatial clustering"
    elif association_mode == "tracklet_hungarian":
        objects, nodes, edges, tracklets, matches, rejected = associate(
            valid, views, config)
        algorithm = "tracklet + Hungarian"
    else:
        raise ValueError(f"Unsupported association mode: {association_mode}")
    validation = validate(objects, nodes, edges, valid, views)
    for path, records in zip(paths[:3], (objects, nodes, edges)): atomic_write(path, records)
    qualities = Counter(x["position_quality"] for x in objects); statuses = Counter(x["association_status"] for x in objects); passes = Counter(x["travel_pass"] for x in tracklets)
    report = {"dataset_id": config["dataset_id"], "stage": "M5_object_association", "status": "complete",
              "updated_at": datetime.now(timezone.utc).isoformat(), "algorithm": algorithm,
              "config_fingerprint": fingerprint, "association_config": config["association"],
              "input": {"observations_total": len(all_observations), "valid_observations_associated": len(valid),
                        "rejected_observations_preserved": len(all_observations)-len(valid)},
              "object_count": len(objects), "tracklet_counts": {"forward": passes["forward"], "return": passes["return"], "mixed": passes["mixed"]},
              "cross_pass_matches": len(matches), "object_quality_counts": {"multi_pass": qualities["multi_pass"],
              "single_sided": qualities["single_sided"], "ambiguous": statuses["ambiguous"]},
              "objects": [{k: x[k] for k in ("object_id", "canonical_label", "observation_count", "cluster_diameter_m", "position_quality", "association_status", "observation_ids")} for x in objects],
              "candidate_rejections": rejected, "validation": validation}
    atomic_write(paths[-1], report, False)
    manifest_path = dataset / "manifest.json"; manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("stages", {})["object_association"] = {"status": "complete", "objects_file": "objects/objects.jsonl", "objects_json_file": "objects/objects.json", "nodes_file": "graph/nodes.jsonl", "nodes_json_file": "graph/nodes.json", "edges_file": "graph/edges.jsonl", "edges_json_file": "graph/edges.json", "report_file": "reports/association_report.json", "object_count": len(objects), "algorithm": algorithm}
    atomic_write(manifest_path, manifest, False)
    return report


def main(argv=None):
    from .paths import default_dataset
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--dataset", type=Path, default=default_dataset()); parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv); print(json.dumps(run(args.dataset, args.force), indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
