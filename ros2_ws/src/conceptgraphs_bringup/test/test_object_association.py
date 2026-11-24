import numpy as np
from conceptgraphs_bringup.object_association import (
    associate, associate_spatial, validate,
)


def config():
    return {"association": {"tracklet_max_distance_m": 0.5, "tracklet_max_diameter_m": 0.75,
        "cross_pass_candidate_radius_m": 1.0, "maximum_size_difference": 0.6,
        "size_cost_weight": 0.25, "cross_pass_maximum_cost": 1.1,
        "representative_weights": {"confidence": .35, "near_depth": .2,
        "valid_depth_ratio": .3, "mask_quality": .15}}}


def obs(number, view, x, label="yellow", aisle="aisle_01"):
    return {"observation_id": f"obs_{number:06d}_001", "view_id": view,
        "dataset_id": "dataset", "greenhouse_id": "greenhouse_01", "aisle_id": aisle,
        "canonical_label": label, "projection_status": "valid", "centroid_3d": [x,0,0],
        "bbox_3d_min": [x-.2,-.2,-.2], "bbox_3d_max": [x+.2,.2,.2],
        "bbox_2d_xyxy": [0,0,20,20], "mask_area_pixels": 300,
        "sam_confidence": .9, "median_depth_m": 1.5, "valid_depth_ratio": .8}


def view(identifier, timestamp, travel_pass):
    return {"view_id": identifier, "timestamp_ns": timestamp, "travel_pass": travel_pass}


def test_tracklet_hungarian_matches_two_passes_and_uses_midpoint():
    observations = [obs(1,"v1",0), obs(2,"v2",.1), obs(3,"v3",.7), obs(4,"v4",.8)]
    views = {"v1":view("v1",1,"forward"), "v2":view("v2",2,"forward"),
             "v3":view("v3",3,"return"), "v4":view("v4",4,"return")}
    objects, nodes, edges, tracklets, matches, _ = associate(observations, views, config())
    assert len(tracklets) == 2 and len(matches) == 1 and len(objects) == 1
    assert objects[0]["position_quality"] == "multi_pass"
    assert objects[0]["centroid_3d"] == [0.4, 0.0, 0.0]
    assert len(edges) == 4
    assert validate(objects, nodes, edges, observations, views)["valid"]


def test_does_not_transitively_chain_tracklet():
    observations = [obs(1,"v1",0), obs(2,"v2",.45), obs(3,"v3",.9)]
    views = {f"v{i}":view(f"v{i}",i,"forward") for i in range(1,4)}
    objects, _, _, tracklets, _, _ = associate(observations, views, config())
    assert len(tracklets) == 2
    assert sorted(x["observation_count"] for x in objects) == [1,2]
    assert all(x["position_quality"] == "single_sided" for x in objects)
    assert all(x["association_status"] == "ambiguous" for x in objects)


def test_separates_label_and_aisle():
    observations = [obs(1,"v1",0,"yellow","a1"), obs(2,"v2",.01,"red","a1"),
                    obs(3,"v3",.02,"yellow","a2")]
    views = {f"v{i}":view(f"v{i}",i,"forward") for i in range(1,4)}
    objects, _, _, tracklets, _, _ = associate(observations, views, config())
    assert len(tracklets) == 3 and len(objects) == 3


def test_unassigned_aisle_is_preserved_and_sorted_safely():
    observations = [obs(1, "v1", 0, aisle=None),
                    obs(2, "v2", .01, aisle="aisle_01")]
    views = {"v1": view("v1", 1, "forward"),
             "v2": view("v2", 2, "forward")}
    objects, _, _, tracklets, _, _ = associate(observations, views, config())
    assert len(tracklets) == 2 and len(objects) == 2
    assert {item["aisle_id"] for item in objects} == {None, "aisle_01"}
    assert objects[0]["aisle_id"] is None


def test_representative_prefers_near_high_quality_view():
    a, b = obs(1,"v1",0), obs(2,"v2",.1)
    a["median_depth_m"], a["valid_depth_ratio"], a["sam_confidence"] = 2.8, .3, .8
    b["median_depth_m"], b["valid_depth_ratio"], b["sam_confidence"] = 1.0, .9, .95
    views = {"v1":view("v1",1,"forward"), "v2":view("v2",2,"forward")}
    objects, *_ = associate([a,b], views, config())
    assert objects[0]["representative_observation_id"] == b["observation_id"]


def test_spatial_mode_merges_across_aisle_and_pass_metadata():
    cfg = config()
    cfg["association"]["spatial_cluster_radius_m"] = 1.5
    observations = [
        obs(1, "v1", 0, "red", None),
        obs(2, "v2", .6, "red", "aisle_02"),
        obs(3, "v3", 1.1, "red", "aisle_02"),
        obs(4, "v4", 5.0, "yellow", "aisle_02"),
        obs(5, "v5", 5.5, "yellow", "aisle_02"),
        obs(6, "v6", 10.0, "yellow", "aisle_03"),
    ]
    views = {
        f"v{i}": view(f"v{i}", i, "forward" if i % 2 else "return")
        for i in range(1, 7)
    }
    objects, _, _, _, _, _ = associate_spatial(
        observations, views, cfg)
    assert len(objects) == 3
    assert sorted((item["canonical_label"], item["observation_count"])
                  for item in objects) == [
                      ("red", 3), ("yellow", 1), ("yellow", 2)]
