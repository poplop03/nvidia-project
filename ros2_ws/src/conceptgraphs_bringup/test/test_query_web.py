from conceptgraphs_bringup.hmsg_database import build_database
from conceptgraphs_bringup.query_web import api_payload, create_app, reuses_context
from test_hmsg_database import fixture_dataset


def result(found=True):
    rows = [] if not found else [{
        "object_id": "obj1", "canonical_label": "red_plastic_container", "aisle_id": "a1",
        "frame_id": "camera_init", "similarity": .8, "semantic_confidence": .9,
        "centroid_3d": [1, 2, 3], "bbox_3d_min": [0, 1, 2], "bbox_3d_max": [2, 3, 4],
        "representative_view_id": "v1", "representative_rgb_path": "keyframes/rgb/v1.jpg",
        "representative_crop_path": "observations/crops/o1.jpg", "caption": "A red box.",
        "traversability": {"object_semantic_class": "non_traversable_obstacle",
                            "path_assessment": "passable", "can_robot_pass": True,
                            "reasoning_summary": "The object is beside the path.",
                            "visible_obstacles": [], "requires_geometric_verification": True},
    }]
    retrieval = {"query_id": "q1", "query": "red box", "results": rows, "ambiguous": False,
                 "verification": None, "latency_sec": .1,
                 "provenance": {"requires_geometric_verification": False}}
    return {"status": "success" if found else "not_found", "answer": "Natural answer.",
            "answer_source": "gemma", "intent": "find_object", "results": rows,
            "verified_facts": {"objects": rows}, "retrieval": retrieval}


def test_web_api_and_real_graph(tmp_path, monkeypatch):
    root = fixture_dataset(tmp_path); build_database(root)
    app = create_app(root)
    assert any(route.path == "/api/query" for route in app.routes)
    body = api_payload(root, result())
    assert body["matched_object"]["object_id"] == "obj1"
    assert body["answer_source"] == "gemma"
    assert "attributes" in body["matched_object"]
    assert "physical_properties" in body["matched_object"]
    node_types = {node["type"] for node in body["scene_graph"]["nodes"]}
    relations = {edge["relation"] for edge in body["scene_graph"]["edges"]}
    assert node_types == {"Greenhouse", "Aisle", "Object", "View"}
    assert {"CONTAINS", "OBSERVES"} <= relations
    assert body["scene_graph"]["focus_node_id"] == "obj1"
    assert len(body["related_images"]) >= 2
    assert {image["kind"] for image in body["related_images"]} >= {"crop", "view"}
    assert any(route.path == "/api/status" for route in app.routes)


def test_not_found_and_context_language(tmp_path, monkeypatch):
    root = fixture_dataset(tmp_path); build_database(root)
    body = api_payload(root, result(False))
    assert body["status"] == "not_found"
    assert reuses_context("Can the robot pass this object?")
    assert not reuses_context("Find object_000001")
