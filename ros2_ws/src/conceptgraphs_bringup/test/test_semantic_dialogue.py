import json

from conceptgraphs_bringup.hmsg_database import build_database
from conceptgraphs_bringup.semantic_dialogue import (
    clarification_for,
    compose_answer,
    fallback_answer,
    interact,
    response_spec,
    route_channel,
    route_intent,
    spatial_facts,
)
from test_hmsg_database import fixture_dataset


def add_layout(root, tmp_path):
    layout = tmp_path / "layout.yaml"
    layout.write_text("""mode: single_aisle
greenhouse: {id: g1, frame_id: camera_init}
aisles:
  - id: a1
    centerline: [[0.0, 0.0], [30.0, 0.0]]
    head_point: [0.0, 0.0]
    tail_point: [30.0, 0.0]
    polygon_xy: []
""")
    (root / "pipeline_config.yaml").write_text("layout_path: " + str(layout) + "\n")


def test_dialogue_intent_routes_and_clarification():
    assert route_intent("What objects are in the map?") == "list_objects"
    assert route_intent("Where is the red container?") == "locate_object"
    assert route_intent("Can the robot pass this object?") == "assess_traversability"
    assert clarification_for("màu giống quả dưa hấu")
    assert clarification_for("màu giống ruột quả dưa hấu") is None
    assert route_channel("Bạn có thể làm được gì?") == "general_chat"
    assert route_channel("Red container ở đâu?") == "map_query"


def test_six_demo_questions_have_explicit_intents_and_formats():
    assert route_intent("Where is the ladder?") == "locate_object"
    assert response_spec("Where is the ladder?", "locate_object")["format"] == "topological"
    assert route_intent("Give me the xyz coordinates of the ladder.") == "locate_object"
    assert response_spec("Give me the xyz coordinates of the ladder.", "locate_object")["format"] == "xyz"
    assert response_spec("Which aisle contains the ladder?", "locate_object")["format"] == "aisle"
    assert response_spec("How far is the ladder from the aisle head?", "locate_object")["format"] == "distance_from_head"
    assert route_intent("Can the robot pass the ladder?") == "assess_traversability"
    assert route_intent("What physical properties does the ladder have?") == "describe_object_properties"


def test_intent_specific_fallbacks_answer_requested_fields():
    base = {"intent": "locate_object", "response_spec": {"format": "xyz"}, "objects": [{
        "object_id": "object_000004", "label": "ladder", "aisle_id": "aisle_03",
        "frame_id": "camera_init", "centroid_3d": [15.589, 10.288, .304],
    }]}
    answer = fallback_answer("coordinates", base)
    assert "15.59" in answer and "10.29" in answer and "0.30" in answer
    assert "camera_init" in answer

    properties = {"intent": "describe_object_properties",
                  "response_spec": {"format": "physical_properties"}, "objects": [{
                      "object_id": "object_000004", "label": "ladder", "aisle_id": "aisle_03",
                      "color": "silver", "material": "metal", "shape": "ladder frame",
                      "physical_properties": {"rigidity": "rigid", "deformability": "low",
                                               "compressibility": "low", "fragility": "low",
                                               "estimated_mass_class": "heavy"},
                  }]}
    answer = fallback_answer("properties", properties)
    assert "metal" in answer and "rigid" in answer and "low deformability" in answer


def test_traversability_fallback_separates_object_from_route():
    facts = {"intent": "assess_traversability", "response_spec": {"format": "traversability"},
             "objects": [{"object_id": "object_000004", "label": "ladder", "aisle_id": "aisle_03",
                          "object_traversability": "non_traversable_obstacle",
                          "path_geometry": {"can_robot_pass": True, "available_clearance_m": 2.09}}]}
    answer = fallback_answer("Can the robot pass?", facts)
    assert "should not drive over or through" in answer
    assert "route around it is estimated to be passable" in answer


def test_natural_numeric_formatting_is_accepted():
    from conceptgraphs_bringup.semantic_dialogue import answer_contains_number
    assert answer_contains_number("Coordinates are 15.589, 10.288, 0.304 metres", 15.59)
    assert answer_contains_number("It is 21.09 metres from the head", 21.1, tolerance=.11)
    assert not answer_contains_number("It is 22.4 metres from the head", 21.1, tolerance=.11)


def test_spatial_facts_come_from_layout(tmp_path):
    root = fixture_dataset(tmp_path); add_layout(root, tmp_path)
    facts = spatial_facts(root, {"aisle_id": "a1", "centroid_3d": [18.0, 1.0, 0.0]})
    assert facts["section"] == "middle"
    assert facts["tendency"] == "toward_tail"
    assert facts["distance_from_head_m"] == 18.0
    assert facts["source"] == "layout_geometry"


def test_composer_rejects_unknown_ids_and_falls_back(tmp_path, monkeypatch):
    facts = {"intent": "find_object", "objects": [{"object_id": "obj1", "label": "red_box", "aisle_id": "a1"}]}
    cfg = {"ollama": {}}
    monkeypatch.setattr("conceptgraphs_bringup.semantic_dialogue._load_prompts", lambda: {"response_prompt": "answer"})
    monkeypatch.setattr("conceptgraphs_bringup.semantic_dialogue.available_ollama", lambda cfg: ("base", "gemma", "digest"))
    monkeypatch.setattr("conceptgraphs_bringup.semantic_dialogue.ollama_json", lambda *a, **k: {
        "answer": "Invented", "mentioned_object_ids": ["unknown"], "confidence": .9,
        "requires_geometric_verification": False})
    value = compose_answer(tmp_path, "find red", facts, cfg)
    assert value["source"] == "template_fallback"
    assert "red box" in value["answer"]


def test_list_fallback_mentions_every_verified_object():
    facts = {"intent": "list_objects", "objects": [
        {"object_id": "one", "label": "red_container"},
        {"object_id": "two", "label": "yellow_container"},
    ]}
    answer = fallback_answer("list", facts)
    assert "2 mapped objects" in answer
    assert "red container" in answer and "yellow container" in answer


def test_composer_rejects_false_not_found(tmp_path, monkeypatch):
    facts = {"intent": "find_object", "objects": [{"object_id": "obj1", "label": "red_container", "aisle_id": "a1"}]}
    monkeypatch.setattr("conceptgraphs_bringup.semantic_dialogue._load_prompts", lambda: {"response_prompt": "answer"})
    monkeypatch.setattr("conceptgraphs_bringup.semantic_dialogue.available_ollama", lambda cfg: ("base", "gemma", "digest"))
    monkeypatch.setattr("conceptgraphs_bringup.semantic_dialogue.ollama_json", lambda *a, **k: {
        "answer": "No matching container was found.", "mentioned_object_ids": [], "confidence": .9,
        "requires_geometric_verification": False})
    value = compose_answer(tmp_path, "like watermelon flesh", facts, {"ollama": {}})
    assert value["source"] == "template_fallback"
    assert "best verified match" in value["answer"]


def test_not_found_never_calls_gemma(tmp_path, monkeypatch):
    monkeypatch.setattr("conceptgraphs_bringup.semantic_dialogue.ollama_json",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call Gemma")))
    value = compose_answer(tmp_path, "find blue", {"intent": "find_object", "objects": []}, {}, True)
    assert value["source"] == "verified_not_found"
    assert "could not find" in value["answer"]


def test_manual_chat_never_calls_retrieval(tmp_path, monkeypatch):
    root = fixture_dataset(tmp_path); build_database(root)
    monkeypatch.setattr("conceptgraphs_bringup.hmsg_retrieval.load_config", lambda dataset: {"ollama": {}})
    monkeypatch.setattr("conceptgraphs_bringup.semantic_dialogue.general_chat",
                        lambda *a, **k: {"answer": "Xin chào!", "source": "gemma_general_chat", "confidence": 1.0})
    monkeypatch.setattr("conceptgraphs_bringup.semantic_dialogue.dialogue",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("retrieval must not run")))
    value = interact(root, "Bạn có thể làm được gì?", mode="chat", history=[])
    assert value["response_type"] == "general_chat"
    assert value["answer"] == "Xin chào!"


def test_manual_map_always_uses_retrieval_path(tmp_path, monkeypatch):
    root = fixture_dataset(tmp_path); build_database(root)
    monkeypatch.setattr("conceptgraphs_bringup.hmsg_retrieval.load_config", lambda dataset: {"ollama": {}})
    monkeypatch.setattr("conceptgraphs_bringup.semantic_dialogue.dialogue",
                        lambda *a, **k: {"status": "success", "intent": "find_object", "answer": "found",
                                             "answer_source": "template", "results": [{"object_id": "obj1"}]})
    value = interact(root, "hello", mode="map")
    assert value["response_type"] == "map_query"
    assert value["routing"]["reason"] == "manual_map"
