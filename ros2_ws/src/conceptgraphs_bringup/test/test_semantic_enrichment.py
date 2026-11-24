import subprocess
import sys
from pathlib import Path

import numpy as np

from conceptgraphs_bringup.semantic_enrichment import (
    embedding_manifest,
    pydantic_validate,
    validate_outputs,
)

PYTHON = sys.executable
WORKER = Path(__file__).parents[1] / "scripts/semantic_schema_worker.py"


def structured_caption(assessment="passable", blocking=None, traversable=None):
    blocking = blocking or []
    traversable = traversable or []
    can_pass = {"passable": True, "blocked": False, "uncertain": None}[assessment]
    return {
        "scene_summary": "A greenhouse aisle with one container.",
        "physical_objects": [
            {
                "category": "plastic container",
                "color": "red",
                "material": "plastic",
                "shape": "rectangular",
                "state": "on the ground",
                "relative_position": "right side of the aisle",
                "occlusion": "low",
                "physical_properties": {
                    "rigidity": "rigid",
                    "deformability": "low",
                    "compressibility": "low",
                    "fragility": "medium",
                    "estimated_mass_class": "unknown",
                    "anchored_to_ground": False,
                    "confidence": 0.82,
                    "evidence": "Visible rigid plastic walls.",
                },
                "object_traversability": {
                    "semantic_class": "non_traversable_obstacle",
                    "pass_over_recommendation": "forbidden",
                    "contact_recommendation": "avoid",
                    "navigation_cost": "lethal",
                    "potential_risks": ["collision", "object_damage"],
                    "requires_geometric_verification": True,
                    "confidence": 0.88,
                },
            }
        ],
        "greenhouse_context": {
            "crop_type_visible": "unknown",
            "irrigation_equipment_visible": False,
            "ground_condition": "dry soil",
        },
        "traversability_cues": {
            "vlm_path_assessment": assessment,
            "can_robot_pass": can_pass,
            "blocking_object_ids": blocking,
            "traversable_object_ids": traversable,
            "visible_obstacles": [],
            "reasoning_summary": "The visible corridor remains open.",
            "requires_geometric_verification": True,
            "confidence": 0.9,
        },
        "uncertain_attributes": ["estimated mass"],
    }


def test_pydantic_nested_object_schema_and_validation():
    parsed = pydantic_validate(PYTHON, WORKER, "object", structured_caption())
    properties = parsed["physical_objects"][0]["physical_properties"]
    assert properties["estimated_mass_class"] == "unknown"


def test_pydantic_rejects_geometry_override_and_bad_confidence():
    value = structured_caption()
    value["object_id"] = "evil"
    value["physical_objects"][0]["physical_properties"]["confidence"] = 2
    try:
        pydantic_validate(PYTHON, WORKER, "object", value)
        assert False
    except ValueError:
        pass


def test_embedding_manifest_maps_every_vector(tmp_path):
    views = [{"view_id": "v1", "rgb_path": "v1.jpg"}]
    obs = [{"observation_id": "o1", "crop_path": "c.jpg"}]
    objects = [
        {
            "object_id": "obj1",
            "observation_ids": ["o1"],
            "representative_view_id": "v1",
        }
    ]
    result = embedding_manifest(tmp_path, objects, obs, views)
    assert result["views"][0]["view_id"] == "v1"
    assert [x["entry_type"] for x in result["objects"]] == [
        "object_crop_mean",
        "object_representative_view",
    ]


def test_output_reference_validation(tmp_path):
    (tmp_path / "image.jpg").write_bytes(b"x")
    objects = [{"object_id": "o"}]
    views = [{"view_id": "v"}]
    parsed = structured_caption(traversable=["o"])
    base = {
        "status": "complete",
        "input_image_paths": ["image.jpg"],
        "input_ids": {"object_ids": ["o"]},
        "parsed": parsed,
    }
    result = validate_outputs(
        tmp_path,
        [{"object_id": "o"} | base],
        [{"view_id": "v"} | base],
        [{"view_id": "v"} | base],
        objects,
        views,
    )
    assert result["path_and_reference_integrity"]


def test_pydantic_rejects_contradictory_traversability():
    value = structured_caption(
        assessment="passable", blocking=["object_1"], traversable=["object_1"]
    )
    try:
        pydantic_validate(PYTHON, WORKER, "view", value)
        assert False
    except ValueError:
        pass


def test_generated_schema_contains_plan_8_3_sections():
    output = subprocess.check_output([PYTHON, str(WORKER), "schema", "view"], text=True)
    for field in (
        "scene_summary",
        "physical_objects",
        "greenhouse_context",
        "traversability_cues",
        "estimated_mass_class",
        "navigation_cost",
    ):
        assert field in output
