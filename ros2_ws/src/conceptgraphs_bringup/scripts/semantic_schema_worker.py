#!/usr/bin/env python3
"""Pydantic schemas and strict validation for M6 semantic records."""

import argparse
import json
import sys
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PhysicalProperties(StrictModel):
    rigidity: Literal["rigid", "semi_rigid", "soft", "unknown"]
    deformability: Literal["low", "medium", "high", "unknown"]
    compressibility: Literal["low", "medium", "high", "unknown"]
    fragility: Literal["low", "medium", "high", "unknown"]
    estimated_mass_class: Literal["light", "medium", "heavy", "unknown"]
    anchored_to_ground: bool | None
    confidence: float = Field(ge=0, le=1)
    evidence: str


class ObjectTraversability(StrictModel):
    semantic_class: Literal[
        "non_traversable_obstacle",
        "potentially_traversable",
        "traversable",
        "unknown",
    ]
    pass_over_recommendation: Literal["forbidden", "avoid", "allowed", "unknown"]
    contact_recommendation: Literal["avoid", "cautious_contact", "allowed", "unknown"]
    navigation_cost: Literal["free", "low", "high", "lethal", "unknown"]
    potential_risks: list[str]
    requires_geometric_verification: Literal[True]
    confidence: float = Field(ge=0, le=1)


class PhysicalObject(StrictModel):
    category: str
    color: str
    material: str
    shape: str
    state: str
    relative_position: str
    occlusion: Literal["low", "medium", "high", "unknown"]
    physical_properties: PhysicalProperties
    object_traversability: ObjectTraversability


class GreenhouseContext(StrictModel):
    crop_type_visible: str
    irrigation_equipment_visible: bool | None
    ground_condition: str


class VisibleObstacle(StrictModel):
    object_id: str
    category: str
    position_relative_to_path: str
    occupies_drivable_space: bool | None
    assessment: Literal[
        "non_traversable", "potentially_traversable", "traversable", "unknown"
    ]
    reason: str


class TraversabilityCues(StrictModel):
    vlm_path_assessment: Literal["passable", "blocked", "uncertain"]
    can_robot_pass: bool | None = Field(
        description="Exactly true for passable, false for blocked, null for uncertain."
    )
    blocking_object_ids: list[str] = Field(
        description="Must be empty when vlm_path_assessment is passable."
    )
    traversable_object_ids: list[str] = Field(
        description="IDs safe to contact or pass over, not merely objects the robot can drive around."
    )
    visible_obstacles: list[VisibleObstacle]
    reasoning_summary: str
    requires_geometric_verification: Literal[True]
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def consistent_traversability(self):
        blocking = set(self.blocking_object_ids)
        traversable = set(self.traversable_object_ids)
        if blocking & traversable:
            raise ValueError("blocking and traversable IDs must be disjoint")
        if self.vlm_path_assessment == "passable":
            if blocking:
                raise ValueError("passable View cannot list blocking IDs")
            if self.can_robot_pass is not True:
                raise ValueError("passable requires can_robot_pass=true")
        elif self.vlm_path_assessment == "blocked":
            if not blocking:
                raise ValueError("blocked View needs a blocking ID")
            if self.can_robot_pass is not False:
                raise ValueError("blocked requires can_robot_pass=false")
        elif self.can_robot_pass is not None:
            raise ValueError("uncertain requires can_robot_pass=null")

        obstacle_by_id = {item.object_id: item for item in self.visible_obstacles}
        referenced = blocking | traversable
        if not referenced <= set(obstacle_by_id):
            raise ValueError(
                "Every blocking/traversable ID must appear in visible_obstacles"
            )
        if any(
            obstacle_by_id[object_id].assessment != "non_traversable"
            for object_id in blocking
        ):
            raise ValueError("Every blocking ID must be assessed non_traversable")
        if any(
            obstacle_by_id[object_id].assessment
            not in {"traversable", "potentially_traversable"}
            for object_id in traversable
        ):
            raise ValueError("Every traversable ID needs a traversable assessment")
        return self


class StructuredCaption(StrictModel):
    scene_summary: str
    physical_objects: list[PhysicalObject]
    greenhouse_context: GreenhouseContext
    traversability_cues: TraversabilityCues
    uncertain_attributes: list[str]


class ObjectSemantic(StructuredCaption):
    @model_validator(mode="after")
    def exactly_one_target_object(self):
        if len(self.physical_objects) != 1:
            raise ValueError("Object caption must describe exactly one target object")
        return self


class ViewSemantic(StructuredCaption):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["schema", "validate"])
    parser.add_argument("kind", choices=["object", "view"])
    args = parser.parse_args()
    model = ObjectSemantic if args.kind == "object" else ViewSemantic
    if args.action == "schema":
        print(json.dumps(model.model_json_schema(), ensure_ascii=False))
    else:
        value = model.model_validate(json.load(sys.stdin))
        print(value.model_dump_json())


if __name__ == "__main__":
    main()
