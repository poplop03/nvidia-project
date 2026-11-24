import numpy as np
import pytest
import yaml

from conceptgraphs_bringup.sam3_offline import (
    canonical_prompt_map, clip_bbox, config_fingerprint, context_bbox,
    masked_crop, observation_id,
)
from conceptgraphs_bringup.sam3_prompt_config import load_sam3_prompt_config


def test_observation_id_is_stable_and_view_scoped():
    assert observation_id("view_000042", 3) == "obs_000042_003"


def test_bbox_and_context_are_clipped():
    bbox = clip_bbox((-3.2, 4.1, 30.7, 50.2), 25, 40)
    assert bbox == [0, 4, 25, 40]
    assert context_bbox([10, 10, 20, 20], 100, 80, 0.2) == [8, 8, 22, 22]


def test_masked_crop_blacks_out_background():
    image = np.full((4, 4, 3), 100, np.uint8)
    mask = np.zeros((4, 4), np.uint8)
    mask[1:3, 1:3] = 255
    crop = masked_crop(image, mask, [0, 0, 4, 4])
    assert tuple(crop[0, 0]) == (0, 0, 0)
    assert tuple(crop[1, 1]) == (100, 100, 100)


def test_prompt_config_and_fingerprint_are_deterministic():
    config = {"sam3": {"prompts": [
        {"text": "yellow container", "canonical_label": "yellow_container"}
    ]}}
    assert canonical_prompt_map(config) == {"yellow container": "yellow_container"}
    assert config_fingerprint(config["sam3"]) == config_fingerprint(config["sam3"])
    duplicate = {"sam3": {"prompts": [
        {"text": "same", "canonical_label": "one"},
        {"text": "same", "canonical_label": "two"},
    ]}}
    with pytest.raises(ValueError, match="unique"):
        canonical_prompt_map(duplicate)


def test_shared_prompt_config_supports_root_schema(tmp_path):
    path = tmp_path / "prompts.yaml"
    path.write_text(yaml.safe_dump({
        "prompt_version": "test_v1",
        "prompts": [{"text": "blue container",
                     "canonical_label": "blue_container"}],
    }), encoding="utf-8")
    config = load_sam3_prompt_config(path)
    assert config["prompt_version"] == "test_v1"
    assert config["prompts"][0]["canonical_label"] == "blue_container"
