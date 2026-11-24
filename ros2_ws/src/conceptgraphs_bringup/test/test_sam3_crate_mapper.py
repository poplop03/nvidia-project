from pathlib import Path

import pytest

from conceptgraphs_bringup.sam3_crate_mapper import (
    load_prompt_texts,
    marker_color_for_label,
)


def test_load_prompt_texts_uses_shared_sam3_config(tmp_path: Path):
    config = tmp_path / "pipeline.yaml"
    config.write_text(
        "sam3:\n"
            "  prompt_version: test_v1\n"
        "  prompts:\n"
        "    - {text: ladder, canonical_label: ladder}\n"
        "    - {text: wheelbarrow, canonical_label: wheelbarrow}\n",
        encoding="utf-8",
    )
    assert load_prompt_texts(config) == ["ladder", "wheelbarrow"]


def test_load_prompt_texts_rejects_empty_list(tmp_path: Path):
    config = tmp_path / "pipeline.yaml"
    config.write_text(
        "sam3: {prompt_version: test_v1, prompts: []}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="require text and canonical_label"):
        load_prompt_texts(config)


def test_marker_colors_follow_prompt_semantics():
    assert marker_color_for_label("yellow plastic container on the ground") == (0.10, 1.0, 0.10)
    assert marker_color_for_label("red_plastic_container") == (1.0, 0.0, 0.65)
    assert marker_color_for_label("blue plastic container on the ground") == (0.05, 0.30, 1.0)
    assert marker_color_for_label("ladder") == (0.72, 0.20, 0.95)
    assert marker_color_for_label("wheelbarrow") == (0.0, 1.0, 1.0)


def test_unknown_prompt_color_is_stable_and_visible():
    first = marker_color_for_label("watering can")
    assert first == marker_color_for_label("watering can")
    assert all(0.0 <= channel <= 1.0 for channel in first)
