from pathlib import Path

from conceptgraphs_bringup.keyframe_extractor import load_config


CONFIG_ROOT = Path(__file__).parents[1] / "config"


def test_dataset_configs_keep_inputs_and_outputs_isolated():
    single = load_config(CONFIG_ROOT / "datasets/greenhouse_20261029_single_aisle.yaml")
    multi = load_config(CONFIG_ROOT / "datasets/greenhouse_20261029_three_aisles.yaml")

    assert single["dataset_id"] == "greenhouse_20261029_aisle01"
    assert multi["dataset_id"] == "greenhouse_20261029_three_aisles"
    assert single["bag_path"] != multi["bag_path"]
    assert single["output_dir"] != multi["output_dir"]
    assert single["layout_path"] != multi["layout_path"]
    assert single["selection"] == multi["selection"]
    assert single["sam3"] == multi["sam3"]
