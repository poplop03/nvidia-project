from pathlib import Path
import sys

from conceptgraphs_bringup.paths import (
    datasets_dir,
    default_sam3_model,
    package_config_dir,
    project_root,
    resolve_config_path,
    resolve_sam3_model,
    resolve_worker_python,
)


def test_project_root_and_package_config_dir_exist():
    root = project_root()
    assert (root / "AGENTS.md").is_file()
    assert (root / "ros2_ws").is_dir()
    cfg_dir = package_config_dir()
    assert (cfg_dir / "sam3_prompts.yaml").is_file()
    assert (cfg_dir / "layouts/greenhouse_20261029_three_aisles.yaml").is_file()
    assert resolve_sam3_model("/home/orc/SAM3/sam3.pt") == default_sam3_model()


def test_resolve_config_path_handles_stale_absolute_paths():
    stale_layout = "/home/orc/SAM3/ros2_ws/src/conceptgraphs_bringup/config/layouts/greenhouse_20261029_three_aisles.yaml"
    resolved = resolve_config_path(stale_layout)
    assert resolved.is_file()
    assert resolved.name == "greenhouse_20261029_three_aisles.yaml"

    stale_prompt = "/home/orc/SAM3/ros2_ws/src/conceptgraphs_bringup/config/retrieval_prompts_v1.yaml"
    resolved_prompt = resolve_config_path(stale_prompt)
    assert resolved_prompt.is_file()
    assert resolved_prompt.name == "retrieval_prompts_v1.yaml"


def test_resolve_config_path_handles_relative_package_paths():
    resolved = resolve_config_path("layouts/greenhouse_20261029_single_aisle.yaml")
    assert resolved.is_file()
    assert resolved.name == "greenhouse_20261029_single_aisle.yaml"


def test_resolve_worker_python_falls_back_when_configured_missing():
    resolved = resolve_worker_python("/nonexistent/anaconda3/bin/python")
    assert resolved == sys.executable


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("VLM_ROOT", str(tmp_path))
    assert project_root() == tmp_path.resolve()
    monkeypatch.setenv("VLM_DATASETS_DIR", str(tmp_path / "custom_ds"))
    assert datasets_dir() == (tmp_path / "custom_ds").resolve()
    monkeypatch.setenv("VLM_SAM3_MODEL", str(tmp_path / "custom.pt"))
    assert default_sam3_model() == (tmp_path / "custom.pt").resolve()
