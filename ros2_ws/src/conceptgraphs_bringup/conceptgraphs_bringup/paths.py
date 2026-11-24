"""Portable path resolution for the Greenhouse HMSG workspace.

Supports environment overrides so code and tests work across machines and
multi-provider agent environments without hardcoded user home directories.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


def project_root() -> Path:
    """Return the repository root directory."""
    env_root = os.environ.get("VLM_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "ros2_ws").is_dir() and (
            (parent / "AGENTS.md").is_file()
            or (parent / "data").is_dir()
            or (parent / "models").is_dir()
        ):
            return parent
    for parent in here.parents:
        if (parent / "ros2_ws").is_dir():
            return parent
    return Path.cwd().resolve()


def package_config_dir() -> Path:
    """Return the conceptgraphs_bringup config directory (source tree or install share)."""
    source = Path(__file__).resolve().parents[1] / "config"
    if source.is_dir():
        return source
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory("conceptgraphs_bringup")) / "config"


def datasets_dir() -> Path:
    """Return the directory containing extracted HMSG datasets."""
    env_dir = os.environ.get("VLM_DATASETS_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    root = project_root()
    for candidate in (root / "data" / "datasets", root / "ros2_ws" / "datasets"):
        if candidate.is_dir():
            return candidate
    return root / "data" / "datasets"


def bags_dir() -> Path:
    """Return the directory containing raw or processed ROS 2 bags."""
    env_dir = os.environ.get("VLM_BAGS_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    root = project_root()
    for candidate in (root / "data" / "bags", root / "ros2_ws" / "dataset"):
        if candidate.is_dir():
            return candidate
    return root / "data" / "bags"


def default_dataset(name: str = "greenhouse_20261029_three_aisles") -> Path:
    """Return the default HMSG dataset path."""
    base = datasets_dir()
    preferred = base / name
    if preferred.is_dir():
        return preferred
    pilot = base / "greenhouse_20260910_aisle01"
    if pilot.is_dir():
        return pilot
    return preferred


def default_sam3_model() -> Path:
    """Return the default SAM3 weights path."""
    env_model = os.environ.get("VLM_SAM3_MODEL")
    if env_model:
        return Path(env_model).expanduser().resolve()
    root = project_root()
    for candidate in (root / "models" / "sam3.pt", root / "sam3.pt"):
        if candidate.is_file():
            return candidate
    return root / "models" / "sam3.pt"


def _safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _safe_is_executable(path: str) -> bool:
    try:
        return os.access(path, os.X_OK)
    except OSError:
        return False


def resolve_worker_python(configured: str | Path | None = None) -> str:
    """Resolve an executable Python interpreter for isolated worker subprocesses."""
    env_python = os.environ.get("VLM_WORKER_PYTHON")
    if env_python and _safe_is_executable(env_python):
        return env_python
    if configured:
        candidate = str(configured)
        if os.path.isabs(candidate) and _safe_is_executable(candidate):
            return candidate
        if not os.path.isabs(candidate):
            return candidate
    return sys.executable


def resolve_config_path(configured: str | Path, dataset: Path | None = None) -> Path:
    """Resolve a config or layout YAML path, falling back to package config/.

    Allows frozen datasets generated on another machine (with stale absolute
    paths in pipeline_config.yaml) to resolve package-bundled YAML files cleanly
    without modifying dataset artifacts.
    """
    raw = Path(configured)
    if _safe_is_file(raw):
        return raw
    if not raw.is_absolute():
        if dataset is not None and _safe_is_file(dataset / raw):
            return dataset / raw
        pkg_candidate = package_config_dir() / raw
        if _safe_is_file(pkg_candidate):
            return pkg_candidate
    posix = raw.as_posix()
    marker = "/config/"
    if marker in posix:
        subpath = posix.split(marker, 1)[1]
        pkg_candidate = package_config_dir() / subpath
        if _safe_is_file(pkg_candidate):
            return pkg_candidate
    pkg_by_name = package_config_dir() / raw.name
    if _safe_is_file(pkg_by_name):
        return pkg_by_name
    if not raw.is_absolute() and dataset is not None:
        return dataset / raw
    return raw


def resolve_project_path(configured: str | Path) -> Path:
    """Resolve a relative project path against project_root()."""
    raw = Path(configured)
    if raw.is_absolute():
        return raw
    return project_root() / raw


def resolve_sam3_model(configured: str | Path | None = None) -> Path:
    """Resolve a SAM3 model path, falling back to default_sam3_model() if unreadable."""
    if configured:
        candidate = resolve_project_path(configured)
        if _safe_is_file(candidate):
            return candidate
    return default_sam3_model()
