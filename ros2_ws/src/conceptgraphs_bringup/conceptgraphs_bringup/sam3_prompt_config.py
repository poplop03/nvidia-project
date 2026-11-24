"""Shared SAM3 prompt configuration loader for offline and online inference."""

from pathlib import Path

import yaml


def load_sam3_prompt_config(path):
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"Invalid SAM3 prompt config: {path}")
    section = document.get("sam3", document)
    prompts = section.get("prompts", [])
    version = section.get("prompt_version") or document.get("prompt_version")
    if not version:
        raise ValueError(f"Missing prompt_version in {path}")
    if not prompts or any(
            not isinstance(item, dict)
            or not str(item.get("text", "")).strip()
            or not str(item.get("canonical_label", "")).strip()
            for item in prompts):
        raise ValueError(f"Prompts in {path} require text and canonical_label")
    texts = [item["text"].strip() for item in prompts]
    labels = [item["canonical_label"].strip() for item in prompts]
    if len(texts) != len(set(texts)):
        raise ValueError(f"SAM3 prompt texts must be unique in {path}")
    if len(labels) != len(set(labels)):
        raise ValueError(f"SAM3 canonical labels must be unique in {path}")
    return {
        "prompt_version": str(version),
        "prompts": [
            {"text": text, "canonical_label": label}
            for text, label in zip(texts, labels)
        ],
        "prompt_config_path": str(path),
    }
