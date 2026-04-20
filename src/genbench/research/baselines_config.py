"""Read/write baselines.yaml — the auto-maintained SOTA baseline config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

BASELINES_PATH = Path(__file__).parent.parent / "baselines.yaml"


def load_baselines() -> dict[str, Any]:
    """Load baselines.yaml. Returns empty dict if file doesn't exist."""
    if not BASELINES_PATH.exists():
        return {}
    with open(BASELINES_PATH) as f:
        return yaml.safe_load(f) or {}


def save_baselines(data: dict[str, Any]) -> None:
    """Write baselines.yaml."""
    with open(BASELINES_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=True)


def get_known_sota(eval_name: str) -> dict | None:
    """Get the known SOTA baseline for a specific eval."""
    baselines = load_baselines()
    return baselines.get(eval_name)


def update_sota(eval_name: str, sota_info: dict) -> None:
    """Update the SOTA for a single eval."""
    baselines = load_baselines()
    baselines[eval_name] = sota_info
    save_baselines(baselines)
