"""Eval configuration — parameterizes how an eval runs.

Each eval can have multiple configs matching different published benchmarks.
A config specifies: data filters, split strategy, metrics, and expected baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from genbench.types import SplitType


@dataclass
class EvalConfig:
    """Configuration for a specific eval run."""

    name: str
    description: str

    # Data filtering — eval-specific keys interpreted by each eval's load_data()
    filters: dict[str, Any] = field(default_factory=dict)

    # Split strategy
    split_type: SplitType = SplitType.ZERO_SHOT
    split_params: dict[str, Any] = field(default_factory=dict)

    # Scoring
    metrics: list[str] = field(default_factory=lambda: ["auroc"])
    score_on_non_nan_only: bool = True

    # Expected baselines — what published models scored on this exact config
    # e.g. {"alphamissense": {"auroc": 0.940}}
    expected_baselines: dict[str, dict[str, float]] = field(default_factory=dict)

    # Provenance
    paper: str = ""
    doi: str | None = None


def validate_result(
    result_metrics: dict,
    config: EvalConfig,
    model_name: str,
    tolerance: float = 0.05,
) -> list[str]:
    """Check if results match expected baselines. Returns list of warnings."""
    warnings = []

    if model_name not in config.expected_baselines:
        return warnings

    expected = config.expected_baselines[model_name]
    for metric_name, expected_val in expected.items():
        if metric_name not in result_metrics:
            warnings.append(f"Expected metric '{metric_name}' not found in results")
            continue

        actual_metric = result_metrics[metric_name]
        if actual_metric.aggregate is None:
            warnings.append(f"Metric '{metric_name}' has no aggregate value")
            continue

        actual_val = actual_metric.aggregate.estimate
        diff = abs(actual_val - expected_val)
        if diff > tolerance:
            warnings.append(
                f"{metric_name}: got {actual_val:.4f}, expected {expected_val:.4f} "
                f"(diff {diff:.4f} > tolerance {tolerance}). "
                f"Paper: {config.paper}"
            )

    return warnings
