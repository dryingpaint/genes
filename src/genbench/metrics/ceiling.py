"""Ceiling-normalized metrics."""

from __future__ import annotations

from genbench.types import CeilingInfo


def ceiling_normalized(
    metric_value: float,
    ceiling_value: float,
    ceiling_source: str,
) -> CeilingInfo:
    """Compute ceiling-normalized performance.

    Args:
        metric_value: The observed metric (e.g., R²).
        ceiling_value: The theoretical ceiling (e.g., h²_SNP).
        ceiling_source: Citation or description of ceiling source.

    Returns:
        CeilingInfo with the normalized performance ratio.
    """
    if ceiling_value <= 0:
        raise ValueError(f"Ceiling must be positive, got {ceiling_value}")

    normalized = metric_value / ceiling_value

    return CeilingInfo(
        ceiling_value=ceiling_value,
        ceiling_source=ceiling_source,
        normalized_performance=normalized,
    )
