"""Per-ancestry metric stratification.

There is no code path that returns aggregate-only results.
Every metric must be reported per ancestry group.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from genbench.metrics.confidence import bootstrap_ci
from genbench.types import AncestryStratifiedMetric, MetricValue


def stratify_by_ancestry(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ancestry_labels: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    min_samples: int = 30,
) -> AncestryStratifiedMetric:
    """Compute a metric separately for each ancestry group.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        ancestry_labels: Ancestry group label per sample.
        metric_fn: Metric function(y_true, y_pred) -> float.
        n_bootstrap: Bootstrap iterations per group.
        min_samples: Minimum samples per group to compute metric.

    Returns:
        AncestryStratifiedMetric with per-ancestry and aggregate values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ancestry_labels = np.asarray(ancestry_labels)

    per_ancestry: dict[str, MetricValue] = {}

    for group in np.unique(ancestry_labels):
        mask = ancestry_labels == group
        n = mask.sum()

        if n < min_samples:
            per_ancestry[str(group)] = MetricValue(
                estimate=float("nan"),
                ci_lower=float("nan"),
                ci_upper=float("nan"),
                n=int(n),
            )
            continue

        mv = bootstrap_ci(
            metric_fn,
            y_true[mask],
            y_pred[mask],
            n_bootstrap=n_bootstrap,
        )
        per_ancestry[str(group)] = mv

    # Aggregate across all samples
    aggregate = bootstrap_ci(metric_fn, y_true, y_pred, n_bootstrap=n_bootstrap)

    return AncestryStratifiedMetric(
        per_ancestry=per_ancestry,
        aggregate=aggregate,
    )


def portability_ratio(
    metric_eur: float,
    metric_target: float,
) -> float:
    """Compute portability ratio = metric(target) / metric(EUR).

    Expected ranges (from BENCHMARKS.md):
        AFR: 0.25–0.50
        SAS: 0.50–0.70
        EAS: 0.40–0.60
    """
    if metric_eur == 0:
        return 0.0
    return metric_target / metric_eur
