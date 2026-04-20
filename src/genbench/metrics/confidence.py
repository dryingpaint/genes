"""Bootstrap confidence interval computation."""

from __future__ import annotations

from typing import Callable

import numpy as np

from genbench.types import MetricValue


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> MetricValue:
    """Compute a metric with bootstrap confidence interval.

    Args:
        metric_fn: Function(y_true, y_pred) -> float.
        y_true: Ground truth values.
        y_pred: Predicted values.
        n_bootstrap: Number of bootstrap samples.
        alpha: Significance level (0.05 = 95% CI).
        seed: Random seed.

    Returns:
        MetricValue with point estimate and CI bounds.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    point_estimate = float(metric_fn(y_true, y_pred))

    rng = np.random.default_rng(seed)
    bootstrap_estimates = np.empty(n_bootstrap)

    # Cap bootstrap sample size for large datasets to keep runtime ~seconds
    sample_size = min(n, 50_000)

    for i in range(n_bootstrap):
        idx = rng.choice(n, size=sample_size, replace=True)
        try:
            bootstrap_estimates[i] = metric_fn(y_true[idx], y_pred[idx])
        except (ValueError, ZeroDivisionError):
            bootstrap_estimates[i] = np.nan

    valid = bootstrap_estimates[~np.isnan(bootstrap_estimates)]
    if len(valid) < n_bootstrap * 0.5:
        ci_lower = ci_upper = point_estimate
    else:
        ci_lower = float(np.percentile(valid, 100 * alpha / 2))
        ci_upper = float(np.percentile(valid, 100 * (1 - alpha / 2)))

    return MetricValue(
        estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n=n,
    )
