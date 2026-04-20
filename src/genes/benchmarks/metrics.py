"""Metrics for the benchmark harness.

All metrics return point estimate + 95% CI via bootstrap (n_boot=1000).
Follows the specification in docs/phenotype-suite/BENCHMARKS.md.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from scipy.stats import spearmanr, pearsonr

from genes.benchmarks.types import MetricValue


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap a metric to get (estimate, ci_lower, ci_upper)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    estimate = metric_fn(y_true, y_pred)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boots[i] = metric_fn(y_true[idx], y_pred[idx])
        except (ValueError, ZeroDivisionError):
            boots[i] = np.nan
    boots = boots[~np.isnan(boots)]
    if len(boots) == 0:
        return estimate, np.nan, np.nan
    ci_lower = float(np.percentile(boots, 100 * alpha / 2))
    ci_upper = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(estimate), ci_lower, ci_upper


def auroc(y_true: ArrayLike, y_score: ArrayLike, n_boot: int = 1000) -> MetricValue:
    """Area under ROC curve with bootstrap 95% CI."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    est, lo, hi = _bootstrap_ci(y_true, y_score, roc_auc_score, n_boot=n_boot)
    return MetricValue(estimate=est, ci_lower=lo, ci_upper=hi, n=len(y_true))


def auprc(y_true: ArrayLike, y_score: ArrayLike, n_boot: int = 1000) -> MetricValue:
    """Area under precision-recall curve with bootstrap 95% CI."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    est, lo, hi = _bootstrap_ci(y_true, y_score, average_precision_score, n_boot=n_boot)
    return MetricValue(estimate=est, ci_lower=lo, ci_upper=hi, n=len(y_true))


def balanced_acc(y_true: ArrayLike, y_pred: ArrayLike, n_boot: int = 1000) -> MetricValue:
    """Balanced accuracy with bootstrap 95% CI."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    est, lo, hi = _bootstrap_ci(y_true, y_pred, balanced_accuracy_score, n_boot=n_boot)
    return MetricValue(estimate=est, ci_lower=lo, ci_upper=hi, n=len(y_true))


def spearman_rho(y_true: ArrayLike, y_pred: ArrayLike, n_boot: int = 1000) -> MetricValue:
    """Spearman rank correlation with bootstrap 95% CI."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    def _spearman(a, b):
        return spearmanr(a, b).statistic

    est, lo, hi = _bootstrap_ci(y_true, y_pred, _spearman, n_boot=n_boot)
    return MetricValue(estimate=est, ci_lower=lo, ci_upper=hi, n=len(y_true))


def pearson_r(y_true: ArrayLike, y_pred: ArrayLike, n_boot: int = 1000) -> MetricValue:
    """Pearson correlation with bootstrap 95% CI."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    def _pearson(a, b):
        return pearsonr(a, b).statistic

    est, lo, hi = _bootstrap_ci(y_true, y_pred, _pearson, n_boot=n_boot)
    return MetricValue(estimate=est, ci_lower=lo, ci_upper=hi, n=len(y_true))


def incremental_r2(
    y_true: ArrayLike,
    y_pred_full: ArrayLike,
    y_pred_base: ArrayLike,
) -> float:
    """R² of full model minus R² of base model (covariates only)."""
    y_true = np.asarray(y_true)
    ss_res_full = np.sum((y_true - np.asarray(y_pred_full)) ** 2)
    ss_res_base = np.sum((y_true - np.asarray(y_pred_base)) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    r2_full = 1 - ss_res_full / ss_tot
    r2_base = 1 - ss_res_base / ss_tot
    return float(r2_full - r2_base)


def ceiling_normalized_r2(r2: float, h2_snp: float) -> float:
    """What fraction of achievable variance is the model capturing."""
    if h2_snp <= 0:
        return 0.0
    return r2 / h2_snp


def portability_ratio(metric_target: float, metric_eur: float) -> float:
    """Ratio of performance in target ancestry vs EUR."""
    if metric_eur <= 0:
        return 0.0
    return metric_target / metric_eur


def direction_of_effect_accuracy(
    true_direction: ArrayLike, pred_direction: ArrayLike
) -> MetricValue:
    """Fraction of variants where predicted direction matches observed."""
    true_dir = np.sign(np.asarray(true_direction))
    pred_dir = np.sign(np.asarray(pred_direction))
    acc = float(np.mean(true_dir == pred_dir))
    n = len(true_dir)
    # Wilson score interval for proportion
    z = 1.96
    denom = 1 + z**2 / n
    center = (acc + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((acc * (1 - acc) + z**2 / (4 * n)) / n) / denom
    return MetricValue(
        estimate=acc, ci_lower=float(center - margin), ci_upper=float(center + margin), n=n
    )


def calibration_slope_intercept(
    y_true: ArrayLike, y_prob: ArrayLike
) -> tuple[float, float]:
    """Logistic calibration: slope and intercept of observed vs predicted."""
    from sklearn.linear_model import LogisticRegression

    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    # Logit of predicted probabilities
    eps = 1e-10
    logit_p = np.log(np.clip(y_prob, eps, 1 - eps) / (1 - np.clip(y_prob, eps, 1 - eps)))
    lr = LogisticRegression(fit_intercept=True, C=np.inf, max_iter=1000)
    lr.fit(logit_p.reshape(-1, 1), y_true)
    slope = float(lr.coef_[0, 0])
    intercept = float(lr.intercept_[0])
    return slope, intercept


def brier_score(y_true: ArrayLike, y_prob: ArrayLike, n_boot: int = 1000) -> MetricValue:
    """Brier score (lower is better) with bootstrap 95% CI."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    est, lo, hi = _bootstrap_ci(y_true, y_prob, brier_score_loss, n_boot=n_boot)
    return MetricValue(estimate=est, ci_lower=lo, ci_upper=hi, n=len(y_true))
