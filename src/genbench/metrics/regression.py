"""Regression metrics for continuous prediction tasks (Tiers 1-3)."""

from __future__ import annotations

import numpy as np
from scipy import stats

from genbench.metrics.confidence import bootstrap_ci
from genbench.types import MetricValue


def pearson_r(
    y_true: np.ndarray, y_pred: np.ndarray, n_bootstrap: int = 1000
) -> MetricValue:
    """Cross-individual Pearson correlation with bootstrap CI.

    This is the Huang/Sasse test metric for eQTL prediction.
    """

    def _pearson(yt: np.ndarray, yp: np.ndarray) -> float:
        if len(yt) < 3:
            return 0.0
        r, _ = stats.pearsonr(yt, yp)
        return float(r)

    return bootstrap_ci(_pearson, y_true, y_pred, n_bootstrap=n_bootstrap)


def spearman_rho(
    y_true: np.ndarray, y_pred: np.ndarray, n_bootstrap: int = 1000
) -> MetricValue:
    """Spearman rank correlation with bootstrap CI.

    Primary metric for DMS and eQTL effect sizes.
    """

    def _spearman(yt: np.ndarray, yp: np.ndarray) -> float:
        if len(yt) < 3:
            return 0.0
        rho, _ = stats.spearmanr(yt, yp)
        return float(rho)

    return bootstrap_ci(_spearman, y_true, y_pred, n_bootstrap=n_bootstrap)


def incremental_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_covariates: np.ndarray | None = None,
    n_bootstrap: int = 1000,
) -> MetricValue:
    """Incremental R² = R²(model + covariates) - R²(covariates only).

    Measures the variance explained by the genetic predictor beyond demographics.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    def _incr_r2(yt: np.ndarray, yp: np.ndarray) -> float:
        if y_covariates is None:
            # No covariates — just return R²
            ss_res = np.sum((yt - yp) ** 2)
            ss_tot = np.sum((yt - yt.mean()) ** 2)
            if ss_tot == 0:
                return 0.0
            return float(1 - ss_res / ss_tot)

        # R² with covariates only
        lr_cov = LinearRegression().fit(y_covariates, yt)
        r2_cov = r2_score(yt, lr_cov.predict(y_covariates))

        # R² with covariates + model predictions
        X_full = np.column_stack([y_covariates, yp])
        lr_full = LinearRegression().fit(X_full, yt)
        r2_full = r2_score(yt, lr_full.predict(X_full))

        return float(r2_full - r2_cov)

    return bootstrap_ci(_incr_r2, y_true, y_pred, n_bootstrap=n_bootstrap)


def direction_of_effect_accuracy(
    true_effects: np.ndarray, pred_effects: np.ndarray, n_bootstrap: int = 1000
) -> MetricValue:
    """Fraction of variants where predicted direction matches observed direction."""

    def _doe(yt: np.ndarray, yp: np.ndarray) -> float:
        # Exclude zero effects
        nonzero = (yt != 0) & (yp != 0)
        if nonzero.sum() == 0:
            return 0.5
        same_sign = np.sign(yt[nonzero]) == np.sign(yp[nonzero])
        return float(same_sign.mean())

    return bootstrap_ci(_doe, true_effects, pred_effects, n_bootstrap=n_bootstrap)


def mae(y_true: np.ndarray, y_pred: np.ndarray, n_bootstrap: int = 1000) -> MetricValue:
    """Mean absolute error with bootstrap CI."""

    def _mae(yt: np.ndarray, yp: np.ndarray) -> float:
        return float(np.mean(np.abs(yt - yp)))

    return bootstrap_ci(_mae, y_true, y_pred, n_bootstrap=n_bootstrap)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray, n_bootstrap: int = 1000) -> MetricValue:
    """Standard R² (coefficient of determination)."""

    def _r2(yt: np.ndarray, yp: np.ndarray) -> float:
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1 - ss_res / ss_tot)

    return bootstrap_ci(_r2, y_true, y_pred, n_bootstrap=n_bootstrap)
