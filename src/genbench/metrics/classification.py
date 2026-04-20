"""Classification metrics for Tier 0 and disease endpoints."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_curve,
    roc_auc_score,
)

from genbench.metrics.confidence import bootstrap_ci
from genbench.types import MetricValue


def auroc(y_true: np.ndarray, y_score: np.ndarray, n_bootstrap: int = 1000) -> MetricValue:
    """Area under ROC curve with bootstrap CI."""
    return bootstrap_ci(
        lambda yt, ys: float(roc_auc_score(yt, ys)),
        y_true,
        y_score,
        n_bootstrap=n_bootstrap,
    )


def auprc(y_true: np.ndarray, y_score: np.ndarray, n_bootstrap: int = 1000) -> MetricValue:
    """Area under precision-recall curve with bootstrap CI."""
    return bootstrap_ci(
        lambda yt, ys: float(average_precision_score(yt, ys)),
        y_true,
        y_score,
        n_bootstrap=n_bootstrap,
    )


def balanced_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, n_bootstrap: int = 1000
) -> MetricValue:
    """Balanced accuracy (sensitivity + specificity) / 2."""
    return bootstrap_ci(
        lambda yt, yp: float(balanced_accuracy_score(yt, yp)),
        y_true,
        y_pred,
        n_bootstrap=n_bootstrap,
    )


def ppv_at_sensitivity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_sensitivity: float = 0.90,
) -> MetricValue:
    """Positive predictive value (precision) at a fixed sensitivity (recall) threshold."""

    def _ppv_at_sens(yt: np.ndarray, ys: np.ndarray) -> float:
        precision, recall, _ = precision_recall_curve(yt, ys)
        # Find threshold where recall >= target
        valid = recall >= target_sensitivity
        if not valid.any():
            return 0.0
        return float(precision[valid][-1])

    return bootstrap_ci(_ppv_at_sens, y_true, y_score)


def npv_at_specificity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_specificity: float = 0.99,
) -> MetricValue:
    """Negative predictive value at a fixed specificity threshold."""

    def _npv_at_spec(yt: np.ndarray, ys: np.ndarray) -> float:
        # Sort by score ascending
        order = np.argsort(ys)
        yt_sorted = yt[order]
        ys_sorted = ys[order]

        n_neg = (yt == 0).sum()
        if n_neg == 0:
            return 0.0

        # Find threshold where specificity >= target
        for i in range(len(ys_sorted)):
            threshold = ys_sorted[i]
            predicted_neg = ys <= threshold
            tn = ((yt == 0) & predicted_neg).sum()
            spec = tn / n_neg
            if spec >= target_specificity:
                fn = ((yt == 1) & predicted_neg).sum()
                neg_total = predicted_neg.sum()
                if neg_total == 0:
                    return 0.0
                return float(tn / neg_total)
        return 0.0

    return bootstrap_ci(_npv_at_spec, y_true, y_score)


def calibration_slope_intercept(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Calibration slope and intercept via logistic regression of outcome on predicted probability."""
    from sklearn.linear_model import LogisticRegression

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob).reshape(-1, 1)

    # Logit of predicted probability
    eps = 1e-10
    logit_prob = np.log(np.clip(y_prob, eps, 1 - eps) / (1 - np.clip(y_prob, eps, 1 - eps)))

    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(logit_prob, y_true)

    return {
        "slope": float(lr.coef_[0, 0]),
        "intercept": float(lr.intercept_[0]),
    }


def nri(
    y_true: np.ndarray,
    old_risk: np.ndarray,
    new_risk: np.ndarray,
    threshold: float = 0.5,
) -> MetricValue:
    """Net Reclassification Improvement.

    Measures improvement in reclassification when moving from old_risk to new_risk.
    """

    def _nri(yt: np.ndarray, risks: np.ndarray) -> float:
        # risks is [old_risk, new_risk] stacked
        n = len(yt)
        old = risks[:n]
        new = risks[n:]

        old_class = (old >= threshold).astype(int)
        new_class = (new >= threshold).astype(int)

        cases = yt == 1
        controls = yt == 0

        # Event NRI: proportion of cases correctly reclassified up
        if cases.sum() > 0:
            up_cases = ((new_class == 1) & (old_class == 0) & cases).sum()
            down_cases = ((new_class == 0) & (old_class == 1) & cases).sum()
            event_nri = (up_cases - down_cases) / cases.sum()
        else:
            event_nri = 0.0

        # Non-event NRI: proportion of controls correctly reclassified down
        if controls.sum() > 0:
            down_controls = ((new_class == 0) & (old_class == 1) & controls).sum()
            up_controls = ((new_class == 1) & (old_class == 0) & controls).sum()
            nonevent_nri = (down_controls - up_controls) / controls.sum()
        else:
            nonevent_nri = 0.0

        return float(event_nri + nonevent_nri)

    # Stack old and new risks for bootstrap
    combined = np.concatenate([old_risk, new_risk])
    return bootstrap_ci(_nri, y_true, combined)
