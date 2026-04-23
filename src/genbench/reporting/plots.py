"""Visualization for eval inputs and outputs.

Generates matplotlib plots for classification evals (ROC, PR, score distributions),
regression evals (predicted vs actual, per-assay distributions), and variant calling
evals (precision/recall summaries).

All functions return a matplotlib Figure and optionally save to disk.

Usage:
    from genbench.reporting.plots import plot_classification, plot_regression

    fig = plot_classification(y_true, y_pred, title="ClinVar AlphaMissense")
    fig.savefig("clinvar_roc.png", dpi=150, bbox_inches="tight")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Lazy import matplotlib to avoid import errors when not installed
_MPL_AVAILABLE = None


def _check_matplotlib():
    global _MPL_AVAILABLE
    if _MPL_AVAILABLE is None:
        try:
            import matplotlib
            matplotlib.use("Agg")  # non-interactive backend
            _MPL_AVAILABLE = True
        except ImportError:
            _MPL_AVAILABLE = False
    if not _MPL_AVAILABLE:
        raise ImportError(
            "matplotlib required for plots. Install: pip install matplotlib seaborn"
        )


# ---------------------------------------------------------------------------
# Classification eval plots (ClinVar, BRCA1 SGE)
# ---------------------------------------------------------------------------


def plot_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Classification Eval",
    save_path: str | None = None,
) -> Any:
    """Generate a 2x2 panel for classification evals.

    Panels:
      1. ROC curve with AUC
      2. Precision-Recall curve with AUPRC
      3. Score distribution by true class
      4. Calibration (predicted probability vs observed frequency)
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    # Filter NaN predictions
    valid = ~np.isnan(y_pred)
    yt, yp = y_true[valid], y_pred[valid]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # --- Panel 1: ROC curve ---
    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(yt, yp)
    auc_val = roc_auc_score(yt, yp)
    ax.plot(fpr, tpr, color="#2563eb", linewidth=2, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # --- Panel 2: Precision-Recall curve ---
    ax = axes[0, 1]
    precision, recall, _ = precision_recall_curve(yt, yp)
    ap = average_precision_score(yt, yp)
    ax.plot(recall, precision, color="#dc2626", linewidth=2, label=f"AP = {ap:.4f}")
    prevalence = yt.mean()
    ax.axhline(y=prevalence, linestyle="--", color="#94a3b8", linewidth=1,
               label=f"Baseline = {prevalence:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # --- Panel 3: Score distribution ---
    ax = axes[1, 0]
    pos_scores = yp[yt == 1]
    neg_scores = yp[yt == 0]
    bins = np.linspace(0, 1, 50)
    ax.hist(neg_scores, bins=bins, alpha=0.6, color="#22c55e", label=f"Benign (n={len(neg_scores)})",
            density=True, edgecolor="white", linewidth=0.5)
    ax.hist(pos_scores, bins=bins, alpha=0.6, color="#ef4444", label=f"Pathogenic (n={len(pos_scores)})",
            density=True, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Predicted Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution by Class")
    ax.legend()

    # --- Panel 4: Calibration ---
    ax = axes[1, 1]
    try:
        prob_true, prob_pred = calibration_curve(yt, yp, n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, "o-", color="#8b5cf6", linewidth=2, markersize=6,
                label="Model")
        ax.plot([0, 1], [0, 1], "--", color="#94a3b8", linewidth=1, label="Perfect")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Frequency")
        ax.set_title("Calibration")
        ax.legend(loc="lower right")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    except Exception:
        ax.text(0.5, 0.5, "Insufficient data\nfor calibration",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("Calibration")

    # --- Summary stats annotation ---
    n_total = len(y_pred)
    n_scored = int(valid.sum())
    coverage = n_scored / n_total if n_total > 0 else 0
    fig.text(0.5, 0.01,
             f"N={n_total}  Scored={n_scored} ({coverage:.1%})  "
             f"Pos={int(yt.sum())}  Neg={int((1-yt).sum())}  "
             f"AUROC={auc_val:.4f}  AUPRC={ap:.4f}",
             ha="center", fontsize=10, color="#64748b")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Regression eval plots (DMS / ProteinGym)
# ---------------------------------------------------------------------------


def plot_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Regression Eval",
    save_path: str | None = None,
) -> Any:
    """Generate a 1x2 panel for regression evals.

    Panels:
      1. Predicted vs actual scatter
      2. Residual distribution
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    valid = ~np.isnan(y_pred) & ~np.isnan(y_true)
    yt, yp = y_true[valid], y_pred[valid]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # --- Panel 1: Scatter ---
    ax = axes[0]
    rho, _ = spearmanr(yt, yp)

    # Subsample for plotting if too many points
    if len(yt) > 5000:
        idx = np.random.default_rng(42).choice(len(yt), 5000, replace=False)
        plot_yt, plot_yp = yt[idx], yp[idx]
    else:
        plot_yt, plot_yp = yt, yp

    ax.scatter(plot_yt, plot_yp, alpha=0.3, s=8, color="#2563eb", edgecolors="none")
    ax.set_xlabel("True Score")
    ax.set_ylabel("Predicted Score")
    ax.set_title(f"Predicted vs Actual (Spearman rho = {rho:.4f})")

    # Fit line
    z = np.polyfit(plot_yt, plot_yp, 1)
    x_line = np.linspace(plot_yt.min(), plot_yt.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "--", color="#ef4444", linewidth=1.5)

    # --- Panel 2: Residual distribution ---
    ax = axes[1]
    residuals = yp - yt
    ax.hist(residuals, bins=50, color="#8b5cf6", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#ef4444", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Residual (Predicted - True)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Distribution (median={np.median(residuals):.3f})")

    fig.text(0.5, 0.01,
             f"N={len(yt)}  Spearman rho={rho:.4f}  "
             f"MAE={np.mean(np.abs(residuals)):.4f}",
             ha="center", fontsize=10, color="#64748b")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_per_assay_rho(
    rhos: list[float],
    assay_names: list[str] | None = None,
    title: str = "Per-Assay Spearman Rho",
    save_path: str | None = None,
) -> Any:
    """Distribution of per-assay Spearman correlations (DMS eval)."""
    _check_matplotlib()
    import matplotlib.pyplot as plt

    rhos_arr = np.array(rhos)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # --- Panel 1: Histogram ---
    ax = axes[0]
    ax.hist(rhos_arr, bins=30, color="#2563eb", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axvline(np.mean(rhos_arr), color="#ef4444", linestyle="--", linewidth=2,
               label=f"Mean = {np.mean(rhos_arr):.4f}")
    ax.axvline(np.median(rhos_arr), color="#22c55e", linestyle="--", linewidth=2,
               label=f"Median = {np.median(rhos_arr):.4f}")
    ax.set_xlabel("Spearman Rho")
    ax.set_ylabel("Number of Assays")
    ax.set_title("Distribution")
    ax.legend()

    # --- Panel 2: Sorted bar chart (top/bottom 20) ---
    ax = axes[1]
    sorted_idx = np.argsort(rhos_arr)
    n_show = min(20, len(rhos_arr))

    # Bottom 10 + top 10
    show_idx = np.concatenate([sorted_idx[:n_show // 2], sorted_idx[-(n_show // 2):]])
    show_rhos = rhos_arr[show_idx]
    if assay_names:
        show_names = [assay_names[i][:25] for i in show_idx]
    else:
        show_names = [f"Assay {i}" for i in show_idx]

    colors = ["#ef4444" if r < 0.3 else "#f59e0b" if r < 0.5 else "#22c55e" for r in show_rhos]
    ax.barh(range(len(show_rhos)), show_rhos, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(show_rhos)))
    ax.set_yticklabels(show_names, fontsize=7)
    ax.set_xlabel("Spearman Rho")
    ax.set_title(f"Best/Worst Assays (of {len(rhos_arr)})")

    fig.text(0.5, 0.01,
             f"N={len(rhos_arr)} assays  Mean={np.mean(rhos_arr):.4f}  "
             f"Median={np.median(rhos_arr):.4f}  "
             f"Min={np.min(rhos_arr):.4f}  Max={np.max(rhos_arr):.4f}",
             ha="center", fontsize=10, color="#64748b")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Variant calling plots (GIAB)
# ---------------------------------------------------------------------------


def plot_variant_calling(
    metrics: dict[str, dict[str, float]],
    title: str = "Variant Calling Accuracy",
    save_path: str | None = None,
) -> Any:
    """Bar chart of precision/recall/F1 for SNVs and indels.

    Args:
        metrics: {"snv": {"precision": 0.99, "recall": 0.99, "f1": 0.99},
                  "indel": {"precision": 0.98, "recall": 0.97, "f1": 0.97}}
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    variant_types = list(metrics.keys())
    metric_names = ["precision", "recall", "f1"]
    x = np.arange(len(variant_types))
    width = 0.25
    colors = ["#2563eb", "#22c55e", "#ef4444"]

    for i, metric in enumerate(metric_names):
        vals = [metrics[vt].get(metric, 0) for vt in variant_types]
        bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(), color=colors[i],
                      edgecolor="white", linewidth=0.5)
        # Add value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Variant Type")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1")
    ax.set_xticks(x + width)
    ax.set_xticklabels([vt.upper() for vt in variant_types])
    ax.legend()
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Input data summary plots
# ---------------------------------------------------------------------------


def plot_input_summary(
    labels: np.ndarray,
    metadata: dict[str, Any] | None = None,
    title: str = "Input Data Summary",
    save_path: str | None = None,
) -> Any:
    """Visualize the input data: class balance, sample counts.

    Args:
        labels: Ground truth labels (0/1 for classification, continuous for regression).
        metadata: Optional dict with extra info (e.g., gene counts, chrom distribution).
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    is_binary = set(np.unique(labels[~np.isnan(labels)])).issubset({0, 1, 0.0, 1.0})
    metadata = metadata or {}

    if is_binary:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    fig.suptitle(title, fontsize=14, fontweight="bold")

    if is_binary:
        # --- Panel 1: Class balance pie ---
        ax = axes[0]
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos
        ax.pie([n_neg, n_pos], labels=[f"Negative\n(n={n_neg})", f"Positive\n(n={n_pos})"],
               colors=["#22c55e", "#ef4444"], autopct="%1.1f%%",
               startangle=90, textprops={"fontsize": 10})
        ax.set_title("Class Balance")

        # --- Panel 2: Label distribution bar ---
        ax = axes[1]
        ax.bar(["Negative (0)", "Positive (1)"], [n_neg, n_pos],
               color=["#22c55e", "#ef4444"], edgecolor="white")
        ax.set_ylabel("Count")
        ax.set_title(f"N = {len(labels)}")
        for i, v in enumerate([n_neg, n_pos]):
            ax.text(i, v + max(n_neg, n_pos) * 0.02, str(v), ha="center", fontsize=11)
    else:
        # --- Continuous labels ---
        ax = axes[0]
        ax.hist(labels[~np.isnan(labels)], bins=50, color="#2563eb", alpha=0.7,
                edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Label Value")
        ax.set_ylabel("Count")
        ax.set_title("Label Distribution")

        ax = axes[1]
        ax.boxplot(labels[~np.isnan(labels)], orientation="vertical")
        ax.set_ylabel("Label Value")
        ax.set_title(f"N = {len(labels)}  "
                     f"Mean = {np.nanmean(labels):.3f}  "
                     f"Std = {np.nanstd(labels):.3f}")

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Model comparison plots
# ---------------------------------------------------------------------------


def plot_model_comparison(
    results: list[dict[str, Any]],
    metric: str = "auroc",
    title: str = "Model Comparison",
    save_path: str | None = None,
) -> Any:
    """Bar chart comparing multiple models on the same eval.

    Args:
        results: List of dicts with "model_name" and "metrics" keys.
            metrics should be {metric_name: float_value}.
        metric: Which metric to compare.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    models = [r["model_name"] for r in results]
    values = [r["metrics"].get(metric, 0) for r in results]

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.5), 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    colors = ["#2563eb", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6", "#06b6d4", "#ec4899"]
    bar_colors = [colors[i % len(colors)] for i in range(len(models))]

    bars = ax.bar(models, values, color=bar_colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} by Model")

    # Set reasonable y-axis
    if values:
        min_val = min(v for v in values if v > 0) if any(v > 0 for v in values) else 0
        ax.set_ylim(max(0, min_val - 0.1), min(1.05, max(values) + 0.1))

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# History / trend plot
# ---------------------------------------------------------------------------


def plot_metric_history(
    history: list[tuple[str, float]],
    metric: str = "auroc",
    title: str = "Performance Over Time",
    save_path: str | None = None,
) -> Any:
    """Line chart of a metric over time.

    Args:
        history: List of (timestamp_str, value) tuples.
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    from datetime import datetime

    if not history:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No history data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        return fig

    timestamps = [h[0][:19] for h in history]
    values = [h[1] for h in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax.plot(range(len(values)), values, "o-", color="#2563eb", linewidth=2, markersize=6)
    ax.set_xticks(range(len(timestamps)))
    ax.set_xticklabels(timestamps, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric} Over Time ({len(values)} runs)")

    if len(values) > 1:
        ax.axhline(y=np.mean(values), linestyle="--", color="#94a3b8",
                   label=f"Mean = {np.mean(values):.4f}")
        ax.legend()

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
