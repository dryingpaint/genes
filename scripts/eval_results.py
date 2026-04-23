"""View, compare, and visualize eval results.

Usage:
    # List all stored results
    python scripts/eval_results.py --list

    # List results for a specific eval
    python scripts/eval_results.py --list --eval clinvar

    # Compare all models on an eval
    python scripts/eval_results.py --compare --eval clinvar --config all_snv

    # Show performance history for a model
    python scripts/eval_results.py --history --eval clinvar --model alphamissense --metric auroc

    # Regenerate plots from stored predictions
    python scripts/eval_results.py --plot results/clinvar/alphamissense/clinvar_all_snv_alphamissense_predictions.npz

    # Generate comparison chart
    python scripts/eval_results.py --compare-plot --eval clinvar --metric auroc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for local execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def cmd_list(args):
    from genbench.reporting.store import ResultStore
    store = ResultStore()
    print(store.summary_table(eval_name=args.eval or None))


def cmd_compare(args):
    from genbench.reporting.store import ResultStore
    store = ResultStore()

    if not args.eval:
        print("Error: --eval required for compare")
        sys.exit(1)

    runs = store.compare(args.eval, config=args.config or None)
    if not runs:
        print(f"No results found for eval '{args.eval}'")
        return

    # Print comparison table
    print(f"\nComparison: {args.eval}" + (f" [{args.config}]" if args.config else ""))
    print("=" * 80)

    # Collect all metric names
    all_metrics = set()
    for run in runs:
        all_metrics.update(run.metrics.keys())
    metric_names = sorted(all_metrics)

    # Header
    header = f"{'Model':25s}"
    for m in metric_names:
        header += f"  {m:>12s}"
    print(header)
    print("-" * len(header))

    # Rows
    for run in sorted(runs, key=lambda r: r.model_name):
        row = f"{run.model_name:25s}"
        for m in metric_names:
            val = run.metrics.get(m)
            if val is not None and val == val:
                row += f"  {val:12.4f}"
            else:
                row += f"  {'N/A':>12s}"
        print(row)

    print(f"\n{len(runs)} model(s)")


def cmd_history(args):
    from genbench.reporting.store import ResultStore
    store = ResultStore()

    if not args.eval or not args.model or not args.metric:
        print("Error: --eval, --model, and --metric required for history")
        sys.exit(1)

    history = store.history(args.eval, args.model, args.metric, config=args.config)
    if not history:
        print(f"No history for {args.eval}/{args.model}/{args.metric}")
        return

    print(f"\nHistory: {args.eval} / {args.model} / {args.metric}")
    print("=" * 60)
    for ts, val in history:
        print(f"  {ts[:19]}  {val:.4f}")

    if len(history) > 1:
        vals = [h[1] for h in history]
        print(f"\n  Mean: {sum(vals)/len(vals):.4f}  "
              f"Min: {min(vals):.4f}  Max: {max(vals):.4f}")

    # Generate plot if matplotlib available
    if args.save_plot:
        try:
            from genbench.reporting.plots import plot_metric_history
            plot_metric_history(
                history, metric=args.metric,
                title=f"{args.eval} / {args.model}: {args.metric} over time",
                save_path=args.save_plot,
            )
            print(f"\n  Plot saved: {args.save_plot}")
        except ImportError:
            print("  (matplotlib not installed, skipping plot)")


def cmd_plot(args):
    """Regenerate plots from a saved predictions .npz file."""
    npz_path = Path(args.plot)
    if not npz_path.exists():
        print(f"Error: file not found: {npz_path}")
        sys.exit(1)

    import numpy as np
    data = np.load(npz_path)
    y_true = data["y_true"]
    y_pred = data["y_pred"]

    output_dir = str(npz_path.parent)
    stem = npz_path.stem

    try:
        from genbench.reporting.plots import (
            plot_classification,
            plot_input_summary,
            plot_regression,
        )
    except ImportError:
        print("Error: matplotlib required. Install: pip install matplotlib")
        sys.exit(1)

    # Determine type
    unique_labels = set(np.unique(y_true[~np.isnan(y_true)]))
    is_binary = unique_labels.issubset({0, 1, 0.0, 1.0})

    # Input summary
    input_path = f"{output_dir}/{stem}_inputs.png"
    plot_input_summary(y_true, title=f"Input Data: {stem}", save_path=input_path)
    print(f"  Saved: {input_path}")

    if is_binary:
        plot_path = f"{output_dir}/{stem}_classification.png"
        plot_classification(y_true, y_pred, title=stem, save_path=plot_path)
        print(f"  Saved: {plot_path}")
    else:
        plot_path = f"{output_dir}/{stem}_regression.png"
        plot_regression(y_true, y_pred, title=stem, save_path=plot_path)
        print(f"  Saved: {plot_path}")

    import matplotlib.pyplot as plt
    plt.close("all")


def cmd_compare_plot(args):
    """Generate a bar chart comparing models on an eval."""
    from genbench.reporting.store import ResultStore
    store = ResultStore()

    if not args.eval or not args.metric:
        print("Error: --eval and --metric required for compare-plot")
        sys.exit(1)

    runs = store.compare(args.eval, config=args.config)
    if not runs:
        print(f"No results for {args.eval}")
        return

    try:
        from genbench.reporting.plots import plot_model_comparison
    except ImportError:
        print("Error: matplotlib required. Install: pip install matplotlib")
        sys.exit(1)

    results = [{"model_name": r.model_name, "metrics": r.metrics} for r in runs]
    save_path = args.save_plot or f"results/{args.eval}_comparison.png"

    plot_model_comparison(
        results, metric=args.metric,
        title=f"{args.eval}: {args.metric} by Model",
        save_path=save_path,
    )
    print(f"  Saved: {save_path}")

    import matplotlib.pyplot as plt
    plt.close("all")


def main():
    parser = argparse.ArgumentParser(description="View and compare eval results")
    parser.add_argument("--list", action="store_true", help="List stored results")
    parser.add_argument("--compare", action="store_true", help="Compare models on an eval")
    parser.add_argument("--history", action="store_true", help="Show metric history")
    parser.add_argument("--plot", type=str, help="Regenerate plots from .npz file")
    parser.add_argument("--compare-plot", action="store_true", help="Generate comparison chart")

    parser.add_argument("--eval", type=str, default="", help="Filter by eval name")
    parser.add_argument("--model", type=str, default="", help="Filter by model name")
    parser.add_argument("--config", type=str, default="", help="Filter by config name")
    parser.add_argument("--metric", type=str, default="auroc", help="Metric for history/comparison")
    parser.add_argument("--save-plot", type=str, default="", help="Path to save plot")

    args = parser.parse_args()

    if args.plot:
        cmd_plot(args)
    elif args.compare_plot:
        cmd_compare_plot(args)
    elif args.compare:
        cmd_compare(args)
    elif args.history:
        cmd_history(args)
    else:
        cmd_list(args)


if __name__ == "__main__":
    main()
