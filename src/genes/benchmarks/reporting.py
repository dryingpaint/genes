"""Report generation for benchmark results.

Produces markdown reports following the template in docs/phenotype-suite/BENCHMARKS.md.
"""

from __future__ import annotations

from genes.benchmarks.types import (
    AncestryStratifiedMetric,
    BaselineComparison,
    BenchmarkResult,
    CeilingInfo,
)


def _format_metric(m) -> str:
    """Format a MetricValue as 'estimate (CI lower–upper)'."""
    return f"{m.estimate:.4f} ({m.ci_lower:.4f}–{m.ci_upper:.4f})"


def result_to_markdown(result: BenchmarkResult) -> str:
    """Render a single BenchmarkResult as markdown."""
    lines = [
        f"## {result.task_id}",
        "",
        f"**Model:** {result.model_name}  ",
        f"**Split:** {result.split_type}  ",
        f"**Gate:** {result.gate or 'N/A'}  ",
        f"**Timestamp:** {result.timestamp.isoformat()}  ",
        "",
    ]

    # Anti-leakage
    lines.append("### Anti-leakage checks")
    if result.leakage_report.checks:
        for name, check in result.leakage_report.checks.items():
            status = "PASS" if check.passed else "**FAIL**"
            lines.append(f"- {name}: {status} — {check.details}")
    else:
        lines.append("- No checks configured")
    lines.append("")

    # Metrics table
    if result.metrics:
        lines.append("### Results")
        lines.append("")

        # Collect all ancestry groups across all metrics
        ancestries = set()
        for metric in result.metrics.values():
            ancestries.update(metric.per_ancestry.keys())
        ancestries = sorted(ancestries)

        if ancestries:
            header = "| Metric | " + " | ".join(f"{a}" for a in ancestries)
            if any(m.aggregate for m in result.metrics.values()):
                header += " | Aggregate"
            header += " |"
            lines.append(header)

            sep = "|---|" + "|".join("---" for _ in ancestries)
            if any(m.aggregate for m in result.metrics.values()):
                sep += "|---"
            sep += "|"
            lines.append(sep)

            for name, metric in result.metrics.items():
                row = f"| {name} | "
                for a in ancestries:
                    if a in metric.per_ancestry:
                        row += f"{_format_metric(metric.per_ancestry[a])} | "
                    else:
                        row += "— | "
                if metric.aggregate:
                    row += f"{_format_metric(metric.aggregate)} |"
                elif any(m.aggregate for m in result.metrics.values()):
                    row += "— |"
                lines.append(row)
        else:
            # No per-ancestry, just aggregate
            lines.append("| Metric | Value |")
            lines.append("|---|---|")
            for name, metric in result.metrics.items():
                if metric.aggregate:
                    lines.append(f"| {name} | {_format_metric(metric.aggregate)} |")
        lines.append("")

    # Ceiling analysis
    if result.ceiling:
        lines.append("### Ceiling analysis")
        lines.append(f"- Ceiling: {result.ceiling.ceiling_value:.4f} ({result.ceiling.ceiling_source})")
        lines.append(f"- Normalized performance: {result.ceiling.normalized_performance:.4f}")
        lines.append(f"- Gap to ceiling: {result.ceiling.ceiling_value - result.ceiling.normalized_performance * result.ceiling.ceiling_value:.4f}")
        lines.append("")

    # Sample sizes
    if result.n_samples:
        lines.append("### Sample sizes")
        for ancestry, n in sorted(result.n_samples.items()):
            lines.append(f"- {ancestry}: {n:,}")
        lines.append("")

    return "\n".join(lines)


def comparison_to_markdown(comparison: BaselineComparison) -> str:
    """Render a BaselineComparison as markdown."""
    lines = [
        f"## {comparison.task_id} — Model vs Baselines",
        "",
    ]

    # Collect all metrics and models
    all_results = {"**Model**": comparison.model_result}
    for name, br in comparison.baseline_results.items():
        label = name
        if name in comparison.circularity_flags:
            label += f" (circularity: {comparison.circularity_flags[name]})"
        all_results[label] = br

    # Find common metrics
    all_metric_names = set()
    for br in all_results.values():
        all_metric_names.update(br.metrics.keys())

    if all_metric_names:
        lines.append("| Model | " + " | ".join(sorted(all_metric_names)) + " |")
        lines.append("|---|" + "|".join("---" for _ in all_metric_names) + "|")
        for label, br in all_results.items():
            row = f"| {label} | "
            for mn in sorted(all_metric_names):
                if mn in br.metrics and br.metrics[mn].aggregate:
                    row += f"{_format_metric(br.metrics[mn].aggregate)} | "
                else:
                    row += "— | "
            lines.append(row)
    lines.append("")

    return "\n".join(lines)
