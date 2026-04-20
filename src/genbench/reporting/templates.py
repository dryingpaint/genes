"""Markdown report renderer.

Produces the reporting template specified in BENCHMARKS.md.
"""

from __future__ import annotations

from datetime import UTC, datetime

from genbench.types import BenchmarkResult, GateResult


def render_report(results: list[BenchmarkResult]) -> str:
    """Render a full benchmark report from results.

    Groups results by tier and produces the spec-mandated format:
    setup, results table, ceiling analysis, failure modes.
    """
    sections = []
    sections.append(f"# Genome-to-Phenotype Benchmark Report")
    sections.append(f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}\n")

    # Group by tier
    tier_groups: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        tier = r.task_id.split(".")[0]  # e.g., "tier0"
        tier_groups.setdefault(tier, []).append(r)

    # Tier 0
    if "tier0" in tier_groups:
        sections.append(_render_tier0(tier_groups["tier0"]))

    # Tier 1
    if "tier1" in tier_groups:
        sections.append(_render_tier1(tier_groups["tier1"]))

    # Tier 3
    if "tier3" in tier_groups:
        sections.append(_render_tier3(tier_groups["tier3"]))

    return "\n\n".join(sections)


def _render_tier0(results: list[BenchmarkResult]) -> str:
    lines = ["## Tier 0: Sanity Checks\n"]

    for r in results:
        gate_str = ""
        if r.gate == GateResult.PASS:
            gate_str = " [PASS]"
        elif r.gate == GateResult.FAIL:
            gate_str = f" [FAIL: {r.gate_reason}]"

        lines.append(f"### {r.task_id}{gate_str}\n")
        lines.append(f"- Model: {r.model_name}")
        lines.append(f"- Split: {r.split_type.value}")
        lines.append(f"- N: {r.n_samples}")
        lines.append(f"- Anti-leakage: {'all passed' if r.leakage_report.all_passed else 'FAILED'}")

        if r.metrics:
            lines.append("\n| Metric | Value (95% CI) |")
            lines.append("|---|---|")
            for metric_name, metric_val in r.metrics.items():
                if metric_val.aggregate:
                    mv = metric_val.aggregate
                    lines.append(
                        f"| {metric_name} | {mv.estimate:.4f} "
                        f"({mv.ci_lower:.4f}–{mv.ci_upper:.4f}), N={mv.n} |"
                    )
        lines.append("")

    return "\n".join(lines)


def _render_tier1(results: list[BenchmarkResult]) -> str:
    lines = ["## Tier 1: Molecular Phenotypes\n"]

    # Separate aggregate from per-assay
    aggregates = [r for r in results if "aggregate" in r.task_id]
    per_task = [r for r in results if "aggregate" not in r.task_id]

    for r in aggregates:
        lines.append(f"### {r.task_id}\n")
        lines.append(f"- Model: {r.model_name}")
        lines.append(f"- N samples: {r.n_samples}")

        if r.ceiling:
            lines.append(f"- Ceiling: {r.ceiling.ceiling_value:.3f} ({r.ceiling.ceiling_source})")
            lines.append(f"- Ceiling-normalized: {r.ceiling.normalized_performance:.3f}")

        if r.metrics:
            lines.append("\n| Metric | Value |")
            lines.append("|---|---|")
            for name, val in r.metrics.items():
                if val.aggregate:
                    lines.append(f"| {name} | {val.aggregate.estimate:.4f} |")

        lines.append(f"\n*{len(per_task)} individual assays/tasks evaluated*\n")

    return "\n".join(lines)


def _render_tier3(results: list[BenchmarkResult]) -> str:
    lines = ["## Tier 3: Model Organisms\n"]

    # Group by organism
    organisms: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        parts = r.task_id.split(".")
        org = parts[1] if len(parts) > 1 else "unknown"
        organisms.setdefault(org, []).append(r)

    for org, org_results in organisms.items():
        lines.append(f"### {org.upper()}\n")

        # Summary table
        valid = [r for r in org_results if r.metrics.get("cv_r2")]
        if valid:
            lines.append("| Trait | CV R² (mean ± SD) | N | Ceiling |")
            lines.append("|---|---|---|---|")
            for r in sorted(valid, key=lambda x: x.task_id):
                trait = r.metadata.get("trait", r.task_id)
                mv = r.metrics["cv_r2"].aggregate
                ceil_str = ""
                if r.ceiling:
                    ceil_str = f"{r.ceiling.ceiling_value:.2f} ({r.ceiling.normalized_performance:.1%})"
                lines.append(
                    f"| {trait} | {mv.estimate:.3f} ± {(mv.ci_upper - mv.estimate) / 1.96:.3f} "
                    f"| {mv.n} | {ceil_str} |"
                )

        skipped = [r for r in org_results if r.metadata.get("skipped")]
        if skipped:
            lines.append(f"\n*{len(skipped)} traits skipped (insufficient samples)*")

        lines.append("")

    return "\n".join(lines)
