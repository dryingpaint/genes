"""CLI entry point for running benchmarks.

Usage:
    python -m genes.benchmarks.runner --tiers 0,1,3 --output-dir ./results
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from genes.benchmarks.types import BenchmarkResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run genome-to-phenotype benchmarks")
    parser.add_argument(
        "--tiers",
        type=str,
        default="0",
        help="Comma-separated tier numbers to run (e.g., '0', '0,1', '0,1,2,3')",
    )
    parser.add_argument(
        "--subtasks",
        type=str,
        default="all",
        help="Comma-separated subtask IDs (e.g., '0a,0b') or 'all'",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Directory to write result reports",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json", "both"],
        default="both",
        help="Output format",
    )
    return parser.parse_args()


# Registry of benchmark functions per tier/subtask
BENCHMARK_REGISTRY: dict[str, dict[str, callable]] = {
    "0": {
        "0a": None,  # tier0.mendelian_pathogenicity — to be implemented
        "0b": None,  # tier0.pgx_diplotyping
        "0c": None,  # tier0.hla_typing
        "0d": None,  # tier0.variant_calling
    },
    "1": {
        "1a": None,  # tier1.eqtl
        "1b": None,  # tier1.caqtl
        "1c": None,  # tier1.sqtl
        "1d": None,  # tier1.dms
        "1e": None,  # tier1.brca1_sge
    },
    "2": {
        "2a": None,  # tier2.quantitative_traits
        "2b": None,  # tier2.disease_endpoints
        "2c": None,  # tier2.externally_visible
        "2d": None,  # tier2.portability
    },
    "3": {
        "3a": None,  # tier3.dgrp
        "3b": None,  # tier3.arabidopsis
    },
}


def run_benchmarks(tiers: list[str], subtasks: list[str], output_dir: Path) -> list[BenchmarkResult]:
    """Run requested benchmarks and return results."""
    results: list[BenchmarkResult] = []

    for tier in tiers:
        if tier not in BENCHMARK_REGISTRY:
            print(f"Warning: unknown tier '{tier}', skipping")
            continue

        tier_tasks = BENCHMARK_REGISTRY[tier]
        tasks_to_run = tier_tasks if subtasks == ["all"] else {
            k: v for k, v in tier_tasks.items() if k in subtasks
        }

        for task_id, bench_fn in tasks_to_run.items():
            if bench_fn is None:
                print(f"  [{task_id}] Not yet implemented, skipping")
                continue
            print(f"  [{task_id}] Running...")
            result = bench_fn()
            results.append(result)
            print(f"  [{task_id}] Done — gate: {result.gate}")

    return results


def write_results(results: list[BenchmarkResult], output_dir: Path, fmt: str) -> None:
    """Write results to disk in requested format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if fmt in ("json", "both"):
        json_path = output_dir / f"results_{timestamp}.json"
        data = [r.model_dump(mode="json") for r in results]
        json_path.write_text(json.dumps(data, indent=2, default=str))
        print(f"JSON results written to {json_path}")

    if fmt in ("markdown", "both"):
        from genes.benchmarks.reporting import result_to_markdown

        md_path = output_dir / f"results_{timestamp}.md"
        sections = [result_to_markdown(r) for r in results]
        md_path.write_text(
            f"# Benchmark Results — {timestamp}\n\n" + "\n---\n\n".join(sections)
        )
        print(f"Markdown report written to {md_path}")


def main() -> None:
    args = parse_args()
    tiers = [t.strip() for t in args.tiers.split(",")]
    subtasks = [s.strip() for s in args.subtasks.split(",")] if args.subtasks != "all" else ["all"]
    output_dir = Path(args.output_dir)

    print(f"Running benchmarks: tiers={tiers}, subtasks={subtasks}")
    results = run_benchmarks(tiers, subtasks, output_dir)

    if results:
        write_results(results, output_dir, args.format)
        passed = sum(1 for r in results if r.gate and r.gate == "pass")
        print(f"\n{passed}/{len(results)} benchmarks passed")
    else:
        print("No benchmarks were run (all requested tasks are unimplemented)")


if __name__ == "__main__":
    main()
