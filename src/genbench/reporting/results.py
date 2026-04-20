"""Result storage: JSON-lines on the results volume."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from genbench.config import RESULTS_PATH
from genbench.types import BenchmarkResult


def save_results(results: list[BenchmarkResult]) -> str:
    """Append results as JSON-lines to the results volume.

    File structure: {RESULTS_PATH}/{run_timestamp}.jsonl
    Each line is a serialized BenchmarkResult.

    Returns:
        Path to the written file.
    """
    base = Path(RESULTS_PATH)
    base.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    filepath = base / f"run_{timestamp}.jsonl"

    with open(filepath, "a") as f:
        for result in results:
            f.write(result.model_dump_json() + "\n")

    return str(filepath)


def load_results(
    run_file: str | None = None,
    task_id_prefix: str | None = None,
    model_name: str | None = None,
) -> list[BenchmarkResult]:
    """Load results from JSON-lines files.

    Args:
        run_file: Specific run file to load. If None, loads latest.
        task_id_prefix: Filter by task ID prefix (e.g., "tier0").
        model_name: Filter by model name.

    Returns:
        List of BenchmarkResult objects.
    """
    base = Path(RESULTS_PATH)

    if run_file:
        files = [Path(run_file)]
    else:
        files = sorted(base.glob("run_*.jsonl"), reverse=True)
        if not files:
            return []
        files = [files[0]]  # latest

    results = []
    for filepath in files:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                result = BenchmarkResult.model_validate_json(line)

                if task_id_prefix and not result.task_id.startswith(task_id_prefix):
                    continue
                if model_name and result.model_name != model_name:
                    continue

                results.append(result)

    return results
