"""Local result store — persistent eval tracking across runs.

Stores results as JSONL locally (not on Modal volumes) so you can query
history, compare models, and track performance over time.

Usage:
    from genbench.reporting.store import ResultStore

    store = ResultStore()                         # default: results/eval_history.jsonl
    store.record(benchmark_result)                # append a result
    store.list_runs()                             # all runs
    store.list_runs(eval_name="clinvar")          # filter by eval
    store.latest("clinvar", "alphamissense")      # most recent result
    store.compare("clinvar", config="all_snv")    # all models on one config
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from genbench.types import BenchmarkResult


class EvalRun:
    """A lightweight view of a stored result for display/comparison."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    @property
    def task_id(self) -> str:
        return self._data["task_id"]

    @property
    def model_name(self) -> str:
        return self._data["model_name"]

    @property
    def timestamp(self) -> str:
        return self._data.get("timestamp", "")

    @property
    def config(self) -> str:
        return self._data.get("metadata", {}).get("config", "default")

    @property
    def metrics(self) -> dict[str, float]:
        """Flat dict of metric_name → estimate value."""
        result = {}
        for name, metric_data in self._data.get("metrics", {}).items():
            agg = metric_data.get("aggregate")
            if agg and isinstance(agg, dict):
                est = agg.get("estimate")
                if est is not None:
                    result[name] = est
        return result

    @property
    def n_samples(self) -> dict[str, int]:
        return self._data.get("n_samples", {})

    @property
    def metadata(self) -> dict[str, Any]:
        return self._data.get("metadata", {})

    @property
    def predictions_path(self) -> str | None:
        return self._data.get("metadata", {}).get("predictions_path")

    def summary_line(self) -> str:
        """One-line summary for display."""
        metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in self.metrics.items()
                                if isinstance(v, float) and v == v)
        ts = self.timestamp[:19] if self.timestamp else "?"
        return f"{ts}  {self.task_id:30s}  {self.model_name:20s}  {metrics_str}"

    def to_dict(self) -> dict[str, Any]:
        return self._data


class ResultStore:
    """Append-only JSONL store for benchmark results."""

    def __init__(self, path: str | Path | None = None):
        if path is None:
            path = Path("results") / "eval_history.jsonl"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, result: BenchmarkResult, predictions_path: str | None = None) -> None:
        """Append a BenchmarkResult to the store.

        Args:
            result: The result to store.
            predictions_path: Optional path to saved predictions (npz file).
        """
        data = json.loads(result.model_dump_json())
        if predictions_path:
            data.setdefault("metadata", {})["predictions_path"] = predictions_path
        with open(self.path, "a") as f:
            f.write(json.dumps(data) + "\n")

    def _load_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        results = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results

    def list_runs(
        self,
        eval_name: str | None = None,
        model_name: str | None = None,
        config: str | None = None,
        limit: int = 50,
    ) -> list[EvalRun]:
        """Query stored results with optional filters."""
        runs = []
        for data in self._load_all():
            if eval_name:
                task = data.get("task_id", "")
                if not task.startswith(eval_name) and eval_name not in task:
                    continue
            if model_name and data.get("model_name") != model_name:
                continue
            if config:
                run_config = data.get("metadata", {}).get("config", "default")
                if run_config != config:
                    continue
            runs.append(EvalRun(data))

        # Sort by timestamp descending (most recent first)
        runs.sort(key=lambda r: r.timestamp, reverse=True)
        return runs[:limit]

    def latest(self, eval_name: str, model_name: str, config: str | None = None) -> EvalRun | None:
        """Get the most recent result for a specific eval/model combo."""
        runs = self.list_runs(eval_name=eval_name, model_name=model_name, config=config, limit=1)
        return runs[0] if runs else None

    def compare(
        self,
        eval_name: str,
        config: str | None = None,
        latest_only: bool = True,
    ) -> list[EvalRun]:
        """Get results for all models on a given eval/config for comparison.

        If latest_only=True, returns only the most recent run per model.
        """
        runs = self.list_runs(eval_name=eval_name, config=config, limit=1000)

        if latest_only:
            seen: dict[str, EvalRun] = {}
            for run in runs:
                key = run.model_name
                if key not in seen:
                    seen[key] = run
            return list(seen.values())

        return runs

    def history(
        self,
        eval_name: str,
        model_name: str,
        metric: str,
        config: str | None = None,
    ) -> list[tuple[str, float]]:
        """Get the time series of a metric for a specific eval/model.

        Returns list of (timestamp, value) tuples, chronologically ordered.
        """
        runs = self.list_runs(eval_name=eval_name, model_name=model_name, config=config, limit=1000)
        points = []
        for run in reversed(runs):  # reverse to get chronological order
            val = run.metrics.get(metric)
            if val is not None and val == val:  # not NaN
                points.append((run.timestamp, val))
        return points

    def summary_table(self, eval_name: str | None = None) -> str:
        """Render a text summary table of stored results."""
        runs = self.list_runs(eval_name=eval_name)
        if not runs:
            return "No results stored."

        lines = [
            f"{'Timestamp':19s}  {'Task':30s}  {'Model':20s}  Metrics",
            "-" * 100,
        ]
        for run in runs:
            lines.append(run.summary_line())
        return "\n".join(lines)
