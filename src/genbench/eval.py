"""Eval base class — one subclass per benchmark.

Each eval can have multiple configs (EvalConfig) matching different published
benchmarks. The config parameterizes data filtering, split strategy, metrics,
and expected baseline scores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from genbench.eval_config import EvalConfig
from genbench.model import Model
from genbench.types import (
    AncestryStratifiedMetric,
    BenchmarkResult,
    LeakageReport,
    SplitType,
)


@dataclass
class EvalResult:
    """Extended result that carries predictions alongside the BenchmarkResult.

    This is returned by evaluate() when save_predictions=True, enabling
    downstream visualization and analysis of model outputs.
    """

    benchmark: BenchmarkResult
    y_true: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    input_metadata: dict[str, Any] = field(default_factory=dict)
    predictions_path: str | None = None
    plots_dir: str | None = None

    def save_predictions(self, output_dir: str) -> str:
        """Save y_true/y_pred to a .npz file for later visualization."""
        if self.y_true is None or self.y_pred is None:
            return ""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        task_safe = self.benchmark.task_id.replace("/", "_")
        model_safe = self.benchmark.model_name
        fname = f"{task_safe}_{model_safe}_predictions.npz"
        filepath = path / fname

        np.savez(
            filepath,
            y_true=self.y_true,
            y_pred=self.y_pred,
        )
        self.predictions_path = str(filepath)
        return str(filepath)

    def generate_plots(self, output_dir: str) -> list[str]:
        """Generate visualization plots and save to output_dir.

        Returns list of saved plot paths.
        """
        if self.y_true is None or self.y_pred is None:
            return []

        try:
            from genbench.reporting.plots import (
                plot_classification,
                plot_input_summary,
                plot_regression,
            )
        except ImportError:
            return []

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.plots_dir = str(path)

        task_safe = self.benchmark.task_id.replace("/", "_")
        model_safe = self.benchmark.model_name
        prefix = f"{task_safe}_{model_safe}"
        saved = []

        # Input summary
        try:
            input_path = str(path / f"{prefix}_inputs.png")
            plot_input_summary(
                self.y_true,
                metadata=self.input_metadata,
                title=f"Input Data: {self.benchmark.task_id}",
                save_path=input_path,
            )
            saved.append(input_path)
        except Exception:
            pass

        # Determine eval type from labels
        unique_labels = set(np.unique(self.y_true[~np.isnan(self.y_true)]))
        is_binary = unique_labels.issubset({0, 1, 0.0, 1.0})

        if is_binary:
            try:
                output_path = str(path / f"{prefix}_classification.png")
                plot_classification(
                    self.y_true, self.y_pred,
                    title=f"{self.benchmark.task_id} — {model_safe}",
                    save_path=output_path,
                )
                saved.append(output_path)
            except Exception:
                pass
        else:
            try:
                output_path = str(path / f"{prefix}_regression.png")
                plot_regression(
                    self.y_true, self.y_pred,
                    title=f"{self.benchmark.task_id} — {model_safe}",
                    save_path=output_path,
                )
                saved.append(output_path)
            except Exception:
                pass

        import matplotlib.pyplot as plt
        plt.close("all")

        return saved


class Eval(ABC):
    """Abstract base for all evaluations.

    Subclasses implement data loading, splitting, input preparation,
    and scoring. The evaluate() method orchestrates the full pipeline.
    """

    # --- Identity (override as class attributes) ---

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def tier(self) -> int: ...

    @property
    @abstractmethod
    def split_type(self) -> SplitType: ...

    @property
    @abstractmethod
    def default_baselines(self) -> list[str]: ...

    @property
    def expected_ceiling(self) -> float | dict[str, float] | None:
        return None

    # --- Configs ---

    configs: dict[str, EvalConfig] = {}
    default_config: str = "default"

    def get_config(self, config_name: str | None = None) -> EvalConfig:
        """Resolve a config by name. Falls back to default_config."""
        if config_name is None:
            config_name = self.default_config
        if config_name not in self.configs:
            available = ", ".join(sorted(self.configs.keys())) or "(none)"
            raise KeyError(
                f"Unknown config '{config_name}' for eval '{self.name}'. Available: {available}"
            )
        return self.configs[config_name]

    def list_configs(self) -> list[str]:
        return sorted(self.configs.keys())

    # --- Data & Splits ---

    @abstractmethod
    def load_data(self, config: EvalConfig) -> Any:
        """Load and filter dataset based on config."""
        ...

    @abstractmethod
    def get_splits(self, data: Any, config: EvalConfig) -> dict[str, Any]:
        """Return dict with at least 'test' key."""
        ...

    @abstractmethod
    def make_inputs(self, data: Any, split_data: Any) -> dict[str, Any]:
        """Prepare the inputs dict passed to model.predict()."""
        ...

    @abstractmethod
    def get_labels(self, data: Any, split_data: Any) -> np.ndarray:
        """Extract ground truth for a split."""
        ...

    @abstractmethod
    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, config: EvalConfig
    ) -> dict[str, AncestryStratifiedMetric]:
        """Compute metrics specified by config."""
        ...

    # --- Defaults ---

    def verify_leakage(self, splits: dict[str, Any], config: EvalConfig) -> LeakageReport:
        return LeakageReport()

    def is_available(self) -> bool:
        try:
            config = self.get_config()
            self.load_data(config)
            return True
        except (FileNotFoundError, OSError, KeyError):
            return False

    # --- Main entry point ---

    def evaluate(
        self,
        model: Model,
        config_name: str | None = None,
        save_predictions: bool = False,
        output_dir: str | None = None,
    ) -> BenchmarkResult | EvalResult:
        """Full pipeline: load -> split -> predict -> score -> validate.

        Args:
            model: The model to evaluate.
            config_name: Which config to use. Defaults to self.default_config.
            save_predictions: If True, returns EvalResult with y_true/y_pred
                and optionally saves predictions and plots to output_dir.
            output_dir: Directory for predictions and plots. Defaults to
                results/{eval_name}/{model_name}/.
        """
        config = self.get_config(config_name)

        data = self.load_data(config)
        splits = self.get_splits(data, config)
        leakage_report = self.verify_leakage(splits, config)

        test_data = splits["test"]
        inputs = self.make_inputs(data, test_data)
        y_true = self.get_labels(data, test_data)

        y_pred = model.predict(inputs)
        y_pred = np.asarray(y_pred, dtype=float)

        if len(y_pred) != len(y_true):
            raise ValueError(
                f"Model returned {len(y_pred)} predictions but eval has {len(y_true)} labels"
            )

        metrics = self.score(y_true, y_pred, config)

        # Validate against expected baselines
        from genbench.eval_config import validate_result
        warnings = validate_result(metrics, config, model.name)
        for w in warnings:
            print(f"  WARNING: {w}")

        task_id = f"{self.name}/{config.name}" if config.name != "default" else self.name

        result = BenchmarkResult(
            task_id=task_id,
            model_name=model.name,
            split_type=config.split_type,
            leakage_report=leakage_report,
            metrics=metrics,
            ceiling=self._make_ceiling(metrics),
            n_samples={"ALL": len(y_true)},
            metadata={
                "tier": self.tier,
                "description": self.description,
                "config": config.name,
                "config_description": config.description,
                "paper": config.paper,
                "expected_baselines": config.expected_baselines,
            },
        )

        if save_predictions:
            eval_result = EvalResult(
                benchmark=result,
                y_true=y_true,
                y_pred=y_pred,
                input_metadata={
                    "n_total": len(y_true),
                    "n_scored": int((~np.isnan(y_pred)).sum()),
                    "eval_name": self.name,
                    "config": config.name,
                    "model": model.name,
                },
            )

            if output_dir is None:
                output_dir = f"results/{self.name}/{model.name}"

            eval_result.save_predictions(output_dir)
            eval_result.generate_plots(output_dir)

            return eval_result

        return result

    def _make_ceiling(self, metrics: dict) -> Any:
        from genbench.metrics.ceiling import ceiling_normalized

        ceiling = self.expected_ceiling
        if ceiling is None:
            return None

        for metric_val in metrics.values():
            if metric_val.aggregate and not np.isnan(metric_val.aggregate.estimate):
                ceil_val = ceiling if isinstance(ceiling, (int, float)) else None
                if ceil_val:
                    return ceiling_normalized(
                        metric_val.aggregate.estimate, ceil_val, f"{self.name} theoretical ceiling"
                    )
                break
        return None
