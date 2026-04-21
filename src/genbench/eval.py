"""Eval base class — one subclass per benchmark.

Each eval can have multiple configs (EvalConfig) matching different published
benchmarks. The config parameterizes data filtering, split strategy, metrics,
and expected baseline scores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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

    def evaluate(self, model: Model, config_name: str | None = None) -> BenchmarkResult:
        """Full pipeline: load → split → predict → score → validate.

        Args:
            model: The model to evaluate.
            config_name: Which config to use. Defaults to self.default_config.
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

        return BenchmarkResult(
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
