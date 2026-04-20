"""Eval base class — one subclass per benchmark."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

import numpy as np

from genbench.model import Model
from genbench.splits.leakage import LeakageError
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
    def name(self) -> str:
        """Short identifier: 'clinvar', 'dms', 'dgrp', etc."""
        ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def tier(self) -> int:
        """0-3. For reporting grouping only, not execution gating."""
        ...

    @property
    @abstractmethod
    def split_type(self) -> SplitType: ...

    @property
    @abstractmethod
    def default_baselines(self) -> list[str]:
        """Model names to compare against by default."""
        ...

    @property
    def expected_ceiling(self) -> float | dict[str, float] | None:
        """Theoretical performance ceiling. Override in subclasses."""
        return None

    # --- Data & Splits ---

    @abstractmethod
    def load_data(self) -> Any:
        """Load dataset. Raises FileNotFoundError if data not available."""
        ...

    @abstractmethod
    def get_splits(self, data: Any) -> dict[str, Any]:
        """Return dict with at least 'test' key. May include 'train', 'val'."""
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
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, AncestryStratifiedMetric]:
        """Compute eval-specific metrics."""
        ...

    # --- Defaults (override if needed) ---

    def verify_leakage(self, splits: dict[str, Any]) -> LeakageReport:
        """Run anti-leakage checks. Override to add eval-specific checks."""
        return LeakageReport()

    def is_available(self) -> bool:
        """Check if data exists. Returns False if load_data would fail."""
        try:
            self.load_data()
            return True
        except (FileNotFoundError, OSError):
            return False

    # --- Main entry point ---

    def evaluate(self, model: Model) -> BenchmarkResult:
        """Full pipeline: load → split → predict → score → leakage check.

        This method should rarely need overriding.
        """
        data = self.load_data()
        splits = self.get_splits(data)
        leakage_report = self.verify_leakage(splits)

        test_data = splits["test"]
        inputs = self.make_inputs(data, test_data)
        y_true = self.get_labels(data, test_data)

        y_pred = model.predict(inputs)

        # Handle missing predictions
        y_pred = np.asarray(y_pred, dtype=float)
        if len(y_pred) != len(y_true):
            raise ValueError(
                f"Model returned {len(y_pred)} predictions but eval has {len(y_true)} labels"
            )

        metrics = self.score(y_true, y_pred)

        return BenchmarkResult(
            task_id=self.name,
            model_name=model.name,
            split_type=self.split_type,
            leakage_report=leakage_report,
            metrics=metrics,
            ceiling=self._make_ceiling(metrics),
            n_samples={"ALL": len(y_true)},
            metadata={
                "tier": self.tier,
                "description": self.description,
            },
        )

    def _make_ceiling(self, metrics: dict) -> Any:
        """Build CeilingInfo from expected_ceiling if set."""
        from genbench.metrics.ceiling import ceiling_normalized

        ceiling = self.expected_ceiling
        if ceiling is None:
            return None

        # Use the first metric's aggregate estimate for normalization
        for metric_val in metrics.values():
            if metric_val.aggregate and not np.isnan(metric_val.aggregate.estimate):
                ceil_val = ceiling if isinstance(ceiling, (int, float)) else None
                if ceil_val:
                    return ceiling_normalized(
                        metric_val.aggregate.estimate, ceil_val, f"{self.name} theoretical ceiling"
                    )
                break
        return None
