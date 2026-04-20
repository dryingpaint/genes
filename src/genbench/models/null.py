"""Null baselines: random, mean, frequency predictors."""

from __future__ import annotations

from typing import Any

import numpy as np

from genbench.registry import register_model


@register_model("null")
class NullModel:
    """Random uniform predictions. The floor for any eval."""

    name = "null"

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        n = _infer_n(inputs)
        return np.random.default_rng(42).uniform(0, 1, n)


@register_model("mean")
class MeanModel:
    """Predict the mean of training labels (if provided in inputs)."""

    name = "mean"

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        n = _infer_n(inputs)
        train_mean = inputs.get("train_mean", 0.5)
        return np.full(n, train_mean)


def _infer_n(inputs: dict[str, Any]) -> int:
    """Infer number of samples from inputs dict."""
    for key in ["chroms", "positions", "variants", "sample_ids"]:
        if key in inputs:
            return len(inputs[key])
    if "genotype_matrix" in inputs:
        return inputs["genotype_matrix"].shape[0]
    if "n" in inputs:
        return inputs["n"]
    raise ValueError("Cannot infer sample count from inputs")
