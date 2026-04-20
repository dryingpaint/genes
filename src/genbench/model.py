"""Model protocol — anything that makes predictions."""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Model(Protocol):
    """Minimal contract for a model that can be evaluated.

    The inputs dict is eval-specific:
    - Variant evals: {"chroms": [...], "positions": [...], "refs": [...], "alts": [...]}
    - Protein evals: {"sequence": str, "variants": [...]}
    - Genomic prediction: {"genotype_matrix": np.ndarray, "sample_ids": [...]}
    """

    @property
    def name(self) -> str: ...

    def predict(self, inputs: dict[str, Any]) -> np.ndarray: ...


class FunctionModel:
    """Wrap a bare callable as a Model."""

    def __init__(self, name: str, fn: Callable[[dict[str, Any]], np.ndarray]):
        self._name = name
        self._fn = fn

    @property
    def name(self) -> str:
        return self._name

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        return self._fn(inputs)
