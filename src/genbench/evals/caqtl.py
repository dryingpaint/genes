"""Chromatin accessibility QTL eval.

Predict variant effects on ATAC-seq chromatin accessibility peaks.
Evaluates how well models capture the regulatory impact of genetic
variants on open chromatin regions using ENCODE4 and GTEx ATAC-seq data.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from genbench.eval import Eval
from genbench.eval_config import EvalConfig
from genbench.registry import register_eval
from genbench.types import AncestryStratifiedMetric, SplitType


@register_eval("caqtl")
class CaqtlEval(Eval):
    name = "caqtl"
    description = "Predict variant effect on ATAC-seq peaks (caQTL)"
    tier = 1
    split_type = SplitType.INDIVIDUAL
    default_baselines = ["chrombpnet", "null"]
    expected_ceiling = None

    configs = {
        "default": EvalConfig(
            name="default",
            description="ENCODE4/GTEx ATAC-seq caQTL prediction",
            split_type=SplitType.INDIVIDUAL,
            expected_baselines={"chrombpnet": {}},
        ),
    }
    default_config = "default"

    def load_data(self, config: EvalConfig) -> Any:
        raise FileNotFoundError(
            "ENCODE4/GTEx ATAC-seq data not available. Requires ENCODE4 ATAC-seq "
            "peak calls and GTEx ATAC-seq QTL summary statistics."
        )

    def get_splits(self, data: Any, config: EvalConfig) -> dict[str, Any]:
        raise NotImplementedError("Requires ENCODE4/GTEx ATAC-seq data")

    def make_inputs(self, data: Any, split_data: Any) -> dict[str, Any]:
        raise NotImplementedError("Requires ENCODE4/GTEx ATAC-seq data")

    def get_labels(self, data: Any, split_data: Any) -> np.ndarray:
        raise NotImplementedError("Requires ENCODE4/GTEx ATAC-seq data")

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, config: EvalConfig
    ) -> dict[str, AncestryStratifiedMetric]:
        raise NotImplementedError("Requires ENCODE4/GTEx ATAC-seq data")
