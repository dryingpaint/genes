"""GTEx cis-eQTL prediction eval.

Predict per-gene expression levels from local genotype context using the
GTEx v8 cis-eQTL dataset. Evaluates correlation between predicted and
observed expression across individuals, benchmarked against median cis-h^2
as the theoretical ceiling.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from genbench.eval import Eval
from genbench.eval_config import EvalConfig
from genbench.registry import register_eval
from genbench.types import AncestryStratifiedMetric, SplitType


@register_eval("eqtl")
class EqtlEval(Eval):
    name = "eqtl"
    description = "Predict per-gene expression from genotype (GTEx v8 cis-eQTL)"
    tier = 1
    split_type = SplitType.INDIVIDUAL
    default_baselines = ["predixcan", "null"]
    expected_ceiling = 0.15  # median cis-h^2

    configs = {
        "default": EvalConfig(
            name="default",
            description="GTEx v8 cis-eQTL cross-individual prediction",
            split_type=SplitType.INDIVIDUAL,
            expected_baselines={"predixcan": {"cross_individual_r": 0.10}},
        ),
    }
    default_config = "default"

    def load_data(self, config: EvalConfig) -> Any:
        raise FileNotFoundError(
            "GTEx v8 data not available. Requires dbGaP-authorized access to "
            "GTEx v8 genotype and expression matrices (phs000424.v8)."
        )

    def get_splits(self, data: Any, config: EvalConfig) -> dict[str, Any]:
        raise NotImplementedError("Requires GTEx v8 data (dbGaP)")

    def make_inputs(self, data: Any, split_data: Any) -> dict[str, Any]:
        raise NotImplementedError("Requires GTEx v8 data (dbGaP)")

    def get_labels(self, data: Any, split_data: Any) -> np.ndarray:
        raise NotImplementedError("Requires GTEx v8 data (dbGaP)")

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, config: EvalConfig
    ) -> dict[str, AncestryStratifiedMetric]:
        raise NotImplementedError("Requires GTEx v8 data (dbGaP)")
