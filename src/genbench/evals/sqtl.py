"""Splicing QTL eval.

Predict variant effects on mRNA splicing quantified as percent-spliced-in
(PSI) changes. Uses GTEx v8 sQTL associations to evaluate how accurately
models predict splice-altering variant effects across tissues.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from genbench.eval import Eval
from genbench.eval_config import EvalConfig
from genbench.registry import register_eval
from genbench.types import AncestryStratifiedMetric, SplitType


@register_eval("sqtl")
class SqtlEval(Eval):
    name = "sqtl"
    description = "Predict variant effect on splicing PSI (GTEx v8 sQTL)"
    tier = 1
    split_type = SplitType.INDIVIDUAL
    default_baselines = ["spliceai", "null"]
    expected_ceiling = None

    configs = {
        "default": EvalConfig(
            name="default",
            description="GTEx v8 sQTL splicing prediction",
            split_type=SplitType.INDIVIDUAL,
            expected_baselines={"spliceai": {}},
        ),
    }
    default_config = "default"

    def load_data(self, config: EvalConfig) -> Any:
        raise FileNotFoundError(
            "GTEx v8 sQTL data not available. Requires dbGaP-authorized access "
            "to GTEx v8 splice QTL summary statistics and junction-level "
            "quantifications (phs000424.v8)."
        )

    def get_splits(self, data: Any, config: EvalConfig) -> dict[str, Any]:
        raise NotImplementedError("Requires GTEx v8 sQTL data")

    def make_inputs(self, data: Any, split_data: Any) -> dict[str, Any]:
        raise NotImplementedError("Requires GTEx v8 sQTL data")

    def get_labels(self, data: Any, split_data: Any) -> np.ndarray:
        raise NotImplementedError("Requires GTEx v8 sQTL data")

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, config: EvalConfig
    ) -> dict[str, AncestryStratifiedMetric]:
        raise NotImplementedError("Requires GTEx v8 sQTL data")
