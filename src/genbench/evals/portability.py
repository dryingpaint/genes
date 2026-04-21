"""Ancestry portability analysis eval.

Cross-ancestry evaluation of polygenic risk scores. Measures how PRS
performance degrades when models trained on one ancestry group are
applied to others. Uses UKB multi-ancestry cohort to quantify the
portability gap — a key equity metric for genomic prediction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from genbench.eval import Eval
from genbench.eval_config import EvalConfig
from genbench.registry import register_eval
from genbench.types import AncestryStratifiedMetric, SplitType


@register_eval("portability")
class PortabilityEval(Eval):
    name = "portability"
    description = "Cross-ancestry PRS portability evaluation"
    tier = 2
    split_type = SplitType.INDIVIDUAL
    default_baselines = ["prscsx", "null"]
    expected_ceiling = None

    configs = {
        "default": EvalConfig(
            name="default",
            description="Cross-ancestry PRS portability analysis",
            split_type=SplitType.INDIVIDUAL,
            expected_baselines={"prscsx": {}},
        ),
    }
    default_config = "default"

    def load_data(self, config: EvalConfig) -> Any:
        raise FileNotFoundError(
            "UKB multi-ancestry data not available. Requires approved UKB "
            "application with genotype and phenotype data across EUR, AFR, "
            "SAS, EAS, and AMR ancestry groups."
        )

    def get_splits(self, data: Any, config: EvalConfig) -> dict[str, Any]:
        raise NotImplementedError("Requires UKB multi-ancestry data")

    def make_inputs(self, data: Any, split_data: Any) -> dict[str, Any]:
        raise NotImplementedError("Requires UKB multi-ancestry data")

    def get_labels(self, data: Any, split_data: Any) -> np.ndarray:
        raise NotImplementedError("Requires UKB multi-ancestry data")

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, config: EvalConfig
    ) -> dict[str, AncestryStratifiedMetric]:
        raise NotImplementedError("Requires UKB multi-ancestry data")
