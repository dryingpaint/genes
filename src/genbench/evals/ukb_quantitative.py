"""UKB quantitative traits eval.

Polygenic risk score (PRS) prediction of 13 continuous traits including
height, BMI, blood pressure, and biomarkers from UK Biobank. Performance
ceiling varies per trait based on SNP heritability (H2_SNP from config).
Evaluates cross-ancestry PRS accuracy with ancestry-stratified metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from genbench.eval import Eval
from genbench.eval_config import EvalConfig
from genbench.registry import register_eval
from genbench.types import AncestryStratifiedMetric, SplitType


@register_eval("ukb_quantitative")
class UkbQuantitativeEval(Eval):
    name = "ukb_quantitative"
    description = "PRS prediction of 13 quantitative traits (height, BMI, etc.) from UKB"
    tier = 2
    split_type = SplitType.INDIVIDUAL
    default_baselines = ["sbayesrc", "null"]
    expected_ceiling = None  # varies per trait, loaded from config H2_SNP

    configs = {
        "default": EvalConfig(
            name="default",
            description="UKB quantitative trait PRS prediction",
            split_type=SplitType.INDIVIDUAL,
            expected_baselines={"sbayesrc": {"height_r2": 0.35}},
        ),
    }
    default_config = "default"

    def load_data(self, config: EvalConfig) -> Any:
        raise FileNotFoundError(
            "UK Biobank data not available. Requires approved UKB application "
            "with access to genotype data and phenotype fields for the 13 "
            "quantitative traits."
        )

    def get_splits(self, data: Any, config: EvalConfig) -> dict[str, Any]:
        raise NotImplementedError("Requires UKB access")

    def make_inputs(self, data: Any, split_data: Any) -> dict[str, Any]:
        raise NotImplementedError("Requires UKB access")

    def get_labels(self, data: Any, split_data: Any) -> np.ndarray:
        raise NotImplementedError("Requires UKB access")

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, config: EvalConfig
    ) -> dict[str, AncestryStratifiedMetric]:
        raise NotImplementedError("Requires UKB access")
