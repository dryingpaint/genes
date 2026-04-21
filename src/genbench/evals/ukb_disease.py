"""UKB disease endpoints eval.

Polygenic risk score (PRS) classification of 10 common diseases from
UK Biobank, including type 2 diabetes, coronary artery disease, and
breast cancer. Evaluates discrimination (AUROC) and calibration of
PRS-based disease risk prediction with ancestry-stratified reporting.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from genbench.eval import Eval
from genbench.eval_config import EvalConfig
from genbench.registry import register_eval
from genbench.types import AncestryStratifiedMetric, SplitType


@register_eval("ukb_disease")
class UkbDiseaseEval(Eval):
    name = "ukb_disease"
    description = "PRS classification of 10 disease endpoints from UKB"
    tier = 2
    split_type = SplitType.INDIVIDUAL
    default_baselines = ["sbayesrc", "null"]
    expected_ceiling = None

    configs = {
        "default": EvalConfig(
            name="default",
            description="UKB disease endpoint PRS classification",
            split_type=SplitType.INDIVIDUAL,
            expected_baselines={"sbayesrc": {"cad_auc": 0.80}},
        ),
    }
    default_config = "default"

    def load_data(self, config: EvalConfig) -> Any:
        raise FileNotFoundError(
            "UK Biobank data not available. Requires approved UKB application "
            "with access to genotype data and ICD-10/self-reported disease "
            "phenotypes for the 10 disease endpoints."
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
