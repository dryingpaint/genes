"""BRCA1 SGE (saturation genome editing) eval.

Binary functional classification of 3,893 BRCA1 SNVs using saturation
genome editing ground truth from Findlay et al. (2018). Each variant is
labeled functional or non-functional based on cell viability assays.
Data is downloaded but requires xlsx parsing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from genbench.eval import Eval
from genbench.registry import register_eval
from genbench.types import AncestryStratifiedMetric, SplitType


@register_eval("brca1_sge")
class Brca1SgeEval(Eval):
    name = "brca1_sge"
    description = "Binary functional classification of 3,893 BRCA1 SNVs (SGE)"
    tier = 1
    split_type = SplitType.ZERO_SHOT
    default_baselines = ["evo2", "alphamissense", "null"]
    expected_ceiling = 0.97

    def load_data(self) -> Any:
        raise FileNotFoundError(
            "BRCA1 SGE data needs xlsx parsing. The Findlay et al. supplementary "
            "table is downloaded but requires openpyxl to extract the 3,893 SNV "
            "functional classifications."
        )

    def get_splits(self, data: Any) -> dict[str, Any]:
        raise NotImplementedError("Requires BRCA1 SGE xlsx parsing")

    def make_inputs(self, data: Any, split_data: Any) -> dict[str, Any]:
        raise NotImplementedError("Requires BRCA1 SGE xlsx parsing")

    def get_labels(self, data: Any, split_data: Any) -> np.ndarray:
        raise NotImplementedError("Requires BRCA1 SGE xlsx parsing")

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, AncestryStratifiedMetric]:
        raise NotImplementedError("Requires BRCA1 SGE xlsx parsing")
