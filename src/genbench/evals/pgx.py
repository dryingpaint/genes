"""PGx diplotyping eval.

Pharmacogenomic star-allele diplotype calling accuracy evaluated against
the GeT-RM (Genetic Testing Reference Materials) truth set from CDC.
Measures concordance of called diplotypes for key pharmacogenes (CYP2D6,
CYP2C19, etc.) used in clinical dosing decisions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from genbench.eval import Eval
from genbench.eval_config import EvalConfig
from genbench.registry import register_eval
from genbench.types import AncestryStratifiedMetric, SplitType


@register_eval("pgx")
class PgxEval(Eval):
    name = "pgx"
    description = "Star-allele diplotype calling on GeT-RM truth set"
    tier = 0
    split_type = SplitType.ZERO_SHOT
    default_baselines = ["pharmcat", "null"]
    expected_ceiling = 0.97

    configs = {
        "default": EvalConfig(
            name="default",
            description="GeT-RM diplotype calling accuracy",
            split_type=SplitType.ZERO_SHOT,
            expected_baselines={"pharmcat": {"accuracy": 0.97}},
        ),
    }
    default_config = "default"

    def load_data(self, config: EvalConfig) -> Any:
        raise FileNotFoundError(
            "GeT-RM truth set not available. Request access from CDC Genetic "
            "Testing Reference Materials program (https://www.cdc.gov/labquality/get-rm/)."
        )

    def get_splits(self, data: Any, config: EvalConfig) -> dict[str, Any]:
        raise NotImplementedError("Requires GeT-RM truth set")

    def make_inputs(self, data: Any, split_data: Any) -> dict[str, Any]:
        raise NotImplementedError("Requires GeT-RM truth set")

    def get_labels(self, data: Any, split_data: Any) -> np.ndarray:
        raise NotImplementedError("Requires GeT-RM truth set")

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, config: EvalConfig
    ) -> dict[str, AncestryStratifiedMetric]:
        raise NotImplementedError("Requires GeT-RM truth set")
