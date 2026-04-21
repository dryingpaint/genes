"""HLA typing eval.

HLA allele calling at 2-field resolution evaluated on 1000 Genomes Phase 3
samples. Measures accuracy of Class I (HLA-A, -B, -C) and Class II (HLA-DRB1,
-DQB1) typing from whole-genome sequencing data against validated truth types.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from genbench.eval import Eval
from genbench.eval_config import EvalConfig
from genbench.registry import register_eval
from genbench.types import AncestryStratifiedMetric, SplitType


@register_eval("hla")
class HlaEval(Eval):
    name = "hla"
    description = "HLA allele calling at 2-field resolution on 1KG Phase 3"
    tier = 0
    split_type = SplitType.INDIVIDUAL
    default_baselines = ["hla_la", "null"]
    expected_ceiling = 0.99

    configs = {
        "default": EvalConfig(
            name="default",
            description="HLA typing concordance on 1KG Phase 3",
            split_type=SplitType.INDIVIDUAL,
            expected_baselines={"hla_la": {"concordance": 0.99}},
        ),
    }
    default_config = "default"

    def load_data(self, config: EvalConfig) -> Any:
        raise FileNotFoundError(
            "1KG HLA truth types not available. Requires 1000 Genomes Phase 3 "
            "HLA reference panel with validated 2-field resolution types."
        )

    def get_splits(self, data: Any, config: EvalConfig) -> dict[str, Any]:
        raise NotImplementedError("Requires 1KG HLA truth types")

    def make_inputs(self, data: Any, split_data: Any) -> dict[str, Any]:
        raise NotImplementedError("Requires 1KG HLA truth types")

    def get_labels(self, data: Any, split_data: Any) -> np.ndarray:
        raise NotImplementedError("Requires 1KG HLA truth types")

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, config: EvalConfig
    ) -> dict[str, AncestryStratifiedMetric]:
        raise NotImplementedError("Requires 1KG HLA truth types")
