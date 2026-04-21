"""1001 Genomes Arabidopsis genomic prediction eval.

Genomic prediction in 1,135 Arabidopsis thaliana accessions from the
1001 Genomes Project. Evaluates prediction of quantitative traits
(flowering time, rosette diameter, etc.) using leave-n-out cross-validation
with phenotypes from AraPheno as a plant model organism benchmark.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from genbench.eval import Eval
from genbench.eval_config import EvalConfig
from genbench.registry import register_eval
from genbench.types import AncestryStratifiedMetric, SplitType


@register_eval("arabidopsis")
class ArabidopsisEval(Eval):
    name = "arabidopsis"
    description = "Genomic prediction in 1,135 Arabidopsis accessions (1001 Genomes)"
    tier = 3
    split_type = SplitType.LEAVE_N_OUT
    default_baselines = ["gblup", "elastic_net", "null"]
    expected_ceiling = 0.70  # flowering time H^2

    configs = {
        "default": EvalConfig(
            name="default",
            description="1001 Genomes Arabidopsis genomic prediction (flowering time)",
            split_type=SplitType.LEAVE_N_OUT,
            expected_baselines={"gblup": {"flowering_r2": 0.55}},
        ),
    }
    default_config = "default"

    def load_data(self, config: EvalConfig) -> Any:
        raise FileNotFoundError(
            "1001 Genomes Arabidopsis data not available. Requires the 1001 "
            "Genomes VCF (1001genomes.org) and AraPheno phenotype database "
            "(arapheno.1001genomes.org)."
        )

    def get_splits(self, data: Any, config: EvalConfig) -> dict[str, Any]:
        raise NotImplementedError("Requires 1001 Genomes VCF + AraPheno phenotypes")

    def make_inputs(self, data: Any, split_data: Any) -> dict[str, Any]:
        raise NotImplementedError("Requires 1001 Genomes VCF + AraPheno phenotypes")

    def get_labels(self, data: Any, split_data: Any) -> np.ndarray:
        raise NotImplementedError("Requires 1001 Genomes VCF + AraPheno phenotypes")

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, config: EvalConfig
    ) -> dict[str, AncestryStratifiedMetric]:
        raise NotImplementedError("Requires 1001 Genomes VCF + AraPheno phenotypes")
