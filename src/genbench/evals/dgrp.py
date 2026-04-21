"""DGRP Drosophila genomic prediction eval.

Genomic prediction in 205 inbred Drosophila melanogaster lines from the
DGRP2 (Drosophila Genetic Reference Panel). Evaluates prediction of
quantitative traits (starvation resistance, startle response, etc.) using
leave-n-out cross-validation as a model organism benchmark.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from genbench.eval import Eval
from genbench.eval_config import EvalConfig
from genbench.registry import register_eval
from genbench.types import AncestryStratifiedMetric, SplitType


@register_eval("dgrp")
class DgrpEval(Eval):
    name = "dgrp"
    description = "Genomic prediction in 205 inbred Drosophila lines (DGRP2)"
    tier = 3
    split_type = SplitType.LEAVE_N_OUT
    default_baselines = ["gblup", "elastic_net", "null"]
    expected_ceiling = 0.60  # starvation resistance H^2

    configs = {
        "default": EvalConfig(
            name="default",
            description="DGRP2 genomic prediction (starvation resistance)",
            split_type=SplitType.LEAVE_N_OUT,
            expected_baselines={"gblup": {"starvation_r2": 0.50}},
        ),
    }
    default_config = "default"

    def load_data(self, config: EvalConfig) -> Any:
        raise FileNotFoundError(
            "DGRP2 data not available. Requires DGRP2 genotype VCF and "
            "phenotype files from dgrp2.gnets.ncsu.edu (server currently down). "
            "See Mackay et al. (2012) Nature."
        )

    def get_splits(self, data: Any, config: EvalConfig) -> dict[str, Any]:
        raise NotImplementedError("Requires DGRP2 data (server currently down)")

    def make_inputs(self, data: Any, split_data: Any) -> dict[str, Any]:
        raise NotImplementedError("Requires DGRP2 data (server currently down)")

    def get_labels(self, data: Any, split_data: Any) -> np.ndarray:
        raise NotImplementedError("Requires DGRP2 data (server currently down)")

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, config: EvalConfig
    ) -> dict[str, AncestryStratifiedMetric]:
        raise NotImplementedError("Requires DGRP2 data (server currently down)")
