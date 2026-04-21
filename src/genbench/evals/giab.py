"""GIAB variant calling eval.

Germline SNV/indel accuracy evaluated against NIST Genome in a Bottle
truth sets using hap.py-style benchmarking. Measures precision, recall,
and F1 for SNVs and indels separately on HG001-HG007 truth samples.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from genbench.eval import Eval
from genbench.eval_config import EvalConfig
from genbench.registry import register_eval
from genbench.types import AncestryStratifiedMetric, SplitType


@register_eval("giab")
class GiabEval(Eval):
    name = "giab"
    description = "Germline SNV/indel accuracy on NIST truth sets"
    tier = 0
    split_type = SplitType.ZERO_SHOT
    default_baselines = ["deepvariant", "null"]
    expected_ceiling = 0.999  # SNV F1

    configs = {
        "default": EvalConfig(
            name="default",
            description="GIAB SNV/indel accuracy on NIST truth sets",
            split_type=SplitType.ZERO_SHOT,
            expected_baselines={"deepvariant": {"snv_f1": 0.999}},
        ),
    }
    default_config = "default"

    def load_data(self, config: EvalConfig) -> Any:
        raise FileNotFoundError(
            "GIAB truth sets not available. Requires NIST hap.py-compatible "
            "truth VCFs and high-confidence BED files for HG001-HG007, plus "
            "query VCFs to benchmark."
        )

    def get_splits(self, data: Any, config: EvalConfig) -> dict[str, Any]:
        raise NotImplementedError("Requires hap.py + query VCFs")

    def make_inputs(self, data: Any, split_data: Any) -> dict[str, Any]:
        raise NotImplementedError("Requires hap.py + query VCFs")

    def get_labels(self, data: Any, split_data: Any) -> np.ndarray:
        raise NotImplementedError("Requires hap.py + query VCFs")

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, config: EvalConfig
    ) -> dict[str, AncestryStratifiedMetric]:
        raise NotImplementedError("Requires hap.py + query VCFs")
