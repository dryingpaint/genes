"""HIrisPlex-S visible traits eval.

Predict eye, hair, and skin color from a small panel of SNPs using the
HIrisPlex-S system. Evaluated on 1000 Genomes samples with phenotype
annotations. Tests whether genomic models can capture well-characterized
pigmentation genetics as a Tier 2 sanity check.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from genbench.eval import Eval
from genbench.registry import register_eval
from genbench.types import AncestryStratifiedMetric, SplitType


@register_eval("hirisplex")
class HirisplexEval(Eval):
    name = "hirisplex"
    description = "Eye/hair/skin color prediction from SNPs (HIrisPlex-S)"
    tier = 2
    split_type = SplitType.ZERO_SHOT
    default_baselines = ["hirisplex_s", "null"]
    expected_ceiling = 0.90

    def load_data(self) -> Any:
        raise FileNotFoundError(
            "HIrisPlex data not available. Requires 1000 Genomes genotypes at "
            "HIrisPlex-S SNP positions plus matched phenotype annotations for "
            "eye, hair, and skin color."
        )

    def get_splits(self, data: Any) -> dict[str, Any]:
        raise NotImplementedError("Requires 1KG + phenotype data")

    def make_inputs(self, data: Any, split_data: Any) -> dict[str, Any]:
        raise NotImplementedError("Requires 1KG + phenotype data")

    def get_labels(self, data: Any, split_data: Any) -> np.ndarray:
        raise NotImplementedError("Requires 1KG + phenotype data")

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, AncestryStratifiedMetric]:
        raise NotImplementedError("Requires 1KG + phenotype data")
