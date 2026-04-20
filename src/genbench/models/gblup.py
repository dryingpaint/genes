"""Genomic BLUP via sklearn (pure Python fallback for model organisms)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from genbench.registry import register_model


@register_model("gblup")
class GblupModel:
    """Ridge regression on genotype matrix — approximates GBLUP without GCTA dependency."""

    name = "gblup"

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        X_train = inputs["genotype_matrix_train"]
        y_train = inputs["labels_train"]
        X_test = inputs["genotype_matrix_test"]

        # Compute genomic relationship matrix via linear kernel
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        return model.predict(X_test)


@register_model("elastic_net")
class ElasticNetModel:
    """Elastic net on genotype matrix — strong linear baseline for genomic prediction."""

    name = "elastic_net"

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        from sklearn.linear_model import ElasticNet

        X_train = inputs["genotype_matrix_train"]
        y_train = inputs["labels_train"]
        X_test = inputs["genotype_matrix_test"]

        model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)
        model.fit(X_train, y_train)
        return model.predict(X_test)
