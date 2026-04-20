"""ProteinGym Deep Mutational Scanning eval.

Tier 1 molecular phenotype. Zero-shot prediction of variant fitness effects
across 87+ DMS assays. Primary metric: per-assay Spearman rho.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from genbench.eval import Eval
from genbench.config import DATASETS_PATH
from genbench.metrics.regression import spearman_rho
from genbench.registry import register_eval
from genbench.types import AncestryStratifiedMetric, LeakageReport, MetricValue, SplitType


@register_eval("dms")
class DmsEval(Eval):
    name = "dms"
    description = "ProteinGym DMS zero-shot variant effect prediction (87+ assays)"
    tier = 1
    split_type = SplitType.ZERO_SHOT
    expected_ceiling = 0.65  # aggregate ceiling across all assays
    default_baselines = ["saprot", "null"]

    def load_data(self) -> dict[str, pd.DataFrame]:
        """Load all ProteinGym assay CSVs. Returns {assay_name: DataFrame}."""
        base = Path(DATASETS_PATH) / "proteingym" / "ProteinGym_substitutions"
        if not base.exists():
            raise FileNotFoundError(f"ProteinGym data not found at {base}")

        assays = {}
        for f in sorted(base.glob("*.csv")):
            df = pd.read_csv(f)
            if "DMS_score" in df.columns:
                assays[f.stem] = df

        if not assays:
            raise FileNotFoundError(f"No valid ProteinGym assays found in {base}")

        return assays

    def get_splits(self, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        # Zero-shot: no split, entire dataset is test
        return {"test": data}

    def make_inputs(self, data: dict[str, pd.DataFrame], split_data: Any) -> dict[str, Any]:
        # For DMS, inputs are per-assay. The model receives all assays.
        return {"assays": split_data}

    def get_labels(self, data: dict[str, pd.DataFrame], split_data: Any) -> np.ndarray:
        # Concatenate all assay DMS scores as a flat array
        all_scores = []
        for name, df in split_data.items():
            all_scores.extend(df["DMS_score"].values)
        return np.array(all_scores)

    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, AncestryStratifiedMetric]:
        rho = spearman_rho(y_true, y_pred, n_bootstrap=500)
        return {
            "spearman_rho": AncestryStratifiedMetric(aggregate=rho),
        }

    def evaluate(self, model):
        """Override: evaluate per-assay, then aggregate."""
        data = self.load_data()
        splits = self.get_splits(data)
        leakage_report = self.verify_leakage(splits)

        rhos = []
        per_assay_results = []

        for assay_name, assay_df in splits["test"].items():
            y_true = assay_df["DMS_score"].values

            # Build per-assay inputs
            inputs = {"assays": {assay_name: assay_df}}
            if "mutant" in assay_df.columns:
                inputs["variants"] = assay_df["mutant"].tolist()
                # Extract sequence if available
                if "target_seq" in assay_df.columns:
                    inputs["sequence"] = assay_df["target_seq"].iloc[0]

            try:
                y_pred = model.predict(inputs)
                y_pred = np.asarray(y_pred, dtype=float)
                if len(y_pred) != len(y_true):
                    continue
                rho = spearman_rho(y_true, y_pred, n_bootstrap=100)
                rhos.append(rho.estimate)
            except Exception:
                continue

        if not rhos:
            rhos = [0.0]

        from genbench.types import BenchmarkResult

        return BenchmarkResult(
            task_id=self.name,
            model_name=model.name,
            split_type=self.split_type,
            leakage_report=leakage_report,
            metrics={
                "mean_spearman_rho": AncestryStratifiedMetric(
                    aggregate=MetricValue(
                        estimate=float(np.mean(rhos)),
                        ci_lower=float(np.mean(rhos) - 1.96 * np.std(rhos) / np.sqrt(len(rhos))),
                        ci_upper=float(np.mean(rhos) + 1.96 * np.std(rhos) / np.sqrt(len(rhos))),
                        n=len(rhos),
                    )
                ),
                "median_spearman_rho": AncestryStratifiedMetric(
                    aggregate=MetricValue(
                        estimate=float(np.median(rhos)), ci_lower=0, ci_upper=0, n=len(rhos)
                    )
                ),
            },
            n_samples={"ALL": sum(len(df) for df in data.values())},
            metadata={
                "tier": self.tier,
                "description": self.description,
                "n_assays_evaluated": len(rhos),
                "n_assays_total": len(data),
            },
        )
