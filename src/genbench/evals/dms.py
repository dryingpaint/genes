"""ProteinGym Deep Mutational Scanning eval.

Configs:
  - substitutions_v1: 217 DMS assays, per-assay Spearman rho aggregated.
    Expected: SaProt-650M mean rho ~0.473.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from genbench.eval import Eval
from genbench.eval_config import EvalConfig
from genbench.config import DATASETS_PATH
from genbench.metrics.regression import spearman_rho
from genbench.registry import register_eval
from genbench.types import AncestryStratifiedMetric, LeakageReport, MetricValue, SplitType


@register_eval("dms")
class DmsEval(Eval):
    name = "dms"
    description = "ProteinGym DMS zero-shot variant effect prediction"
    tier = 1
    split_type = SplitType.ZERO_SHOT
    expected_ceiling = 0.65
    default_baselines = ["saprot", "null"]
    default_config = "substitutions_v0.1"

    configs = {
        "substitutions_v0.1": EvalConfig(
            name="substitutions_v0.1",
            description="87 DMS assays (ProteinGym v0.1, SaProt's published eval)",
            filters={"assay_type": "substitutions", "proteingym_version": "v0.1"},
            split_type=SplitType.ZERO_SHOT,
            metrics=["mean_spearman_rho", "median_spearman_rho"],
            expected_baselines={
                "saprot": {"mean_spearman_rho": 0.473},
            },
            paper="Su et al. ICLR 2024 (SaProt on ProteinGym v0.1)",
        ),
        "substitutions_v1": EvalConfig(
            name="substitutions_v1",
            description="217 DMS assays (ProteinGym v1, current leaderboard)",
            filters={"assay_type": "substitutions", "proteingym_version": "v1"},
            split_type=SplitType.ZERO_SHOT,
            metrics=["mean_spearman_rho", "median_spearman_rho"],
            expected_baselines={},
            paper="Notin et al. NeurIPS 2023 / ProteinGym v1",
        ),
    }

    def load_data(self, config: EvalConfig) -> dict[str, pd.DataFrame]:
        version = config.filters.get("proteingym_version", "v1")

        # Try versioned path first, then unversioned fallback
        for search_path in [
            Path(DATASETS_PATH) / "proteingym" / version / "ProteinGym_substitutions",
            Path(DATASETS_PATH) / "proteingym" / "ProteinGym_substitutions",
        ]:
            if search_path.exists():
                base = search_path
                break
        else:
            raise FileNotFoundError(
                f"ProteinGym {version} data not found. Run: modal run scripts/ingest_all.py"
            )

        assays = {}
        for f in sorted(base.glob("*.csv")):
            df = pd.read_csv(f)
            if "DMS_score" in df.columns:
                assays[f.stem] = df

        if not assays:
            raise FileNotFoundError(f"No valid ProteinGym assays found in {base}")
        return assays

    def get_splits(self, data: dict[str, pd.DataFrame], config: EvalConfig) -> dict[str, Any]:
        return {"test": data}

    def make_inputs(self, data: dict[str, pd.DataFrame], split_data: Any) -> dict[str, Any]:
        return {"assays": split_data}

    def get_labels(self, data: dict[str, pd.DataFrame], split_data: Any) -> np.ndarray:
        all_scores = []
        for df in split_data.values():
            all_scores.extend(df["DMS_score"].values)
        return np.array(all_scores)

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, config: EvalConfig
    ) -> dict[str, AncestryStratifiedMetric]:
        rho = spearman_rho(y_true, y_pred, n_bootstrap=500)
        return {"spearman_rho": AncestryStratifiedMetric(aggregate=rho)}

    def evaluate(self, model, config_name: str | None = None):
        """Override: evaluate per-assay, then aggregate."""
        config = self.get_config(config_name)
        data = self.load_data(config)
        splits = self.get_splits(data, config)
        leakage_report = self.verify_leakage(splits, config)

        rhos = []
        for assay_name, assay_df in splits["test"].items():
            y_true = assay_df["DMS_score"].values
            inputs = {"assays": {assay_name: assay_df}, "n": len(assay_df)}
            if "mutant" in assay_df.columns:
                inputs["variants"] = assay_df["mutant"].tolist()
            if "target_seq" in assay_df.columns:
                inputs["sequence"] = assay_df["target_seq"].iloc[0]

            try:
                y_pred = np.asarray(model.predict(inputs), dtype=float)
                if len(y_pred) != len(y_true):
                    continue
                rho = spearman_rho(y_true, y_pred, n_bootstrap=100)
                rhos.append(rho.estimate)
            except Exception:
                continue

        if not rhos:
            rhos = [0.0]

        mean_rho = float(np.mean(rhos))
        median_rho = float(np.median(rhos))

        metrics = {
            "mean_spearman_rho": AncestryStratifiedMetric(
                aggregate=MetricValue(
                    estimate=mean_rho,
                    ci_lower=mean_rho - 1.96 * np.std(rhos) / max(np.sqrt(len(rhos)), 1),
                    ci_upper=mean_rho + 1.96 * np.std(rhos) / max(np.sqrt(len(rhos)), 1),
                    n=len(rhos),
                )
            ),
            "median_spearman_rho": AncestryStratifiedMetric(
                aggregate=MetricValue(estimate=median_rho, ci_lower=0, ci_upper=0, n=len(rhos))
            ),
        }

        from genbench.eval_config import validate_result
        for w in validate_result(metrics, config, model.name):
            print(f"  WARNING: {w}")

        task_id = f"{self.name}/{config.name}" if config.name != "default" else self.name

        from genbench.types import BenchmarkResult
        return BenchmarkResult(
            task_id=task_id,
            model_name=model.name,
            split_type=config.split_type,
            leakage_report=leakage_report,
            metrics=metrics,
            n_samples={"ALL": sum(len(df) for df in data.values())},
            metadata={
                "tier": self.tier,
                "config": config.name,
                "paper": config.paper,
                "n_assays_evaluated": len(rhos),
                "n_assays_total": len(data),
                "expected_baselines": config.expected_baselines,
            },
        )
