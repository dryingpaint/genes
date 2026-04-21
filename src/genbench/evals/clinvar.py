"""ClinVar Mendelian Variant Pathogenicity eval.

Tier 0 sanity check. Binary classification: pathogenic vs benign for known
disease-associated variants. Temporal split prevents circularity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from genbench.config import DATASETS_PATH, TEMPORAL_CUTOFF
from genbench.eval import Eval
from genbench.metrics.classification import auprc, auroc, balanced_accuracy
from genbench.registry import register_eval
from genbench.splits.leakage import verify_no_leakage
from genbench.splits.temporal import make_temporal_split
from genbench.types import AncestryStratifiedMetric, LeakageReport, SplitType


@register_eval("clinvar")
class ClinVarEval(Eval):
    name = "clinvar"
    description = "Mendelian variant pathogenicity (ClinVar 3-star+ P/LP/B/LB, temporal split)"
    tier = 0
    split_type = SplitType.TEMPORAL
    expected_ceiling = 0.97  # missense AUROC ceiling
    default_baselines = ["alphamissense", "null"]

    def load_data(self) -> pd.DataFrame:
        path = Path(DATASETS_PATH) / "clinvar" / "variant_summary.txt.gz"
        if not path.exists():
            raise FileNotFoundError(f"ClinVar data not found at {path}")

        df = pd.read_csv(path, sep="\t", dtype={"Chromosome": str}, low_memory=False)

        # GRCh38 only
        df = df[df["Assembly"] == "GRCh38"]

        # Reviewed submissions
        reviewed = {
            "criteria provided, single submitter",
            "criteria provided, multiple submitters, no conflicts",
            "reviewed by expert panel",
            "practice guideline",
        }
        df = df[df["ReviewStatus"].isin(reviewed)]

        # P/LP vs B/LB via ClinSigSimple (1 = pathogenic, 0 = benign)
        df = df[df["ClinSigSimple"].isin([0, 1])]
        df["label"] = df["ClinSigSimple"].astype(int)

        # Filter to SNVs (single nucleotide variants) — required for missense-focused
        # models like AlphaMissense. Indels, deletions, etc. are excluded.
        df = df[df["Type"] == "single nucleotide variant"]

        # Parse dates
        df["submission_date"] = pd.to_datetime(df["LastEvaluated"], format="mixed", errors="coerce")
        df = df.dropna(subset=["submission_date"])

        return df

    def get_splits(self, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        train, test = make_temporal_split(data, date_column="submission_date", cutoff=TEMPORAL_CUTOFF)
        return {"train": train, "test": test}

    def make_inputs(self, data: pd.DataFrame, split_data: pd.DataFrame) -> dict[str, Any]:
        # Normalize chromosomes to chr* format (ClinVar uses "7", most tools use "chr7")
        chroms = [f"chr{c}" if not str(c).startswith("chr") else str(c)
                  for c in split_data["Chromosome"]]
        return {
            "chroms": chroms,
            "positions": split_data["PositionVCF"].tolist(),
            "refs": split_data["ReferenceAlleleVCF"].tolist(),
            "alts": split_data["AlternateAlleleVCF"].tolist(),
            "n": len(split_data),
        }

    def get_labels(self, data: pd.DataFrame, split_data: pd.DataFrame) -> np.ndarray:
        return split_data["label"].values

    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, AncestryStratifiedMetric]:
        # Only score variants where the model returned a prediction.
        # Missense-only models (AlphaMissense) return NaN for non-missense variants —
        # we exclude those rather than penalizing the model for not covering them.
        scored_mask = ~np.isnan(y_pred)
        n_scored = scored_mask.sum()
        n_total = len(y_pred)

        if n_scored == 0:
            from genbench.types import MetricValue
            nan_mv = MetricValue(estimate=float("nan"), ci_lower=0, ci_upper=0, n=0)
            return {
                "auroc": AncestryStratifiedMetric(aggregate=nan_mv),
                "auprc": AncestryStratifiedMetric(aggregate=nan_mv),
                "coverage": AncestryStratifiedMetric(
                    aggregate=MetricValue(estimate=0.0, ci_lower=0, ci_upper=0, n=n_total)
                ),
            }

        yt = y_true[scored_mask]
        yp = y_pred[scored_mask]

        from genbench.types import MetricValue
        return {
            "auroc": AncestryStratifiedMetric(aggregate=auroc(yt, yp, n_bootstrap=500)),
            "auprc": AncestryStratifiedMetric(aggregate=auprc(yt, yp, n_bootstrap=500)),
            "balanced_accuracy": AncestryStratifiedMetric(
                aggregate=balanced_accuracy(yt, (yp > 0.5).astype(int), n_bootstrap=500)
            ),
            "coverage": AncestryStratifiedMetric(
                aggregate=MetricValue(
                    estimate=n_scored / n_total, ci_lower=0, ci_upper=0, n=n_total,
                )
            ),
        }

    def verify_leakage(self, splits: dict[str, Any]) -> LeakageReport:
        return verify_no_leakage(
            ["temporal"],
            train_dates=splits["train"]["submission_date"],
            test_dates=splits["test"]["submission_date"],
            temporal_cutoff=TEMPORAL_CUTOFF,
        )
