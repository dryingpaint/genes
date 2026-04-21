"""ClinVar Mendelian Variant Pathogenicity eval.

Configs:
  - alphamissense_balanced: Published AlphaMissense eval (Cheng et al. Science 2023).
    18,924 missense variants balanced per gene. Expected AUROC 0.940.
  - all_snv: All reviewed ClinVar SNVs with temporal split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from genbench.config import DATASETS_PATH, TEMPORAL_CUTOFF
from genbench.eval import Eval
from genbench.eval_config import EvalConfig
from genbench.metrics.classification import auprc, auroc, balanced_accuracy
from genbench.registry import register_eval
from genbench.splits.leakage import verify_no_leakage
from genbench.splits.temporal import make_temporal_split
from genbench.types import AncestryStratifiedMetric, LeakageReport, MetricValue, SplitType


@register_eval("clinvar")
class ClinVarEval(Eval):
    name = "clinvar"
    description = "Mendelian variant pathogenicity (ClinVar)"
    tier = 0
    split_type = SplitType.TEMPORAL
    default_baselines = ["alphamissense", "null"]
    expected_ceiling = 0.97

    default_config = "all_snv"

    configs = {
        "alphamissense_balanced": EvalConfig(
            name="alphamissense_balanced",
            description=(
                "AlphaMissense published eval: missense variants balanced per gene "
                "(genes with ≥5 P and ≥5 B). 612 genes, ~18,924 variants."
            ),
            filters={
                "variant_type": "missense",
                "balanced": True,
                "min_per_gene": 5,
                "min_review_stars": 2,  # multiple submitters or expert panel
            },
            split_type=SplitType.ZERO_SHOT,
            split_params={},
            metrics=["auroc"],
            score_on_non_nan_only=True,
            expected_baselines={
                "alphamissense": {"auroc": 0.940},
            },
            paper="Cheng et al. Science 2023",
            doi="10.1126/science.adg7492",
        ),
        "all_snv": EvalConfig(
            name="all_snv",
            description="All reviewed ClinVar SNVs, temporal split at 2023-12-31",
            filters={"variant_type": "snv"},
            split_type=SplitType.TEMPORAL,
            split_params={"cutoff": "2023-12-31"},
            metrics=["auroc", "auprc", "balanced_accuracy"],
            score_on_non_nan_only=True,
            expected_baselines={},
            paper="internal",
        ),
    }

    def _load_base(self) -> pd.DataFrame:
        """Load and do common filtering (GRCh38, reviewed, P/LP vs B/LB)."""
        path = Path(DATASETS_PATH) / "clinvar" / "variant_summary.txt.gz"
        if not path.exists():
            raise FileNotFoundError(f"ClinVar data not found at {path}")

        df = pd.read_csv(path, sep="\t", dtype={"Chromosome": str}, low_memory=False)
        df = df[df["Assembly"] == "GRCh38"]

        reviewed = {
            "criteria provided, single submitter",
            "criteria provided, multiple submitters, no conflicts",
            "reviewed by expert panel",
            "practice guideline",
        }
        df = df[df["ReviewStatus"].isin(reviewed)]
        df = df[df["ClinSigSimple"].isin([0, 1])]
        df["label"] = df["ClinSigSimple"].astype(int)

        df["submission_date"] = pd.to_datetime(df["LastEvaluated"], format="mixed", errors="coerce")
        df = df.dropna(subset=["submission_date"])

        return df

    def load_data(self, config: EvalConfig) -> pd.DataFrame:
        df = self._load_base()
        filters = config.filters

        # Filter by review quality (star count)
        min_stars = filters.get("min_review_stars")
        if min_stars and min_stars >= 2:
            # 2+ stars = multiple submitters, expert panel, or practice guideline
            high_quality = {
                "criteria provided, multiple submitters, no conflicts",
                "reviewed by expert panel",
                "practice guideline",
            }
            before = len(df)
            df = df[df["ReviewStatus"].isin(high_quality)]
            print(f"  Review filter (≥{min_stars} stars): {len(df)}/{before} variants")

        # Filter by variant type
        vtype = filters.get("variant_type", "snv")
        if vtype == "snv":
            df = df[df["Type"] == "single nucleotide variant"]
        elif vtype == "missense":
            df = df[df["Type"] == "single nucleotide variant"]
            df = self._filter_to_am_missense(df)

        # Balance per gene if requested
        if filters.get("balanced"):
            min_per_gene = filters.get("min_per_gene", 5)
            df = self._balance_per_gene(df, min_per_gene)

        return df

    def _filter_to_am_missense(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only variants present in the AlphaMissense TSV (i.e., true missense)."""
        from genbench.models.alphamissense import _load_index

        try:
            index = _load_index()
        except FileNotFoundError:
            print("  WARNING: AlphaMissense TSV not available, skipping missense filter")
            return df

        # Normalize ClinVar chroms to chr* format
        chroms = [f"chr{c}" if not str(c).startswith("chr") else str(c)
                  for c in df["Chromosome"]]

        mask = [
            (chrom, int(pos), ref, alt) in index
            for chrom, pos, ref, alt in zip(
                chroms, df["PositionVCF"], df["ReferenceAlleleVCF"], df["AlternateAlleleVCF"]
            )
        ]
        filtered = df[mask]
        print(f"  Missense filter: {len(filtered)}/{len(df)} variants have AlphaMissense scores")
        return filtered

    def _balance_per_gene(self, df: pd.DataFrame, min_per_gene: int) -> pd.DataFrame:
        """Subsample to equal P/B per gene, keeping only genes with enough of each."""
        rng = np.random.default_rng(42)
        balanced_rows = []

        for gene, group in df.groupby("GeneSymbol"):
            pathogenic = group[group["label"] == 1]
            benign = group[group["label"] == 0]

            if len(pathogenic) < min_per_gene or len(benign) < min_per_gene:
                continue

            n = min(len(pathogenic), len(benign))
            p_idx = rng.choice(len(pathogenic), size=n, replace=False)
            b_idx = rng.choice(len(benign), size=n, replace=False)
            balanced_rows.append(pathogenic.iloc[p_idx])
            balanced_rows.append(benign.iloc[b_idx])

        if not balanced_rows:
            raise ValueError("No genes have enough P and B variants for balancing")

        return pd.concat(balanced_rows).reset_index(drop=True)

    def get_splits(self, data: pd.DataFrame, config: EvalConfig) -> dict[str, pd.DataFrame]:
        if config.split_type == SplitType.TEMPORAL:
            cutoff = config.split_params.get("cutoff", TEMPORAL_CUTOFF)
            train, test = make_temporal_split(data, date_column="submission_date", cutoff=cutoff)
            return {"train": train, "test": test}
        else:
            # Zero-shot: entire dataset is test (e.g., alphamissense_balanced)
            return {"test": data}

    def make_inputs(self, data: pd.DataFrame, split_data: pd.DataFrame) -> dict[str, Any]:
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

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, config: EvalConfig
    ) -> dict[str, AncestryStratifiedMetric]:
        # Filter to scored variants (non-NaN) if config says so
        if config.score_on_non_nan_only:
            scored = ~np.isnan(y_pred)
            n_total = len(y_pred)
            n_scored = scored.sum()

            if n_scored == 0:
                nan_mv = MetricValue(estimate=float("nan"), ci_lower=0, ci_upper=0, n=0)
                return {"auroc": AncestryStratifiedMetric(aggregate=nan_mv)}

            y_true = y_true[scored]
            y_pred = y_pred[scored]
        else:
            n_total = n_scored = len(y_pred)
            y_pred = np.where(np.isnan(y_pred), 0.5, y_pred)

        result = {}

        if "auroc" in config.metrics:
            result["auroc"] = AncestryStratifiedMetric(
                aggregate=auroc(y_true, y_pred, n_bootstrap=500)
            )
        if "auprc" in config.metrics:
            result["auprc"] = AncestryStratifiedMetric(
                aggregate=auprc(y_true, y_pred, n_bootstrap=500)
            )
        if "balanced_accuracy" in config.metrics:
            result["balanced_accuracy"] = AncestryStratifiedMetric(
                aggregate=balanced_accuracy(y_true, (y_pred > 0.5).astype(int), n_bootstrap=500)
            )

        # Always report coverage
        result["coverage"] = AncestryStratifiedMetric(
            aggregate=MetricValue(
                estimate=n_scored / n_total if n_total > 0 else 0,
                ci_lower=0, ci_upper=0, n=n_total,
            )
        )

        return result

    def verify_leakage(self, splits: dict[str, Any], config: EvalConfig) -> LeakageReport:
        if config.split_type == SplitType.TEMPORAL and "train" in splits:
            return verify_no_leakage(
                ["temporal"],
                train_dates=splits["train"]["submission_date"],
                test_dates=splits["test"]["submission_date"],
                temporal_cutoff=config.split_params.get("cutoff", TEMPORAL_CUTOFF),
            )
        return LeakageReport()
