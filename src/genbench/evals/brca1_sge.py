"""BRCA1 SGE (saturation genome editing) eval.

Binary functional classification of ~3,893 BRCA1 SNVs using saturation
genome editing ground truth from Findlay et al. (Nature 2018). Each variant
is labeled functional or non-functional based on cell viability assays.

Configs:
  - lof_vs_func: Binary LOF vs FUNC (excludes intermediates). ~3,645 variants.
  - lof_vs_func_int: Binary LOF vs FUNC+INT (intermediates grouped with functional). ~3,893 variants.

BRCA1 coordinates: chr17:43,044,295-43,125,483 (GRCh38, negative strand).
The Findlay xlsx uses hg19 positions; we liftover to GRCh38 using the known
BRCA1 offset (hg19 chr17 → GRCh38 chr17 for this locus: pos_38 = pos_19 - 226,942).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from genbench.config import DATASETS_PATH
from genbench.eval import Eval
from genbench.eval_config import EvalConfig
from genbench.metrics.classification import auroc, balanced_accuracy
from genbench.registry import register_eval
from genbench.types import AncestryStratifiedMetric, MetricValue, SplitType

# BRCA1 hg19→GRCh38 liftover offset for chr17.
# hg19 chr17 is 81,195,210 bp; GRCh38 chr17 is 83,257,441 bp.
# For the BRCA1 locus specifically (chr17:41,196,312-41,277,500 in hg19,
# chr17:43,044,295-43,125,483 in GRCh38), the offset is +1,847,983.
_HG19_TO_HG38_BRCA1_OFFSET = 1_847_983


def _load_findlay_xlsx(base: Path) -> pd.DataFrame:
    """Parse the Findlay 2018 supplementary table xlsx.

    Expected columns (header at row 3, i.e. header=2):
      chromosome, position (hg19), reference, alt, function.score.mean, func.class
    """
    xlsx_path = base / "findlay2018_supp_table2.xlsx"
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"BRCA1 SGE data not found at {xlsx_path}. Run: modal run scripts/ingest_all.py"
        )

    df = pd.read_excel(xlsx_path, header=2, engine="openpyxl")

    # Normalize column names (handle slight variations)
    col_map = {}
    for col in df.columns:
        cl = str(col).lower().strip()
        if "chromosome" in cl or cl == "chr":
            col_map[col] = "chrom"
        elif "position" in cl and "hg19" in cl:
            col_map[col] = "pos_hg19"
        elif cl in ("position", "pos") and "pos_hg19" not in col_map.values():
            col_map[col] = "pos_hg19"
        elif cl in ("reference", "ref"):
            col_map[col] = "ref"
        elif cl == "alt" or cl == "alternate":
            col_map[col] = "alt"
        elif "function.score" in cl or "function_score" in cl:
            col_map[col] = "function_score"
        elif "func.class" in cl or "func_class" in cl or "classification" in cl:
            col_map[col] = "func_class"
    df = df.rename(columns=col_map)

    required = {"chrom", "pos_hg19", "ref", "alt", "func_class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in Findlay xlsx: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Drop rows without classification
    df = df.dropna(subset=["func_class"])

    # Liftover hg19 → GRCh38
    df["pos"] = df["pos_hg19"].astype(int) + _HG19_TO_HG38_BRCA1_OFFSET
    df["chrom"] = "chr17"

    # Standardize classification values
    df["func_class"] = df["func_class"].str.upper().str.strip()

    print(f"  Loaded {len(df)} BRCA1 SGE variants")
    print(f"  Classifications: {df['func_class'].value_counts().to_dict()}")

    return df


@register_eval("brca1_sge")
class Brca1SgeEval(Eval):
    name = "brca1_sge"
    description = "Binary functional classification of BRCA1 SNVs (SGE)"
    tier = 1
    split_type = SplitType.ZERO_SHOT
    default_baselines = ["evo2", "alphamissense", "null"]
    expected_ceiling = 0.97

    configs = {
        "lof_vs_func": EvalConfig(
            name="lof_vs_func",
            description=(
                "Binary: LOF vs FUNC only (intermediates excluded). "
                "Cleaner labels, ~3,645 variants."
            ),
            filters={"include_intermediate": False},
            split_type=SplitType.ZERO_SHOT,
            metrics=["auroc", "balanced_accuracy"],
            expected_baselines={
                "evo2": {"auroc": 0.95},
            },
            paper="Findlay et al. Nature 2018 / Brixi et al. Nature 2025 (Evo 2)",
            doi="10.1038/s41586-018-0461-z",
        ),
        "lof_vs_func_int": EvalConfig(
            name="lof_vs_func_int",
            description=(
                "Binary: LOF vs FUNC+INT (intermediates grouped with functional). "
                "All ~3,893 variants."
            ),
            filters={"include_intermediate": True},
            split_type=SplitType.ZERO_SHOT,
            metrics=["auroc", "balanced_accuracy"],
            expected_baselines={},
            paper="Findlay et al. Nature 2018",
            doi="10.1038/s41586-018-0461-z",
        ),
    }
    default_config = "lof_vs_func"

    def load_data(self, config: EvalConfig) -> pd.DataFrame:
        base = Path(DATASETS_PATH) / "brca1_sge"
        df = _load_findlay_xlsx(base)

        include_int = config.filters.get("include_intermediate", False)

        if include_int:
            # LOF=1 (pathogenic), FUNC+INT=0 (benign/functional)
            df["label"] = (df["func_class"] == "LOF").astype(int)
        else:
            # Exclude intermediates
            df = df[df["func_class"].isin(["LOF", "FUNC"])].copy()
            df["label"] = (df["func_class"] == "LOF").astype(int)

        print(f"  After filtering: {len(df)} variants (LOF={df['label'].sum()}, FUNC={len(df) - df['label'].sum()})")
        return df

    def get_splits(self, data: pd.DataFrame, config: EvalConfig) -> dict[str, pd.DataFrame]:
        # Zero-shot: entire dataset is test
        return {"test": data}

    def make_inputs(self, data: pd.DataFrame, split_data: pd.DataFrame) -> dict[str, Any]:
        """Prepare variant inputs for model.predict().

        Models like AlphaMissense expect chroms/positions/refs/alts.
        Models like Evo 2 expect reference_sequence + variants list.
        We provide both formats.
        """
        inputs: dict[str, Any] = {
            "chroms": split_data["chrom"].tolist(),
            "positions": split_data["pos"].tolist(),
            "refs": split_data["ref"].tolist(),
            "alts": split_data["alt"].tolist(),
            "n": len(split_data),
        }

        # For sequence-based models (Evo 2), also provide variant tuples
        if "pos_hg19" in split_data.columns:
            inputs["variants"] = list(zip(
                split_data["pos"].tolist(),
                split_data["ref"].tolist(),
                split_data["alt"].tolist(),
            ))

        return inputs

    def get_labels(self, data: pd.DataFrame, split_data: pd.DataFrame) -> np.ndarray:
        return split_data["label"].values

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, config: EvalConfig
    ) -> dict[str, AncestryStratifiedMetric]:
        # Handle NaN predictions (model doesn't cover variant)
        scored = ~np.isnan(y_pred)
        n_total = len(y_pred)
        n_scored = int(scored.sum())

        if n_scored == 0:
            nan_mv = MetricValue(estimate=float("nan"), ci_lower=0, ci_upper=0, n=0)
            return {"auroc": AncestryStratifiedMetric(aggregate=nan_mv)}

        yt = y_true[scored]
        yp = y_pred[scored]

        result: dict[str, AncestryStratifiedMetric] = {}

        if "auroc" in config.metrics:
            result["auroc"] = AncestryStratifiedMetric(
                aggregate=auroc(yt, yp, n_bootstrap=500)
            )

        if "balanced_accuracy" in config.metrics:
            # Threshold at 0.5 for binary classification
            yp_binary = (yp > 0.5).astype(int)
            result["balanced_accuracy"] = AncestryStratifiedMetric(
                aggregate=balanced_accuracy(yt, yp_binary, n_bootstrap=500)
            )

        # Report coverage
        result["coverage"] = AncestryStratifiedMetric(
            aggregate=MetricValue(
                estimate=n_scored / n_total if n_total > 0 else 0,
                ci_lower=0, ci_upper=0, n=n_total,
            )
        )

        return result
