"""AlphaMissense pre-computed score lookup."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from genbench.config import DATASETS_PATH
from genbench.registry import register_model

_CACHE: pd.DataFrame | None = None


def _load() -> pd.DataFrame:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    path = Path(DATASETS_PATH) / "baseline_scores" / "AlphaMissense_hg38.tsv.gz"
    if not path.exists():
        raise FileNotFoundError(f"AlphaMissense scores not found at {path}")
    _CACHE = pd.read_csv(path, sep="\t", comment="#", dtype={"#CHROM": str, "POS": int})
    return _CACHE


@register_model("alphamissense")
class AlphaMissenseModel:
    """Pathogenicity scores for all human missense variants. Unsupervised (no ClinVar circularity)."""

    name = "alphamissense"

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        df = _load()
        scores = []
        for chrom, pos, ref, alt in zip(
            inputs["chroms"], inputs["positions"], inputs["refs"], inputs["alts"]
        ):
            mask = (
                (df["#CHROM"] == str(chrom))
                & (df["POS"] == int(pos))
                & (df["REF"] == ref)
                & (df["ALT"] == alt)
            )
            matches = df.loc[mask, "am_pathogenicity"]
            scores.append(float(matches.iloc[0]) if len(matches) > 0 else np.nan)
        return np.array(scores)
