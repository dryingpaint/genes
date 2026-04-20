"""GPN-MSA pre-computed score lookup. SOTA on non-coding variant effect prediction."""

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
    path = Path(DATASETS_PATH) / "baseline_scores" / "gpn_msa_scores.tsv.gz"
    if not path.exists():
        raise FileNotFoundError(f"GPN-MSA scores not found at {path}")
    _CACHE = pd.read_csv(path, sep="\t", dtype={"chrom": str})
    return _CACHE


@register_model("gpn_msa")
class GpnMsaModel:
    """Log-likelihood ratio scores from 100-way vertebrate MSA. Higher = more deleterious."""

    name = "gpn_msa"

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        df = _load()
        scores = []
        for chrom, pos, ref, alt in zip(
            inputs["chroms"], inputs["positions"], inputs["refs"], inputs["alts"]
        ):
            mask = (
                (df["chrom"] == str(chrom))
                & (df["pos"] == int(pos))
                & (df["ref"] == ref)
                & (df["alt"] == alt)
            )
            matches = df.loc[mask, "llr"]
            scores.append(float(matches.iloc[0]) if len(matches) > 0 else np.nan)
        return np.array(scores)
