"""SpliceAI pre-computed delta score lookup. Lazy per-chromosome loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from genbench.config import DATASETS_PATH
from genbench.registry import register_model

_CHROM_CACHE: dict[str, pd.DataFrame] = {}


def _load_chrom(chrom: str) -> pd.DataFrame:
    if chrom in _CHROM_CACHE:
        return _CHROM_CACHE[chrom]
    path = Path(DATASETS_PATH) / "baseline_scores" / "spliceai" / f"spliceai_scores.raw.snv.{chrom}.tsv.gz"
    if not path.exists():
        raise FileNotFoundError(f"SpliceAI scores not found for {chrom}")
    df = pd.read_csv(path, sep="\t", dtype={"CHROM": str})
    _CHROM_CACHE[chrom] = df
    return df


@register_model("spliceai")
class SpliceAiModel:
    """Max delta score across acceptor/donor gain/loss. Higher = more splice-disruptive."""

    name = "spliceai"

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        scores = []
        for chrom, pos, ref, alt in zip(
            inputs["chroms"], inputs["positions"], inputs["refs"], inputs["alts"]
        ):
            try:
                df = _load_chrom(str(chrom))
            except FileNotFoundError:
                scores.append(np.nan)
                continue
            mask = (
                (df["CHROM"] == str(chrom))
                & (df["POS"] == int(pos))
                & (df["REF"] == ref)
                & (df["ALT"] == alt)
            )
            matches = df[mask]
            if len(matches) > 0:
                row = matches.iloc[0]
                scores.append(float(max(
                    row.get("DS_AG", 0), row.get("DS_AL", 0),
                    row.get("DS_DG", 0), row.get("DS_DL", 0),
                )))
            else:
                scores.append(np.nan)
        return np.array(scores)
