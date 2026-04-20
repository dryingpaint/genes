"""Chromosome hold-out split for molecular phenotype benchmarks."""

from __future__ import annotations

import pandas as pd

from genbench.config import HOLDOUT_CHROMOSOMES


def make_chromosome_split(
    df: pd.DataFrame,
    chrom_column: str = "chrom",
    holdout: tuple[str, ...] = HOLDOUT_CHROMOSOMES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe by chromosome.

    Holds out specified chromosomes for testing. All other chromosomes go to training.

    Returns (train, test).
    """
    # Normalize chromosome names (handle both "chr8" and "8")
    chroms = df[chrom_column].astype(str)
    holdout_normalized = set()
    for c in holdout:
        holdout_normalized.add(c)
        holdout_normalized.add(c.replace("chr", ""))
        holdout_normalized.add(f"chr{c}" if not c.startswith("chr") else c)

    test_mask = chroms.isin(holdout_normalized)
    train = df[~test_mask].copy()
    test = df[test_mask].copy()

    if len(test) == 0:
        raise ValueError(
            f"Chromosome split produced empty test set. "
            f"Holdout chroms {holdout} not found in column '{chrom_column}'. "
            f"Unique values: {chroms.unique()[:20]}"
        )

    return train, test
