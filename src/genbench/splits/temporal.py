"""Temporal split for ClinVar-based benchmarks."""

from __future__ import annotations

import pandas as pd


def make_temporal_split(
    df: pd.DataFrame,
    date_column: str = "submission_date",
    cutoff: str = "2023-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe by date.

    Returns (train, test) where train contains rows with date <= cutoff
    and test contains rows with date > cutoff.
    """
    cutoff_date = pd.Timestamp(cutoff)
    dates = pd.to_datetime(df[date_column])

    train_mask = dates <= cutoff_date
    test_mask = dates > cutoff_date

    train = df[train_mask].copy()
    test = df[test_mask].copy()

    if len(test) == 0:
        raise ValueError(
            f"Temporal split produced empty test set. "
            f"All {len(df)} rows have {date_column} <= {cutoff}. "
            f"Date range: {dates.min()} to {dates.max()}"
        )

    return train, test
