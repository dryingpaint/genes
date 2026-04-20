"""Leave-N-out cross-validation for model organism benchmarks."""

from __future__ import annotations

import numpy as np


def make_leave_n_out_folds(
    sample_ids: list[str],
    n_out: int,
    n_repeats: int = 10,
    stratify_by: np.ndarray | None = None,
    seed: int = 42,
) -> list[tuple[list[str], list[str]]]:
    """Generate repeated leave-N-out CV folds.

    Args:
        sample_ids: Identifiers for each sample/line/accession.
        n_out: Number of samples to hold out per fold.
        n_repeats: Number of random repeats.
        stratify_by: Optional group labels for proportional representation
            in each fold (e.g., genetic subpopulation).
        seed: Random seed for reproducibility.

    Returns:
        List of (train_ids, test_ids) tuples.
    """
    rng = np.random.default_rng(seed)
    ids = np.array(sample_ids)
    n = len(ids)

    if n_out >= n:
        raise ValueError(f"n_out ({n_out}) must be less than n_samples ({n})")

    folds = []
    for _ in range(n_repeats):
        if stratify_by is not None:
            # Proportional stratified sampling
            test_indices = _stratified_sample(stratify_by, n_out, rng)
        else:
            test_indices = rng.choice(n, size=n_out, replace=False)

        train_mask = np.ones(n, dtype=bool)
        train_mask[test_indices] = False

        train_ids = ids[train_mask].tolist()
        test_ids = ids[test_indices].tolist()
        folds.append((train_ids, test_ids))

    return folds


def _stratified_sample(
    groups: np.ndarray,
    n_out: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample n_out indices with proportional representation per group."""
    unique_groups, counts = np.unique(groups, return_counts=True)
    proportions = counts / counts.sum()
    per_group = np.round(proportions * n_out).astype(int)

    # Adjust rounding to hit exact n_out
    diff = n_out - per_group.sum()
    if diff > 0:
        # Add to largest groups first
        order = np.argsort(-per_group)
        for i in range(diff):
            per_group[order[i % len(order)]] += 1
    elif diff < 0:
        order = np.argsort(per_group)
        for i in range(-diff):
            idx = order[i % len(order)]
            if per_group[idx] > 0:
                per_group[idx] -= 1

    selected = []
    for group, n_select in zip(unique_groups, per_group):
        group_indices = np.where(groups == group)[0]
        chosen = rng.choice(group_indices, size=min(n_select, len(group_indices)), replace=False)
        selected.extend(chosen)

    return np.array(selected)
