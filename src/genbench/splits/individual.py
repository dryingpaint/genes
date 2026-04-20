"""Individual hold-out split with KING relatedness filtering.

Used for GTEx (Phase 2) and UKB (Phase 3). Implemented now, integration-tested later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from genbench.config import KING_THRESHOLD


def make_individual_split(
    sample_ids: list[str],
    ancestry_labels: dict[str, str],
    kinship_matrix: pd.DataFrame | None = None,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    king_threshold: float = KING_THRESHOLD,
    seed: int = 42,
) -> dict[str, dict[str, list[str]]]:
    """Split individuals into train/val/test, stratified by ancestry, with relatedness filtering.

    Args:
        sample_ids: List of sample identifiers.
        ancestry_labels: Mapping of sample_id -> ancestry group (from PCA, not self-report).
        kinship_matrix: Optional KING kinship matrix as a DataFrame indexed by sample_id.
            If provided, no pair across train/test splits will have kinship > king_threshold.
        ratios: (train, val, test) proportions. Must sum to 1.
        king_threshold: KING kinship threshold for relatedness filtering.
        seed: Random seed.

    Returns:
        Nested dict: {ancestry: {"train": [...], "val": [...], "test": [...]}}
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")

    rng = np.random.default_rng(seed)
    result: dict[str, dict[str, list[str]]] = {}

    # Group by ancestry
    ancestry_groups: dict[str, list[str]] = {}
    for sid in sample_ids:
        anc = ancestry_labels.get(sid, "UNKNOWN")
        ancestry_groups.setdefault(anc, []).append(sid)

    for ancestry, ids in ancestry_groups.items():
        ids_arr = np.array(ids)
        rng.shuffle(ids_arr)

        n = len(ids_arr)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])

        train_ids = ids_arr[:n_train].tolist()
        val_ids = ids_arr[n_train : n_train + n_val].tolist()
        test_ids = ids_arr[n_train + n_val :].tolist()

        # Remove related individuals across splits
        if kinship_matrix is not None:
            train_ids, val_ids, test_ids = _remove_related(
                train_ids, val_ids, test_ids, kinship_matrix, king_threshold
            )

        result[ancestry] = {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        }

    return result


def _remove_related(
    train: list[str],
    val: list[str],
    test: list[str],
    kinship: pd.DataFrame,
    threshold: float,
) -> tuple[list[str], list[str], list[str]]:
    """Remove one member of each related pair that crosses split boundaries.

    When a related pair spans splits, the individual in the smaller split is moved
    to the training set (largest split) to preserve test/val integrity.
    """
    test_set = set(test)
    val_set = set(val)
    train_set = set(train)

    # Check test vs train/val
    to_remove_from_test = set()
    for t_id in list(test_set):
        if t_id not in kinship.index:
            continue
        for other in list(train_set | val_set):
            if other not in kinship.columns:
                continue
            if kinship.loc[t_id, other] > threshold:
                to_remove_from_test.add(t_id)
                break

    test_set -= to_remove_from_test
    train_set |= to_remove_from_test

    # Check val vs train
    to_remove_from_val = set()
    for v_id in list(val_set):
        if v_id not in kinship.index:
            continue
        for other in list(train_set):
            if other not in kinship.columns:
                continue
            if kinship.loc[v_id, other] > threshold:
                to_remove_from_val.add(v_id)
                break

    val_set -= to_remove_from_val
    train_set |= to_remove_from_val

    return list(train_set), list(val_set), list(test_set)
