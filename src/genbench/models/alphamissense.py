"""AlphaMissense pre-computed score lookup.

71.7M human missense variants. On first load, builds a dict index
for O(1) lookup per variant. Takes ~10 min and ~5GB RAM on first call.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import numpy as np

from genbench.config import DATASETS_PATH
from genbench.registry import register_model

_INDEX: dict[tuple[str, int, str, str], float] | None = None


def _load_index() -> dict[tuple[str, int, str, str], float]:
    global _INDEX
    if _INDEX is not None:
        return _INDEX

    path = Path(DATASETS_PATH) / "baseline_scores" / "AlphaMissense_hg38.tsv.gz"
    if not path.exists():
        raise FileNotFoundError(f"AlphaMissense scores not found at {path}")

    print(f"Loading AlphaMissense index from {path} (this takes ~10 min on first run)...")
    index: dict[tuple[str, int, str, str], float] = {}

    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            # Columns: CHROM, POS, REF, ALT, genome, uniprot_id,
            #          transcript_id, protein_variant, am_pathogenicity, am_class
            try:
                chrom = parts[0]
                pos = int(parts[1])
                ref = parts[2]
                alt = parts[3]
                score = float(parts[8])
                index[(chrom, pos, ref, alt)] = score
            except (IndexError, ValueError):
                continue

    print(f"AlphaMissense index loaded: {len(index)} variants")
    _INDEX = index
    return _INDEX


@register_model("alphamissense")
class AlphaMissenseModel:
    """Pathogenicity scores for all 71.7M human missense variants.

    Unsupervised (no ClinVar label leakage). AUROC ~0.94 on ClinVar.
    Thresholds: likely_benign < 0.34, likely_pathogenic > 0.564.
    """

    name = "alphamissense"

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        index = _load_index()
        scores = np.full(len(inputs["chroms"]), np.nan)

        for i, (chrom, pos, ref, alt) in enumerate(
            zip(inputs["chroms"], inputs["positions"], inputs["refs"], inputs["alts"])
        ):
            key = (str(chrom), int(pos), str(ref), str(alt))
            if key in index:
                scores[i] = index[key]

        return scores
