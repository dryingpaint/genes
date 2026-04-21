"""Test AlphaMissense model returns exact scores from the actual TSV.

The fixture file contains 30 variants with scores extracted directly from
AlphaMissense_hg38.tsv.gz on Modal. These tests verify the model's lookup
logic is correct by checking it reproduces those exact values.

Tests marked @pytest.mark.modal require the AlphaMissense TSV on the Modal
volume. Run with: uv run modal run scripts/run_eval.py --eval clinvar --model alphamissense
"""

import csv
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
AM_VECTORS = FIXTURES / "alphamissense_test_vectors.csv"


def _load_vectors() -> list[dict]:
    with open(AM_VECTORS) as f:
        return list(csv.DictReader(f))


def _tsv_available() -> bool:
    """Check if AlphaMissense TSV is available locally (it usually isn't)."""
    from genbench.config import DATASETS_PATH
    path = Path(DATASETS_PATH) / "baseline_scores" / "AlphaMissense_hg38.tsv.gz"
    return path.exists()


class TestAlphaMissenseModelLocal:
    """Tests that run without the TSV — validate fixture and model interface."""

    def test_fixture_has_30_variants(self):
        assert len(_load_vectors()) == 30

    def test_fixture_scores_match_thresholds(self):
        """Every likely_pathogenic > 0.564, every likely_benign < 0.34."""
        for r in _load_vectors():
            score = float(r["am_pathogenicity"])
            if r["am_class"] == "likely_pathogenic":
                assert score > 0.564
            elif r["am_class"] == "likely_benign":
                assert score < 0.34

    def test_model_registers(self):
        from genbench.registry import get_model
        m = get_model("alphamissense")
        assert m.name == "alphamissense"

    def test_model_raises_without_tsv(self):
        """Model should raise FileNotFoundError when TSV is missing."""
        if _tsv_available():
            pytest.skip("TSV is available — this test is for environments without it")
        from genbench.registry import get_model
        m = get_model("alphamissense")
        with pytest.raises(FileNotFoundError):
            m.predict({"chroms": ["chr17"], "positions": [7669613], "refs": ["T"], "alts": ["G"]})


@pytest.mark.skipif(not _tsv_available(), reason="AlphaMissense TSV not available locally")
class TestAlphaMissenseModelWithTSV:
    """Tests that require the actual AlphaMissense TSV.

    These run on Modal or any environment where the TSV is at
    /data/datasets/baseline_scores/AlphaMissense_hg38.tsv.gz
    """

    def test_exact_scores_match_fixture(self):
        """Model must return the exact scores we extracted from the TSV."""
        from genbench.registry import get_model

        vectors = _load_vectors()
        m = get_model("alphamissense")
        predictions = m.predict({
            "chroms": [r["chrom"] for r in vectors],
            "positions": [int(r["pos"]) for r in vectors],
            "refs": [r["ref"] for r in vectors],
            "alts": [r["alt"] for r in vectors],
        })

        for i, r in enumerate(vectors):
            expected = float(r["am_pathogenicity"])
            actual = predictions[i]
            assert actual == pytest.approx(expected, abs=0.0001), (
                f"{r['protein_variant']} ({r['uniprot_id']}): "
                f"expected {expected}, got {actual}"
            )

    def test_auroc_on_fixture_is_perfect(self):
        """AUROC on our 30 cleanly-separated variants should be 1.0."""
        from sklearn.metrics import roc_auc_score
        from genbench.registry import get_model

        vectors = _load_vectors()
        m = get_model("alphamissense")
        scores = m.predict({
            "chroms": [r["chrom"] for r in vectors],
            "positions": [int(r["pos"]) for r in vectors],
            "refs": [r["ref"] for r in vectors],
            "alts": [r["alt"] for r in vectors],
        })
        labels = np.array([int(r["label"]) for r in vectors])
        assert roc_auc_score(labels, scores) == 1.0

    def test_missing_variant_returns_nan(self):
        """Variants not in the TSV should return NaN, not crash."""
        from genbench.registry import get_model

        m = get_model("alphamissense")
        scores = m.predict({
            "chroms": ["chrX"],
            "positions": [999999999],
            "refs": ["A"],
            "alts": ["T"],
        })
        assert np.isnan(scores[0])
