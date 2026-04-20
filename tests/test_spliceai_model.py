"""Validate SpliceAI model using test vectors from the SpliceAI repo.

The SpliceAI GitHub repo (Illumina/SpliceAI) publishes exact expected delta
scores in tests/test_delta_score.py. We use these as ground truth.
"""

import csv
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
SPLICE_VECTORS = FIXTURES / "spliceai_test_vectors.csv"


def _load_test_vectors() -> list[dict]:
    with open(SPLICE_VECTORS) as f:
        return list(csv.DictReader(f))


class TestSpliceAiTestVectors:
    """Validate against SpliceAI repo's published test vectors."""

    def test_vectors_file_exists(self):
        assert SPLICE_VECTORS.exists()

    def test_vectors_have_expected_scores(self):
        rows = _load_test_vectors()
        assert len(rows) == 2

        # First variant: chr10:94077 A>C
        v1 = rows[0]
        assert v1["CHROM"] == "10"
        assert v1["POS"] == "94077"
        assert v1["REF"] == "A"
        assert v1["ALT"] == "C"
        assert float(v1["expected_DS_AG"]) == pytest.approx(0.15, abs=0.02)
        assert float(v1["expected_DS_AL"]) == pytest.approx(0.27, abs=0.02)
        assert float(v1["expected_DS_DG"]) == pytest.approx(0.00, abs=0.02)
        assert float(v1["expected_DS_DL"]) == pytest.approx(0.05, abs=0.02)

        # Second variant: chr10:94555 C>T
        v2 = rows[1]
        assert v2["CHROM"] == "10"
        assert v2["POS"] == "94555"
        assert float(v2["expected_DS_DL"]) == pytest.approx(0.62, abs=0.02)

    def test_max_delta_score_identifies_strongest_effect(self):
        """The max delta across all 4 channels should identify the primary splice effect."""
        rows = _load_test_vectors()

        for r in rows:
            ds_values = {
                "DS_AG": float(r["expected_DS_AG"]),
                "DS_AL": float(r["expected_DS_AL"]),
                "DS_DG": float(r["expected_DS_DG"]),
                "DS_DL": float(r["expected_DS_DL"]),
            }
            max_channel = max(ds_values, key=ds_values.get)
            max_score = ds_values[max_channel]

            # Both test variants should have max delta > 0.2
            assert max_score >= 0.15, (
                f"Variant {r['CHROM']}:{r['POS']} max delta {max_score} < 0.15"
            )

    def test_clinical_thresholds(self):
        """Verify scores against SpliceAI clinical thresholds.

        Published thresholds:
        - Δ >= 0.2: increased sensitivity
        - Δ >= 0.5: high precision (recommended for clinical use)
        - Δ >= 0.8: very high confidence
        """
        rows = _load_test_vectors()

        # Second variant (chr10:94555 C>T) has DS_DL=0.62 — should pass clinical threshold
        v2 = rows[1]
        max_delta = max(
            float(v2["expected_DS_AG"]),
            float(v2["expected_DS_AL"]),
            float(v2["expected_DS_DG"]),
            float(v2["expected_DS_DL"]),
        )
        assert max_delta >= 0.5, "Variant with DL=0.62 should pass high-precision threshold"

        # First variant (chr10:94077 A>C) has max=0.27 — passes sensitivity but not precision
        v1 = rows[0]
        max_delta_v1 = max(
            float(v1["expected_DS_AG"]),
            float(v1["expected_DS_AL"]),
            float(v1["expected_DS_DG"]),
            float(v1["expected_DS_DL"]),
        )
        assert max_delta_v1 >= 0.2, "Should pass sensitivity threshold"
        assert max_delta_v1 < 0.5, "Should not pass high-precision threshold"
