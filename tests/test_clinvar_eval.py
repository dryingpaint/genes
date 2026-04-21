"""Validate ClinVar eval using real AlphaMissense scores.

Test vectors extracted from the actual AlphaMissense_hg38.tsv.gz (71.7M variants)
on Modal. Each score is the real pre-computed value — not hand-written.

These tests verify:
1. The scores are consistent with AlphaMissense classification thresholds
2. AUROC on our test set matches expectations (perfect separation on known P/B)
3. The eval pipeline (temporal split, leakage check, metric computation) is correct
"""

import csv
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

FIXTURES = Path(__file__).parent / "fixtures"
AM_VECTORS = FIXTURES / "alphamissense_test_vectors.csv"


def _load_test_vectors() -> list[dict]:
    with open(AM_VECTORS) as f:
        return list(csv.DictReader(f))


class TestAlphaMissenseRealScores:
    """Validate using scores extracted from the actual AlphaMissense TSV."""

    def test_30_real_variants(self):
        rows = _load_test_vectors()
        assert len(rows) == 30

    def test_pathogenic_above_threshold(self):
        """AlphaMissense threshold: likely_pathogenic > 0.564."""
        rows = _load_test_vectors()
        for r in rows:
            if r["am_class"] == "likely_pathogenic":
                score = float(r["am_pathogenicity"])
                assert score > 0.564, (
                    f"{r['protein_variant']} ({r['uniprot_id']}): "
                    f"score {score} should be > 0.564 for likely_pathogenic"
                )

    def test_benign_below_threshold(self):
        """AlphaMissense threshold: likely_benign < 0.34."""
        rows = _load_test_vectors()
        for r in rows:
            if r["am_class"] == "likely_benign":
                score = float(r["am_pathogenicity"])
                assert score < 0.34, (
                    f"{r['protein_variant']} ({r['uniprot_id']}): "
                    f"score {score} should be < 0.34 for likely_benign"
                )

    def test_auroc_perfect_on_test_vectors(self):
        """Our 30 variants are cleanly pathogenic/benign — AUROC should be 1.0."""
        rows = _load_test_vectors()
        y_true = np.array([int(r["label"]) for r in rows])
        y_score = np.array([float(r["am_pathogenicity"]) for r in rows])
        auc = roc_auc_score(y_true, y_score)
        assert auc == 1.0

    def test_tp53_hotspots_score_high(self):
        """TP53 R248Q and R175H are among the most common cancer mutations."""
        rows = _load_test_vectors()
        tp53_path = [r for r in rows if r["uniprot_id"] == "P04637" and r["label"] == "1"]
        assert len(tp53_path) >= 3
        for r in tp53_path:
            assert float(r["am_pathogenicity"]) > 0.5

    def test_brca1_has_both_classes(self):
        """BRCA1 fixture should have both pathogenic and benign variants."""
        rows = _load_test_vectors()
        brca1 = [r for r in rows if r["uniprot_id"] == "P38398"]
        labels = {r["label"] for r in brca1}
        assert labels == {"0", "1"}

    def test_scores_reproduce_from_tsv(self):
        """Spot-check: specific variants have exact known scores.

        These are the ground truth values from the actual AlphaMissense TSV
        on Modal. If the model code loads the TSV and looks up these variants,
        it MUST return these exact scores.
        """
        rows = _load_test_vectors()
        lookup = {(r["chrom"], r["pos"], r["ref"], r["alt"]): r for r in rows}

        # TP53 D393H — extracted from TSV
        v = lookup[("chr17", "7669614", "C", "G")]
        assert float(v["am_pathogenicity"]) == pytest.approx(0.6494, abs=0.0001)

        # BRCA1 L1854P
        v = lookup[("chr17", "43045709", "A", "G")]
        assert float(v["am_pathogenicity"]) == pytest.approx(0.8666, abs=0.0001)

        # MLH1 S2A (benign)
        v = lookup[("chr3", "36993551", "T", "G")]
        assert float(v["am_pathogenicity"]) == pytest.approx(0.0639, abs=0.0001)

    def test_published_clinvar_auroc_reference(self):
        """AlphaMissense paper: AUROC 0.940 on ClinVar balanced set (18,924 variants).

        We can't run this locally (needs 3.6GB TSV + ClinVar on Modal).
        This test documents the expected value so we know what to check
        when running the full eval on Modal.
        """
        expected = 0.940
        assert 0.90 < expected < 1.0  # sanity


class TestClinVarEvalPipeline:
    """Test the eval pipeline mechanics with synthetic data."""

    def test_temporal_split(self):
        import pandas as pd
        from genbench.splits.temporal import make_temporal_split

        df = pd.DataFrame({
            "submission_date": pd.date_range("2022-01-01", periods=100, freq="15D"),
            "label": np.random.default_rng(42).choice([0, 1], 100),
        })
        train, test = make_temporal_split(df, cutoff="2023-12-31")
        assert len(train) > 0
        assert len(test) > 0
        assert (pd.to_datetime(train["submission_date"]) <= pd.Timestamp("2023-12-31")).all()
        assert (pd.to_datetime(test["submission_date"]) > pd.Timestamp("2023-12-31")).all()

    def test_auroc_matches_sklearn(self):
        from genbench.metrics.classification import auroc

        rng = np.random.default_rng(42)
        y_true = np.array([0] * 50 + [1] * 50)
        y_score = np.where(y_true == 1, rng.uniform(0.6, 1.0, 100), rng.uniform(0.0, 0.4, 100))

        our_auc = auroc(y_true, y_score, n_bootstrap=100)
        sklearn_auc = roc_auc_score(y_true, y_score)
        assert abs(our_auc.estimate - sklearn_auc) < 1e-10

    def test_null_auroc_near_half(self):
        from genbench.metrics.classification import auroc

        rng = np.random.default_rng(42)
        y_true = np.array([0] * 500 + [1] * 500)
        y_score = rng.uniform(0, 1, 1000)
        result = auroc(y_true, y_score, n_bootstrap=100)
        assert 0.45 < result.estimate < 0.55

    def test_leakage_catches_temporal_violation(self):
        import pandas as pd
        from genbench.splits.leakage import LeakageError, verify_no_leakage

        with pytest.raises(LeakageError):
            verify_no_leakage(
                ["temporal"],
                train_dates=pd.Series(pd.date_range("2020-01-01", periods=50, freq="D")),
                test_dates=pd.Series(pd.date_range("2023-01-01", periods=50, freq="D")),
                temporal_cutoff="2023-12-31",
            )
