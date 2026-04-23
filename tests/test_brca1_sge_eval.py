"""Tests for the BRCA1 SGE eval.

Tests eval registration and scoring logic locally.
Full BRCA1 SGE benchmarks require Modal + ingested data.
"""

import numpy as np
import pytest

from genbench.evals.brca1_sge import Brca1SgeEval, _HG19_TO_HG38_BRCA1_OFFSET
from genbench.metrics.classification import auroc
from genbench.registry import get_eval
from genbench.types import AncestryStratifiedMetric, MetricValue


class TestBrca1SgeRegistration:
    def test_registered(self):
        ev = get_eval("brca1_sge")
        assert isinstance(ev, Brca1SgeEval)

    def test_configs(self):
        ev = get_eval("brca1_sge")
        configs = ev.list_configs()
        assert "lof_vs_func" in configs
        assert "lof_vs_func_int" in configs

    def test_default_config(self):
        ev = get_eval("brca1_sge")
        assert ev.default_config == "lof_vs_func"

    def test_tier_1(self):
        ev = get_eval("brca1_sge")
        assert ev.tier == 1

    def test_expected_baselines(self):
        ev = get_eval("brca1_sge")
        config = ev.get_config("lof_vs_func")
        assert "evo2" in config.expected_baselines
        assert config.expected_baselines["evo2"]["auroc"] == 0.95

    def test_load_data_raises_without_data(self):
        ev = get_eval("brca1_sge")
        config = ev.get_config("lof_vs_func")
        with pytest.raises(FileNotFoundError, match="BRCA1 SGE data not found"):
            ev.load_data(config)


class TestBrca1SgeScoring:
    """Test the score() method with synthetic data."""

    def test_perfect_auroc(self):
        ev = get_eval("brca1_sge")
        config = ev.get_config("lof_vs_func")

        # Perfect predictions: LOF=1.0, FUNC=0.0
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])

        result = ev.score(y_true, y_pred, config)
        assert "auroc" in result
        assert result["auroc"].aggregate.estimate == pytest.approx(1.0, abs=0.01)

    def test_random_auroc(self):
        ev = get_eval("brca1_sge")
        config = ev.get_config("lof_vs_func")

        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200)
        y_pred = rng.uniform(0, 1, size=200)

        result = ev.score(y_true, y_pred, config)
        assert "auroc" in result
        # Random predictions should be around 0.5
        assert 0.3 < result["auroc"].aggregate.estimate < 0.7

    def test_nan_handling(self):
        ev = get_eval("brca1_sge")
        config = ev.get_config("lof_vs_func")

        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([0.9, np.nan, 0.1, 0.2, np.nan, 0.3])

        result = ev.score(y_true, y_pred, config)
        assert "coverage" in result
        assert result["coverage"].aggregate.estimate == pytest.approx(4 / 6)

    def test_all_nan(self):
        ev = get_eval("brca1_sge")
        config = ev.get_config("lof_vs_func")

        y_true = np.array([1, 0])
        y_pred = np.array([np.nan, np.nan])

        result = ev.score(y_true, y_pred, config)
        assert "auroc" in result
        assert np.isnan(result["auroc"].aggregate.estimate)


class TestLiftover:
    def test_offset_positive(self):
        """BRCA1 GRCh38 positions are larger than hg19 positions."""
        assert _HG19_TO_HG38_BRCA1_OFFSET > 0

    def test_brca1_locus_in_range(self):
        """Verify the liftover puts BRCA1 in the expected GRCh38 range."""
        # BRCA1 hg19: chr17:41,196,312-41,277,500
        # BRCA1 GRCh38: chr17:43,044,295-43,125,483
        hg19_start = 41_196_312
        hg38_start = hg19_start + _HG19_TO_HG38_BRCA1_OFFSET
        assert 43_000_000 < hg38_start < 43_200_000
