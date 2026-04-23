"""Tests for the GIAB variant calling eval.

Tests the VCF comparison logic locally using small synthetic VCFs.
Full GIAB benchmarks require Modal + ingested data (tested via scripts/benchmark_pipeline.py).
"""

import gzip
import tempfile
from pathlib import Path

import numpy as np
import pytest

from genbench.evals.giab import (
    GiabEval,
    _compute_variant_metrics,
    _filter_to_bed,
    _in_bed_regions,
    _load_bed_index,
)
from genbench.registry import get_eval


class TestVariantMetrics:
    def test_perfect_match(self):
        truth = {("chr1", 100, "A", "G"), ("chr1", 200, "C", "T")}
        query = {("chr1", 100, "A", "G"), ("chr1", 200, "C", "T")}
        m = _compute_variant_metrics(truth, query)
        assert m["tp"] == 2
        assert m["fp"] == 0
        assert m["fn"] == 0
        assert m["f1"] == 1.0

    def test_partial_match(self):
        truth = {("chr1", 100, "A", "G"), ("chr1", 200, "C", "T"), ("chr1", 300, "G", "A")}
        query = {("chr1", 100, "A", "G"), ("chr1", 200, "C", "T"), ("chr1", 400, "T", "C")}
        m = _compute_variant_metrics(truth, query)
        assert m["tp"] == 2
        assert m["fp"] == 1  # chr1:400 not in truth
        assert m["fn"] == 1  # chr1:300 not in query
        assert m["precision"] == pytest.approx(2 / 3)
        assert m["recall"] == pytest.approx(2 / 3)

    def test_empty_query(self):
        truth = {("chr1", 100, "A", "G")}
        query = set()
        m = _compute_variant_metrics(truth, query)
        assert m["tp"] == 0
        assert m["fn"] == 1
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0

    def test_empty_truth(self):
        truth = set()
        query = {("chr1", 100, "A", "G")}
        m = _compute_variant_metrics(truth, query)
        assert m["tp"] == 0
        assert m["fp"] == 1
        assert m["precision"] == 0.0
        assert m["f1"] == 0.0

    def test_both_empty(self):
        m = _compute_variant_metrics(set(), set())
        assert m["f1"] == 0.0


class TestBedFiltering:
    def test_load_bed_index(self, tmp_path):
        bed = tmp_path / "test.bed"
        bed.write_text("chr1\t100\t200\nchr1\t300\t400\nchr2\t50\t150\n")
        index = _load_bed_index(str(bed))
        assert "chr1" in index
        assert "chr2" in index
        assert len(index["chr1"]) == 2
        assert len(index["chr2"]) == 1

    def test_in_bed_regions(self, tmp_path):
        bed = tmp_path / "test.bed"
        bed.write_text("chr1\t100\t200\nchr1\t300\t400\n")
        index = _load_bed_index(str(bed))

        # Inside first region
        assert _in_bed_regions("chr1", 150, index) is True
        # Inside second region
        assert _in_bed_regions("chr1", 350, index) is True
        # Outside both regions
        assert _in_bed_regions("chr1", 250, index) is False
        # At boundary (BED is 0-based half-open: [100, 200))
        assert _in_bed_regions("chr1", 100, index) is True
        assert _in_bed_regions("chr1", 199, index) is True
        # Different chrom
        assert _in_bed_regions("chr3", 150, index) is False

    def test_filter_to_bed(self, tmp_path):
        bed = tmp_path / "test.bed"
        bed.write_text("chr1\t100\t200\n")
        index = _load_bed_index(str(bed))

        variants = {
            ("chr1", 151, "A", "G"),  # pos 151 is 1-based VCF → 0-based 150 → inside [100,200)
            ("chr1", 250, "C", "T"),  # outside
            ("chr1", 101, "G", "A"),  # 1-based 101 → 0-based 100 → inside
        }
        filtered = _filter_to_bed(variants, index)
        assert len(filtered) == 2
        assert ("chr1", 250, "C", "T") not in filtered


class TestGiabEvalRegistration:
    def test_registered(self):
        ev = get_eval("giab")
        assert isinstance(ev, GiabEval)

    def test_configs(self):
        ev = get_eval("giab")
        configs = ev.list_configs()
        assert "hg002_snv" in configs
        assert "hg002_indel" in configs
        assert "all_samples" in configs

    def test_tier_0(self):
        ev = get_eval("giab")
        assert ev.tier == 0

    def test_load_data_raises_without_data(self):
        ev = get_eval("giab")
        config = ev.get_config("hg002_snv")
        with pytest.raises(FileNotFoundError, match="GIAB truth VCF not found"):
            ev.load_data(config)

    def test_is_available_false_without_data(self):
        ev = get_eval("giab")
        assert ev.is_available() is False
