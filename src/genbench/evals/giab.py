"""GIAB variant calling eval.

Germline SNV/indel accuracy evaluated against NIST Genome in a Bottle
truth sets. Compares a query VCF against GIAB truth VCFs within
high-confidence BED regions. Reports precision, recall, and F1 for
SNVs and indels separately.

Configs:
  - hg002_snv: HG002 (Ashkenazi Jewish son) SNV accuracy. DeepVariant expected F1 ~0.999.
  - hg002_indel: HG002 indel accuracy. DeepVariant expected F1 ~0.995.
  - all_samples: All 7 GIAB samples, SNV + indel combined.

Unlike score-based evals, GIAB compares variant *sets* (truth vs query VCF),
so it overrides evaluate() instead of using the standard predict→score flow.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from genbench.config import DATASETS_PATH
from genbench.eval import Eval
from genbench.eval_config import EvalConfig
from genbench.registry import register_eval
from genbench.types import (
    AncestryStratifiedMetric,
    BenchmarkResult,
    LeakageReport,
    MetricValue,
    SplitType,
)


# ---------------------------------------------------------------------------
# VCF parsing and comparison utilities
# ---------------------------------------------------------------------------


def _parse_vcf_variants(vcf_path: str) -> tuple[set[tuple], set[tuple]]:
    """Parse a VCF into SNV and indel variant sets.

    Returns:
        (snvs, indels) where each is a set of (chrom, pos, ref, alt) tuples.
        Only PASS (or unfiltered) variants are included.
    """
    import pysam

    snvs: set[tuple[str, int, str, str]] = set()
    indels: set[tuple[str, int, str, str]] = set()

    vcf = pysam.VariantFile(str(vcf_path))
    for rec in vcf:
        # Skip filtered variants (keep PASS and no-filter)
        filters = list(rec.filter.keys())
        if filters and "PASS" not in filters:
            continue

        chrom = rec.chrom
        # Normalize to chr* format
        if not chrom.startswith("chr"):
            chrom = f"chr{chrom}"

        for alt in rec.alts or []:
            if alt == "*" or alt == ".":
                continue
            key = (chrom, rec.pos, rec.ref, alt)
            if len(rec.ref) == 1 and len(alt) == 1:
                snvs.add(key)
            else:
                indels.add(key)

    vcf.close()
    return snvs, indels


def _load_bed_index(bed_path: str) -> dict[str, list[tuple[int, int]]]:
    """Load BED file into a dict of sorted interval lists per chromosome.

    Returns:
        {chrom: [(start, end), ...]} sorted by start position.
    """
    regions: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with open(bed_path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("track"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            if not chrom.startswith("chr"):
                chrom = f"chr{chrom}"
            regions[chrom].append((int(parts[1]), int(parts[2])))

    # Sort by start position for binary search
    for chrom in regions:
        regions[chrom].sort()
    return dict(regions)


def _in_bed_regions(
    chrom: str, pos: int, bed_index: dict[str, list[tuple[int, int]]]
) -> bool:
    """Check if a position falls within any BED region (0-based half-open)."""
    intervals = bed_index.get(chrom)
    if not intervals:
        return False

    # Binary search: find rightmost interval with start <= pos
    starts = [iv[0] for iv in intervals]
    idx = bisect.bisect_right(starts, pos) - 1
    if idx < 0:
        return False

    # Check a few intervals near the insertion point (handles overlapping regions)
    for i in range(max(0, idx - 1), min(len(intervals), idx + 2)):
        start, end = intervals[i]
        if start <= pos < end:
            return True
    return False


def _filter_to_bed(
    variants: set[tuple], bed_index: dict[str, list[tuple[int, int]]]
) -> set[tuple]:
    """Keep only variants within high-confidence BED regions."""
    return {
        v for v in variants
        if _in_bed_regions(v[0], v[1] - 1, bed_index)  # VCF is 1-based, BED is 0-based
    }


def _compute_variant_metrics(
    truth: set[tuple], query: set[tuple]
) -> dict[str, float]:
    """Compute precision, recall, F1 from truth and query variant sets."""
    tp = len(truth & query)
    fp = len(query - truth)
    fn = len(truth - query)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# GIAB sample data paths
# ---------------------------------------------------------------------------

GIAB_SAMPLES = ["HG001", "HG002", "HG003", "HG004", "HG005", "HG006", "HG007"]


def _giab_truth_paths(sample: str) -> dict[str, Path]:
    """Return paths to truth VCF and BED for a GIAB sample."""
    base = Path(DATASETS_PATH) / "giab" / sample
    return {
        "truth_vcf": base / f"{sample}_GRCh38_1_22_v4.2.1_benchmark.vcf.gz",
        "confidence_bed": base / f"{sample}_GRCh38_1_22_v4.2.1_benchmark.bed",
    }


# ---------------------------------------------------------------------------
# GIAB Eval
# ---------------------------------------------------------------------------


@register_eval("giab")
class GiabEval(Eval):
    name = "giab"
    description = "Germline SNV/indel accuracy on NIST truth sets"
    tier = 0
    split_type = SplitType.ZERO_SHOT
    default_baselines = ["deepvariant", "null"]
    expected_ceiling = 0.999  # SNV F1

    configs = {
        "hg002_snv": EvalConfig(
            name="hg002_snv",
            description="HG002 (Ashkenazi Jewish son) SNV calling accuracy",
            filters={"sample": "HG002", "variant_type": "snv"},
            split_type=SplitType.ZERO_SHOT,
            metrics=["snv_f1", "snv_precision", "snv_recall"],
            expected_baselines={"deepvariant": {"snv_f1": 0.999}},
            paper="Olson et al. Nat Biotechnol 2022 (GIAB v4.2.1)",
        ),
        "hg002_indel": EvalConfig(
            name="hg002_indel",
            description="HG002 indel calling accuracy",
            filters={"sample": "HG002", "variant_type": "indel"},
            split_type=SplitType.ZERO_SHOT,
            metrics=["indel_f1", "indel_precision", "indel_recall"],
            expected_baselines={"deepvariant": {"indel_f1": 0.995}},
            paper="Olson et al. Nat Biotechnol 2022 (GIAB v4.2.1)",
        ),
        "all_samples": EvalConfig(
            name="all_samples",
            description="All 7 GIAB samples, SNV + indel metrics",
            filters={"samples": GIAB_SAMPLES},
            split_type=SplitType.ZERO_SHOT,
            metrics=["snv_f1", "indel_f1"],
            expected_baselines={},
            paper="Olson et al. Nat Biotechnol 2022 (GIAB v4.2.1)",
        ),
    }
    default_config = "hg002_snv"

    def load_data(self, config: EvalConfig) -> dict[str, dict[str, Path]]:
        """Load truth VCF and BED paths for requested samples."""
        samples = config.filters.get("samples") or [config.filters.get("sample", "HG002")]
        data = {}
        for sample in samples:
            paths = _giab_truth_paths(sample)
            if not paths["truth_vcf"].exists():
                raise FileNotFoundError(
                    f"GIAB truth VCF not found for {sample} at {paths['truth_vcf']}. "
                    f"Run: modal run scripts/ingest_all.py"
                )
            data[sample] = paths
        return data

    def get_splits(self, data: Any, config: EvalConfig) -> dict[str, Any]:
        return {"test": data}

    def make_inputs(self, data: Any, split_data: Any) -> dict[str, Any]:
        """Return sample paths — the 'model' needs to provide query VCFs."""
        return {"samples": split_data}

    def get_labels(self, data: Any, split_data: Any) -> np.ndarray:
        # Not used — GIAB comparison is set-based, not label-based
        return np.array([])

    def score(
        self, y_true: np.ndarray, y_pred: np.ndarray, config: EvalConfig
    ) -> dict[str, AncestryStratifiedMetric]:
        # Not used — GIAB overrides evaluate()
        return {}

    def evaluate_vcf(
        self,
        query_vcf: str,
        sample: str = "HG002",
        config_name: str | None = None,
    ) -> BenchmarkResult:
        """Benchmark a query VCF against GIAB truth for a single sample.

        This is the primary entry point for GIAB benchmarking. Pass the path
        to a VCF produced by your variant caller (e.g., DeepVariant output).

        Args:
            query_vcf: Path to the query VCF (bgzipped + indexed preferred).
            sample: GIAB sample name (HG001-HG007).
            config_name: Which config to use (determines metrics reported).
        """
        if config_name is None:
            config_name = self.default_config
        config = self.get_config(config_name)

        # Load truth data
        paths = _giab_truth_paths(sample)
        if not paths["truth_vcf"].exists():
            raise FileNotFoundError(
                f"GIAB truth VCF not found for {sample}. Run: modal run scripts/ingest_all.py"
            )

        print(f"  Parsing truth VCF: {paths['truth_vcf']}")
        truth_snvs, truth_indels = _parse_vcf_variants(str(paths["truth_vcf"]))

        print(f"  Parsing query VCF: {query_vcf}")
        query_snvs, query_indels = _parse_vcf_variants(str(query_vcf))

        # Filter to high-confidence regions if BED available
        if paths["confidence_bed"].exists():
            print(f"  Loading confidence regions: {paths['confidence_bed']}")
            bed_index = _load_bed_index(str(paths["confidence_bed"]))
            truth_snvs = _filter_to_bed(truth_snvs, bed_index)
            truth_indels = _filter_to_bed(truth_indels, bed_index)
            query_snvs = _filter_to_bed(query_snvs, bed_index)
            query_indels = _filter_to_bed(query_indels, bed_index)
        else:
            print("  WARNING: No confidence BED file — evaluating all regions")

        print(f"  Truth: {len(truth_snvs)} SNVs, {len(truth_indels)} indels")
        print(f"  Query: {len(query_snvs)} SNVs, {len(query_indels)} indels")

        # Compute metrics
        snv_stats = _compute_variant_metrics(truth_snvs, query_snvs)
        indel_stats = _compute_variant_metrics(truth_indels, query_indels)

        print(f"  SNV:   P={snv_stats['precision']:.4f}  R={snv_stats['recall']:.4f}  F1={snv_stats['f1']:.4f}")
        print(f"  Indel: P={indel_stats['precision']:.4f}  R={indel_stats['recall']:.4f}  F1={indel_stats['f1']:.4f}")

        metrics: dict[str, AncestryStratifiedMetric] = {}

        def _mv(val: float, n: int) -> MetricValue:
            return MetricValue(estimate=val, ci_lower=0, ci_upper=0, n=n)

        n_snv = snv_stats["tp"] + snv_stats["fn"]
        n_indel = indel_stats["tp"] + indel_stats["fn"]

        metrics["snv_precision"] = AncestryStratifiedMetric(aggregate=_mv(snv_stats["precision"], n_snv))
        metrics["snv_recall"] = AncestryStratifiedMetric(aggregate=_mv(snv_stats["recall"], n_snv))
        metrics["snv_f1"] = AncestryStratifiedMetric(aggregate=_mv(snv_stats["f1"], n_snv))
        metrics["indel_precision"] = AncestryStratifiedMetric(aggregate=_mv(indel_stats["precision"], n_indel))
        metrics["indel_recall"] = AncestryStratifiedMetric(aggregate=_mv(indel_stats["recall"], n_indel))
        metrics["indel_f1"] = AncestryStratifiedMetric(aggregate=_mv(indel_stats["f1"], n_indel))

        # Validate against expected baselines
        from genbench.eval_config import validate_result

        for w in validate_result(metrics, config, "deepvariant"):
            print(f"  WARNING: {w}")

        task_id = f"{self.name}/{config.name}" if config.name != "default" else self.name

        return BenchmarkResult(
            task_id=task_id,
            model_name="query_vcf",
            split_type=config.split_type,
            leakage_report=LeakageReport(),
            metrics=metrics,
            n_samples={
                "snv_truth": int(n_snv),
                "snv_query": len(query_snvs),
                "indel_truth": int(n_indel),
                "indel_query": len(query_indels),
            },
            metadata={
                "tier": self.tier,
                "config": config.name,
                "sample": sample,
                "query_vcf": str(query_vcf),
                "paper": config.paper,
                "snv_tp": snv_stats["tp"],
                "snv_fp": snv_stats["fp"],
                "snv_fn": snv_stats["fn"],
                "indel_tp": indel_stats["tp"],
                "indel_fp": indel_stats["fp"],
                "indel_fn": indel_stats["fn"],
            },
        )

    def evaluate(self, model, config_name: str | None = None):
        """Override: GIAB requires a query VCF, not a score prediction.

        The model's predict() should return a dict with 'query_vcf' path(s),
        or use evaluate_vcf() directly for simpler usage.
        """
        config = self.get_config(config_name)
        sample = config.filters.get("sample", "HG002")

        # Try to get query VCF from model
        inputs = {"sample": sample, "n": 0}
        try:
            result = model.predict(inputs)
            if isinstance(result, dict) and "query_vcf" in result:
                return self.evaluate_vcf(result["query_vcf"], sample, config_name)
            elif isinstance(result, str):
                return self.evaluate_vcf(result, sample, config_name)
        except (TypeError, NotImplementedError):
            pass

        raise ValueError(
            f"GIAB eval requires a query VCF. Use evaluate_vcf() directly:\n"
            f"  ev = get_eval('giab')\n"
            f"  result = ev.evaluate_vcf('/path/to/query.vcf.gz', sample='{sample}')"
        )
