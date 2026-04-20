"""Anti-leakage checks for the benchmark harness.

Implements the anti-leakage protocol from docs/phenotype-suite/BENCHMARKS.md.
Every benchmark run must pass these checks before results are reported.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np

from genes.benchmarks.types import LeakageCheck, LeakageReport


def check_relatedness(
    train_ids: set[str],
    test_ids: set[str],
    kinship: dict[tuple[str, str], float],
    threshold: float = 0.0442,
) -> LeakageCheck:
    """Check that no individual in test has KING > threshold with any in train.

    Args:
        train_ids: Sample IDs in training set.
        test_ids: Sample IDs in test set.
        kinship: Dict mapping (id_a, id_b) -> KING kinship coefficient.
        threshold: KING threshold (0.0442 = 2nd degree relatives).

    Returns:
        LeakageCheck with pass/fail and details of violations.
    """
    violations = []
    for (a, b), k in kinship.items():
        if k > threshold:
            if (a in train_ids and b in test_ids) or (b in train_ids and a in test_ids):
                violations.append(f"{a}-{b} (KING={k:.4f})")
    return LeakageCheck(
        name="relatedness",
        passed=len(violations) == 0,
        details=f"{len(violations)} related pairs across splits" if violations else "clean",
    )


def check_temporal_integrity(
    test_variant_ids: set[str],
    cutoff_date: date,
    submission_dates: dict[str, date],
) -> LeakageCheck:
    """Check that no test variant's classification predates the cutoff.

    Args:
        test_variant_ids: Variant IDs in test set.
        cutoff_date: Train/test temporal boundary (e.g., 2024-01-01).
        submission_dates: Dict mapping variant_id -> ClinVar submission date.

    Returns:
        LeakageCheck with pass/fail and count of leaked variants.
    """
    leaked = [
        vid
        for vid in test_variant_ids
        if vid in submission_dates and submission_dates[vid] < cutoff_date
    ]
    return LeakageCheck(
        name="temporal_integrity",
        passed=len(leaked) == 0,
        details=f"{len(leaked)} test variants submitted before {cutoff_date}" if leaked else "clean",
    )


def check_gwas_leakage(
    gwas_cohort_ids: set[str],
    test_ids: set[str],
) -> LeakageCheck:
    """Check that test individuals are not in the GWAS cohort used for PRS weights.

    This is the most common leakage in UKB PRS benchmarks.
    """
    overlap = gwas_cohort_ids & test_ids
    return LeakageCheck(
        name="gwas_leakage",
        passed=len(overlap) == 0,
        details=f"{len(overlap)} test individuals found in GWAS cohort" if overlap else "clean",
    )


def check_chromosome_leakage(
    test_chromosomes: set[str],
    feature_chromosomes: set[str],
) -> LeakageCheck:
    """Check that no features from test chromosomes are used in training.

    For chromosome hold-out splits, verify no trans-chromosome features leak.
    """
    overlap = test_chromosomes & feature_chromosomes
    return LeakageCheck(
        name="chromosome_leakage",
        passed=len(overlap) == 0,
        details=f"Features from test chromosomes {overlap} used in training" if overlap else "clean",
    )


def check_batch_independence(
    test_ids: set[str],
    batch_assignments: dict[str, str],
) -> LeakageCheck:
    """Check that test individuals are not segregated by sequencing batch.

    For molecular phenotypes (GTEx, ENCODE), batch effects can confound results.
    """
    test_batches = {batch_assignments[sid] for sid in test_ids if sid in batch_assignments}
    all_batches = set(batch_assignments.values())
    if len(test_batches) == 1 and len(all_batches) > 1:
        return LeakageCheck(
            name="batch_independence",
            passed=False,
            details=f"All test samples from single batch: {test_batches.pop()}",
        )
    return LeakageCheck(name="batch_independence", passed=True, details="clean")


def run_all_checks(
    train_ids: set[str] | None = None,
    test_ids: set[str] | None = None,
    kinship: dict[tuple[str, str], float] | None = None,
    test_variant_ids: set[str] | None = None,
    cutoff_date: date | None = None,
    submission_dates: dict[str, date] | None = None,
    gwas_cohort_ids: set[str] | None = None,
    test_chromosomes: set[str] | None = None,
    feature_chromosomes: set[str] | None = None,
    batch_assignments: dict[str, str] | None = None,
) -> LeakageReport:
    """Run all applicable leakage checks and return aggregated report."""
    report = LeakageReport()

    if train_ids is not None and test_ids is not None and kinship is not None:
        report.checks["relatedness"] = check_relatedness(train_ids, test_ids, kinship)

    if test_variant_ids is not None and cutoff_date is not None and submission_dates is not None:
        report.checks["temporal_integrity"] = check_temporal_integrity(
            test_variant_ids, cutoff_date, submission_dates
        )

    if gwas_cohort_ids is not None and test_ids is not None:
        report.checks["gwas_leakage"] = check_gwas_leakage(gwas_cohort_ids, test_ids)

    if test_chromosomes is not None and feature_chromosomes is not None:
        report.checks["chromosome_leakage"] = check_chromosome_leakage(
            test_chromosomes, feature_chromosomes
        )

    if test_ids is not None and batch_assignments is not None:
        report.checks["batch_independence"] = check_batch_independence(
            test_ids, batch_assignments
        )

    return report
