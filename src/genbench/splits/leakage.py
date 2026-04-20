"""Anti-leakage verification.

Every benchmark must call verify_no_leakage() and attach the LeakageReport
to its BenchmarkResult. A single failed check blocks reporting.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from genbench.config import KING_THRESHOLD
from genbench.types import LeakageCheck, LeakageReport


def verify_no_leakage(
    checks: list[str],
    *,
    # Data for specific checks — pass only what's needed
    train_ids: list[str] | None = None,
    test_ids: list[str] | None = None,
    kinship_matrix: pd.DataFrame | None = None,
    king_threshold: float = KING_THRESHOLD,
    train_dates: pd.Series | None = None,
    test_dates: pd.Series | None = None,
    temporal_cutoff: str | None = None,
    train_labels: set | None = None,
    test_labels: set | None = None,
    train_chroms: set[str] | None = None,
    test_chroms: set[str] | None = None,
    ancestry_source: str | None = None,
    gwas_sample_ids: set[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> LeakageReport:
    """Run specified anti-leakage checks and return a report.

    Args:
        checks: List of check names to run. Valid names:
            "relatedness", "temporal", "chromosome", "label",
            "population", "summary_stat"

    Returns:
        LeakageReport with per-check results.

    Raises:
        LeakageError if any check fails (hard-fail mode).
    """
    report = LeakageReport()

    dispatch = {
        "relatedness": lambda: _check_relatedness(
            train_ids, test_ids, kinship_matrix, king_threshold
        ),
        "temporal": lambda: _check_temporal(train_dates, test_dates, temporal_cutoff),
        "chromosome": lambda: _check_chromosome(train_chroms, test_chroms),
        "label": lambda: _check_label_leakage(train_labels, test_labels),
        "population": lambda: _check_population_source(ancestry_source),
        "summary_stat": lambda: _check_summary_stat(test_ids, gwas_sample_ids),
    }

    for check_name in checks:
        if check_name not in dispatch:
            report.checks[check_name] = LeakageCheck(
                name=check_name,
                passed=False,
                details=f"Unknown check: {check_name}",
            )
            continue
        report.checks[check_name] = dispatch[check_name]()

    if not report.all_passed:
        failed = [c.name for c in report.checks.values() if not c.passed]
        raise LeakageError(
            f"Anti-leakage checks failed: {failed}. "
            f"Results from this benchmark run will not be reported."
        )

    return report


class LeakageError(Exception):
    """Raised when an anti-leakage check fails."""

    def __init__(self, message: str, report: LeakageReport | None = None):
        super().__init__(message)
        self.report = report


# --- Individual checks ---


def _check_relatedness(
    train_ids: list[str] | None,
    test_ids: list[str] | None,
    kinship: pd.DataFrame | None,
    threshold: float,
) -> LeakageCheck:
    if train_ids is None or test_ids is None:
        return LeakageCheck(
            name="relatedness",
            passed=True,
            details="Skipped: no individual IDs provided (variant-level benchmark)",
        )

    if kinship is None:
        return LeakageCheck(
            name="relatedness",
            passed=False,
            details="Kinship matrix required for relatedness check but not provided",
        )

    violations = []
    test_set = set(test_ids)
    for t_id in test_set:
        if t_id not in kinship.index:
            continue
        for tr_id in train_ids:
            if tr_id not in kinship.columns:
                continue
            k = kinship.loc[t_id, tr_id]
            if k > threshold:
                violations.append((t_id, tr_id, k))

    if violations:
        return LeakageCheck(
            name="relatedness",
            passed=False,
            details=f"{len(violations)} related pairs across splits "
            f"(KING > {threshold}). First 3: {violations[:3]}",
        )

    return LeakageCheck(
        name="relatedness",
        passed=True,
        details=f"No pairs with KING > {threshold} across train/test",
    )


def _check_temporal(
    train_dates: pd.Series | None,
    test_dates: pd.Series | None,
    cutoff: str | None,
) -> LeakageCheck:
    if train_dates is None or test_dates is None or cutoff is None:
        return LeakageCheck(
            name="temporal",
            passed=False,
            details="Temporal check requires train_dates, test_dates, and cutoff",
        )

    cutoff_dt = pd.Timestamp(cutoff)
    train_after = (pd.to_datetime(train_dates) > cutoff_dt).sum()
    test_before = (pd.to_datetime(test_dates) <= cutoff_dt).sum()

    if train_after > 0 or test_before > 0:
        return LeakageCheck(
            name="temporal",
            passed=False,
            details=f"{train_after} train samples after cutoff, "
            f"{test_before} test samples before/at cutoff",
        )

    return LeakageCheck(
        name="temporal",
        passed=True,
        details=f"Temporal split clean at cutoff {cutoff}",
    )


def _check_chromosome(
    train_chroms: set[str] | None,
    test_chroms: set[str] | None,
) -> LeakageCheck:
    if train_chroms is None or test_chroms is None:
        return LeakageCheck(
            name="chromosome",
            passed=False,
            details="Chromosome check requires train_chroms and test_chroms",
        )

    overlap = train_chroms & test_chroms
    if overlap:
        return LeakageCheck(
            name="chromosome",
            passed=False,
            details=f"Chromosomes in both train and test: {overlap}",
        )

    return LeakageCheck(
        name="chromosome",
        passed=True,
        details=f"Train chroms: {sorted(train_chroms)}, test chroms: {sorted(test_chroms)}",
    )


def _check_label_leakage(
    train_labels: set | None,
    test_labels: set | None,
) -> LeakageCheck:
    if train_labels is None or test_labels is None:
        return LeakageCheck(
            name="label",
            passed=True,
            details="Skipped: no label sets provided",
        )

    overlap = train_labels & test_labels
    if overlap:
        return LeakageCheck(
            name="label",
            passed=False,
            details=f"{len(overlap)} labels appear in both train and test sets",
        )

    return LeakageCheck(
        name="label",
        passed=True,
        details="No label overlap between train and test",
    )


def _check_population_source(ancestry_source: str | None) -> LeakageCheck:
    if ancestry_source is None:
        return LeakageCheck(
            name="population",
            passed=False,
            details="Ancestry source not specified",
        )

    valid_sources = {"pca", "pca_clustering", "genetic_pca"}
    invalid_sources = {"self_reported", "self-reported", "ehr", "survey"}

    source_lower = ancestry_source.lower()
    if source_lower in invalid_sources:
        return LeakageCheck(
            name="population",
            passed=False,
            details=f"Ancestry from '{ancestry_source}' — must use genetic PCA, not self-report",
        )

    if source_lower in valid_sources:
        return LeakageCheck(
            name="population",
            passed=True,
            details=f"Ancestry from {ancestry_source}",
        )

    return LeakageCheck(
        name="population",
        passed=True,
        details=f"Ancestry source '{ancestry_source}' — not a known invalid source",
    )


def _check_summary_stat(
    test_ids: list[str] | set[str] | None,
    gwas_sample_ids: set[str] | None,
) -> LeakageCheck:
    if test_ids is None or gwas_sample_ids is None:
        return LeakageCheck(
            name="summary_stat",
            passed=False,
            details="Summary stat check requires test_ids and gwas_sample_ids",
        )

    test_set = set(test_ids)
    overlap = test_set & gwas_sample_ids
    if overlap:
        return LeakageCheck(
            name="summary_stat",
            passed=False,
            details=f"{len(overlap)} test individuals found in GWAS summary stat cohort",
        )

    return LeakageCheck(
        name="summary_stat",
        passed=True,
        details="No test individuals in GWAS summary statistics",
    )
