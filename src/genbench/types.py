"""Core types for the benchmark harness."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Ancestry(str, Enum):
    EUR = "EUR"
    AFR = "AFR"
    SAS = "SAS"
    EAS = "EAS"
    AMR = "AMR"
    ALL = "ALL"


class SplitType(str, Enum):
    TEMPORAL = "temporal"
    INDIVIDUAL = "individual"
    CHROMOSOME = "chromosome"
    LEAVE_N_OUT = "leave_n_out"
    ZERO_SHOT = "zero_shot"


class VariantClass(str, Enum):
    MISSENSE = "missense"
    NONSENSE_FRAMESHIFT = "nonsense_frameshift"
    SPLICE = "splice"
    NONCODING = "noncoding"
    INFRAME_INDEL = "inframe_indel"


# --- Metrics ---


class MetricValue(BaseModel):
    """A single metric with confidence interval."""

    estimate: float
    ci_lower: float
    ci_upper: float
    n: int


class AncestryStratifiedMetric(BaseModel):
    """A metric reported per ancestry group. Aggregate-only is disallowed for Tier 2+."""

    per_ancestry: dict[str, MetricValue] = Field(default_factory=dict)
    aggregate: MetricValue | None = None


class CeilingInfo(BaseModel):
    """Theoretical performance ceiling for a trait/task."""

    ceiling_value: float
    ceiling_source: str
    normalized_performance: float  # metric / ceiling


# --- Anti-leakage ---


class LeakageCheck(BaseModel):
    """Result of a single anti-leakage check."""

    name: str
    passed: bool
    details: str = ""


class LeakageReport(BaseModel):
    """Aggregate anti-leakage verification. Must be attached to every BenchmarkResult."""

    checks: dict[str, LeakageCheck] = Field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks.values())


# --- Results ---


class GateResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"


class BenchmarkResult(BaseModel):
    """The canonical output of every benchmark task."""

    task_id: str  # e.g. "tier0.clinvar.missense"
    model_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    split_type: SplitType
    leakage_report: LeakageReport
    metrics: dict[str, AncestryStratifiedMetric] = Field(default_factory=dict)
    ceiling: CeilingInfo | None = None
    n_samples: dict[str, int] = Field(default_factory=dict)  # ancestry -> count
    gate: GateResult | None = None
    gate_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaselineComparison(BaseModel):
    """Side-by-side comparison of model vs baselines for one task."""

    task_id: str
    model_result: BenchmarkResult
    baseline_results: dict[str, BenchmarkResult]  # baseline_name -> result
    circularity_flags: dict[str, str] = Field(default_factory=dict)  # baseline -> reason


# --- Model submission ---


class ModelSubmission(BaseModel):
    """What a user provides to be benchmarked."""

    model_name: str
    predict_fn: str  # import path to callable: (data) -> scores
    tier_targets: list[str] = Field(default_factory=lambda: ["0", "1", "2", "3"])
    gpu_required: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
