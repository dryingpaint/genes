"""Request and aggregated-result models for the orchestrator."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from genes.orchestrator.spec import Criticality, Mode, Status
from genes.tools._base import ToolResult


class InputType(str, Enum):
    """Supported input modalities. Each maps to a Mode + initial artifact set."""

    GERMLINE_VCF = "germline_vcf"
    GERMLINE_BAM = "germline_bam"
    GERMLINE_FASTQ = "germline_fastq"
    TUMOR_NORMAL_BAM = "tumor_normal_bam"
    TUMOR_ONLY_BAM = "tumor_only_bam"
    SOMATIC_VCF = "somatic_vcf"
    CFDNA_BAM = "cfdna_bam"
    METHYLATION_ARRAY = "methylation_array"


class SequencingPlatform(str, Enum):
    ILLUMINA = "illumina"
    PACBIO = "pacbio"
    ONT = "ont"


class AnalysisRequest(BaseModel):
    """What the user (or Claude) submits to trigger a run."""

    run_id: str
    input_type: InputType
    input_paths: list[str]
    reference_build: str = "GRCh38"
    sequencing_platform: SequencingPlatform = SequencingPlatform.ILLUMINA

    # Optional parameters that gate conditional tools.
    hpo_terms: list[str] | None = None      # Enables Exomiser.
    prs_traits: list[str] | None = None     # Enables PRS for these traits.
    tumor_type: str | None = None           # Threaded to OncoKB + SigProfiler.
    reported_ancestry: str | None = None    # Cross-checked against inferred.

    model_config = ConfigDict(use_enum_values=True)


class PipelineResult(BaseModel):
    """Aggregated output of a run. Status is derived from the criticality-
    weighted statuses of the tools that ran."""

    run_id: str
    mode: Mode
    input_type: InputType
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    tool_results: dict[str, ToolResult] = Field(default_factory=dict)
    # Orchestrator-level errors (mode misconfiguration, no tools applicable,
    # bad initial artifacts). Always poisons status to FAILED.
    pipeline_errors: list[str] = Field(default_factory=list)

    model_config = ConfigDict(use_enum_values=True)

    @property
    def status(self) -> Status:
        if self.pipeline_errors:
            return Status.FAILED

        critical_bad = any(
            r.status in (Status.FAILED, Status.TIMEOUT, Status.SKIPPED)
            for r in self.tool_results.values()
            if r.criticality == Criticality.CRITICAL
        )
        if critical_bad:
            return Status.FAILED

        any_bad = any(
            r.status in (Status.FAILED, Status.TIMEOUT, Status.SKIPPED)
            for r in self.tool_results.values()
            if r.criticality != Criticality.OPTIONAL
        )
        any_partial = any(
            r.status == Status.PARTIAL
            for r in self.tool_results.values()
            if r.criticality != Criticality.OPTIONAL
        )
        if any_bad or any_partial:
            return Status.PARTIAL
        return Status.OK

    @property
    def runtime_s(self) -> float | None:
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()

    def finalize(self) -> None:
        self.completed_at = datetime.now(timezone.utc)

    def by_status(self, status: Status) -> list[str]:
        return [name for name, r in self.tool_results.items() if r.status == status]

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "mode": self.mode,
            "runtime_s": self.runtime_s,
            "ok": self.by_status(Status.OK),
            "partial": self.by_status(Status.PARTIAL),
            "failed": self.by_status(Status.FAILED),
            "timeout": self.by_status(Status.TIMEOUT),
            "skipped": self.by_status(Status.SKIPPED),
            "pipeline_errors": self.pipeline_errors,
        }
