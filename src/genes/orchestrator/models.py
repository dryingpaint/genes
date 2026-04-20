"""Pydantic models for the orchestrator: requests, results, and pipeline configuration."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from genes.tools._base import ToolResult


class InputType(str, Enum):
    """Supported input modalities. Each maps to a specific pipeline DAG."""

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
    """What the user (or Claude) submits to trigger a pipeline run."""

    run_id: str
    input_type: InputType
    input_paths: list[str]
    reference_build: str = "GRCh38"
    sequencing_platform: SequencingPlatform = SequencingPlatform.ILLUMINA

    # Optional parameters that enable conditional pipeline branches
    hpo_terms: list[str] | None = None  # Enables Exomiser (Mendelian prioritization)
    prs_traits: list[str] | None = None  # Which PRS to compute (e.g. ["height", "CAD", "T2D"])
    tumor_type: str | None = None  # For somatic signature context

    # Ancestry (if known; otherwise inferred)
    reported_ancestry: str | None = None

    model_config = ConfigDict(use_enum_values=True)


class PipelineResult(BaseModel):
    """Aggregated output of a complete pipeline run."""

    run_id: str
    input_type: InputType
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    tool_results: dict[str, ToolResult] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0 and all(r.success for r in self.tool_results.values())

    @property
    def tools_run(self) -> list[str]:
        return list(self.tool_results.keys())

    @property
    def runtime_seconds(self) -> float | None:
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()

    def finalize(self) -> None:
        """Mark the pipeline as complete."""
        self.completed_at = datetime.now(timezone.utc)
