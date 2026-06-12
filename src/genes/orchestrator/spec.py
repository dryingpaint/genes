"""Tool specs and the global registry.

A ToolSpec is the contract between a tool and the scheduler: what kind of
artifact does this tool consume, what does it produce, how long should it
get, what reference files does it touch, and what does its failure mean for
the pipeline as a whole. Tools register themselves at import time. The
scheduler discovers them through the registry and builds an execution DAG
from their declared consumes/produces.

There is no per-pipeline routing logic. The DAG that runs is whatever the
specs describe for the requested mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class Mode(str, Enum):
    """Top-level analysis mode, derived from AnalysisRequest.input_type."""
    GERMLINE = "germline"
    SOMATIC = "somatic"
    CFDNA = "cfdna"
    METHYLATION = "methylation"


class Artifact(str, Enum):
    """Named artifacts that flow through the pipeline DAG.

    Tools declare which artifacts they consume and which they produce; the
    scheduler treats each tool as a node in a DAG keyed on these labels.
    """
    BAM = "bam"
    TUMOR_BAM = "tumor_bam"
    NORMAL_BAM = "normal_bam"
    CFDNA_BAM = "cfdna_bam"
    CFDNA_BISULFITE_BAM = "cfdna_bisulfite_bam"
    VCF = "vcf"
    ANNOTATED_VCF = "annotated_vcf"
    METHYLATION_MATRIX = "methylation_matrix"


class Criticality(str, Enum):
    """How much a tool's failure poisons the pipeline.

    - CRITICAL: callers, annotators. If this fails, downstream tools can't
      do meaningful work and the pipeline reports FAILED.
    - STANDARD: substantive interpretation tools. Failure makes the
      pipeline PARTIAL but doesn't invalidate other findings.
    - OPTIONAL: convenience layers (traits readout, cfDNA add-ons). Failure
      leaves the pipeline OK.
    """
    CRITICAL = "critical"
    STANDARD = "standard"
    OPTIONAL = "optional"


class Status(str, Enum):
    """Per-tool result status. PipelineResult.status is a roll-up of these
    weighted by Criticality.
    """
    OK = "ok"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


def _always(_: Any) -> bool:
    return True


@dataclass(frozen=True)
class ToolSpec:
    """Declarative contract for a tool."""

    name: str
    version: str
    modes: tuple[Mode, ...]
    consumes: tuple[Artifact, ...]
    produces: tuple[Artifact, ...] = ()
    optional_consumes: tuple[Artifact, ...] = ()
    criticality: Criticality = Criticality.STANDARD
    timeout_s: int = 1800
    reference_artifacts: tuple[str, ...] = ()
    request_kwargs: tuple[str, ...] = ()
    # Runtime predicate: receives the AnalysisRequest. If False, the tool
    # is dropped entirely. Use this for tools gated on optional inputs
    # (Exomiser requires HPO terms; PRS requires prs_traits; UXM requires
    # a bisulfite BAM).
    gate: Callable[[Any], bool] = field(default=_always)


# name -> (spec, modal Function)
REGISTRY: dict[str, tuple[ToolSpec, Any]] = {}


def register(spec: ToolSpec, fn: Any) -> None:
    """Register a tool. Idempotent on re-import."""
    REGISTRY[spec.name] = (spec, fn)
