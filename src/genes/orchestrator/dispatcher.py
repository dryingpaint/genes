"""Single entry point for the orchestrator.

Kept as a thin alias so callers (scripts, Claude integration, tests) have
one obvious import. All real logic lives in scheduler.run_pipeline.
"""

from __future__ import annotations

from genes.orchestrator.models import AnalysisRequest, PipelineResult
from genes.orchestrator.scheduler import run_pipeline

__all__ = ["AnalysisRequest", "PipelineResult", "run_pipeline"]
