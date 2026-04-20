"""Dispatcher — main entry point for the genome-to-phenotype pipeline.

Detects the input type from the AnalysisRequest and routes to the
appropriate pipeline DAG. Returns a unified PipelineResult.
"""

from __future__ import annotations

import logging

from genes.orchestrator.models import AnalysisRequest, InputType, PipelineResult
from genes.orchestrator.pipelines import (
    cfdna_pipeline,
    germline_bam_pipeline,
    germline_vcf_pipeline,
    methylation_pipeline,
    somatic_pipeline,
    somatic_vcf_pipeline,
)

logger = logging.getLogger(__name__)

# Map input types to pipeline functions
_PIPELINE_DISPATCH: dict = {
    InputType.GERMLINE_VCF: germline_vcf_pipeline,
    InputType.GERMLINE_BAM: germline_bam_pipeline,
    InputType.GERMLINE_FASTQ: germline_bam_pipeline,  # FASTQ -> align -> BAM pipeline
    InputType.TUMOR_NORMAL_BAM: somatic_pipeline,
    InputType.TUMOR_ONLY_BAM: somatic_pipeline,
    InputType.SOMATIC_VCF: somatic_vcf_pipeline,
    InputType.CFDNA_BAM: cfdna_pipeline,
    InputType.METHYLATION_ARRAY: methylation_pipeline,
}


def run_pipeline(request: AnalysisRequest) -> PipelineResult:
    """Route an analysis request to the appropriate pipeline and execute it.

    This is the single entry point that Claude (or any caller) uses to trigger
    a full analysis. It validates the request, selects the right pipeline DAG,
    and returns the aggregated results.

    Args:
        request: The analysis request specifying input type, paths, and options.

    Returns:
        PipelineResult with tool outputs, errors, and summary.
    """
    logger.info(
        "Starting pipeline for run_id=%s, input_type=%s",
        request.run_id,
        request.input_type,
    )

    # Validate input paths are provided
    if not request.input_paths:
        result = PipelineResult(run_id=request.run_id, input_type=request.input_type)
        result.errors.append("No input paths provided.")
        result.finalize()
        return result

    # Look up the pipeline function
    pipeline_fn = _PIPELINE_DISPATCH.get(InputType(request.input_type))
    if pipeline_fn is None:
        result = PipelineResult(run_id=request.run_id, input_type=request.input_type)
        result.errors.append(f"Unsupported input type: {request.input_type}")
        result.finalize()
        return result

    # Execute the pipeline with top-level error handling
    try:
        result = pipeline_fn(request)
    except Exception as exc:
        logger.exception("Pipeline failed for run_id=%s", request.run_id)
        result = PipelineResult(run_id=request.run_id, input_type=request.input_type)
        result.errors.append(f"Pipeline execution failed: {exc}")
        result.finalize()

    # Build a high-level summary
    result.summary["tools_run"] = result.tools_run
    result.summary["tools_succeeded"] = [
        name for name, tr in result.tool_results.items() if tr.success
    ]
    result.summary["tools_failed"] = [
        name for name, tr in result.tool_results.items() if not tr.success
    ]
    result.summary["total_runtime_seconds"] = result.runtime_seconds

    logger.info(
        "Pipeline complete for run_id=%s: %d tools run, %d succeeded, %d failed",
        request.run_id,
        len(result.tools_run),
        len(result.summary["tools_succeeded"]),
        len(result.summary["tools_failed"]),
    )

    return result
