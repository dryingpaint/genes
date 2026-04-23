"""Pipeline DAGs for the genome-to-phenotype toolkit.

Each pipeline function takes an AnalysisRequest and returns a PipelineResult.
Tools are launched in parallel via Modal's .spawn()/.get() pattern wherever
the DAG allows. Each pipeline captures tool failures without crashing.
"""

from __future__ import annotations

import logging
from typing import Any

from genes.orchestrator.models import AnalysisRequest, PipelineResult
from genes.tools._base import ToolResult

logger = logging.getLogger(__name__)


def _safe_spawn(tool_fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Spawn a Modal function, returning the FunctionCall handle or None on error."""
    try:
        return tool_fn.spawn(*args, **kwargs)
    except Exception as exc:
        logger.error("Failed to spawn %s: %s", getattr(tool_fn, "__name__", tool_fn), exc)
        return None


def _safe_get(handle: Any, tool_name: str, timeout: int = 1800) -> ToolResult | None:
    """Collect a spawned result with timeout, returning error ToolResult on failure."""
    if handle is None:
        return None
    try:
        return handle.get(timeout=timeout)
    except TimeoutError:
        logger.warning("Tool %s timed out after %ds", tool_name, timeout)
        from datetime import datetime, timezone
        return ToolResult(
            tool_name=tool_name,
            version="unknown",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            errors=[f"Timed out after {timeout}s — still running on Modal"],
            warnings=["Result may complete later; check Modal dashboard"],
        )
    except Exception as exc:
        logger.error("Tool %s failed during execution: %s", tool_name, exc)
        from datetime import datetime, timezone
        return ToolResult(
            tool_name=tool_name,
            version="unknown",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            errors=[str(exc)],
        )


# Per-tool timeout for .get() — VEP gets longer since it processes the full VCF
_TOOL_TIMEOUTS = {
    "vep": 7200,       # 2 hours — full WGS annotation is slow
    "alphamissense": 1800,
    "spliceai": 1800,
    "gpn_msa": 7200,  # GPN-MSA tabix lookups on full WGS take 1-2 hours
    "evee": 1800,
    "pharmcat": 600,
    "traits": 300,
    "ancestry": 600,
    "prs": 600,
    "exomiser": 3600,
}


def _collect(
    result: PipelineResult,
    futures: dict[str, Any],
) -> None:
    """Gather all spawned futures into the PipelineResult with per-tool timeouts."""
    for name, handle in futures.items():
        timeout = _TOOL_TIMEOUTS.get(name, 1800)
        tool_result = _safe_get(handle, name, timeout=timeout)
        if tool_result is not None:
            result.tool_results[name] = tool_result
            if not tool_result.success:
                result.errors.append(f"Tool {name} reported errors: {tool_result.errors}")


# ---------------------------------------------------------------------------
# Pipeline: Germline VCF
# ---------------------------------------------------------------------------


def germline_vcf_pipeline(request: AnalysisRequest) -> PipelineResult:
    """All tools run in parallel on the raw VCF. VEP is no longer a bottleneck.

    AlphaMissense/SpliceAI/GPN-MSA do their own lookups from pre-computed scores
    and don't need VEP-annotated output. VEP runs alongside everything else.
    """
    from genes.tools import vep, alphamissense, spliceai, gpn_msa, pharmcat, prs, ancestry
    from genes.tools import traits, evee

    result = PipelineResult(run_id=request.run_id, input_type=request.input_type)
    vcf_path = request.input_paths[0]

    # All tools in parallel
    futures: dict[str, Any] = {}
    futures["vep"] = _safe_spawn(vep.annotate, vcf_path, request.run_id)
    futures["alphamissense"] = _safe_spawn(
        alphamissense.alphamissense_lookup, request.run_id, vcf_path=vcf_path
    )
    futures["spliceai"] = _safe_spawn(spliceai.spliceai_lookup, vcf_path, request.run_id)
    futures["gpn_msa"] = _safe_spawn(gpn_msa.gpn_msa_lookup, vcf_path, request.run_id)
    futures["evee"] = _safe_spawn(evee.lookup_evee_scores, vcf_path, request.run_id)
    futures["pharmcat"] = _safe_spawn(pharmcat.run, vcf_path, request.run_id)
    futures["traits"] = _safe_spawn(traits.lookup_traits, vcf_path, request.run_id)
    futures["ancestry"] = _safe_spawn(ancestry.infer_ancestry, vcf_path, request.run_id)

    if request.prs_traits:
        futures["prs"] = _safe_spawn(
            prs.calculate, vcf_path, request.run_id, request.prs_traits
        )

    if request.hpo_terms:
        try:
            from genes.tools import exomiser
            futures["exomiser"] = _safe_spawn(
                exomiser.prioritize_variants, vcf_path, request.run_id,
                hpo_terms=request.hpo_terms
            )
        except ImportError:
            result.errors.append("Exomiser tool not yet available; skipping.")

    _collect(result, futures)
    result.finalize()
    return result


# ---------------------------------------------------------------------------
# Pipeline: Germline BAM
# ---------------------------------------------------------------------------


def germline_bam_pipeline(request: AnalysisRequest) -> PipelineResult:
    """DeepVariant -> germline_vcf_pipeline + [Cyrius, HLA] in parallel."""
    from genes.tools import deepvariant, cyrius, hla

    result = PipelineResult(run_id=request.run_id, input_type=request.input_type)
    bam_path = request.input_paths[0]

    # Step 1: Variant calling with DeepVariant
    try:
        dv_result = deepvariant.call_variants.remote(bam_path, request.run_id)
        result.tool_results["deepvariant"] = dv_result
    except Exception as exc:
        result.errors.append(f"DeepVariant failed: {exc}")
        result.finalize()
        return result

    if not dv_result.success or not dv_result.output_paths:
        result.errors.append("DeepVariant did not produce output VCF.")
        result.finalize()
        return result

    called_vcf = dv_result.output_paths[0]

    # Step 2: Run germline VCF pipeline on the called VCF, plus BAM-specific tools
    # Launch BAM-specific tools in parallel with the VCF sub-pipeline
    bam_futures: dict[str, Any] = {}
    bam_futures["cyrius"] = _safe_spawn(cyrius.call_cyp2d6, bam_path, request.run_id)
    bam_futures["hla"] = _safe_spawn(hla.type_hla, bam_path, request.run_id)

    # Run the VCF sub-pipeline (this blocks but internally parallelizes)
    vcf_request = request.model_copy(update={"input_paths": [called_vcf]})
    vcf_result = germline_vcf_pipeline(vcf_request)

    # Merge VCF pipeline results into our result
    result.tool_results.update(vcf_result.tool_results)
    result.errors.extend(vcf_result.errors)

    # Collect BAM-specific results
    _collect(result, bam_futures)
    result.finalize()
    return result


# ---------------------------------------------------------------------------
# Pipeline: Somatic (tumor-normal BAM)
# ---------------------------------------------------------------------------


def somatic_pipeline(request: AnalysisRequest) -> PipelineResult:
    """Mutect2 -> VEP -> [OncoKB/CIViC, SigProfiler, MSIsensor] in parallel."""
    from genes.tools import vep

    result = PipelineResult(run_id=request.run_id, input_type=request.input_type)
    tumor_bam = request.input_paths[0]
    normal_bam = request.input_paths[1] if len(request.input_paths) > 1 else None

    # Step 1: Somatic variant calling with Mutect2
    try:
        from genes.tools import mutect2

        mutect_kwargs: dict[str, Any] = {
            "tumor_bam": tumor_bam,
            "run_id": request.run_id,
        }
        if normal_bam:
            mutect_kwargs["normal_bam"] = normal_bam
        mutect_result = mutect2.call_somatic.remote(**mutect_kwargs)
        result.tool_results["mutect2"] = mutect_result
    except ImportError:
        result.errors.append("Mutect2 tool not yet available.")
        result.finalize()
        return result
    except Exception as exc:
        result.errors.append(f"Mutect2 failed: {exc}")
        result.finalize()
        return result

    if not mutect_result.success or not mutect_result.output_paths:
        result.errors.append("Mutect2 did not produce output VCF.")
        result.finalize()
        return result

    somatic_vcf = mutect_result.output_paths[0]

    # Step 2: VEP annotation
    try:
        vep_result = vep.annotate.remote(somatic_vcf, request.run_id)
        result.tool_results["vep"] = vep_result
    except Exception as exc:
        result.errors.append(f"VEP failed: {exc}")
        result.finalize()
        return result

    annotated_vcf = vep_result.output_summary.get("annotated_vcf", somatic_vcf)

    # Step 3: Parallel somatic interpretation tools
    futures: dict[str, Any] = {}

    try:
        from genes.tools import sigprofiler

        futures["sigprofiler"] = _safe_spawn(
            sigprofiler.analyze, somatic_vcf, request.run_id,
            tumor_type=request.tumor_type,
        )
    except ImportError:
        result.errors.append("SigProfiler tool not yet available; skipping.")

    try:
        from genes.tools import oncokb_civic

        futures["oncokb_civic"] = _safe_spawn(
            oncokb_civic.annotate, annotated_vcf, request.run_id,
            tumor_type=request.tumor_type,
        )
    except ImportError:
        result.errors.append("OncoKB/CIViC tool not yet available; skipping.")

    try:
        from genes.tools import msisensor

        futures["msisensor"] = _safe_spawn(
            msisensor.score, tumor_bam, request.run_id,
            normal_bam=normal_bam,
        )
    except ImportError:
        result.errors.append("MSIsensor tool not yet available; skipping.")

    _collect(result, futures)
    result.finalize()
    return result


# ---------------------------------------------------------------------------
# Pipeline: Somatic VCF (pre-called)
# ---------------------------------------------------------------------------


def somatic_vcf_pipeline(request: AnalysisRequest) -> PipelineResult:
    """VEP -> [OncoKB/CIViC, SigProfiler] in parallel."""
    from genes.tools import vep

    result = PipelineResult(run_id=request.run_id, input_type=request.input_type)
    vcf_path = request.input_paths[0]

    # Step 1: VEP annotation
    try:
        vep_result = vep.annotate.remote(vcf_path, request.run_id)
        result.tool_results["vep"] = vep_result
    except Exception as exc:
        result.errors.append(f"VEP failed: {exc}")
        result.finalize()
        return result

    annotated_vcf = vep_result.output_summary.get("annotated_vcf", vcf_path)

    # Step 2: Parallel somatic interpretation
    futures: dict[str, Any] = {}

    try:
        from genes.tools import oncokb_civic

        futures["oncokb_civic"] = _safe_spawn(
            oncokb_civic.annotate, annotated_vcf, request.run_id,
            tumor_type=request.tumor_type,
        )
    except ImportError:
        result.errors.append("OncoKB/CIViC tool not yet available; skipping.")

    try:
        from genes.tools import sigprofiler

        futures["sigprofiler"] = _safe_spawn(
            sigprofiler.analyze, vcf_path, request.run_id,
            tumor_type=request.tumor_type,
        )
    except ImportError:
        result.errors.append("SigProfiler tool not yet available; skipping.")

    _collect(result, futures)
    result.finalize()
    return result


# ---------------------------------------------------------------------------
# Pipeline: cfDNA
# ---------------------------------------------------------------------------


def cfdna_pipeline(request: AnalysisRequest) -> PipelineResult:
    """[ichorCNA, Griffin] in parallel; + UXM if bisulfite."""
    from genes.tools import cfdna

    result = PipelineResult(run_id=request.run_id, input_type=request.input_type)
    bam_path = request.input_paths[0]

    # Determine if this is bisulfite-sequenced (check for second path or metadata)
    is_bisulfite = len(request.input_paths) > 1 and "bisulfite" in request.input_paths[1].lower()
    bisulfite_bam = request.input_paths[1] if is_bisulfite else None

    # Launch parallel cfDNA tools
    futures: dict[str, Any] = {}
    futures["ichorcna"] = _safe_spawn(cfdna.run_ichorcna, bam_path, request.run_id)
    futures["griffin"] = _safe_spawn(cfdna.run_griffin, bam_path, request.run_id)

    if bisulfite_bam:
        futures["uxm"] = _safe_spawn(cfdna.run_uxm, bisulfite_bam, request.run_id)

    _collect(result, futures)
    result.finalize()
    return result


# ---------------------------------------------------------------------------
# Pipeline: Methylation array
# ---------------------------------------------------------------------------


def methylation_pipeline(request: AnalysisRequest) -> PipelineResult:
    """Biolearn epigenetic clocks on a methylation beta-value matrix."""
    from genes.tools import biolearn

    result = PipelineResult(run_id=request.run_id, input_type=request.input_type)
    methylation_path = request.input_paths[0]

    try:
        biolearn_result = biolearn.compute_clocks.remote(
            methylation_path,
            request.run_id,
        )
        result.tool_results["biolearn"] = biolearn_result
    except Exception as exc:
        result.errors.append(f"Biolearn failed: {exc}")

    result.finalize()
    return result
