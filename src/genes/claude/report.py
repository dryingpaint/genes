"""Report synthesis — converts pipeline results into structured context for Claude.

Provides:
  - build_report_context(): Converts PipelineResult into a text block for Claude
  - SYSTEM_PROMPT: Instructions for Claude on interpreting genomic results
  - Per-pipeline report templates
"""

from __future__ import annotations

from genes.orchestrator.models import InputType, PipelineResult

# ---------------------------------------------------------------------------
# System prompt for Claude when generating genomic reports
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a clinical genomics assistant helping to interpret genome analysis results.
You have access to outputs from a comprehensive bioinformatics pipeline. Follow these
guidelines strictly:

## Communication principles
- Present findings clearly, starting with the most clinically significant results.
- Always state the level of evidence and confidence for each finding.
- Use plain language first, then provide technical details in parentheses.
- Distinguish between diagnostic findings, incidental findings, and variants of uncertain significance (VUS).
- Never make a diagnosis. Present findings that a clinician can use for diagnosis.

## ACMG variant classification
- When presenting ACMG codes (e.g., PM2, PP3, BP4), explain what each code means.
- Clearly state the final classification: Pathogenic, Likely Pathogenic, VUS, Likely Benign, Benign.
- For VUS: explicitly state that these do not confirm or rule out a condition.
- Group variants by gene and clinical relevance, not by chromosomal position.

## Ancestry and population context
- If inferred ancestry differs from reported ancestry, flag this prominently as it
  affects PRS interpretation and allele frequency context.
- PRS percentiles are only valid within the reference population. State which
  population was used and note limitations for underrepresented ancestries.
- Allele frequencies should reference gnomAD population-specific values when available.

## Pharmacogenomics
- Present star alleles with their associated metabolizer status.
- Flag high-priority drug-gene interactions (e.g., CYP2D6/codeine, HLA-B*57:01/abacavir).
- Note that PharmCAT results should be confirmed with targeted clinical PGx testing.

## Somatic / tumor reporting
- Distinguish driver mutations from passenger mutations.
- Present OncoKB levels of evidence for actionable variants.
- Report mutational signatures with their associated etiology.
- MSI status should be reported as MSI-H, MSS, or MSI-L with the numeric score.

## cfDNA reporting
- Report tumor fraction with clinical context (detectable vs. undetectable).
- ichorCNA tumor fraction < 3% is generally considered below reliable detection.
- Griffin nucleosome footprinting results reflect regulatory activity, not mutations.
- UXM cell-of-origin results indicate tissue contribution to cfDNA, not tumor location.

## Epigenetic clocks
- Report biological age alongside chronological age.
- Age acceleration > 5 years is generally considered clinically meaningful.
- DunedinPACE measures pace of aging (1.0 = average); interpret relative to 1.0.
- Note that epigenetic clocks have different training sets and may disagree.

## Uncertainty and limitations
- Always disclose tool failures or missing data explicitly.
- If a tool failed, explain what information is missing and its clinical impact.
- Note that computational predictions (SpliceAI, AlphaMissense) are not clinical-grade
  without functional validation.
- ClinVar classifications can change; note the review status (star rating).
"""

# ---------------------------------------------------------------------------
# Report context builder
# ---------------------------------------------------------------------------


def build_report_context(result: PipelineResult) -> str:
    """Convert a PipelineResult into a structured context string for Claude.

    The output is a markdown-formatted text block that Claude can use to
    generate a human-readable genomic report.
    """
    input_type = InputType(result.input_type)
    template_fn = _TEMPLATE_MAP.get(input_type, _generic_report)
    return template_fn(result)


# ---------------------------------------------------------------------------
# Template: Germline report
# ---------------------------------------------------------------------------


def _germline_report(result: PipelineResult) -> str:
    sections: list[str] = []
    sections.append(f"# Germline Analysis Report\n**Run ID:** {result.run_id}\n")

    # Pipeline status
    sections.append(_pipeline_status_section(result))

    # VEP summary
    if "vep" in result.tool_results:
        vep = result.tool_results["vep"]
        sections.append(
            f"## Variant Annotation (VEP {vep.version})\n"
            f"- Variants annotated: {vep.output_summary.get('variant_count', 'N/A')}\n"
        )

    # Variant effect predictors
    predictor_tools = ["alphamissense", "spliceai", "gpn_msa"]
    predictor_sections = []
    for name in predictor_tools:
        if name in result.tool_results:
            tr = result.tool_results[name]
            predictor_sections.append(f"### {tr.tool_name} (v{tr.version})")
            if tr.output_summary:
                for key, val in tr.output_summary.items():
                    predictor_sections.append(f"- {key}: {val}")
            if tr.warnings:
                predictor_sections.append(f"- Warnings: {'; '.join(tr.warnings)}")
    if predictor_sections:
        sections.append("## Variant Effect Predictions\n" + "\n".join(predictor_sections))

    # Pharmacogenomics
    if "pharmcat" in result.tool_results:
        tr = result.tool_results["pharmcat"]
        sections.append("## Pharmacogenomics (PharmCAT)")
        if tr.output_summary:
            for key, val in tr.output_summary.items():
                sections.append(f"- {key}: {val}")

    # PRS
    if "prs" in result.tool_results:
        tr = result.tool_results["prs"]
        sections.append("## Polygenic Risk Scores")
        if tr.output_summary:
            for key, val in tr.output_summary.items():
                sections.append(f"- {key}: {val}")

    # Ancestry
    if "ancestry" in result.tool_results:
        tr = result.tool_results["ancestry"]
        sections.append("## Ancestry Inference")
        if tr.output_summary:
            for key, val in tr.output_summary.items():
                sections.append(f"- {key}: {val}")

    # HLA typing
    if "hla" in result.tool_results:
        tr = result.tool_results["hla"]
        sections.append("## HLA Typing")
        if tr.output_summary:
            for key, val in tr.output_summary.items():
                sections.append(f"- {key}: {val}")

    # CYP2D6 (Cyrius)
    if "cyrius" in result.tool_results:
        tr = result.tool_results["cyrius"]
        sections.append("## CYP2D6 Star Alleles (Cyrius)")
        if tr.output_summary:
            for key, val in tr.output_summary.items():
                sections.append(f"- {key}: {val}")

    # Exomiser
    if "exomiser" in result.tool_results:
        tr = result.tool_results["exomiser"]
        sections.append("## Mendelian Disease Prioritization (Exomiser)")
        if tr.output_summary:
            for key, val in tr.output_summary.items():
                sections.append(f"- {key}: {val}")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Template: Somatic report
# ---------------------------------------------------------------------------


def _somatic_report(result: PipelineResult) -> str:
    sections: list[str] = []
    sections.append(f"# Somatic Analysis Report\n**Run ID:** {result.run_id}\n")
    sections.append(_pipeline_status_section(result))

    # Mutect2
    if "mutect2" in result.tool_results:
        tr = result.tool_results["mutect2"]
        sections.append("## Somatic Variant Calling (Mutect2)")
        if tr.output_summary:
            for key, val in tr.output_summary.items():
                sections.append(f"- {key}: {val}")

    # VEP
    if "vep" in result.tool_results:
        vep = result.tool_results["vep"]
        sections.append(
            f"## Variant Annotation (VEP {vep.version})\n"
            f"- Variants annotated: {vep.output_summary.get('variant_count', 'N/A')}\n"
        )

    # OncoKB/CIViC
    if "oncokb_civic" in result.tool_results:
        tr = result.tool_results["oncokb_civic"]
        sections.append("## Actionable Variants (OncoKB/CIViC)")
        if tr.output_summary:
            for key, val in tr.output_summary.items():
                sections.append(f"- {key}: {val}")

    # SigProfiler
    if "sigprofiler" in result.tool_results:
        tr = result.tool_results["sigprofiler"]
        sections.append("## Mutational Signatures (SigProfiler)")
        if tr.output_summary:
            for key, val in tr.output_summary.items():
                sections.append(f"- {key}: {val}")

    # MSIsensor
    if "msisensor" in result.tool_results:
        tr = result.tool_results["msisensor"]
        sections.append("## Microsatellite Instability (MSIsensor)")
        if tr.output_summary:
            for key, val in tr.output_summary.items():
                sections.append(f"- {key}: {val}")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Template: cfDNA report
# ---------------------------------------------------------------------------


def _cfdna_report(result: PipelineResult) -> str:
    sections: list[str] = []
    sections.append(f"# cfDNA Liquid Biopsy Report\n**Run ID:** {result.run_id}\n")
    sections.append(_pipeline_status_section(result))

    # ichorCNA
    if "ichorcna" in result.tool_results:
        tr = result.tool_results["ichorcna"]
        sections.append("## Tumor Fraction (ichorCNA)")
        tf = tr.output_summary.get("tumor_fraction")
        if tf is not None:
            pct = tf * 100 if tf < 1 else tf
            sections.append(f"- Estimated tumor fraction: {pct:.1f}%")
            if pct < 3:
                sections.append(
                    "- **Note:** Tumor fraction < 3% is below reliable detection threshold."
                )
        if "ploidy" in tr.output_summary:
            sections.append(f"- Estimated ploidy: {tr.output_summary['ploidy']}")

    # Griffin
    if "griffin" in result.tool_results:
        tr = result.tool_results["griffin"]
        sections.append("## Nucleosome Footprinting (Griffin)")
        if tr.output_summary:
            for key, val in tr.output_summary.items():
                sections.append(f"- {key}: {val}")

    # UXM
    if "uxm" in result.tool_results:
        tr = result.tool_results["uxm"]
        sections.append("## Cell-of-Origin Deconvolution (UXM)")
        fractions = tr.output_summary.get("cell_type_fractions", {})
        if fractions:
            # Sort by fraction descending
            sorted_fracs = sorted(fractions.items(), key=lambda x: x[1], reverse=True)
            for tissue, frac in sorted_fracs[:10]:  # Top 10
                pct = frac * 100 if frac < 1 else frac
                sections.append(f"- {tissue}: {pct:.1f}%")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Template: Methylation report
# ---------------------------------------------------------------------------


def _methylation_report(result: PipelineResult) -> str:
    sections: list[str] = []
    sections.append(f"# Epigenetic Clock Report\n**Run ID:** {result.run_id}\n")
    sections.append(_pipeline_status_section(result))

    if "biolearn" in result.tool_results:
        tr = result.tool_results["biolearn"]
        sections.append(f"## Epigenetic Clocks (Biolearn v{tr.version})")
        sections.append(
            f"- Probes used: {tr.output_summary.get('num_probes', 'N/A')}\n"
            f"- Samples: {tr.output_summary.get('num_samples', 'N/A')}"
        )

        clocks = tr.output_summary.get("clocks", {})
        for clock_name, clock_data in clocks.items():
            sections.append(f"### {clock_name}")
            preds = clock_data.get("predicted_ages", {})
            for sample, age in preds.items():
                sections.append(f"- {sample}: predicted age = {age}")
            accel = clock_data.get("age_acceleration")
            if accel is not None:
                sections.append(f"- Age acceleration: {accel:+.1f} years")
                if abs(accel) > 5:
                    sections.append("- **Clinically notable age acceleration (>5 years)**")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Generic report (fallback)
# ---------------------------------------------------------------------------


def _generic_report(result: PipelineResult) -> str:
    sections: list[str] = []
    sections.append(f"# Analysis Report\n**Run ID:** {result.run_id}\n**Type:** {result.input_type}\n")
    sections.append(_pipeline_status_section(result))

    for name, tr in result.tool_results.items():
        sections.append(f"## {tr.tool_name} (v{tr.version})")
        if tr.output_summary:
            for key, val in tr.output_summary.items():
                sections.append(f"- {key}: {val}")
        if tr.errors:
            sections.append(f"- ERRORS: {'; '.join(tr.errors)}")
        if tr.warnings:
            sections.append(f"- Warnings: {'; '.join(tr.warnings)}")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _pipeline_status_section(result: PipelineResult) -> str:
    """Generate a pipeline status summary section."""
    lines = ["## Pipeline Status"]
    lines.append(f"- Tools run: {', '.join(result.tools_run) or 'none'}")

    succeeded = [n for n, tr in result.tool_results.items() if tr.success]
    failed = [n for n, tr in result.tool_results.items() if not tr.success]

    lines.append(f"- Succeeded: {', '.join(succeeded) or 'none'}")
    if failed:
        lines.append(f"- **Failed: {', '.join(failed)}**")
    if result.errors:
        lines.append("- Pipeline errors:")
        for err in result.errors:
            lines.append(f"  - {err}")
    if result.runtime_seconds is not None:
        lines.append(f"- Total runtime: {result.runtime_seconds:.1f}s")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Template dispatch map
# ---------------------------------------------------------------------------

_TEMPLATE_MAP = {
    InputType.GERMLINE_VCF: _germline_report,
    InputType.GERMLINE_BAM: _germline_report,
    InputType.GERMLINE_FASTQ: _germline_report,
    InputType.TUMOR_NORMAL_BAM: _somatic_report,
    InputType.TUMOR_ONLY_BAM: _somatic_report,
    InputType.SOMATIC_VCF: _somatic_report,
    InputType.CFDNA_BAM: _cfdna_report,
    InputType.METHYLATION_ARRAY: _methylation_report,
}
