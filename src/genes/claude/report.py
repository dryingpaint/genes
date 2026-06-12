"""Report synthesis — convert a PipelineResult into structured context for Claude.

Generic over the registered tools: we no longer hand-roll a per-modality
template. Each ToolResult carries a `payload` dict that the renderer dumps
into the markdown block under its section header. The pipeline's
criticality-weighted Status is surfaced front-and-center so the report
acknowledges partial / failed runs.
"""

from __future__ import annotations

from genes.orchestrator.models import PipelineResult
from genes.orchestrator.spec import Criticality, Mode, Status
from genes.tools._base import ToolResult


SYSTEM_PROMPT = """\
You are a clinical genomics assistant helping to interpret genome analysis results.
You have access to outputs from a comprehensive bioinformatics pipeline. Follow these
guidelines strictly:

## Communication principles
- Lead with the most clinically significant result. Plain language first, technical
  detail in parentheses.
- Distinguish diagnostic findings, incidental findings, and variants of uncertain
  significance (VUS).
- Never make a diagnosis. Present findings a clinician can use for diagnosis.
- If the pipeline status is PARTIAL or FAILED, open with what's missing and what
  inference is unsafe as a result.

## ACMG variant classification
- Present ACMG codes (PM2, PP3, BP4, ...), explain each, then state the final class:
  Pathogenic, Likely Pathogenic, VUS, Likely Benign, Benign.
- For VUS: explicitly say these do not confirm or rule out a condition.
- Group variants by gene + clinical relevance, not by position.

## Ancestry and population context
- Flag any mismatch between reported and inferred ancestry prominently — it affects
  PRS interpretation and allele-frequency context.
- PRS percentiles are valid only within the reference population. State which
  population was used; note limitations for underrepresented ancestries.

## Pharmacogenomics
- Present star alleles with metabolizer phenotype.
- Flag high-priority drug-gene interactions (CYP2D6/codeine, HLA-B*57:01/abacavir).
- PharmCAT recommendations should be confirmed with targeted clinical PGx testing.

## Somatic / tumor reporting
- Distinguish driver from passenger mutations.
- Present OncoKB evidence levels for actionable variants.
- Report mutational signatures with their etiology.
- MSI: MSI-H, MSS, or MSI-L with numeric score.

## cfDNA reporting
- Tumor fraction with clinical context (detectable vs undetectable).
- ichorCNA TF < 3% is below reliable detection.
- Griffin nucleosome footprinting reflects regulatory activity, not mutations.
- UXM cell-of-origin reflects tissue contribution to cfDNA, not tumor location.

## Epigenetic clocks
- Report biological age alongside chronological age.
- Age acceleration > 5 years is clinically meaningful.
- DunedinPACE measures pace of aging (1.0 = average); interpret relative to 1.0.

## Uncertainty and limitations
- Disclose tool failures or missing data explicitly. State the clinical impact.
- Computational predictions (SpliceAI, AlphaMissense, GPN-MSA) are NOT clinical-grade
  without functional validation.
- ClinVar classifications can change — note review status (star rating).
- Note any tool that ran in PARTIAL status: warnings may indicate missing reference
  data or capped processing.
"""


# ---------------------------------------------------------------------------
# Headline sections
# ---------------------------------------------------------------------------


_STATUS_BLURB = {
    Status.OK: "All applicable tools completed successfully.",
    Status.PARTIAL: (
        "Pipeline completed with some tools failing or partial. Interpret results "
        "with the failure list in mind — affected findings may be incomplete."
    ),
    Status.FAILED: (
        "Pipeline failed: at least one critical tool did not complete. Do not "
        "interpret downstream results as a clean read of this sample."
    ),
    Status.SKIPPED: "No tools ran.",
    Status.TIMEOUT: "Pipeline timed out.",
}


def _headline(result: PipelineResult) -> str:
    status = result.status
    lines = [
        f"# Genome Analysis Report",
        f"**Run ID:** `{result.run_id}`  ",
        f"**Mode:** {Mode(result.mode).value}  ",
        f"**Status:** `{status.value.upper()}` — {_STATUS_BLURB[status]}  ",
    ]
    if result.runtime_s is not None:
        lines.append(f"**Runtime:** {result.runtime_s:.0f}s")
    return "\n".join(lines)


def _status_breakdown(result: PipelineResult) -> str:
    lines = ["## Tool Status Breakdown"]
    buckets = [
        (Status.OK, "OK"),
        (Status.PARTIAL, "Partial"),
        (Status.FAILED, "Failed"),
        (Status.TIMEOUT, "Timed out"),
        (Status.SKIPPED, "Skipped (upstream missing)"),
    ]
    for status, label in buckets:
        names = [
            f"{n} ({Criticality(r.criticality).value})"
            for n, r in result.tool_results.items()
            if r.status == status
        ]
        if names:
            lines.append(f"- **{label}:** {', '.join(names)}")
    if result.pipeline_errors:
        lines.append(f"- **Pipeline errors:** {'; '.join(result.pipeline_errors)}")
    return "\n".join(lines)


def _tool_section(name: str, r: ToolResult) -> str:
    """One section per tool. Renders payload + warnings/errors verbatim."""
    head = f"## {name} (v{r.version}) — {r.status.value.upper()}"
    lines = [head]

    if r.criticality == Criticality.CRITICAL:
        lines.append("_Critical tool._")
    elif r.criticality == Criticality.OPTIONAL:
        lines.append("_Optional tool._")

    if r.runtime_s is not None:
        lines.append(f"- Runtime: {r.runtime_s:.1f}s")

    if r.output_paths:
        for art, path in r.output_paths.items():
            lines.append(f"- Produced `{art}`: `{path}`")

    if r.payload:
        lines.append("")
        lines.append("**Results:**")
        for key, val in r.payload.items():
            if isinstance(val, (list, dict)) and len(str(val)) > 400:
                lines.append(f"- {key}: _(large; {len(val) if hasattr(val, '__len__') else '?'} items)_")
            else:
                lines.append(f"- {key}: {val}")

    if r.warnings:
        lines.append("")
        lines.append("**Warnings:**")
        for w in r.warnings:
            lines.append(f"- {w}")

    if r.errors:
        lines.append("")
        lines.append("**Errors:**")
        for e in r.errors:
            lines.append(f"- {e}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ordering — render tools in a meaningful order per mode
# ---------------------------------------------------------------------------


_ORDER: dict[Mode, list[str]] = {
    Mode.GERMLINE: [
        "deepvariant", "vep",
        "alphamissense", "spliceai", "gpn_msa", "evee",
        "exomiser", "pharmcat",
        "cyrius", "hla",
        "ancestry", "prs", "traits",
        "annotsv", "classifycnv",
    ],
    Mode.SOMATIC: [
        "mutect2", "vep",
        "msisensor", "sigprofiler",
        "alphamissense", "spliceai", "gpn_msa",
        "oncokb_civic",
    ],
    Mode.CFDNA: ["ichorcna", "griffin", "uxm"],
    Mode.METHYLATION: ["biolearn"],
}


def _ordered_tools(result: PipelineResult) -> list[tuple[str, ToolResult]]:
    mode = Mode(result.mode)
    order = _ORDER.get(mode, [])
    seen: set[str] = set()
    out: list[tuple[str, ToolResult]] = []
    for name in order:
        if name in result.tool_results:
            out.append((name, result.tool_results[name]))
            seen.add(name)
    # Append any unexpected tools (e.g. new registrations) at the end.
    for name, r in result.tool_results.items():
        if name not in seen:
            out.append((name, r))
    return out


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def build_report_context(result: PipelineResult) -> str:
    """Convert a PipelineResult into a markdown context block for Claude."""
    sections = [_headline(result), _status_breakdown(result)]
    for name, r in _ordered_tools(result):
        sections.append(_tool_section(name, r))
    return "\n\n".join(sections)
