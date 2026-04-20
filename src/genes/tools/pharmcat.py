"""PharmCAT pharmacogenomics tool wrapper.

Runs PharmCAT on a VCF to produce star-allele diplotype calls and
CPIC-guideline-based drug recommendations. PharmCAT covers ~20 key
pharmacogenes (CYP2D6, CYP2C19, DPYD, TPMT, UGT1A1, etc.) and maps
diplotypes to metabolizer phenotypes and actionable prescribing guidance.
"""

from __future__ import annotations

import json
from pathlib import Path

from genes.app import app
from genes.infra.images import image_java
from genes.infra.volumes import (
    MOUNT_REFERENCE,
    MOUNT_WORKDIR,
    vol_reference,
    vol_workdir,
)
from genes.tools._base import ToolResult, ToolTimer, ensure_dir, run_cmd

PHARMCAT_JAR = "/opt/pharmcat.jar"


def _parse_pharmcat_report(report_path: Path) -> dict:
    """Extract diplotypes and recommendations from the PharmCAT JSON report."""
    with open(report_path) as fh:
        report = json.load(fh)

    diplotypes: list[dict] = []
    recommendations: list[dict] = []

    # Parse gene calls
    for gene_call in report.get("geneCalls", []):
        gene = gene_call.get("gene", "")
        diplotype = gene_call.get("diplotype", "")
        phenotype = gene_call.get("phenotype", "")
        activity_score = gene_call.get("activityScore")
        if gene:
            entry = {
                "gene": gene,
                "diplotype": diplotype,
                "phenotype": phenotype,
            }
            if activity_score is not None:
                entry["activity_score"] = activity_score
            diplotypes.append(entry)

    # Parse drug recommendations
    for drug_report in report.get("drugReports", []):
        drug = drug_report.get("drug", "")
        source = drug_report.get("source", "")
        classification = drug_report.get("classification", "")
        guideline_url = drug_report.get("guidelineUrl", "")
        implications = drug_report.get("implications", [])
        rec_text = drug_report.get("recommendation", "")
        if drug:
            recommendations.append({
                "drug": drug,
                "source": source,
                "classification": classification,
                "guideline_url": guideline_url,
                "implications": implications,
                "recommendation": rec_text,
            })

    return {
        "diplotypes": diplotypes,
        "recommendations": recommendations,
        "n_genes_called": len(diplotypes),
        "n_drug_recommendations": len(recommendations),
    }


@app.function(
    image=image_java,
    volumes={
        MOUNT_REFERENCE: vol_reference,
        MOUNT_WORKDIR: vol_workdir,
    },
    timeout=3600,
    cpu=2,
    memory=8192,
)
def run(
    vcf_path: str,
    run_id: str,
    *,
    sample_id: str | None = None,
) -> ToolResult:
    """Run PharmCAT on a VCF to get pharmacogenomic diplotypes and recommendations.

    Parameters
    ----------
    vcf_path:
        Path to input VCF (must contain PGx gene regions).
    run_id:
        Unique run identifier.
    sample_id:
        Optional sample ID to select from a multi-sample VCF.
    """
    out_dir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/pharmcat")
    warnings: list[str] = []
    errors: list[str] = []

    # PharmCAT v2 requires its preprocessor for WGS VCFs.
    # The preprocessor normalizes, filters to PGx regions, and handles multi-allelic sites.
    actual_vcf = vcf_path
    try:
        # Decompress if needed
        if vcf_path.endswith(".gz"):
            import subprocess as _sp
            decompressed = str(out_dir / "input.vcf")
            _sp.run(["bcftools", "view", vcf_path, "-O", "v", "-o", decompressed], check=True)
            actual_vcf = decompressed

        # Run PharmCAT preprocessor
        preprocess_cmd = [
            "java", "-cp", PHARMCAT_JAR,
            "org.pharmgkb.pharmcat.VcfPreprocessor",
            "-vcf", actual_vcf,
            "-o", str(out_dir),
        ]
        run_cmd(preprocess_cmd, timeout=600)

        # Find preprocessed VCF
        preprocessed = list(out_dir.glob("*.preprocessed.vcf"))
        if preprocessed:
            actual_vcf = str(preprocessed[0])
            warnings.append(f"Used PharmCAT preprocessor: {preprocessed[0].name}")
        else:
            warnings.append("PharmCAT preprocessor produced no output; using raw VCF")
    except Exception as e:
        warnings.append(f"PharmCAT preprocessing failed ({e}); using raw VCF")

    cmd = [
        "java", "-jar", PHARMCAT_JAR,
        "-vcf", actual_vcf,
        "-o", str(out_dir),
        "-reporterJson",
    ]

    if sample_id:
        cmd.extend(["-sample", sample_id])

    with ToolTimer() as timer:
        try:
            proc = run_cmd(cmd, check=False, timeout=3500)
            if proc.returncode != 0:
                errors.append(f"PharmCAT exited with code {proc.returncode}")
                if proc.stderr:
                    errors.append(proc.stderr.strip()[:2000])
                if proc.stdout:
                    warnings.append(proc.stdout.strip()[:2000])
            elif proc.stderr:
                for line in proc.stderr.strip().splitlines():
                    line_lower = line.lower()
                    if "warn" in line_lower:
                        warnings.append(line.strip())
                    elif "error" in line_lower:
                        errors.append(line.strip())
        except Exception as e:
            errors.append(str(e))

    # Collect output files
    output_paths: list[str] = []
    output_summary: dict = {}

    json_reports = list(out_dir.glob("*.report.json"))
    html_reports = list(out_dir.glob("*.report.html"))

    for f in json_reports + html_reports:
        output_paths.append(str(f))

    # Parse structured results from the JSON report
    if json_reports:
        try:
            parsed = _parse_pharmcat_report(json_reports[0])
            output_summary = parsed
        except Exception as e:
            warnings.append(f"Failed to parse PharmCAT report: {e}")
            output_summary = {"parse_error": str(e)}
    elif not errors:
        warnings.append("No PharmCAT report generated; check input VCF coverage.")

    return ToolResult(
        tool_name="pharmcat",
        version="2.13.0",
        started_at=timer.started_at,
        completed_at=timer.completed_at,
        input_summary={
            "vcf_path": vcf_path,
            "run_id": run_id,
            "sample_id": sample_id,
        },
        output_paths=output_paths,
        output_summary=output_summary,
        errors=errors,
        warnings=warnings,
    )
