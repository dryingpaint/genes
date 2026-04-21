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
    """Extract diplotypes and recommendations from the PharmCAT v2 JSON report.

    PharmCAT v2 report structure varies — try multiple key paths.
    """
    with open(report_path) as fh:
        report = json.load(fh)

    diplotypes: list[dict] = []
    recommendations: list[dict] = []

    # PharmCAT v2.13: top-level "genes" list
    gene_reports = (
        report.get("genes", [])
        or report.get("reportContext", {}).get("geneReports", [])
        or report.get("geneCalls", [])
    )
    for gr in gene_reports:
        gene = gr.get("gene", gr.get("geneSymbol", ""))
        # Diplotype may be nested under recommendationDiplotypes or directly
        diplotype_obj = gr.get("recommendationDiplotypes", [{}])
        if isinstance(diplotype_obj, list) and diplotype_obj:
            diplotype = diplotype_obj[0].get("label", "")
            activity = diplotype_obj[0].get("activityScore", None)
        else:
            diplotype = gr.get("diplotype", gr.get("printDiplotype", ""))
            activity = gr.get("activityScore")

        phenotype = gr.get("phenotype", gr.get("phenotypes", {}).get("term", ""))
        if gene:
            entry = {"gene": gene, "diplotype": diplotype, "phenotype": phenotype}
            if activity is not None:
                entry["activity_score"] = activity
            diplotypes.append(entry)

    # PharmCAT v2.13: top-level "drugs" list
    drug_reports = (
        report.get("drugs", [])
        or report.get("prescribingGuidanceReports", [])
        or report.get("drugReports", [])
    )
    for dr in drug_reports:
        drug = dr.get("drug", dr.get("drugName", ""))
        source = dr.get("source", "")
        classification = dr.get("classification", "")
        rec_text = dr.get("recommendation", dr.get("prescribingInfo", ""))
        if drug:
            recommendations.append({
                "drug": drug,
                "source": source,
                "classification": classification,
                "recommendation": str(rec_text)[:500],
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

    # PharmCAT can't handle full WGS VCFs (OutOfMemoryError on readAllBytes).
    # Filter to PGx gene regions first with bcftools, then decompress.
    actual_vcf = vcf_path
    try:
        # PharmCAT PGx regions (CPIC genes on GRCh38) — major loci
        pgx_regions = (
            "chr1:97078528-97110987 "   # DPYD
            "chr7:99245817-99277621 "   # CYP3A5
            "chr7:99354604-99464528 "   # CYP3A4
            "chr10:94761900-94853547 "  # CYP2C19
            "chr10:94938658-94990091 "  # CYP2C9
            "chr10:94942205-94989978 "  # CYP2C8
            "chr13:48037580-48070790 "  # NUDT15
            "chr16:31093097-31110537 "  # VKORC1
            "chr19:38924339-38947371 "  # RYR1
            "chr19:15879372-15884328 "  # CYP4F2
            "chr22:42126499-42130881 "  # MTHFR
            "chr6:18130809-18232467 "   # TPMT
            "chr22:42512500-42551899 "  # CYP2D6 region
        )
        filtered_vcf = str(out_dir / "pgx_filtered.vcf")
        run_cmd([
            "bcftools", "view", vcf_path,
            "-r", pgx_regions.strip().replace(" ", ","),
            "-O", "v", "-o", filtered_vcf,
        ], timeout=120)
        actual_vcf = filtered_vcf
        warnings.append("Filtered VCF to PGx regions before PharmCAT")
    except Exception as e:
        warnings.append(f"PGx region filtering failed ({e}); trying with full VCF")
        # Fallback: decompress full VCF (may OOM)
        if vcf_path.endswith(".gz"):
            decompressed = str(out_dir / "input.vcf")
            run_cmd(["bcftools", "view", vcf_path, "-O", "v", "-o", decompressed], timeout=600)
            actual_vcf = decompressed

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
            # Debug: include top-level report keys for troubleshooting
            with open(json_reports[0]) as _fh:
                _raw = json.load(_fh)
                output_summary["_report_keys"] = list(_raw.keys())[:20]
                # Dump first gene entry for structure debugging
                genes_list = _raw.get("genes", [])
                if genes_list:
                    first_gene = genes_list[0]
                    output_summary["_first_gene_keys"] = list(first_gene.keys()) if isinstance(first_gene, dict) else str(type(first_gene))
                    output_summary["_first_gene_sample"] = json.dumps(first_gene, default=str)[:1000]
                drugs_list = _raw.get("drugs", [])
                if drugs_list:
                    output_summary["_first_drug_keys"] = list(drugs_list[0].keys()) if isinstance(drugs_list[0], dict) else str(type(drugs_list[0]))
                    output_summary["_first_drug_sample"] = json.dumps(drugs_list[0], default=str)[:1000]
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
