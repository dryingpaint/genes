"""Exomiser — Mendelian disease gene/variant prioritization.

Runs the Exomiser Java CLI to prioritize variants from a VCF using
patient phenotype (HPO terms), cross-species phenotype comparisons,
and variant pathogenicity data.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from genes.app import app
from genes.infra.images import image_java
from genes.infra.volumes import (
    MOUNT_EXOMISER,
    MOUNT_REFERENCE,
    MOUNT_WORKDIR,
    vol_exomiser,
    vol_reference,
    vol_workdir,
)
from genes.tools._base import ToolResult, ToolTimer, ensure_dir, run_cmd

_EXOMISER_JAR = "/opt/exomiser-cli-14.0.0/exomiser-cli-14.0.0.jar"
_EXOMISER_DATA = f"{MOUNT_EXOMISER}/2402_hg38"

_HPO_PATTERN = re.compile(r"^HP:\d{7}$")


def _build_analysis_yaml(
    vcf_path: str,
    hpo_terms: list[str],
    outdir: Path,
    inheritance_modes: list[str] | None = None,
    frequency_threshold: float = 0.01,
) -> str:
    """Build the Exomiser analysis YAML configuration."""
    if inheritance_modes is None:
        inheritance_modes = [
            "AUTOSOMAL_DOMINANT",
            "AUTOSOMAL_RECESSIVE",
            "X_DOMINANT",
            "X_RECESSIVE",
        ]

    analysis = {
        "analysis": {
            "vcf": vcf_path,
            "genomeAssembly": "hg38",
            "hpoIds": hpo_terms,
            "inheritanceModes": {mode: 0.1 for mode in inheritance_modes},
            "analysisMode": "PASS_ONLY",
            "frequencySources": [
                "THOUSAND_GENOMES",
                "TOPMED",
                "UK10K",
                "ESP_ALL",
                "GNOMAD_E_AFR", "GNOMAD_E_AMR", "GNOMAD_E_EAS",
                "GNOMAD_E_NFE", "GNOMAD_E_SAS",
                "GNOMAD_G_AFR", "GNOMAD_G_AMR", "GNOMAD_G_EAS",
                "GNOMAD_G_NFE", "GNOMAD_G_SAS",
            ],
            "pathogenicitySources": [
                "POLYPHEN", "MUTATION_TASTER", "SIFT", "CADD", "REVEL", "MVP",
            ],
            "steps": [
                {"variantEffectFilter": {"remove": ["UPSTREAM_GENE_VARIANT", "DOWNSTREAM_GENE_VARIANT", "INTERGENIC_VARIANT"]}},
                {"frequencyFilter": {"maxFrequency": frequency_threshold * 100}},
                {"pathogenicityFilter": {"keepNonPathogenic": True}},
                {"inheritanceFilter": {}},
                {"omimPrioritiser": {}},
                {"hiPhivePrioritiser": {}},
            ],
        },
        "outputOptions": {
            "outputContributingVariantsOnly": False,
            "numGenes": 50,
            "outputPrefix": str(outdir / "exomiser_results"),
            "outputFormats": ["HTML", "JSON", "TSV_GENE", "TSV_VARIANT"],
        },
    }
    yaml_path = str(outdir / "analysis.yml")
    with open(yaml_path, "w") as fh:
        import yaml  # Available in Modal container via image_java pip deps

        yaml.dump(analysis, fh, default_flow_style=False)
    return yaml_path


@app.function(
    image=image_java,
    volumes={
        MOUNT_EXOMISER: vol_exomiser,
        MOUNT_REFERENCE: vol_reference,
        MOUNT_WORKDIR: vol_workdir,
    },
    timeout=3600,
    memory=16384,
)
def prioritize_variants(
    vcf_path: str,
    hpo_terms: list[str],
    run_id: str,
    inheritance_modes: list[str] | None = None,
    frequency_threshold: float = 0.01,
) -> ToolResult:
    """Run Exomiser to prioritize variants for Mendelian disease diagnosis.

    Args:
        vcf_path: Path to the input VCF (single proband or family).
        hpo_terms: List of HPO term IDs (e.g. ["HP:0001250", "HP:0001263"]).
        run_id: Unique identifier for this run.
        inheritance_modes: Inheritance modes to test (default: AD, AR, XD, XR).
        frequency_threshold: Maximum population allele frequency (default 0.01).

    Returns:
        ToolResult with top prioritized genes and variants.
    """
    outdir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/exomiser")

    with ToolTimer() as timer:
        errors: list[str] = []
        warnings: list[str] = []
        summary: dict = {}

        # Validate inputs
        if not Path(vcf_path).exists():
            errors.append(f"VCF file not found: {vcf_path}")

        invalid_hpo = [t for t in hpo_terms if not _HPO_PATTERN.match(t)]
        if invalid_hpo:
            errors.append(f"Invalid HPO terms: {invalid_hpo}")

        if not hpo_terms:
            errors.append("At least one HPO term is required.")

        if not errors:
            yaml_path = _build_analysis_yaml(
                vcf_path=vcf_path,
                hpo_terms=hpo_terms,
                outdir=outdir,
                inheritance_modes=inheritance_modes,
                frequency_threshold=frequency_threshold,
            )

            cmd = [
                "java", "-Xmx12g",
                f"-Dexomiser.data-directory={_EXOMISER_DATA}",
                "-jar", _EXOMISER_JAR,
                "--analysis", yaml_path,
            ]

            try:
                run_cmd(cmd, timeout=3400)
            except Exception as exc:
                errors.append(f"Exomiser execution failed: {exc}")

        # Parse results
        output_paths: list[str] = []
        results_json = outdir / "exomiser_results.json"
        results_html = outdir / "exomiser_results.html"
        results_gene_tsv = outdir / "exomiser_results.genes.tsv"
        results_var_tsv = outdir / "exomiser_results.variants.tsv"

        for p in [results_json, results_html, results_gene_tsv, results_var_tsv]:
            if p.exists():
                output_paths.append(str(p))

        if results_json.exists():
            try:
                with open(results_json) as fh:
                    data = json.load(fh)
                # Extract top genes
                top_genes = []
                for gene_result in data[:10]:
                    top_genes.append({
                        "rank": gene_result.get("geneRank"),
                        "gene_symbol": gene_result.get("geneSymbol"),
                        "gene_id": gene_result.get("geneId"),
                        "combined_score": gene_result.get("combinedScore"),
                        "phenotype_score": gene_result.get("phenotypeScore"),
                        "variant_score": gene_result.get("variantScore"),
                        "moi": gene_result.get("modeOfInheritance"),
                    })
                summary["top_genes"] = top_genes
                summary["total_genes_scored"] = len(data)
            except Exception as exc:
                warnings.append(f"Failed to parse Exomiser JSON: {exc}")

        vol_workdir.commit()

    return ToolResult(
        tool_name="exomiser",
        version="14.0.0",
        started_at=timer.started_at,
        completed_at=timer.completed_at,
        input_summary={
            "vcf_path": vcf_path,
            "hpo_terms": hpo_terms,
            "run_id": run_id,
            "inheritance_modes": inheritance_modes or ["AD", "AR", "XD", "XR"],
            "frequency_threshold": frequency_threshold,
        },
        output_paths=output_paths,
        output_summary=summary,
        errors=errors,
        warnings=warnings,
    )
