"""ClassifyCNV — ACMG/ClinGen-based copy number variant classification.

Classifies CNVs according to the ACMG/ClinGen technical standards for
interpretation of copy number gains and losses (Riggs et al., 2020).
"""

from __future__ import annotations

from pathlib import Path

from genes.app import app
from genes.infra.images import image_python_bio
from genes.infra.volumes import (
    MOUNT_CLINICAL,
    MOUNT_WORKDIR,
    vol_clinical,
    vol_workdir,
)
from genes.orchestrator.spec import Artifact, Criticality, Mode, ToolSpec, register
from genes.tools._base import ToolResult, ToolTimer, build_result, ensure_dir, run_cmd

_CLASSIFYCNV_BIN = "/opt/ClassifyCNV/ClassifyCNV.py"
_CLINVAR_CNV = f"{MOUNT_CLINICAL}/clinvar/clinvar_cnv.tsv"
_CLINGEN_DOSAGE = f"{MOUNT_CLINICAL}/clingen/dosage_sensitivity.tsv"


@app.function(
    image=image_python_bio,
    volumes={
        MOUNT_CLINICAL: vol_clinical,
        MOUNT_WORKDIR: vol_workdir,
    },
    timeout=1800,
    cpu=2,
    memory=8192,
)
def classify_cnv(
    cnv_input_path: str,
    run_id: str,
    *,
    genome_build: str = "GRCh38",
) -> ToolResult:
    """Classify CNVs using ACMG/ClinGen criteria.

    Accepts a BED file with columns: chr, start, end, type (DUP/DEL)
    or a VCF with CNV records. Outputs ACMG classification per CNV.

    Args:
        cnv_input_path: Path to CNV BED or VCF file.
        run_id: Unique run identifier.
        genome_build: Reference genome build (hg19 or hg38).
    """
    outdir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/classifycnv")

    with ToolTimer() as timer:
        errors: list[str] = []
        warnings: list[str] = []

        if not Path(cnv_input_path).exists():
            errors.append(f"CNV input file not found: {cnv_input_path}")

        # Convert genome build naming for ClassifyCNV
        build_arg = "hg38" if "38" in genome_build else "hg19"

        if not errors:
            cmd = [
                "python", _CLASSIFYCNV_BIN,
                "--infile", cnv_input_path,
                "--GenomeBuild", build_arg,
                "--outdir", str(outdir),
            ]
            try:
                run_cmd(cmd, timeout=1500)
            except Exception as exc:
                errors.append(f"ClassifyCNV failed: {exc}")

        # Parse results
        summary: dict = {}
        output_paths: list[str] = []
        results_file = outdir / "Scoresheet.txt"

        if results_file.exists():
            output_paths.append(str(results_file))
            try:
                import polars as pl

                df = pl.read_csv(str(results_file), separator="\t", infer_schema_length=0)
                summary["total_cnvs_classified"] = len(df)

                # Count by classification
                if "Classification" in df.columns:
                    class_counts = df.group_by("Classification").len().to_dicts()
                    summary["classification_counts"] = {
                        row["Classification"]: row["len"] for row in class_counts
                    }

                    # Flag pathogenic/likely pathogenic
                    pathogenic = df.filter(
                        pl.col("Classification").is_in([
                            "Pathogenic", "Likely pathogenic",
                            "pathogenic", "likely_pathogenic",
                        ])
                    )
                    if len(pathogenic) > 0:
                        summary["pathogenic_cnv_count"] = len(pathogenic)
                        # Collect gene info if available
                        gene_col = next(
                            (c for c in df.columns if c.lower() in {"gene", "genes", "gene_name"}),
                            None,
                        )
                        if gene_col:
                            summary["pathogenic_cnv_genes"] = (
                                pathogenic.select(gene_col)
                                .unique()
                                .to_series()
                                .to_list()
                            )

                # Count by type (DUP/DEL)
                type_col = next(
                    (c for c in df.columns if c.lower() in {"type", "cnv_type", "sv_type"}),
                    None,
                )
                if type_col:
                    type_counts = df.group_by(type_col).len().to_dicts()
                    summary["cnv_type_counts"] = {
                        row[type_col]: row["len"] for row in type_counts
                    }

            except Exception:
                warnings.append("Could not parse ClassifyCNV results for summary.")

        vol_workdir.commit()

    return build_result(
        SPEC, timer,
        inputs={Artifact.VCF.value: cnv_input_path},
        payload={
            "genome_build": genome_build,
            "results": summary,
            "files": output_paths,
        },
        errors=errors,
        warnings=warnings,
    )


SPEC = ToolSpec(
    name="classifycnv",
    version="1.1.1",
    modes=(Mode.GERMLINE,),
    consumes=(Artifact.VCF,),
    criticality=Criticality.OPTIONAL,
    timeout_s=1800,
    reference_artifacts=("reference_genome", "clinvar"),
)
register(SPEC, classify_cnv)
