"""AnnotSV structural variant annotation.

Annotates structural variants (SVs) from VCF or BED input using AnnotSV,
providing gene annotations, known SV databases overlap, and ACMG-based
pathogenicity classification for structural variants.
"""

from __future__ import annotations

from pathlib import Path

from genes.app import app
from genes.infra.images import image_annotsv
from genes.infra.volumes import (
    MOUNT_CLINICAL,
    MOUNT_WORKDIR,
    vol_clinical,
    vol_workdir,
)
from genes.orchestrator.spec import Artifact, Criticality, Mode, ToolSpec, register
from genes.tools._base import ToolResult, ToolTimer, build_result, ensure_dir, run_cmd

_ANNOTSV_BIN = "/opt/AnnotSV-3.4.2/bin/AnnotSV"
_ANNOTSV_ANNOTATIONS = f"{MOUNT_CLINICAL}/annotsv/annotations"


@app.function(
    image=image_annotsv,
    volumes={
        MOUNT_CLINICAL: vol_clinical,
        MOUNT_WORKDIR: vol_workdir,
    },
    timeout=3600,
    cpu=4,
    memory=8192,
)
def annotate_sv(
    sv_input_path: str,
    run_id: str,
    *,
    genome_build: str = "GRCh38",
    sv_min_size: int = 50,
    annotation_mode: str = "both",
) -> ToolResult:
    """Run AnnotSV on a structural variant VCF or BED file.

    Args:
        sv_input_path: Path to SV VCF or BED file.
        run_id: Unique run identifier.
        genome_build: Reference genome build (GRCh37 or GRCh38).
        sv_min_size: Minimum SV size in bp to annotate (default 50).
        annotation_mode: 'full', 'split', or 'both' (default 'both').
    """
    outdir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/annotsv")
    output_prefix = str(outdir / run_id)

    with ToolTimer() as timer:
        errors: list[str] = []
        warnings: list[str] = []

        if not Path(sv_input_path).exists():
            errors.append(f"SV input file not found: {sv_input_path}")

        # Determine input format from extension
        input_ext = Path(sv_input_path).suffix.lower()
        if input_ext not in {".vcf", ".gz", ".bed"}:
            warnings.append(
                f"Unexpected file extension {input_ext}; AnnotSV expects .vcf, .vcf.gz, or .bed"
            )

        if not errors:
            cmd = [
                _ANNOTSV_BIN,
                "-SVinputFile", sv_input_path,
                "-outputFile", output_prefix,
                "-genomeBuild", genome_build,
                "-SVminSize", str(sv_min_size),
                "-annotationMode", annotation_mode,
                "-overlap", "70",
                "-reciprocal", "yes",
            ]

            # Use custom annotations dir if available
            if Path(_ANNOTSV_ANNOTATIONS).exists():
                cmd.extend(["-annotationsDir", _ANNOTSV_ANNOTATIONS])

            try:
                run_cmd(cmd, timeout=3000)
            except Exception as exc:
                errors.append(f"AnnotSV failed: {exc}")

        # Parse results
        summary: dict = {}
        output_paths: list[str] = []
        results_tsv = Path(f"{output_prefix}.annotated.tsv")

        if results_tsv.exists():
            output_paths.append(str(results_tsv))
            try:
                import polars as pl

                df = pl.read_csv(str(results_tsv), separator="\t", infer_schema_length=0)
                summary["total_sv_annotations"] = len(df)

                # Count by SV type
                if "SV_type" in df.columns:
                    type_counts = df.group_by("SV_type").len().to_dicts()
                    summary["sv_type_counts"] = {
                        row["SV_type"]: row["len"] for row in type_counts
                    }

                # Count ACMG classifications
                if "ACMG_class" in df.columns:
                    acmg_counts = (
                        df.filter(pl.col("ACMG_class") != "")
                        .group_by("ACMG_class")
                        .len()
                        .to_dicts()
                    )
                    summary["acmg_class_counts"] = {
                        row["ACMG_class"]: row["len"] for row in acmg_counts
                    }

                    # Flag pathogenic/likely pathogenic SVs
                    pathogenic = df.filter(
                        pl.col("ACMG_class").is_in(["pathogenic", "likely_pathogenic"])
                    )
                    if len(pathogenic) > 0:
                        summary["pathogenic_sv_count"] = len(pathogenic)
                        summary["pathogenic_sv_genes"] = (
                            pathogenic.select("Gene_name")
                            .unique()
                            .to_series()
                            .to_list()
                        )

            except Exception:
                warnings.append("Could not parse AnnotSV results TSV for summary.")

        vol_workdir.commit()

    return build_result(
        SPEC, timer,
        inputs={Artifact.VCF.value: sv_input_path},
        payload={
            "genome_build": genome_build,
            "sv_min_size": sv_min_size,
            "annotation_mode": annotation_mode,
            "results": summary,
            "files": output_paths,
        },
        errors=errors,
        warnings=warnings,
    )


SPEC = ToolSpec(
    name="annotsv",
    version="3.4.2",
    modes=(Mode.GERMLINE,),
    consumes=(Artifact.VCF,),
    criticality=Criticality.OPTIONAL,
    timeout_s=3600,
    reference_artifacts=("reference_genome", "clinvar"),
)
register(SPEC, annotate_sv)
