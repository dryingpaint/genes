"""Cyrius — CYP2D6 star-allele diplotype calling.

Calls CYP2D6 star alleles from a WGS BAM file using the Cyrius package,
which handles the complex CYP2D6 locus (paralog CYP2D7, gene conversions,
hybrids, and copy-number variation).
"""

from __future__ import annotations

import json
from pathlib import Path

from genes.app import app
from genes.infra.images import image_python_bio
from genes.infra.volumes import (
    MOUNT_REFERENCE,
    MOUNT_WORKDIR,
    vol_reference,
    vol_workdir,
)
from genes.orchestrator.spec import Artifact, Criticality, Mode, ToolSpec, register
from genes.tools._base import ToolResult, ToolTimer, build_result, ensure_dir, run_cmd

_REFERENCE_FASTA = f"{MOUNT_REFERENCE}/GRCh38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna"


@app.function(
    image=image_python_bio,
    volumes={
        MOUNT_REFERENCE: vol_reference,
        MOUNT_WORKDIR: vol_workdir,
    },
    timeout=1800,
)
def call_cyp2d6(
    bam_path: str,
    run_id: str,
    *,
    genome_build: str = "38",
) -> ToolResult:
    """Call CYP2D6 diplotypes from a WGS BAM file.

    Args:
        bam_path: Path to the input BAM file (must be indexed).
        run_id: Unique identifier for this run.
        genome_build: Reference build, "38" (default) or "37".

    Returns:
        ToolResult with CYP2D6 diplotype and activity score.
    """
    outdir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/cyrius")
    output_prefix = str(outdir / "cyp2d6")

    with ToolTimer() as timer:
        errors: list[str] = []
        warnings: list[str] = []
        summary: dict = {}

        if not Path(bam_path).exists():
            errors.append(f"BAM file not found: {bam_path}")

        if not errors:
            # Cyrius is invoked via its CLI entry point: star_caller
            cmd = [
                "python", "-m", "cyrius.star_caller",
                "--manifest", bam_path,
                "--genome", genome_build,
                "--prefix", output_prefix,
                "--outDir", str(outdir),
                "--threads", "4",
            ]

            try:
                run_cmd(cmd, timeout=1500)
            except Exception as exc:
                errors.append(f"Cyrius execution failed: {exc}")

        # Parse results
        output_paths: list[str] = []
        result_json = Path(f"{output_prefix}.json")
        if result_json.exists():
            output_paths.append(str(result_json))
            try:
                with open(result_json) as fh:
                    data = json.load(fh)
                # Cyrius outputs a dict keyed by sample name
                for sample_name, result in data.items():
                    summary["sample"] = sample_name
                    summary["diplotype"] = result.get("Diplotype", "No call")
                    summary["genotype"] = result.get("Genotype", "")
                    summary["activity_score"] = result.get("Activity_Score")
                    summary["phenotype"] = result.get("Phenotype", "")
                    summary["filter"] = result.get("Filter", "PASS")
                    break  # single-sample
            except Exception as exc:
                warnings.append(f"Failed to parse Cyrius output: {exc}")

        tsv_path = Path(f"{output_prefix}.tsv")
        if tsv_path.exists():
            output_paths.append(str(tsv_path))

        vol_workdir.commit()

    return build_result(
        SPEC, timer,
        inputs={Artifact.BAM.value: bam_path},
        payload={"genome_build": genome_build, "results": summary, "files": output_paths},
        errors=errors,
        warnings=warnings,
    )


SPEC = ToolSpec(
    name="cyrius",
    version="1.1",
    modes=(Mode.GERMLINE,),
    consumes=(Artifact.BAM,),
    criticality=Criticality.STANDARD,
    timeout_s=900,
    reference_artifacts=("reference_genome",),
)
register(SPEC, call_cyp2d6)
