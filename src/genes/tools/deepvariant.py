"""DeepVariant — GPU germline variant calling.

Runs Google DeepVariant on a BAM file to produce a germline VCF.
Supports WGS, WES, PACBIO, and ONT_R104 model types.
"""

from __future__ import annotations

from pathlib import Path

import modal

from genes.app import app
from genes.infra.images import image_gpu_jax
from genes.infra.volumes import (
    MOUNT_MODELS,
    MOUNT_REFERENCE,
    MOUNT_WORKDIR,
    vol_models,
    vol_reference,
    vol_workdir,
)
from genes.infra.provenance import PROVENANCE_KEY, stamp
from genes.tools._base import ToolResult, ToolTimer, ensure_dir, run_cmd

_VALID_MODEL_TYPES = {"WGS", "WES", "PACBIO", "ONT_R104"}
_REFERENCE_FASTA = f"{MOUNT_REFERENCE}/GRCh38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna"


@app.function(
    image=image_gpu_jax,
    gpu="A100",
    volumes={
        MOUNT_REFERENCE: vol_reference,
        MOUNT_MODELS: vol_models,
        MOUNT_WORKDIR: vol_workdir,
    },
    timeout=7200,
)
def call_variants(
    bam_path: str,
    run_id: str,
    model_type: str = "WGS",
    regions_bed: str | None = None,
    num_shards: int = 1,
) -> ToolResult:
    """Run DeepVariant on a single BAM to produce a germline VCF.

    Args:
        bam_path: Path to the input BAM file (must be indexed).
        run_id: Unique identifier for this run; outputs go to /work/{run_id}/.
        model_type: One of WGS, WES, PACBIO, ONT_R104.
        regions_bed: Optional BED file to restrict calling regions.
        num_shards: Number of shards for make_examples (default 1).

    Returns:
        ToolResult with VCF and gVCF output paths.
    """
    if model_type not in _VALID_MODEL_TYPES:
        raise ValueError(f"model_type must be one of {_VALID_MODEL_TYPES}, got {model_type!r}")

    outdir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/deepvariant")
    output_vcf = str(outdir / "output.vcf.gz")
    output_gvcf = str(outdir / "output.g.vcf.gz")

    with ToolTimer() as timer:
        errors: list[str] = []
        warnings: list[str] = []

        # Validate inputs
        if not Path(bam_path).exists():
            errors.append(f"BAM file not found: {bam_path}")
        if not Path(_REFERENCE_FASTA).exists():
            errors.append(f"Reference FASTA not found: {_REFERENCE_FASTA}")

        if not errors:
            cmd = [
                "/opt/deepvariant/bin/run_deepvariant",
                f"--model_type={model_type}",
                f"--ref={_REFERENCE_FASTA}",
                f"--reads={bam_path}",
                f"--output_vcf={output_vcf}",
                f"--output_gvcf={output_gvcf}",
                f"--num_shards={num_shards}",
                f"--intermediate_results_dir={outdir}/tmp",
            ]
            if regions_bed:
                cmd.append(f"--regions={regions_bed}")

            try:
                run_cmd(cmd, timeout=7000)
            except Exception as exc:
                errors.append(str(exc))

        # Collect output summary
        output_paths = []
        summary: dict = {}
        if Path(output_vcf).exists():
            output_paths.append(output_vcf)
            # Count variants in VCF
            try:
                result = run_cmd(["bcftools", "stats", output_vcf], timeout=120)
                for line in result.stdout.splitlines():
                    if line.startswith("SN") and "number of records:" in line:
                        summary["total_variants"] = int(line.strip().split("\t")[-1])
                    elif line.startswith("SN") and "number of SNPs:" in line:
                        summary["snps"] = int(line.strip().split("\t")[-1])
                    elif line.startswith("SN") and "number of indels:" in line:
                        summary["indels"] = int(line.strip().split("\t")[-1])
            except Exception:
                warnings.append("Could not compute VCF stats.")

        if Path(output_gvcf).exists():
            output_paths.append(output_gvcf)

        vol_workdir.commit()

    return ToolResult(
        tool_name="deepvariant",
        version="1.6.1",
        started_at=timer.started_at,
        completed_at=timer.completed_at,
        input_summary={
            "bam_path": bam_path,
            "run_id": run_id,
            "model_type": model_type,
            "regions_bed": regions_bed,
            PROVENANCE_KEY: stamp("reference_genome", "deepvariant_model"),
        },
        output_paths=output_paths,
        output_summary=summary,
        errors=errors,
        warnings=warnings,
    )
