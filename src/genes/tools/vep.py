"""Ensembl VEP variant annotation tool wrapper.

Runs Ensembl Variant Effect Predictor (VEP) on a VCF file using the
pre-built cache and plugin data. Outputs an annotated VCF with consequence
predictions, gene annotations, and optional AlphaMissense scores.
"""

from __future__ import annotations

from pathlib import Path

from genes.app import app
from genes.infra.images import image_vep
from genes.infra.volumes import (
    MOUNT_CLINICAL,
    MOUNT_REFERENCE,
    MOUNT_VEP,
    MOUNT_WORKDIR,
    vol_clinical,
    vol_reference,
    vol_vep,
    vol_workdir,
)
from genes.tools._base import ToolResult, ToolTimer, ensure_dir, run_cmd

VEP_CACHE_DIR = f"{MOUNT_VEP}/cache"
FASTA_PATH = f"{MOUNT_REFERENCE}/GRCh38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna"
ALPHAMISSENSE_PLUGIN = f"{MOUNT_VEP}/plugins/AlphaMissense_hg38.tsv.gz"
CLINVAR_VCF = f"{MOUNT_CLINICAL}/clinvar/clinvar.vcf.gz"


@app.function(
    image=image_vep,
    volumes={
        MOUNT_REFERENCE: vol_reference,
        MOUNT_VEP: vol_vep,
        MOUNT_CLINICAL: vol_clinical,
        MOUNT_WORKDIR: vol_workdir,
    },
    timeout=7200,
    cpu=4,
    memory=16384,
)
def annotate(
    vcf_path: str,
    run_id: str,
    *,
    assembly: str = "GRCh38",
    use_alphamissense: bool = True,
    extra_flags: list[str] | None = None,
) -> ToolResult:
    """Run VEP on a VCF file and return an annotated VCF.

    Parameters
    ----------
    vcf_path:
        Path to input VCF (must be on a mounted volume).
    run_id:
        Unique run identifier for output directory isolation.
    assembly:
        Genome assembly (default GRCh38).
    use_alphamissense:
        Whether to enable the AlphaMissense VEP plugin.
    extra_flags:
        Additional VEP CLI flags to append.
    """
    out_dir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/vep")
    input_vcf = Path(vcf_path)
    output_vcf = out_dir / f"{input_vcf.stem}.vep.vcf"
    output_stats = out_dir / f"{input_vcf.stem}.vep.html"

    warnings: list[str] = []

    cmd = [
        "vep",
        "--input_file", str(input_vcf),
        "--output_file", str(output_vcf),
        "--stats_file", str(output_stats),
        "--assembly", assembly,
        "--cache",
        "--dir_cache", VEP_CACHE_DIR,
        "--merged",
        "--vcf",
        "--offline",
        "--fasta", FASTA_PATH,
        "--force_overwrite",
        "--no_escape",
        "--hgvs",
        "--symbol",
        "--canonical",
        "--biotype",
        "--regulatory",
        "--protein",
        "--af",
        "--af_gnomade",
        "--max_af",
        "--sift", "b",
        "--polyphen", "b",
        "--numbers",
        "--domains",
        "--fork", "4",
    ]

    # AlphaMissense plugin (optional — file may not exist yet on first run)
    if use_alphamissense and Path(ALPHAMISSENSE_PLUGIN).exists():
        cmd.extend([
            "--plugin", f"AlphaMissense,file={ALPHAMISSENSE_PLUGIN}",
        ])
    elif use_alphamissense:
        warnings.append(
            f"AlphaMissense plugin data not found at {ALPHAMISSENSE_PLUGIN}; "
            "skipping plugin."
        )

    # ClinVar custom annotation
    if Path(CLINVAR_VCF).exists():
        cmd.extend([
            "--custom",
            f"{CLINVAR_VCF},ClinVar,vcf,exact,0,CLNSIG,CLNREVSTAT,CLNDN",
        ])

    if extra_flags:
        cmd.extend(extra_flags)

    with ToolTimer() as timer:
        errors: list[str] = []
        try:
            proc = run_cmd(cmd, timeout=7000)
            if proc.stderr:
                for line in proc.stderr.strip().splitlines():
                    if "WARNING" in line.upper():
                        warnings.append(line.strip())
        except Exception as e:
            errors.append(str(e))

    # Count annotated variants from the output VCF
    variant_count = 0
    output_paths: list[str] = []
    if output_vcf.exists():
        output_paths.append(str(output_vcf))
        with open(output_vcf) as fh:
            variant_count = sum(1 for line in fh if not line.startswith("#"))
    if output_stats.exists():
        output_paths.append(str(output_stats))

    return ToolResult(
        tool_name="vep",
        version="112.0",
        started_at=timer.started_at,
        completed_at=timer.completed_at,
        input_summary={
            "vcf_path": vcf_path,
            "run_id": run_id,
            "assembly": assembly,
            "use_alphamissense": use_alphamissense,
        },
        output_paths=output_paths,
        output_summary={
            "annotated_vcf": str(output_vcf),
            "variant_count": variant_count,
        },
        errors=errors,
        warnings=warnings,
    )
