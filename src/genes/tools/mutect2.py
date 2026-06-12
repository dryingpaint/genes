"""Mutect2 — GATK4 somatic variant calling.

Runs the full GATK4 Mutect2 somatic calling pipeline: Mutect2 caller,
LearnReadOrientationModel, GetPileupSummaries, CalculateContamination,
and FilterMutectCalls.
"""

from __future__ import annotations

from pathlib import Path

from genes.app import app
from genes.infra.images import image_java
from genes.infra.volumes import (
    MOUNT_POPGEN,
    MOUNT_REFERENCE,
    MOUNT_WORKDIR,
    vol_popgen,
    vol_reference,
    vol_workdir,
)
from genes.orchestrator.spec import Artifact, Criticality, Mode, ToolSpec, register
from genes.tools._base import ToolResult, ToolTimer, build_result, ensure_dir, run_cmd

_GATK = "/opt/gatk-4.5.0.0/gatk"
_REFERENCE_FASTA = f"{MOUNT_REFERENCE}/GRCh38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna"
_GNOMAD_AF = f"{MOUNT_POPGEN}/gnomad/af-only-gnomad.hg38.vcf.gz"
_PON = f"{MOUNT_POPGEN}/gatk/1000g_pon.hg38.vcf.gz"


@app.function(
    image=image_java,
    volumes={
        MOUNT_REFERENCE: vol_reference,
        MOUNT_POPGEN: vol_popgen,
        MOUNT_WORKDIR: vol_workdir,
    },
    timeout=14400,
    cpu=8,
    memory=32768,
)
def call_somatic_variants(
    tumor_bam: str,
    run_id: str,
    *,
    normal_bam: str | None = None,
    intervals: str | None = None,
    tumor_sample: str = "TUMOR",
    normal_sample: str = "NORMAL",
) -> ToolResult:
    """Run the GATK4 Mutect2 somatic variant calling pipeline.

    Args:
        tumor_bam: Path to the tumor BAM file (indexed).
        run_id: Unique identifier for this run.
        normal_bam: Path to the matched normal BAM (optional; tumor-only mode if absent).
        intervals: BED/interval_list to restrict calling regions.
        tumor_sample: Tumor sample name in the BAM header.
        normal_sample: Normal sample name in the BAM header.

    Returns:
        ToolResult with filtered somatic VCF.
    """
    outdir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/mutect2")

    raw_vcf = str(outdir / "unfiltered.vcf.gz")
    f1r2_tar = str(outdir / "f1r2.tar.gz")
    orientation_model = str(outdir / "read-orientation-model.tar.gz")
    tumor_pileups = str(outdir / "tumor-pileups.table")
    normal_pileups = str(outdir / "normal-pileups.table")
    contamination_table = str(outdir / "contamination.table")
    segments_table = str(outdir / "segments.table")
    filtered_vcf = str(outdir / "filtered.vcf.gz")
    stats_file = str(outdir / "unfiltered.vcf.gz.stats")

    with ToolTimer() as timer:
        errors: list[str] = []
        warnings: list[str] = []
        summary: dict = {}
        output_paths: list[str] = []

        # Validate inputs
        if not Path(tumor_bam).exists():
            errors.append(f"Tumor BAM not found: {tumor_bam}")
        if normal_bam and not Path(normal_bam).exists():
            errors.append(f"Normal BAM not found: {normal_bam}")
        if not Path(_REFERENCE_FASTA).exists():
            errors.append(f"Reference FASTA not found: {_REFERENCE_FASTA}")

        # Step 1: Mutect2
        if not errors:
            cmd = [
                _GATK, "Mutect2",
                "-R", _REFERENCE_FASTA,
                "-I", tumor_bam,
                "-O", raw_vcf,
                "--f1r2-tar-gz", f1r2_tar,
                "--native-pair-hmm-threads", "8",
            ]
            if normal_bam:
                cmd.extend(["-I", normal_bam, "-normal", normal_sample])
            if Path(_GNOMAD_AF).exists():
                cmd.extend(["--germline-resource", _GNOMAD_AF])
            if Path(_PON).exists():
                cmd.extend(["--panel-of-normals", _PON])
            if intervals:
                cmd.extend(["-L", intervals])

            try:
                run_cmd(cmd, timeout=10800)
            except Exception as exc:
                errors.append(f"Mutect2 calling failed: {exc}")

        # Step 2: LearnReadOrientationModel
        if not errors and Path(f1r2_tar).exists():
            try:
                run_cmd([
                    _GATK, "LearnReadOrientationModel",
                    "-I", f1r2_tar,
                    "-O", orientation_model,
                ], timeout=1800)
            except Exception as exc:
                warnings.append(f"LearnReadOrientationModel failed: {exc}")

        # Step 3: GetPileupSummaries (for contamination estimation)
        if not errors and Path(_GNOMAD_AF).exists():
            try:
                run_cmd([
                    _GATK, "GetPileupSummaries",
                    "-I", tumor_bam,
                    "-V", _GNOMAD_AF,
                    "-L", _GNOMAD_AF,
                    "-O", tumor_pileups,
                ], timeout=3600)
            except Exception as exc:
                warnings.append(f"GetPileupSummaries (tumor) failed: {exc}")

            if normal_bam:
                try:
                    run_cmd([
                        _GATK, "GetPileupSummaries",
                        "-I", normal_bam,
                        "-V", _GNOMAD_AF,
                        "-L", _GNOMAD_AF,
                        "-O", normal_pileups,
                    ], timeout=3600)
                except Exception as exc:
                    warnings.append(f"GetPileupSummaries (normal) failed: {exc}")

        # Step 4: CalculateContamination
        if not errors and Path(tumor_pileups).exists():
            contam_cmd = [
                _GATK, "CalculateContamination",
                "-I", tumor_pileups,
                "-O", contamination_table,
                "--tumor-segmentation", segments_table,
            ]
            if normal_bam and Path(normal_pileups).exists():
                contam_cmd.extend(["-matched", normal_pileups])
            try:
                run_cmd(contam_cmd, timeout=600)
            except Exception as exc:
                warnings.append(f"CalculateContamination failed: {exc}")

        # Step 5: FilterMutectCalls
        if not errors:
            filter_cmd = [
                _GATK, "FilterMutectCalls",
                "-R", _REFERENCE_FASTA,
                "-V", raw_vcf,
                "-O", filtered_vcf,
                "--stats", stats_file,
            ]
            if Path(orientation_model).exists():
                filter_cmd.extend(["--ob-priors", orientation_model])
            if Path(contamination_table).exists():
                filter_cmd.extend(["--contamination-table", contamination_table])
            if Path(segments_table).exists():
                filter_cmd.extend(["--tumor-segmentation", segments_table])

            try:
                run_cmd(filter_cmd, timeout=1800)
            except Exception as exc:
                errors.append(f"FilterMutectCalls failed: {exc}")

        # Collect outputs and summary
        if Path(filtered_vcf).exists():
            output_paths.append(filtered_vcf)
            try:
                result = run_cmd(["bcftools", "stats", filtered_vcf], timeout=120)
                for line in result.stdout.splitlines():
                    if line.startswith("SN") and "number of records:" in line:
                        summary["total_variants"] = int(line.strip().split("\t")[-1])
                    elif line.startswith("SN") and "number of SNPs:" in line:
                        summary["snps"] = int(line.strip().split("\t")[-1])
                    elif line.startswith("SN") and "number of indels:" in line:
                        summary["indels"] = int(line.strip().split("\t")[-1])

                # Count PASS variants
                pass_result = run_cmd(
                    ["bcftools", "view", "-f", "PASS", filtered_vcf, "-H"],
                    timeout=120,
                )
                summary["pass_variants"] = len(pass_result.stdout.strip().splitlines())
            except Exception:
                warnings.append("Could not compute VCF stats.")

        if Path(raw_vcf).exists():
            output_paths.append(raw_vcf)

        summary["mode"] = "tumor_normal" if normal_bam else "tumor_only"

        vol_workdir.commit()

    vcf_out = filtered_vcf if Path(filtered_vcf).exists() else None
    inputs = {Artifact.TUMOR_BAM.value: tumor_bam}
    if normal_bam:
        inputs[Artifact.NORMAL_BAM.value] = normal_bam
    return build_result(
        SPEC, timer,
        inputs=inputs,
        output_paths={Artifact.VCF.value: vcf_out} if vcf_out else {},
        payload={
            "intervals": intervals,
            "stats": summary,
            "files": output_paths,
        },
        errors=errors,
        warnings=warnings,
    )


SPEC = ToolSpec(
    name="mutect2",
    version="4.5.0.0",
    modes=(Mode.SOMATIC,),
    consumes=(Artifact.TUMOR_BAM,),
    optional_consumes=(Artifact.NORMAL_BAM,),
    produces=(Artifact.VCF,),
    criticality=Criticality.CRITICAL,
    timeout_s=14400,
    reference_artifacts=("reference_genome",),
)
register(SPEC, call_somatic_variants)
