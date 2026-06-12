"""Ensembl VEP variant annotation tool wrapper.

Runs Ensembl Variant Effect Predictor (VEP) on a VCF file using the
pre-built cache and plugin data. Outputs an annotated VCF with consequence
predictions, gene annotations, and optional AlphaMissense scores.

For full-genome VCFs (>500K variants), the VCF is automatically split by
chromosome and each chunk is annotated in parallel, then merged.
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
from genes.orchestrator.spec import Artifact, Criticality, Mode, ToolSpec, register
from genes.tools._base import ToolResult, ToolTimer, build_result, ensure_dir, run_cmd

VEP_CACHE_DIR = f"{MOUNT_VEP}"
FASTA_PATH = f"{MOUNT_REFERENCE}/GRCh38.fa"
ALPHAMISSENSE_PLUGIN = f"{MOUNT_VEP}/plugins/AlphaMissense_hg38.tsv.gz"
CLINVAR_VCF = f"{MOUNT_CLINICAL}/clinvar/clinvar.vcf.gz"

# If a VCF has more variants than this, split by chromosome and run in parallel
_PARALLEL_THRESHOLD = 500_000


def _build_vep_cmd(
    input_vcf: str,
    output_vcf: str,
    output_stats: str,
    assembly: str,
    use_alphamissense: bool,
    extra_flags: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Build VEP command line. Returns (cmd, warnings)."""
    warnings: list[str] = []

    cmd = [
        "vep",
        "--input_file", input_vcf,
        "--output_file", output_vcf,
        "--stats_file", output_stats,
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

    if use_alphamissense and Path(ALPHAMISSENSE_PLUGIN).exists():
        cmd.extend(["--plugin", f"AlphaMissense,file={ALPHAMISSENSE_PLUGIN}"])
    elif use_alphamissense:
        warnings.append(
            f"AlphaMissense plugin data not found at {ALPHAMISSENSE_PLUGIN}; "
            "skipping plugin."
        )

    if Path(CLINVAR_VCF).exists():
        cmd.extend([
            "--custom",
            f"{CLINVAR_VCF},ClinVar,vcf,exact,0,CLNSIG,CLNREVSTAT,CLNDN",
        ])

    if extra_flags:
        cmd.extend(extra_flags)

    return cmd, warnings


def _count_variants(vcf_path: str) -> int:
    """Fast line count of non-header lines in a VCF."""
    count = 0
    import gzip
    opener = gzip.open if vcf_path.endswith(".gz") else open
    with opener(vcf_path, "rt") as fh:
        for line in fh:
            if not line.startswith("#"):
                count += 1
    return count


def _split_vcf_by_chrom(vcf_path: str, out_dir: Path) -> list[tuple[str, str]]:
    """Split a VCF into per-chromosome files. Returns list of (chrom, path)."""
    import gzip

    opener = gzip.open if vcf_path.endswith(".gz") else open
    header_lines: list[str] = []
    chrom_files: dict[str, any] = {}
    chrom_order: list[str] = []

    with opener(vcf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                header_lines.append(line)
                continue
            chrom = line.split("\t", 1)[0]
            if chrom not in chrom_files:
                chrom_path = str(out_dir / f"input_{chrom}.vcf")
                chrom_fh = open(chrom_path, "w")
                for h in header_lines:
                    chrom_fh.write(h)
                chrom_files[chrom] = (chrom_path, chrom_fh)
                chrom_order.append(chrom)
            chrom_files[chrom][1].write(line)

    result = []
    for chrom in chrom_order:
        path, fh = chrom_files[chrom]
        fh.close()
        result.append((chrom, path))

    return result


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
def annotate_chunk(
    vcf_path: str,
    output_vcf: str,
    output_stats: str,
    assembly: str = "GRCh38",
    use_alphamissense: bool = True,
    extra_flags: list[str] | None = None,
) -> dict:
    """Annotate a single VCF chunk with VEP. Used by parallel dispatch."""
    cmd, warnings = _build_vep_cmd(
        vcf_path, output_vcf, output_stats, assembly, use_alphamissense, extra_flags,
    )
    errors: list[str] = []

    try:
        proc = run_cmd(cmd, check=False, timeout=7000)
        if proc.returncode != 0:
            errors.append(f"VEP exited with code {proc.returncode}")
            if proc.stderr:
                errors.append(proc.stderr.strip()[:2000])
        elif proc.stderr:
            for line in proc.stderr.strip().splitlines():
                if "WARNING" in line.upper():
                    warnings.append(line.strip())
    except Exception as e:
        errors.append(str(e))

    variant_count = 0
    if Path(output_vcf).exists():
        with open(output_vcf) as fh:
            variant_count = sum(1 for line in fh if not line.startswith("#"))

    vol_workdir.commit()
    return {"output_vcf": output_vcf, "variant_count": variant_count,
            "errors": errors, "warnings": warnings}


@app.function(
    image=image_vep,
    volumes={
        MOUNT_REFERENCE: vol_reference,
        MOUNT_VEP: vol_vep,
        MOUNT_CLINICAL: vol_clinical,
        MOUNT_WORKDIR: vol_workdir,
    },
    timeout=14400,  # 4 hours for full genome orchestration
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

    For large VCFs (>500K variants), automatically splits by chromosome
    and runs VEP in parallel for each, then merges results.

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

    with ToolTimer() as timer:
        errors: list[str] = []
        warnings: list[str] = []

        # Count variants to decide single vs parallel
        try:
            n_variants = _count_variants(vcf_path)
        except Exception:
            n_variants = 0  # Fall through to single-file mode

        if n_variants > _PARALLEL_THRESHOLD:
            # --- Parallel per-chromosome VEP ---
            warnings.append(
                f"Large VCF ({n_variants:,} variants) — splitting by chromosome "
                f"for parallel annotation"
            )

            chunks_dir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/vep/chunks")
            try:
                chrom_chunks = _split_vcf_by_chrom(vcf_path, chunks_dir)
            except Exception as e:
                errors.append(f"Failed to split VCF: {e}")
                chrom_chunks = []

            if chrom_chunks:
                vol_workdir.commit()

                # Spawn parallel VEP for each chromosome
                handles = {}
                for chrom, chunk_path in chrom_chunks:
                    chunk_out = str(chunks_dir / f"vep_{chrom}.vcf")
                    chunk_stats = str(chunks_dir / f"vep_{chrom}.html")
                    handles[chrom] = annotate_chunk.spawn(
                        chunk_path, chunk_out, chunk_stats,
                        assembly, use_alphamissense, extra_flags,
                    )

                # Collect results
                chunk_vcfs: list[str] = []
                total_variants = 0
                for chrom, handle in handles.items():
                    try:
                        result = handle.get(timeout=7200)
                        if result["errors"]:
                            warnings.append(f"VEP {chrom}: {result['errors'][0][:200]}")
                        else:
                            chunk_vcfs.append(result["output_vcf"])
                            total_variants += result["variant_count"]
                        warnings.extend(result.get("warnings", []))
                    except Exception as e:
                        warnings.append(f"VEP chunk {chrom} failed: {e}")

                # Merge annotated VCFs
                if chunk_vcfs:
                    vol_workdir.reload()
                    try:
                        # Write header from first chunk, then data from all
                        with open(output_vcf, "w") as out_fh:
                            # Header from first file
                            with open(chunk_vcfs[0]) as fh:
                                for line in fh:
                                    if line.startswith("#"):
                                        out_fh.write(line)
                                    else:
                                        out_fh.write(line)
                                        break
                                for line in fh:
                                    out_fh.write(line)

                            # Data only from remaining files
                            for chunk_vcf in chunk_vcfs[1:]:
                                if Path(chunk_vcf).exists():
                                    with open(chunk_vcf) as fh:
                                        for line in fh:
                                            if not line.startswith("#"):
                                                out_fh.write(line)

                        warnings.append(
                            f"Merged {len(chunk_vcfs)} chromosome chunks — "
                            f"{total_variants:,} variants annotated"
                        )
                    except Exception as e:
                        errors.append(f"Failed to merge VEP outputs: {e}")
                else:
                    errors.append("No chromosome chunks completed successfully")
        else:
            # --- Single-file VEP (small VCF) ---
            cmd, cmd_warnings = _build_vep_cmd(
                str(input_vcf), str(output_vcf), str(output_stats),
                assembly, use_alphamissense, extra_flags,
            )
            warnings.extend(cmd_warnings)

            try:
                proc = run_cmd(cmd, check=False, timeout=7000)
                if proc.returncode != 0:
                    errors.append(f"VEP exited with code {proc.returncode}")
                    if proc.stderr:
                        errors.append(proc.stderr.strip()[:2000])
                    if proc.stdout:
                        warnings.append(proc.stdout.strip()[:2000])
                elif proc.stderr:
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

    vol_workdir.commit()

    annotated = str(output_vcf) if output_vcf.exists() else None
    return build_result(
        SPEC, timer,
        inputs={Artifact.VCF.value: vcf_path},
        output_paths={Artifact.ANNOTATED_VCF.value: annotated} if annotated else {},
        payload={
            "assembly": assembly,
            "use_alphamissense": use_alphamissense,
            "variant_count": variant_count,
            "files": output_paths,
        },
        errors=errors,
        warnings=warnings,
    )


SPEC = ToolSpec(
    name="vep",
    version="112.0",
    modes=(Mode.GERMLINE, Mode.SOMATIC),
    consumes=(Artifact.VCF,),
    produces=(Artifact.ANNOTATED_VCF,),
    criticality=Criticality.CRITICAL,
    timeout_s=14400,
    reference_artifacts=("reference_genome", "vep_cache", "alphamissense_plugin", "clinvar"),
)
register(SPEC, annotate)
