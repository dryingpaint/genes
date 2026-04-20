"""HLA typing — HLA*LA for DNA-seq, arcasHLA for RNA-seq.

Dispatches to the appropriate HLA typing tool based on the input data type.
HLA*LA uses a population reference graph approach; arcasHLA uses
genotyping from RNA-seq reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from genes.app import app
from genes.infra.images import image_cpp_tools, image_python_bio
from genes.infra.volumes import (
    MOUNT_HLA,
    MOUNT_REFERENCE,
    MOUNT_WORKDIR,
    vol_hla,
    vol_reference,
    vol_workdir,
)
from genes.tools._base import ToolResult, ToolTimer, ensure_dir, run_cmd

_HLA_LA_GRAPH = f"{MOUNT_HLA}/PRG_MHC_GRCh38_withIMGT"


@app.function(
    image=image_cpp_tools,
    volumes={
        MOUNT_HLA: vol_hla,
        MOUNT_REFERENCE: vol_reference,
        MOUNT_WORKDIR: vol_workdir,
    },
    timeout=7200,
    cpu=8,
    memory=32768,
)
def type_hla_dna(
    bam_path: str,
    run_id: str,
    sample_id: str = "sample",
    threads: int = 8,
) -> ToolResult:
    """Run HLA*LA for HLA typing from DNA-seq (WGS/WES) BAM.

    Args:
        bam_path: Path to the input BAM (must be coordinate-sorted and indexed).
        run_id: Unique identifier for this run.
        sample_id: Sample identifier for output labeling.
        threads: Number of threads for HLA*LA.

    Returns:
        ToolResult with HLA allele calls.
    """
    outdir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/hla_dna")

    with ToolTimer() as timer:
        errors: list[str] = []
        warnings: list[str] = []
        summary: dict = {}

        if not Path(bam_path).exists():
            errors.append(f"BAM file not found: {bam_path}")

        if not errors:
            cmd = [
                "HLA-LA.pl",
                "--BAM", bam_path,
                "--graph", _HLA_LA_GRAPH,
                "--sampleID", sample_id,
                "--maxThreads", str(threads),
                "--workingDir", str(outdir),
            ]
            try:
                run_cmd(cmd, timeout=7000)
            except Exception as exc:
                errors.append(f"HLA*LA execution failed: {exc}")

        # Parse best-guess results
        output_paths: list[str] = []
        best_guess = outdir / sample_id / "hla" / "R1_bestguess_G.txt"
        if best_guess.exists():
            output_paths.append(str(best_guess))
            try:
                alleles: dict[str, list[str]] = {}
                with open(best_guess) as fh:
                    header = fh.readline()
                    for line in fh:
                        parts = line.strip().split("\t")
                        if len(parts) >= 4:
                            locus = parts[0]
                            allele1 = parts[2]
                            allele2 = parts[3]
                            alleles[locus] = [allele1, allele2]
                summary["alleles"] = alleles
                summary["loci_typed"] = len(alleles)
            except Exception as exc:
                warnings.append(f"Failed to parse HLA*LA output: {exc}")

        vol_workdir.commit()

    return ToolResult(
        tool_name="hla_la",
        version="1.0.3",
        started_at=timer.started_at,
        completed_at=timer.completed_at,
        input_summary={
            "bam_path": bam_path,
            "run_id": run_id,
            "sample_id": sample_id,
            "data_type": "dna",
        },
        output_paths=output_paths,
        output_summary=summary,
        errors=errors,
        warnings=warnings,
    )


@app.function(
    image=image_python_bio,
    volumes={
        MOUNT_REFERENCE: vol_reference,
        MOUNT_WORKDIR: vol_workdir,
    },
    timeout=3600,
)
def type_hla_rna(
    bam_path: str,
    run_id: str,
    sample_id: str = "sample",
    threads: int = 8,
) -> ToolResult:
    """Run arcasHLA for HLA typing from RNA-seq BAM.

    Args:
        bam_path: Path to the input BAM (STAR-aligned RNA-seq).
        run_id: Unique identifier for this run.
        sample_id: Sample identifier for output labeling.
        threads: Number of threads.

    Returns:
        ToolResult with HLA allele calls from RNA-seq.
    """
    outdir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/hla_rna")

    with ToolTimer() as timer:
        errors: list[str] = []
        warnings: list[str] = []
        summary: dict = {}

        if not Path(bam_path).exists():
            errors.append(f"BAM file not found: {bam_path}")

        if not errors:
            # Step 1: Extract reads mapping to HLA loci
            extracted = str(outdir / f"{sample_id}.extracted.bam")
            try:
                run_cmd([
                    "arcasHLA", "extract", bam_path,
                    "-o", str(outdir),
                    "-t", str(threads),
                    "-v",
                ], timeout=1800)
            except Exception as exc:
                errors.append(f"arcasHLA extract failed: {exc}")

        if not errors:
            # Step 2: Genotype
            try:
                run_cmd([
                    "arcasHLA", "genotype",
                    str(outdir / f"{Path(bam_path).stem}.extracted.1.fq.gz"),
                    str(outdir / f"{Path(bam_path).stem}.extracted.2.fq.gz"),
                    "-o", str(outdir),
                    "-t", str(threads),
                    "-g", "A,B,C,DPB1,DQB1,DQA1,DRB1",
                ], timeout=1800)
            except Exception as exc:
                errors.append(f"arcasHLA genotype failed: {exc}")

        # Parse results
        output_paths: list[str] = []
        genotype_json = outdir / f"{Path(bam_path).stem}.genotype.json"
        if genotype_json.exists():
            output_paths.append(str(genotype_json))
            try:
                with open(genotype_json) as fh:
                    data = json.load(fh)
                summary["alleles"] = data
                summary["loci_typed"] = len(data)
            except Exception as exc:
                warnings.append(f"Failed to parse arcasHLA output: {exc}")

        vol_workdir.commit()

    return ToolResult(
        tool_name="arcashla",
        version="0.6.0",
        started_at=timer.started_at,
        completed_at=timer.completed_at,
        input_summary={
            "bam_path": bam_path,
            "run_id": run_id,
            "sample_id": sample_id,
            "data_type": "rna",
        },
        output_paths=output_paths,
        output_summary=summary,
        errors=errors,
        warnings=warnings,
    )


@app.function(
    image=image_python_bio,
    volumes={
        MOUNT_WORKDIR: vol_workdir,
    },
    timeout=60,
)
def type_hla(
    bam_path: str,
    run_id: str,
    data_type: str = "dna",
    sample_id: str = "sample",
    threads: int = 8,
) -> ToolResult:
    """Dispatch HLA typing to the appropriate tool based on data type.

    Args:
        bam_path: Path to the input BAM.
        run_id: Unique identifier for this run.
        data_type: "dna" for WGS/WES (uses HLA*LA) or "rna" for RNA-seq (uses arcasHLA).
        sample_id: Sample identifier for output labeling.
        threads: Number of threads.

    Returns:
        ToolResult from the dispatched tool.
    """
    if data_type.lower() == "dna":
        return type_hla_dna.remote(
            bam_path=bam_path,
            run_id=run_id,
            sample_id=sample_id,
            threads=threads,
        )
    elif data_type.lower() == "rna":
        return type_hla_rna.remote(
            bam_path=bam_path,
            run_id=run_id,
            sample_id=sample_id,
            threads=threads,
        )
    else:
        raise ValueError(f"data_type must be 'dna' or 'rna', got {data_type!r}")
