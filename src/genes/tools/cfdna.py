"""cfDNA (cell-free DNA) analysis tools.

Three sub-tools for liquid biopsy analysis:
  - ichorCNA: tumor fraction estimation from shallow WGS cfDNA
  - Griffin: nucleosome footprinting at transcription factor binding sites
  - UXM: cell-of-origin deconvolution from bisulfite-sequenced cfDNA
"""

from __future__ import annotations

import json
from pathlib import Path

from genes.app import app
from genes.infra.images import image_python_bio, image_r
from genes.infra.volumes import (
    MOUNT_CFDNA,
    MOUNT_REFERENCE,
    MOUNT_WORKDIR,
    vol_cfdna,
    vol_reference,
    vol_workdir,
)
from genes.infra.provenance import PROVENANCE_KEY, stamp
from genes.tools._base import ToolResult, ToolTimer, ensure_dir, run_cmd

# ----- Paths within volumes -----
_REFERENCE_FASTA = f"{MOUNT_REFERENCE}/GRCh38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna"
_ICHORCNA_PON = f"{MOUNT_CFDNA}/ichorCNA/pon"
_ICHORCNA_GC_WIG = f"{MOUNT_CFDNA}/ichorCNA/gc_GRCh38_1Mb.wig"
_ICHORCNA_MAP_WIG = f"{MOUNT_CFDNA}/ichorCNA/map_GRCh38_1Mb.wig"
_ICHORCNA_CENTROMERE = f"{MOUNT_CFDNA}/ichorCNA/GRCh38.centromere.txt"
_LOYFER_ATLAS = f"{MOUNT_CFDNA}/uxm/Loyfer_atlas"
_GRIFFIN_SITES_DIR = f"{MOUNT_CFDNA}/griffin/sites"


@app.function(
    image=image_r,
    volumes={
        MOUNT_REFERENCE: vol_reference,
        MOUNT_CFDNA: vol_cfdna,
        MOUNT_WORKDIR: vol_workdir,
    },
    timeout=3600,
    cpu=4,
    memory=16384,
)
def run_ichorcna(
    bam_path: str,
    run_id: str,
    *,
    bin_size: int = 1_000_000,
    normal_panel: str | None = None,
) -> ToolResult:
    """Estimate tumor fraction from shallow-WGS cfDNA using ichorCNA.

    Args:
        bam_path: Path to cfDNA BAM (indexed).
        run_id: Unique run identifier.
        bin_size: Bin size in bp for read-depth binning (default 1 Mb).
        normal_panel: Path to a custom panel-of-normals; defaults to built-in PoN.
    """
    outdir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/ichorcna")
    wig_file = str(outdir / "tumor.wig")
    pon = normal_panel or _ICHORCNA_PON

    with ToolTimer() as timer:
        errors: list[str] = []
        warnings: list[str] = []

        if not Path(bam_path).exists():
            errors.append(f"BAM file not found: {bam_path}")

        if not errors:
            # Step 1: Generate read-count WIG with HMMcopy readCounter
            try:
                run_cmd([
                    "readCounter",
                    "--window", str(bin_size),
                    "--chromosome", "chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,"
                    "chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,"
                    "chr20,chr21,chr22,chrX",
                    bam_path,
                ], timeout=1800)
            except Exception as exc:
                errors.append(f"readCounter failed: {exc}")

        if not errors:
            # Step 2: Run ichorCNA R script
            r_cmd = [
                "Rscript", "/opt/ichorCNA/scripts/runIchorCNA.R",
                "--id", run_id,
                "--WIG", wig_file,
                "--gcWig", _ICHORCNA_GC_WIG,
                "--mapWig", _ICHORCNA_MAP_WIG,
                "--centromere", _ICHORCNA_CENTROMERE,
                "--normalPanel", pon,
                "--outDir", str(outdir),
                "--genomeBuild", "hg38",
                "--genomeStyle", "UCSC",
            ]
            try:
                run_cmd(r_cmd, timeout=2400)
            except Exception as exc:
                errors.append(f"ichorCNA R script failed: {exc}")

        # Parse results
        summary: dict = {}
        output_paths: list[str] = []
        params_file = outdir / f"{run_id}.params.txt"
        if params_file.exists():
            output_paths.append(str(params_file))
            with open(params_file) as fh:
                for line in fh:
                    if "Tumor Fraction" in line:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            try:
                                summary["tumor_fraction"] = float(parts[1])
                            except ValueError:
                                warnings.append(f"Could not parse tumor fraction: {parts[1]}")
                    elif "Ploidy" in line:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            try:
                                summary["ploidy"] = float(parts[1])
                            except ValueError:
                                pass

        seg_file = outdir / f"{run_id}.seg.txt"
        if seg_file.exists():
            output_paths.append(str(seg_file))

        vol_workdir.commit()

    return ToolResult(
        tool_name="ichorcna",
        version="0.5.0",
        started_at=timer.started_at,
        completed_at=timer.completed_at,
        input_summary={
            "bam_path": bam_path,
            "run_id": run_id,
            "bin_size": bin_size,
            PROVENANCE_KEY: stamp("cfdna_refs", "reference_genome"),
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
        MOUNT_CFDNA: vol_cfdna,
        MOUNT_WORKDIR: vol_workdir,
    },
    timeout=3600,
    cpu=4,
    memory=16384,
)
def run_griffin(
    bam_path: str,
    run_id: str,
    *,
    tfbs_list: str | None = None,
    window_size: int = 1000,
) -> ToolResult:
    """Run Griffin nucleosome footprinting on cfDNA BAM.

    Quantifies nucleosome protection at transcription factor binding sites
    (TFBS) to infer gene regulatory activity from cfDNA fragment patterns.

    Args:
        bam_path: Path to cfDNA BAM (indexed).
        run_id: Unique run identifier.
        tfbs_list: Path to TFBS BED file. Defaults to standard ENCODE TFBS list.
        window_size: Window around TFBS center (default 1000 bp).
    """
    outdir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/griffin")
    sites_bed = tfbs_list or f"{_GRIFFIN_SITES_DIR}/ENCODE_TFBS.bed"

    with ToolTimer() as timer:
        errors: list[str] = []
        warnings: list[str] = []

        if not Path(bam_path).exists():
            errors.append(f"BAM file not found: {bam_path}")
        if not Path(sites_bed).exists():
            errors.append(f"TFBS list not found: {sites_bed}")

        if not errors:
            # Griffin runs as a Python module
            cmd = [
                "python", "-m", "griffin",
                "--bam", bam_path,
                "--sites", sites_bed,
                "--reference", _REFERENCE_FASTA,
                "--output_dir", str(outdir),
                "--window_size", str(window_size),
                "--sample_name", run_id,
            ]
            try:
                run_cmd(cmd, timeout=3000)
            except Exception as exc:
                errors.append(f"Griffin failed: {exc}")

        # Parse results
        summary: dict = {}
        output_paths: list[str] = []
        results_tsv = outdir / f"{run_id}.griffin_results.tsv"
        if results_tsv.exists():
            output_paths.append(str(results_tsv))
            try:
                import polars as pl

                df = pl.read_csv(str(results_tsv), separator="\t")
                summary["num_sites_analyzed"] = len(df)
                if "mean_coverage" in df.columns:
                    summary["mean_coverage"] = round(df["mean_coverage"].mean(), 2)
                if "central_coverage" in df.columns:
                    summary["mean_central_coverage"] = round(
                        df["central_coverage"].mean(), 2
                    )
            except Exception:
                warnings.append("Could not parse Griffin results TSV.")

        vol_workdir.commit()

    return ToolResult(
        tool_name="griffin",
        version="1.0.0",
        started_at=timer.started_at,
        completed_at=timer.completed_at,
        input_summary={
            "bam_path": bam_path,
            "run_id": run_id,
            "tfbs_list": sites_bed,
            "window_size": window_size,
            PROVENANCE_KEY: stamp("cfdna_refs", "reference_genome"),
        },
        output_paths=output_paths,
        output_summary=summary,
        errors=errors,
        warnings=warnings,
    )


@app.function(
    image=image_python_bio,
    volumes={
        MOUNT_CFDNA: vol_cfdna,
        MOUNT_WORKDIR: vol_workdir,
    },
    timeout=3600,
    cpu=4,
    memory=16384,
)
def run_uxm(
    bam_path: str,
    run_id: str,
    *,
    atlas_dir: str | None = None,
) -> ToolResult:
    """Run UXM cell-of-origin deconvolution on bisulfite-sequenced cfDNA.

    Uses the Loyfer methylation atlas to estimate tissue-of-origin proportions
    from bisulfite-converted cfDNA reads.

    Args:
        bam_path: Path to bisulfite-sequenced BAM (indexed).
        run_id: Unique run identifier.
        atlas_dir: Path to UXM atlas directory. Defaults to Loyfer atlas.
    """
    outdir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/uxm")
    atlas = atlas_dir or _LOYFER_ATLAS

    with ToolTimer() as timer:
        errors: list[str] = []
        warnings: list[str] = []

        if not Path(bam_path).exists():
            errors.append(f"BAM file not found: {bam_path}")
        if not Path(atlas).exists():
            errors.append(f"UXM atlas not found: {atlas}")

        if not errors:
            cmd = [
                "python", "-m", "uxm",
                "--bam", bam_path,
                "--atlas", atlas,
                "--output_dir", str(outdir),
                "--sample_name", run_id,
            ]
            try:
                run_cmd(cmd, timeout=3000)
            except Exception as exc:
                errors.append(f"UXM deconvolution failed: {exc}")

        # Parse results
        summary: dict = {}
        output_paths: list[str] = []
        results_json = outdir / f"{run_id}.uxm_results.json"
        results_tsv = outdir / f"{run_id}.uxm_results.tsv"

        for candidate in [results_json, results_tsv]:
            if candidate.exists():
                output_paths.append(str(candidate))

        if results_json.exists():
            try:
                with open(results_json) as fh:
                    data = json.load(fh)
                if "cell_type_fractions" in data:
                    summary["cell_type_fractions"] = data["cell_type_fractions"]
                    # Highlight dominant tissue
                    fractions = data["cell_type_fractions"]
                    if fractions:
                        top_tissue = max(fractions, key=fractions.get)
                        summary["dominant_tissue"] = top_tissue
                        summary["dominant_fraction"] = fractions[top_tissue]
            except Exception:
                warnings.append("Could not parse UXM results JSON.")
        elif results_tsv.exists():
            try:
                import polars as pl

                df = pl.read_csv(str(results_tsv), separator="\t")
                if "cell_type" in df.columns and "fraction" in df.columns:
                    fractions = dict(
                        zip(df["cell_type"].to_list(), df["fraction"].to_list())
                    )
                    summary["cell_type_fractions"] = fractions
                    top_tissue = max(fractions, key=fractions.get)
                    summary["dominant_tissue"] = top_tissue
                    summary["dominant_fraction"] = fractions[top_tissue]
            except Exception:
                warnings.append("Could not parse UXM results TSV.")

        vol_workdir.commit()

    return ToolResult(
        tool_name="uxm",
        version="1.0.0",
        started_at=timer.started_at,
        completed_at=timer.completed_at,
        input_summary={
            "bam_path": bam_path,
            "run_id": run_id,
            "atlas_dir": atlas,
            PROVENANCE_KEY: stamp("cfdna_refs"),
        },
        output_paths=output_paths,
        output_summary=summary,
        errors=errors,
        warnings=warnings,
    )
