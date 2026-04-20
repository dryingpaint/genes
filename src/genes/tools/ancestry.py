"""Ancestry inference — flashPCA2 + ADMIXTURE.

Projects a sample VCF onto the 1000 Genomes reference panel via PCA
(flashPCA2/plink2), then estimates admixture proportions using ADMIXTURE.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from genes.app import app
from genes.infra.images import image_cpp_tools
from genes.infra.volumes import (
    MOUNT_POPGEN,
    MOUNT_WORKDIR,
    vol_popgen,
    vol_workdir,
)
from genes.tools._base import ToolResult, ToolTimer, ensure_dir, run_cmd

# 1000 Genomes reference panel (plink2 binary format)
_KG_PREFIX = f"{MOUNT_POPGEN}/1kg/all_phase3_GRCh38"
_KG_POPS = f"{MOUNT_POPGEN}/1kg/integrated_call_samples_v3.20130502.ALL.panel"
_ADMIXTURE_BIN = "/opt/admixture_linux-1.3.0/admixture"


@app.function(
    image=image_cpp_tools,
    volumes={
        MOUNT_POPGEN: vol_popgen,
        MOUNT_WORKDIR: vol_workdir,
    },
    timeout=3600,
    cpu=8,
    memory=32768,
)
def infer_ancestry(
    vcf_path: str,
    run_id: str,
    k_values: list[int] | None = None,
    n_pcs: int = 20,
) -> ToolResult:
    """Infer genetic ancestry from a VCF file.

    Args:
        vcf_path: Path to the input VCF (single sample).
        run_id: Unique identifier for this run.
        k_values: List of K values for ADMIXTURE (default [3, 5, 7]).
        n_pcs: Number of principal components (default 20).

    Returns:
        ToolResult with PCA coordinates and admixture proportions.
    """
    if k_values is None:
        k_values = [3, 5, 7]

    outdir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/ancestry")

    with ToolTimer() as timer:
        errors: list[str] = []
        warnings: list[str] = []
        summary: dict = {}
        output_paths: list[str] = []

        if not Path(vcf_path).exists():
            errors.append(f"VCF file not found: {vcf_path}")

        # Check for 1KG reference panel
        if not Path(f"{_KG_PREFIX}.bed").exists():
            errors.append(
                f"1KG reference panel not found at {_KG_PREFIX}.bed. "
                "Populate the popgen volume with 1000 Genomes Phase 3 plink files."
            )

        sample_prefix = str(outdir / "sample")
        merged_prefix = str(outdir / "merged")

        if not errors:
            # Step 1: Convert sample VCF to plink2 format
            try:
                run_cmd([
                    "plink2",
                    "--vcf", vcf_path,
                    "--make-bed",
                    "--out", sample_prefix,
                    "--allow-extra-chr",
                    "--max-alleles", "2",
                ], timeout=600)
            except Exception as exc:
                errors.append(f"plink2 VCF conversion failed: {exc}")

        if not errors:
            # Step 2: Merge with 1KG reference panel (intersection of SNPs)
            try:
                # Extract overlapping SNPs
                run_cmd([
                    "plink2",
                    "--bfile", sample_prefix,
                    "--bmerge", _KG_PREFIX,
                    "--make-bed",
                    "--out", merged_prefix,
                    "--allow-extra-chr",
                ], timeout=1200)
            except Exception as exc:
                # Common: mismatched alleles. Try with --flip.
                warnings.append(f"Merge attempt 1 failed ({exc}); retrying with SNP extraction.")
                try:
                    run_cmd([
                        "plink2",
                        "--bfile", _KG_PREFIX,
                        "--extract", f"{sample_prefix}.bim",
                        "--bmerge", sample_prefix,
                        "--make-bed",
                        "--out", merged_prefix,
                        "--allow-extra-chr",
                    ], timeout=1200)
                except Exception as exc2:
                    errors.append(f"plink2 merge failed: {exc2}")

        if not errors:
            # Step 3: PCA
            pca_prefix = str(outdir / "pca")
            try:
                run_cmd([
                    "plink2",
                    "--bfile", merged_prefix,
                    "--pca", str(n_pcs),
                    "--out", pca_prefix,
                ], timeout=1200)
            except Exception as exc:
                errors.append(f"PCA failed: {exc}")

            eigenvec_path = f"{pca_prefix}.eigenvec"
            eigenval_path = f"{pca_prefix}.eigenval"
            if Path(eigenvec_path).exists():
                output_paths.append(eigenvec_path)
                # Read sample's PCA coordinates (last row)
                with open(eigenvec_path) as fh:
                    reader = csv.reader(fh, delimiter="\t")
                    header = next(reader, None)
                    rows = list(reader)
                if rows:
                    sample_row = rows[-1]  # sample is appended last
                    summary["pca_coordinates"] = {
                        f"PC{i+1}": float(sample_row[i + 2])
                        for i in range(min(n_pcs, len(sample_row) - 2))
                    }
            if Path(eigenval_path).exists():
                output_paths.append(eigenval_path)

        if not errors:
            # Step 4: ADMIXTURE for each K
            admixture_results: dict[int, dict] = {}
            bed_file = f"{merged_prefix}.bed"
            for k in k_values:
                try:
                    run_cmd(
                        [_ADMIXTURE_BIN, bed_file, str(k), "-j8", "--cv"],
                        timeout=1800,
                    )
                    # ADMIXTURE outputs files in the CWD with .Q suffix
                    q_file = f"{merged_prefix}.{k}.Q"
                    if Path(q_file).exists():
                        output_paths.append(q_file)
                        with open(q_file) as fh:
                            lines = fh.readlines()
                        if lines:
                            # Last line is the sample
                            props = [float(x) for x in lines[-1].strip().split()]
                            admixture_results[k] = {
                                f"pop_{i+1}": round(p, 4) for i, p in enumerate(props)
                            }
                except Exception as exc:
                    warnings.append(f"ADMIXTURE K={k} failed: {exc}")

            summary["admixture"] = admixture_results

        # Write summary JSON
        summary_path = str(outdir / "ancestry_summary.json")
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2)
        output_paths.append(summary_path)

        vol_workdir.commit()

    return ToolResult(
        tool_name="ancestry",
        version="1.0",
        started_at=timer.started_at,
        completed_at=timer.completed_at,
        input_summary={
            "vcf_path": vcf_path,
            "run_id": run_id,
            "k_values": k_values,
            "n_pcs": n_pcs,
        },
        output_paths=output_paths,
        output_summary=summary,
        errors=errors,
        warnings=warnings,
    )
