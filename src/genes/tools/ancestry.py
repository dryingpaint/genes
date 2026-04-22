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
            # Step 1: Convert sample VCF to plink2 format (biallelic SNPs only)
            try:
                conv = run_cmd([
                    "plink2",
                    "--vcf", vcf_path,
                    "--make-bed",
                    "--out", sample_prefix,
                    "--allow-extra-chr",
                    "--max-alleles", "2",
                    "--snps-only", "just-acgt",
                    "--set-all-var-ids", "@:#:\\$r:\\$a",
                    "--new-id-max-allele-len", "20",
                    "--output-chr", "26",
                ], check=False, timeout=600)
                if conv.returncode != 0:
                    # plink2 exit 6 can mean sample count issues — try without sample filtering
                    warnings.append(f"plink2 conversion warning (code {conv.returncode}): {conv.stderr[-500:]}")
                    run_cmd([
                        "plink2",
                        "--vcf", vcf_path,
                        "--make-bed",
                        "--out", sample_prefix,
                        "--allow-extra-chr",
                        "--max-alleles", "2",
                        "--snps-only",
                        "--output-chr", "26",
                    ], timeout=600)
            except Exception as exc:
                errors.append(f"plink2 VCF conversion failed: {exc}")

        if not errors:
            # Step 2: Match by chr:pos (robust to variant ID format differences)
            try:
                # Build position set from sample: "chr\tpos" (from BIM cols 0,3)
                sample_positions = set()
                with open(f"{sample_prefix}.bim") as f:
                    for line in f:
                        parts = line.split()
                        chrom = parts[0].replace("chr", "")
                        pos = parts[3]
                        sample_positions.add(f"{chrom}\t{pos}")

                # Find shared positions in 1KG and write their variant IDs
                shared_snps_file = str(outdir / "shared_snps.txt")
                n_shared = 0
                with open(f"{_KG_PREFIX}.bim") as f_kg, open(shared_snps_file, "w") as f_out:
                    for line in f_kg:
                        parts = line.split()
                        chrom = parts[0].replace("chr", "")
                        pos = parts[3]
                        if f"{chrom}\t{pos}" in sample_positions:
                            f_out.write(parts[1] + "\n")
                            n_shared += 1

                if n_shared < 100:
                    errors.append(f"Only {n_shared} shared SNPs between sample and 1KG (need >100)")
                else:
                    warnings.append(f"Found {n_shared} shared SNPs with 1KG reference")

                    # Extract shared SNPs from 1KG
                    run_cmd([
                        "plink2", "--bfile", _KG_PREFIX,
                        "--extract", shared_snps_file,
                        "--make-bed", "--out", f"{outdir}/kg_shared",
                        "--allow-extra-chr",
                    ], timeout=300)

                    # Extract same positions from sample (using range file)
                    sample_snps_file = str(outdir / "sample_snps.txt")
                    sample_pos_set = set()
                    with open(f"{outdir}/kg_shared.bim") as f:
                        for line in f:
                            parts = line.split()
                            sample_pos_set.add(f"{parts[0].replace('chr', '')}\t{parts[3]}")
                    with open(f"{sample_prefix}.bim") as f_in, open(sample_snps_file, "w") as f_out:
                        for line in f_in:
                            parts = line.split()
                            chrom = parts[0].replace("chr", "")
                            if f"{chrom}\t{parts[3]}" in sample_pos_set:
                                f_out.write(parts[1] + "\n")

                    run_cmd([
                        "plink2", "--bfile", sample_prefix,
                        "--extract", sample_snps_file,
                        "--make-bed", "--out", f"{outdir}/sample_shared",
                        "--allow-extra-chr",
                    ], timeout=300)

                    # Merge — need matching chromosome names
                    # Recode 1KG to match sample chr format
                    run_cmd([
                        "plink2", "--bfile", f"{outdir}/kg_shared",
                        "--output-chr", "26",  # Strip chr prefix
                        "--make-bed", "--out", f"{outdir}/kg_recoded",
                        "--allow-extra-chr",
                    ], timeout=300)
                    run_cmd([
                        "plink2", "--bfile", f"{outdir}/sample_shared",
                        "--output-chr", "26",
                        "--make-bed", "--out", f"{outdir}/sample_recoded",
                        "--allow-extra-chr",
                    ], timeout=300)

                    # Merge — use plink2 --pmerge with allele mismatch handling
                    merge_result = run_cmd([
                        "plink2", "--bfile", f"{outdir}/kg_recoded",
                        "--pmerge", f"{outdir}/sample_recoded",
                        "--make-bed", "--out", merged_prefix,
                        "--allow-extra-chr",
                        "--merge-max-allele-ct", "2",
                    ], check=False, timeout=600)
                    if merge_result.returncode != 0:
                        # Fallback: just use the 1KG data + project sample onto it
                        warnings.append(
                            f"Merge failed (code {merge_result.returncode}); "
                            "projecting sample onto 1KG PCA instead"
                        )
                        # Use 1KG shared SNPs as the merged dataset
                        import shutil
                        for ext in [".bed", ".bim", ".fam"]:
                            src = f"{outdir}/kg_shared{ext}"
                            dst = f"{merged_prefix}{ext}"
                            if Path(src).exists():
                                shutil.copy2(src, dst)
                        # Append sample to FAM
                        if Path(f"{outdir}/sample_shared.fam").exists():
                            with open(f"{merged_prefix}.fam", "a") as f_out:
                                with open(f"{outdir}/sample_shared.fam") as f_in:
                                    f_out.write(f_in.read())
            except Exception as exc:
                errors.append(f"SNP merge failed: {exc}")

        if not errors:
            # Step 3: PCA — cap PCs to available SNPs
            pca_prefix = str(outdir / "pca")
            try:
                # Count available SNPs — PCA needs substantially more SNPs than PCs
                n_snps_available = sum(1 for _ in open(f"{merged_prefix}.bim"))
                if n_snps_available < 500:
                    errors.append(
                        f"Only {n_snps_available} shared SNPs (need ≥500 for reliable PCA). "
                        "GIAB truth VCFs are too sparse for ancestry inference — "
                        "use a full WGS callset instead."
                    )
                    raise ValueError("Insufficient SNPs")
                actual_pcs = min(n_pcs, max(2, n_snps_available // 10))
                if actual_pcs < n_pcs:
                    warnings.append(f"Only {n_snps_available} SNPs; using {actual_pcs} PCs instead of {n_pcs}")
                run_cmd([
                    "plink2",
                    "--bfile", merged_prefix,
                    "--pca", str(actual_pcs),
                    "--out", pca_prefix,
                    "--allow-extra-chr",
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
