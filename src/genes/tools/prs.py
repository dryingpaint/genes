"""Polygenic risk score (PRS) calculation tool wrapper.

Runs plink2 --score against PGS Catalog weight files to compute polygenic
risk scores for one or more traits. Supports multiple traits in a single
call for efficiency. Weight files are expected to live on the popgen
volume under /data/popgen/pgs_catalog/<trait_id>/.
"""

from __future__ import annotations

import csv
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

PGS_CATALOG_DIR = f"{MOUNT_POPGEN}/pgs_catalog"


def _find_weight_file(trait_id: str) -> Path | None:
    """Locate the PGS Catalog scoring file for a trait.

    Looks for files matching common PGS Catalog naming conventions:
      /data/popgen/pgs_catalog/<trait_id>/<PGS_ID>_hmPOS_GRCh38.txt.gz
      /data/popgen/pgs_catalog/<trait_id>/weights.tsv
    """
    trait_dir = Path(PGS_CATALOG_DIR) / trait_id
    if not trait_dir.is_dir():
        return None

    # Prefer harmonized GRCh38 files
    for pattern in ["*_hmPOS_GRCh38.txt.gz", "*_hmPOS_GRCh38.txt", "weights.tsv", "*.sscore"]:
        matches = list(trait_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _parse_plink_score(sscore_path: Path) -> list[dict]:
    """Parse plink2 .sscore output into a list of per-sample score records."""
    results: list[dict] = []
    with open(sscore_path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            sample_id = row.get("#FID", row.get("FID", row.get("#IID", "")))
            iid = row.get("IID", sample_id)
            # plink2 --score columns: SCORE1_AVG or SCORE1_SUM
            score_avg = row.get("SCORE1_AVG")
            score_sum = row.get("SCORE1_SUM")
            n_counted = row.get("ALLELE_CT") or row.get("NAMED_ALLELE_DOSAGE_SUM")
            results.append({
                "sample_id": iid,
                "score_avg": float(score_avg) if score_avg else None,
                "score_sum": float(score_sum) if score_sum else None,
                "n_variants_counted": int(n_counted) if n_counted else None,
            })
    return results


@app.function(
    image=image_cpp_tools,
    volumes={
        MOUNT_POPGEN: vol_popgen,
        MOUNT_WORKDIR: vol_workdir,
    },
    timeout=3600,
    cpu=4,
    memory=16384,
)
def calculate(
    vcf_path: str,
    run_id: str,
    traits: list[str],
    *,
    columns: str = "1,2,4",
    score_col_nums: str | None = None,
) -> ToolResult:
    """Compute polygenic risk scores for one or more traits.

    Parameters
    ----------
    vcf_path:
        Path to input VCF (genotype data).
    run_id:
        Unique run identifier.
    traits:
        List of trait/PGS IDs (e.g., ["PGS000001", "PGS000018"]).
        Must match directory names under the PGS Catalog volume.
    columns:
        plink2 --score column numbers for variant ID, allele, and score
        (default "1,2,4" matching PGS Catalog harmonized format).
    score_col_nums:
        Optional override for which columns contain the effect weights.
    """
    out_dir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/prs")
    warnings: list[str] = []
    errors: list[str] = []
    trait_results: dict[str, list[dict]] = {}
    output_paths: list[str] = []

    with ToolTimer() as timer:
        for trait_id in traits:
            weight_file = _find_weight_file(trait_id)
            if weight_file is None:
                warnings.append(
                    f"No weight file found for trait {trait_id} in "
                    f"{PGS_CATALOG_DIR}/{trait_id}/."
                )
                continue

            sscore_prefix = out_dir / trait_id
            col_args = columns.split(",")

            cmd = [
                "plink2",
                "--vcf", vcf_path,
                "--score", str(weight_file),
                col_args[0], col_args[1], col_args[2],
                "header",
                "list-variants",
                "--out", str(sscore_prefix),
                "--threads", "4",
                "--memory", "12000",
            ]

            if score_col_nums:
                cmd.extend(["--score-col-nums", score_col_nums])

            try:
                proc = run_cmd(cmd, timeout=3000)
                if proc.stderr:
                    for line in proc.stderr.strip().splitlines():
                        if "warning" in line.lower():
                            warnings.append(f"[{trait_id}] {line.strip()}")

                sscore_file = Path(f"{sscore_prefix}.sscore")
                if sscore_file.exists():
                    output_paths.append(str(sscore_file))
                    trait_results[trait_id] = _parse_plink_score(sscore_file)
                else:
                    warnings.append(
                        f"plink2 completed for {trait_id} but no .sscore file produced."
                    )

                # Include the variant list if generated
                variant_list_file = Path(f"{sscore_prefix}.sscore.vars")
                if variant_list_file.exists():
                    output_paths.append(str(variant_list_file))

                # Include the log
                log_file = Path(f"{sscore_prefix}.log")
                if log_file.exists():
                    output_paths.append(str(log_file))

            except Exception as e:
                errors.append(f"[{trait_id}] {e}")

    return ToolResult(
        tool_name="prs",
        version="plink2-alpha5",
        started_at=timer.started_at,
        completed_at=timer.completed_at,
        input_summary={
            "vcf_path": vcf_path,
            "run_id": run_id,
            "traits": traits,
            "n_traits_requested": len(traits),
        },
        output_paths=output_paths,
        output_summary={
            "n_traits_computed": len(trait_results),
            "trait_scores": trait_results,
        },
        errors=errors,
        warnings=warnings,
    )
