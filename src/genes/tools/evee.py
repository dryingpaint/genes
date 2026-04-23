"""EVEE — Cross-type Variant Effect Prediction with Explanations.

Looks up pre-downloaded EVEE/ClinVar scores for variant effect prediction.
EVEE integrates multiple evidence sources and provides human-readable
explanations for each prediction.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from genes.app import app
from genes.infra.images import image_python_bio
from genes.infra.volumes import (
    MOUNT_WORKDIR,
    vol_workdir,
)
from genes.tools._base import ToolResult, ToolTimer, VariantScore, ensure_dir

# Pre-downloaded ClinVar scores with EVEE annotations
_CLINVAR_SCORES_PATH = "/data/precomputed/evee/clinvar_evee_scores.tsv.gz"


def _load_clinvar_index(scores_path: str) -> dict[str, dict]:
    """Load the ClinVar EVEE scores into a lookup dict keyed on chr:pos:ref:alt."""
    import gzip

    index: dict[str, dict] = {}
    path = Path(scores_path)
    if not path.exists():
        return index

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            key = f"{row['chrom']}:{row['pos']}:{row['ref']}:{row['alt']}"
            index[key] = row
    return index


@app.function(
    image=image_python_bio,
    volumes={
        MOUNT_WORKDIR: vol_workdir,
        "/data/precomputed": __import__("genes.infra.volumes", fromlist=["vol_precomputed"]).vol_precomputed,
    },
    timeout=600,
)
def lookup_evee_scores(
    vcf_path: str,
    run_id: str,
) -> ToolResult:
    """Look up EVEE scores from a VCF file.

    Args:
        vcf_path: Path to input VCF.
        run_id: Unique identifier for this run.

    Returns:
        ToolResult with per-variant scores and explanations.
    """
    import pysam

    outdir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/evee")

    with ToolTimer() as timer:
        errors: list[str] = []
        warnings: list[str] = []
        results: list[dict] = []

        # Load index
        try:
            index = _load_clinvar_index(_CLINVAR_SCORES_PATH)
            if not index:
                warnings.append(
                    "ClinVar EVEE scores file is empty or not found; "
                    "returning unscored variants."
                )
        except Exception as exc:
            errors.append(f"Failed to load ClinVar EVEE scores: {exc}")
            index = {}

        # Extract variants from VCF
        variants = []
        try:
            vcf = pysam.VariantFile(vcf_path)
            for rec in vcf:
                for alt in rec.alts or []:
                    variants.append({"chrom": rec.chrom, "pos": rec.pos, "ref": rec.ref, "alt": alt})
            vcf.close()
        except Exception as exc:
            errors.append(f"Failed to read VCF: {exc}")

        for v in variants:
            chrom = str(v.get("chrom", "")).replace("chr", "")
            pos = str(v.get("pos", ""))
            ref = str(v.get("ref", ""))
            alt = str(v.get("alt", ""))

            if not all([chrom, pos, ref, alt]):
                warnings.append(f"Skipping incomplete variant: {v}")
                continue

            key = f"{chrom}:{pos}:{ref}:{alt}"
            match = index.get(key)

            if match:
                score_val = match.get("evee_score") or match.get("score")
                classification = match.get("classification", match.get("clnsig", ""))
                explanation = match.get("explanation", "")
                results.append({
                    "chrom": chrom,
                    "pos": int(pos),
                    "ref": ref,
                    "alt": alt,
                    "gene": match.get("gene", v.get("gene")),
                    "consequence": match.get("consequence", v.get("consequence")),
                    "evee_score": float(score_val) if score_val else None,
                    "classification": classification,
                    "explanation": explanation,
                    "status": "found",
                })
            else:
                results.append({
                    "chrom": chrom,
                    "pos": int(pos),
                    "ref": ref,
                    "alt": alt,
                    "gene": v.get("gene"),
                    "consequence": v.get("consequence"),
                    "evee_score": None,
                    "classification": None,
                    "explanation": None,
                    "status": "not_found",
                })

        # Write output
        output_json = str(outdir / "evee_results.json")
        with open(output_json, "w") as fh:
            json.dump(results, fh, indent=2)

        vol_workdir.commit()

    return ToolResult(
        tool_name="evee",
        version="1.0",
        started_at=timer.started_at,
        completed_at=timer.completed_at,
        input_summary={
            "num_variants": len(variants),
            "run_id": run_id,
        },
        output_paths=[output_json],
        output_summary={
            "total_queried": len(variants),
            "found": sum(1 for r in results if r["status"] == "found"),
            "not_found": sum(1 for r in results if r["status"] == "not_found"),
            "results": results,
        },
        errors=errors,
        warnings=warnings,
    )
