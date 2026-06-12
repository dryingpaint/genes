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
from genes.orchestrator.spec import Artifact, Criticality, Mode, ToolSpec, register
from genes.tools._base import ToolResult, ToolTimer, VariantScore, build_result, ensure_dir

# Pre-downloaded ClinVar scores with EVEE annotations
_CLINVAR_SCORES_PATH = "/data/precomputed/evee/clinvar_evee_scores.tsv.gz"


def _load_clinvar_index(scores_path: str) -> dict[str, dict]:
    """Load the ClinVar EVEE scores into a lookup dict keyed on chr:pos:ref:alt.

    Uses tabix for indexed random access instead of loading the entire file.
    Returns an empty dict if the file doesn't exist — the caller will use
    tabix queries per-variant instead.
    """
    path = Path(scores_path)
    if not path.exists():
        return {}

    # For large files, we use tabix queries instead of loading everything.
    # Return a sentinel to indicate the file exists.
    return {"__tabix_available__": True}


def _query_clinvar_tabix(scores_path: str, chrom: str, pos: int) -> list[dict]:
    """Query ClinVar EVEE scores by position using tabix."""
    import pysam

    results = []
    try:
        tbx = pysam.TabixFile(scores_path)
        # Tabix expects the contig name as stored in the file (no "chr" prefix)
        chrom_raw = chrom.replace("chr", "")
        for row in tbx.fetch(chrom_raw, pos - 1, pos):
            fields = row.split("\t")
            if len(fields) >= 8:
                results.append({
                    "chrom": fields[0],
                    "pos": fields[1],
                    "ref": fields[2],
                    "alt": fields[3],
                    "gene": fields[4],
                    "consequence": fields[5],
                    "classification": fields[6],
                    "score": fields[7],
                    "explanation": fields[8] if len(fields) > 8 else "",
                })
        tbx.close()
    except (ValueError, OSError):
        pass  # Contig not in file
    return results


@app.function(
    image=image_python_bio,
    volumes={
        MOUNT_WORKDIR: vol_workdir,
        "/data/precomputed": __import__("genes.infra.volumes", fromlist=["vol_precomputed"]).vol_precomputed,
    },
    timeout=1800,
    memory=32768,
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

        # Check if ClinVar EVEE scores file exists
        has_scores = Path(_CLINVAR_SCORES_PATH).exists()
        tbi_path = f"{_CLINVAR_SCORES_PATH}.tbi"
        has_index_file = Path(tbi_path).exists()

        if not has_scores:
            warnings.append(
                "ClinVar EVEE scores not populated. "
                "Run populate_evee_clinvar_scores() to enable variant annotations."
            )
        elif not has_index_file:
            warnings.append("ClinVar EVEE scores exist but tabix index (.tbi) is missing.")
            has_scores = False

        n_queried = 0
        n_found = 0

        if has_scores:
            import gzip

            # Load ClinVar index into memory — keyed by "chrom:pos:ref:alt"
            # ~4.4M entries, ~1.5 GB in memory, but fast lookups
            clinvar_index: dict[str, dict] = {}
            try:
                with gzip.open(_CLINVAR_SCORES_PATH, "rt") as fh:
                    header = fh.readline().strip().split("\t")
                    for line in fh:
                        fields = line.strip().split("\t")
                        if len(fields) >= 8:
                            key = f"{fields[0]}:{fields[1]}:{fields[2]}:{fields[3]}"
                            clinvar_index[key] = {
                                "gene": fields[4],
                                "consequence": fields[5],
                                "classification": fields[6],
                                "score": fields[7],
                                "explanation": fields[8] if len(fields) > 8 else "",
                            }
                warnings.append(f"Loaded {len(clinvar_index):,} ClinVar entries")
            except Exception as exc:
                errors.append(f"Failed to load ClinVar index: {exc}")
                clinvar_index = {}

            if clinvar_index:
                # Scan VCF and match against the in-memory index
                try:
                    vcf = pysam.VariantFile(vcf_path)
                    for rec in vcf:
                        chrom_raw = rec.chrom.replace("chr", "")
                        n_queried += 1

                        for alt in (rec.alts or []):
                            key = f"{chrom_raw}:{rec.pos}:{rec.ref}:{alt}"
                            match = clinvar_index.get(key)
                            if match:
                                n_found += 1
                                score_val = match.get("score", "")
                                results.append({
                                    "chrom": chrom_raw,
                                    "pos": rec.pos,
                                    "ref": rec.ref,
                                    "alt": alt,
                                    "gene": match.get("gene", ""),
                                    "consequence": match.get("consequence", ""),
                                    "evee_score": float(score_val) if score_val else None,
                                    "classification": match.get("classification", ""),
                                    "explanation": match.get("explanation", ""),
                                    "status": "found",
                                })

                    vcf.close()
                except Exception as exc:
                    errors.append(f"Failed to query EVEE scores: {exc}")

            # Only keep interesting results (pathogenic/likely_pathogenic)
            # to avoid bloating the output with millions of benign entries
            interesting = [r for r in results
                           if "athogenic" in str(r.get("classification", ""))
                           or "uncertain" in str(r.get("classification", "")).lower()]
            if len(interesting) < len(results):
                warnings.append(
                    f"Found {n_found:,} ClinVar matches total; "
                    f"showing {len(interesting)} pathogenic/uncertain variants"
                )
                results = interesting

        # Write output
        output_json = str(outdir / "evee_results.json")
        with open(output_json, "w") as fh:
            json.dump(results, fh, indent=2)

        vol_workdir.commit()

    return build_result(
        SPEC, timer,
        inputs={Artifact.VCF.value: vcf_path},
        payload={
            "total_queried": n_queried,
            "found": n_found,
            "scored_variants": results[:500],
            "output_json": output_json,
        },
        errors=errors,
        warnings=warnings,
    )


SPEC = ToolSpec(
    name="evee",
    version="1.0",
    modes=(Mode.GERMLINE,),
    consumes=(Artifact.VCF,),
    criticality=Criticality.OPTIONAL,
    timeout_s=1800,
    reference_artifacts=("evee_scores", "clinvar"),
)
register(SPEC, lookup_evee_scores)
