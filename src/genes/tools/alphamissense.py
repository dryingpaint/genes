"""AlphaMissense score lookup tool wrapper.

Queries pre-computed AlphaMissense pathogenicity scores from a
tabix-indexed TSV file. Each variant gets a score in [0, 1] and a
classification (likely_benign, ambiguous, likely_pathogenic).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pysam

from genes.app import app
from genes.infra.images import image_python_bio
from genes.infra.volumes import (
    MOUNT_PRECOMPUTED,
    MOUNT_WORKDIR,
    vol_precomputed,
    vol_workdir,
)
from genes.infra.provenance import PROVENANCE_KEY, stamp
from genes.tools._base import ToolResult, ToolTimer, VariantScore, ensure_dir

# AlphaMissense pre-computed TSV (tabix-indexed)
# Columns: #CHROM  POS  REF  ALT  genome  uniprot_id  transcript_id
#          protein_variant  am_pathogenicity  am_class
AM_TSV = f"{MOUNT_PRECOMPUTED}/alphamissense/AlphaMissense_hg38.tsv.gz"


def _parse_variant_key(chrom: str, pos: int, ref: str, alt: str) -> str:
    """Normalise a variant into a consistent string key."""
    c = chrom.replace("chr", "")
    return f"{c}:{pos}:{ref}:{alt}"


def _lookup_variants(
    variants: list[tuple[str, int, str, str]],
) -> list[VariantScore]:
    """Look up AlphaMissense scores for a list of (chrom, pos, ref, alt) tuples."""
    tbx = pysam.TabixFile(AM_TSV)
    results: list[VariantScore] = []

    for chrom, pos, ref, alt in variants:
        query_chrom = chrom if chrom.startswith("chr") else f"chr{chrom}"
        score_found = False

        try:
            for row in tbx.fetch(query_chrom, pos - 1, pos):
                fields = row.split("\t")
                if len(fields) < 10:
                    continue
                row_ref, row_alt = fields[2], fields[3]
                if row_ref == ref and row_alt == alt:
                    am_score = float(fields[8])
                    am_class = fields[9]
                    results.append(
                        VariantScore(
                            chrom=chrom,
                            pos=pos,
                            ref=ref,
                            alt=alt,
                            gene=fields[6] if fields[6] else None,
                            scores={"am_pathogenicity": am_score},
                            classifications={"am_class": am_class},
                        )
                    )
                    score_found = True
                    break
        except ValueError:
            # Contig not in tabix index
            pass

        if not score_found:
            results.append(
                VariantScore(
                    chrom=chrom,
                    pos=pos,
                    ref=ref,
                    alt=alt,
                    scores={"am_pathogenicity": None},
                    classifications={"am_class": "not_found"},
                )
            )

    tbx.close()
    return results


def _variants_from_vcf(vcf_path: str) -> list[tuple[str, int, str, str]]:
    """Extract variant tuples from a VCF file."""
    variants: list[tuple[str, int, str, str]] = []
    vcf = pysam.VariantFile(vcf_path)
    for rec in vcf:
        for alt in rec.alts or []:
            variants.append((rec.chrom, rec.pos, rec.ref, alt))
    vcf.close()
    return variants


@app.function(
    image=image_python_bio,
    volumes={
        MOUNT_PRECOMPUTED: vol_precomputed,
        MOUNT_WORKDIR: vol_workdir,
    },
    timeout=1800,
    cpu=2,
    memory=8192,
)
def alphamissense_lookup(
    run_id: str,
    *,
    vcf_path: str | None = None,
    variants: list[tuple[str, int, str, str]] | None = None,
) -> ToolResult:
    """Look up AlphaMissense pathogenicity scores.

    Provide either a VCF file path or an explicit list of variant tuples
    ``(chrom, pos, ref, alt)``.

    Parameters
    ----------
    run_id:
        Unique run identifier.
    vcf_path:
        Path to a VCF file on a mounted volume.
    variants:
        Explicit list of ``(chrom, pos, ref, alt)`` tuples.
    """
    out_dir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/alphamissense")
    output_tsv = out_dir / "alphamissense_scores.tsv"
    warnings: list[str] = []
    errors: list[str] = []

    with ToolTimer() as timer:
        try:
            if vcf_path:
                variant_list = _variants_from_vcf(vcf_path)
            elif variants:
                variant_list = variants
            else:
                raise ValueError("Either vcf_path or variants must be provided.")

            if not Path(AM_TSV).exists():
                raise FileNotFoundError(
                    f"AlphaMissense data not found at {AM_TSV}. "
                    "Ensure the precomputed volume is populated."
                )

            scored = _lookup_variants(variant_list)

            # Write results to TSV
            with open(output_tsv, "w", newline="") as fh:
                writer = csv.writer(fh, delimiter="\t")
                writer.writerow([
                    "chrom", "pos", "ref", "alt", "gene",
                    "am_pathogenicity", "am_class",
                ])
                for v in scored:
                    writer.writerow([
                        v.chrom, v.pos, v.ref, v.alt,
                        v.gene or "",
                        v.scores.get("am_pathogenicity", ""),
                        v.classifications.get("am_class", ""),
                    ])

        except Exception as e:
            errors.append(str(e))
            scored = []

    n_found = sum(
        1 for v in scored
        if v.classifications.get("am_class") != "not_found"
    )

    return ToolResult(
        tool_name="alphamissense",
        version="1.0",
        started_at=timer.started_at,
        completed_at=timer.completed_at,
        input_summary={
            "run_id": run_id,
            "vcf_path": vcf_path,
            "n_variants_queried": len(scored),
            PROVENANCE_KEY: stamp("alphamissense_scores"),
        },
        output_paths=[str(output_tsv)] if output_tsv.exists() else [],
        output_summary={
            "n_variants_queried": len(scored),
            "n_scores_found": n_found,
            # Only include variants with actual scores (not the full queried set)
            "scored_variants": [
                v.model_dump() for v in scored
                if v.scores.get("am_pathogenicity") is not None
            ][:1000],  # Cap to avoid bloating JSON
            "pathogenic_count": sum(
                1 for v in scored
                if v.classifications.get("am_class") == "likely_pathogenic"
            ),
            "benign_count": sum(
                1 for v in scored
                if v.classifications.get("am_class") == "likely_benign"
            ),
        },
        errors=errors,
        warnings=warnings,
    )
