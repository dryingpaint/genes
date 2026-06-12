"""GPN-MSA log-likelihood ratio lookup tool wrapper.

Queries pre-computed GPN-MSA scores from a tabix-indexed TSV. GPN-MSA
provides per-variant log-likelihood ratios (LLR) that capture the
evolutionary plausibility of an alternate allele versus the reference
using a genomic pre-trained network over multiple sequence alignments.
Negative LLR indicates the variant is less plausible (potentially
deleterious); positive LLR indicates the variant is at least as
plausible as the reference.
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
from genes.orchestrator.spec import Artifact, Criticality, Mode, ToolSpec, register
from genes.tools._base import ToolResult, ToolTimer, VariantScore, build_result, ensure_dir

# Pre-computed GPN-MSA scores (tabix-indexed TSV.gz)
# Columns: #chrom  pos  ref  alt  llr
GPN_MSA_TSV = f"{MOUNT_PRECOMPUTED}/gpn_msa/gpn_msa_scores_hg38.tsv.gz"

# Classification thresholds (from GPN-MSA paper)
BENIGN_THRESHOLD = 0.0
DELETERIOUS_THRESHOLD = -3.0


def _lookup_variants(
    variants: list[tuple[str, int, str, str]],
) -> list[VariantScore]:
    """Look up GPN-MSA LLR scores for a list of variants."""
    tbx = pysam.TabixFile(GPN_MSA_TSV)
    results: list[VariantScore] = []

    for chrom, pos, ref, alt in variants:
        # GPN-MSA scores use no chr prefix (e.g., "22" not "chr22")
        query_chrom = chrom.replace("chr", "") if chrom.startswith("chr") else chrom
        score_found = False

        try:
            for row in tbx.fetch(query_chrom, pos - 1, pos):
                fields = row.split("\t")
                if len(fields) < 5:
                    continue
                row_ref, row_alt = fields[2], fields[3]
                if row_ref == ref and row_alt == alt:
                    llr = float(fields[4])
                    if llr <= DELETERIOUS_THRESHOLD:
                        classification = "likely_deleterious"
                    elif llr >= BENIGN_THRESHOLD:
                        classification = "likely_benign"
                    else:
                        classification = "uncertain"

                    results.append(
                        VariantScore(
                            chrom=chrom,
                            pos=pos,
                            ref=ref,
                            alt=alt,
                            scores={"gpn_msa_llr": llr},
                            classifications={"gpn_msa_class": classification},
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
                    scores={"gpn_msa_llr": None},
                    classifications={"gpn_msa_class": "not_found"},
                )
            )

    tbx.close()
    return results


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
def gpn_msa_lookup(
    vcf_path: str,
    run_id: str,
) -> ToolResult:
    """Look up pre-computed GPN-MSA log-likelihood ratios for variants in a VCF.

    Parameters
    ----------
    vcf_path:
        Path to input VCF on a mounted volume.
    run_id:
        Unique run identifier.
    """
    out_dir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/gpn_msa")
    output_tsv = out_dir / "gpn_msa_scores.tsv"
    warnings: list[str] = []
    errors: list[str] = []

    with ToolTimer() as timer:
        try:
            if not Path(GPN_MSA_TSV).exists():
                raise FileNotFoundError(
                    f"GPN-MSA data not found at {GPN_MSA_TSV}. "
                    "Ensure the precomputed volume is populated."
                )

            # Extract variants from VCF
            variant_list: list[tuple[str, int, str, str]] = []
            vcf = pysam.VariantFile(vcf_path)
            for rec in vcf:
                for alt_allele in rec.alts or []:
                    variant_list.append((rec.chrom, rec.pos, rec.ref, alt_allele))
            vcf.close()

            # GPN-MSA does sequential tabix lookups — too slow for >500K variants.
            # Sample down to keep runtime under 10 min.
            MAX_VARIANTS = 200_000
            if len(variant_list) > MAX_VARIANTS:
                import random
                random.seed(42)
                sampled = random.sample(variant_list, MAX_VARIANTS)
                warnings.append(
                    f"Sampled {MAX_VARIANTS:,} of {len(variant_list):,} variants "
                    f"for GPN-MSA scoring (full genome too slow for sequential lookups). "
                    f"Run on individual chromosomes for complete coverage."
                )
                variant_list = sampled

            scored = _lookup_variants(variant_list)

            # Write output TSV
            with open(output_tsv, "w", newline="") as fh:
                writer = csv.writer(fh, delimiter="\t")
                writer.writerow([
                    "chrom", "pos", "ref", "alt", "gpn_msa_llr", "gpn_msa_class",
                ])
                for v in scored:
                    writer.writerow([
                        v.chrom, v.pos, v.ref, v.alt,
                        v.scores.get("gpn_msa_llr", ""),
                        v.classifications.get("gpn_msa_class", ""),
                    ])

        except Exception as e:
            errors.append(str(e))
            scored = []

    n_found = sum(
        1 for v in scored
        if v.classifications.get("gpn_msa_class") != "not_found"
    )
    n_deleterious = sum(
        1 for v in scored
        if v.classifications.get("gpn_msa_class") == "likely_deleterious"
    )

    return build_result(
        SPEC, timer,
        inputs={Artifact.VCF.value: vcf_path},
        payload={
            "n_variants_queried": len(scored),
            "n_scores_found": n_found,
            "n_likely_deleterious": n_deleterious,
            "deleterious_variants": [
                v.model_dump() for v in scored
                if v.classifications.get("gpn_msa_class") == "likely_deleterious"
            ][:1000],
            "output_tsv": str(output_tsv) if output_tsv.exists() else None,
        },
        errors=errors,
        warnings=warnings,
    )


SPEC = ToolSpec(
    name="gpn_msa",
    version="1.0",
    modes=(Mode.GERMLINE, Mode.SOMATIC),
    consumes=(Artifact.VCF,),
    criticality=Criticality.STANDARD,
    timeout_s=7200,
    reference_artifacts=("gpn_msa_scores",),
)
register(SPEC, gpn_msa_lookup)
