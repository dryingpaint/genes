"""SpliceAI score lookup tool wrapper.

Queries pre-computed SpliceAI delta scores from a tabix-indexed VCF.
Extracts the four delta scores: DS_AG (acceptor gain), DS_AL (acceptor loss),
DS_DG (donor gain), DS_DL (donor loss). A max delta score >= 0.5
is conventionally considered a significant splice-altering variant.
"""

from __future__ import annotations

import csv
import re
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
from genes.tools._base import ToolResult, ToolTimer, VariantScore, ensure_dir

# Pre-computed SpliceAI scores (tabix-indexed VCF)
SPLICEAI_VCF = f"{MOUNT_PRECOMPUTED}/spliceai/spliceai_scores.raw.snv.hg38.vcf.gz"
SPLICEAI_INDEL_VCF = f"{MOUNT_PRECOMPUTED}/spliceai/spliceai_scores.raw.indel.hg38.vcf.gz"

# SpliceAI INFO field format:
# SpliceAI=ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL
SPLICEAI_PATTERN = re.compile(
    r"SpliceAI=([^|]*)\|([^|]*)\|"
    r"([0-9.]+)\|([0-9.]+)\|([0-9.]+)\|([0-9.]+)\|"
    r"([0-9-]+)\|([0-9-]+)\|([0-9-]+)\|([0-9-]+)"
)

SIGNIFICANCE_THRESHOLD = 0.2
HIGH_THRESHOLD = 0.5


def _parse_spliceai_info(info_str: str, alt: str) -> dict | None:
    """Parse SpliceAI annotation from VCF INFO field."""
    for match in SPLICEAI_PATTERN.finditer(info_str):
        allele = match.group(1)
        # Match if allele equals the ALT or is "." (single-alt record)
        if allele == alt or allele == ".":
            ds_ag = float(match.group(3))
            ds_al = float(match.group(4))
            ds_dg = float(match.group(5))
            ds_dl = float(match.group(6))
            max_ds = max(ds_ag, ds_al, ds_dg, ds_dl)
            return {
                "gene": match.group(2),
                "DS_AG": ds_ag,
                "DS_AL": ds_al,
                "DS_DG": ds_dg,
                "DS_DL": ds_dl,
                "max_DS": max_ds,
                "DP_AG": int(match.group(7)),
                "DP_AL": int(match.group(8)),
                "DP_DG": int(match.group(9)),
                "DP_DL": int(match.group(10)),
            }
    return None


def _lookup_in_file(
    tbx_path: str,
    variants: list[tuple[str, int, str, str]],
) -> dict[str, VariantScore]:
    """Query a single SpliceAI tabix VCF for a batch of variants."""
    results: dict[str, VariantScore] = {}
    if not Path(tbx_path).exists():
        return results

    tbx = pysam.TabixFile(tbx_path)
    for chrom, pos, ref, alt in variants:
        key = f"{chrom}:{pos}:{ref}:{alt}"
        if key in results:
            continue
        query_chrom = chrom.replace("chr", "")
        try:
            for row in tbx.fetch(query_chrom, pos - 1, pos):
                fields = row.split("\t")
                if len(fields) < 8:
                    continue
                row_ref, row_alt = fields[3], fields[4]
                if row_ref == ref and row_alt == alt:
                    parsed = _parse_spliceai_info(fields[7], alt)
                    if parsed:
                        max_ds = parsed["max_DS"]
                        if max_ds >= HIGH_THRESHOLD:
                            classification = "high"
                        elif max_ds >= SIGNIFICANCE_THRESHOLD:
                            classification = "moderate"
                        else:
                            classification = "low"
                        results[key] = VariantScore(
                            chrom=chrom,
                            pos=pos,
                            ref=ref,
                            alt=alt,
                            gene=parsed["gene"],
                            scores={
                                "DS_AG": parsed["DS_AG"],
                                "DS_AL": parsed["DS_AL"],
                                "DS_DG": parsed["DS_DG"],
                                "DS_DL": parsed["DS_DL"],
                                "max_DS": max_ds,
                            },
                            classifications={"spliceai_impact": classification},
                        )
                    break
        except ValueError:
            pass

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
def spliceai_lookup(
    vcf_path: str,
    run_id: str,
) -> ToolResult:
    """Look up pre-computed SpliceAI delta scores for variants in a VCF.

    Parameters
    ----------
    vcf_path:
        Path to input VCF on a mounted volume.
    run_id:
        Unique run identifier.
    """
    out_dir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/spliceai")
    output_tsv = out_dir / "spliceai_scores.tsv"
    warnings: list[str] = []
    errors: list[str] = []

    with ToolTimer() as timer:
        try:
            # Extract variants from input VCF
            variant_list: list[tuple[str, int, str, str]] = []
            vcf = pysam.VariantFile(vcf_path)
            for rec in vcf:
                for alt in rec.alts or []:
                    variant_list.append((rec.chrom, rec.pos, rec.ref, alt))
            vcf.close()

            # Separate SNVs from indels for file routing
            snvs = [
                v for v in variant_list
                if len(v[2]) == 1 and len(v[3]) == 1
            ]
            indels = [
                v for v in variant_list
                if len(v[2]) != 1 or len(v[3]) != 1
            ]

            # Query both pre-computed files
            scored: dict[str, VariantScore] = {}
            scored.update(_lookup_in_file(SPLICEAI_VCF, snvs))
            scored.update(_lookup_in_file(SPLICEAI_INDEL_VCF, indels))

            # Build results list preserving input order, filling missing
            all_results: list[VariantScore] = []
            for chrom, pos, ref, alt in variant_list:
                key = f"{chrom}:{pos}:{ref}:{alt}"
                if key in scored:
                    all_results.append(scored[key])
                else:
                    all_results.append(
                        VariantScore(
                            chrom=chrom,
                            pos=pos,
                            ref=ref,
                            alt=alt,
                            scores={
                                "DS_AG": None, "DS_AL": None,
                                "DS_DG": None, "DS_DL": None,
                                "max_DS": None,
                            },
                            classifications={"spliceai_impact": "not_found"},
                        )
                    )

            # Write output TSV
            with open(output_tsv, "w", newline="") as fh:
                writer = csv.writer(fh, delimiter="\t")
                writer.writerow([
                    "chrom", "pos", "ref", "alt", "gene",
                    "DS_AG", "DS_AL", "DS_DG", "DS_DL", "max_DS",
                    "spliceai_impact",
                ])
                for v in all_results:
                    writer.writerow([
                        v.chrom, v.pos, v.ref, v.alt, v.gene or "",
                        v.scores.get("DS_AG", ""),
                        v.scores.get("DS_AL", ""),
                        v.scores.get("DS_DG", ""),
                        v.scores.get("DS_DL", ""),
                        v.scores.get("max_DS", ""),
                        v.classifications.get("spliceai_impact", ""),
                    ])

        except Exception as e:
            errors.append(str(e))
            all_results = []

    n_found = sum(
        1 for v in all_results
        if v.classifications.get("spliceai_impact") != "not_found"
    )
    n_high = sum(
        1 for v in all_results
        if v.classifications.get("spliceai_impact") == "high"
    )

    return ToolResult(
        tool_name="spliceai",
        version="1.3.1",
        started_at=timer.started_at,
        completed_at=timer.completed_at,
        input_summary={
            "vcf_path": vcf_path,
            "run_id": run_id,
            "n_variants_queried": len(all_results),
        },
        output_paths=[str(output_tsv)] if output_tsv.exists() else [],
        output_summary={
            "n_variants_queried": len(all_results),
            "n_scores_found": n_found,
            "n_high_impact": n_high,
            "scores": [v.model_dump() for v in all_results],
        },
        errors=errors,
        warnings=warnings,
    )
