"""EVE — Evolutionary model of Variant Effect.

Looks up pre-computed EVE scores for protein variants. EVE is an
unsupervised generative model trained on evolutionary sequences that
predicts variant pathogenicity without labels.
"""

from __future__ import annotations

import csv
from pathlib import Path

from genes.app import app
from genes.infra.images import image_python_bio
from genes.infra.volumes import (
    MOUNT_PRECOMPUTED,
    MOUNT_WORKDIR,
    vol_precomputed,
    vol_workdir,
)
from genes.infra.provenance import PROVENANCE_KEY, stamp
from genes.tools._base import ToolResult, ToolTimer, ensure_dir

_EVE_DIR = f"{MOUNT_PRECOMPUTED}/eve"


@app.function(
    image=image_python_bio,
    volumes={
        MOUNT_PRECOMPUTED: vol_precomputed,
        MOUNT_WORKDIR: vol_workdir,
    },
    timeout=600,
)
def lookup_eve_scores(
    variants: list[dict],
    run_id: str,
) -> ToolResult:
    """Look up pre-computed EVE scores for a list of protein variants.

    Args:
        variants: List of dicts, each with keys:
            - gene: gene symbol (e.g. "BRCA1")
            - protein_variant: HGVS protein notation (e.g. "p.Arg1699Trp")
        run_id: Unique identifier for this run.

    Returns:
        ToolResult with scored variants in output_summary.
    """
    outdir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/eve")

    with ToolTimer() as timer:
        errors: list[str] = []
        warnings: list[str] = []
        scored: list[dict] = []

        # Group variants by gene for efficient file access
        by_gene: dict[str, list[dict]] = {}
        for v in variants:
            gene = v.get("gene", "").upper()
            if not gene:
                warnings.append(f"Skipping variant with missing gene: {v}")
                continue
            by_gene.setdefault(gene, []).append(v)

        for gene, gene_variants in by_gene.items():
            score_file = Path(_EVE_DIR) / f"{gene}_scores.csv"
            if not score_file.exists():
                warnings.append(f"No EVE score file found for gene {gene}")
                for v in gene_variants:
                    scored.append({
                        "gene": gene,
                        "variant": v.get("protein_variant", ""),
                        "eve_score": None,
                        "eve_class": None,
                        "status": "gene_not_available",
                    })
                continue

            # Load the gene's score table into a lookup dict keyed on variant notation
            lookup: dict[str, dict] = {}
            with open(score_file, newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    key = row.get("protein_variant", row.get("variant", "")).strip()
                    lookup[key] = row

            for v in gene_variants:
                pvar = v.get("protein_variant", "").strip()
                match = lookup.get(pvar)
                if match:
                    score = match.get("EVE_score") or match.get("eve_score")
                    eve_class_raw = (
                        match.get("EVE_class_75_pct")
                        or match.get("eve_class")
                        or match.get("class75")
                    )
                    scored.append({
                        "gene": gene,
                        "variant": pvar,
                        "eve_score": float(score) if score else None,
                        "eve_class": eve_class_raw,
                        "status": "found",
                    })
                else:
                    scored.append({
                        "gene": gene,
                        "variant": pvar,
                        "eve_score": None,
                        "eve_class": None,
                        "status": "variant_not_found",
                    })

        # Write results to TSV
        output_tsv = str(outdir / "eve_scores.tsv")
        if scored:
            with open(output_tsv, "w", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=["gene", "variant", "eve_score", "eve_class", "status"],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerows(scored)

        vol_workdir.commit()

    return ToolResult(
        tool_name="eve",
        version="1.0",
        started_at=timer.started_at,
        completed_at=timer.completed_at,
        input_summary={
            "num_variants": len(variants),
            "genes_queried": list(by_gene.keys()),
            "run_id": run_id,
            PROVENANCE_KEY: stamp("eve_scores"),
        },
        output_paths=[output_tsv] if scored else [],
        output_summary={
            "total_queried": len(variants),
            "found": sum(1 for s in scored if s["status"] == "found"),
            "not_found": sum(1 for s in scored if s["status"] == "variant_not_found"),
            "gene_not_available": sum(1 for s in scored if s["status"] == "gene_not_available"),
            "scored_variants": scored,
        },
        errors=errors,
        warnings=warnings,
    )
