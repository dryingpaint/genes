"""OncoKB + CIViC — Somatic variant actionability lookup.

Queries the OncoKB REST API and CIViC GraphQL API to annotate somatic
variants with clinical actionability, drug sensitivity, and evidence levels.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import modal

from genes.app import app
from genes.infra.images import image_python_bio
from genes.infra.volumes import (
    MOUNT_WORKDIR,
    vol_workdir,
)
from genes.infra.provenance import PROVENANCE_KEY, stamp
from genes.tools._base import ToolResult, ToolTimer, ensure_dir

_ONCOKB_BASE = "https://www.oncokb.org/api/v1"
_CIVIC_GRAPHQL = "https://civicdb.org/api/graphql"


def _oncokb_annotate_variant(
    gene: str,
    protein_change: str,
    tumor_type: str | None,
    api_key: str,
) -> dict[str, Any]:
    """Query OncoKB for a single variant annotation."""
    params: dict[str, str] = {
        "hugoSymbol": gene,
        "alteration": protein_change,
    }
    if tumor_type:
        params["tumorType"] = tumor_type

    url = f"{_ONCOKB_BASE}/annotate/mutations/byProteinChange?{urlencode(params)}"
    req = Request(url, headers={
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    })

    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        return {
            "oncogenic": data.get("oncogenic", ""),
            "mutation_effect": data.get("mutationEffect", {}).get("knownEffect", ""),
            "highest_sensitive_level": data.get("highestSensitiveLevel", ""),
            "highest_resistance_level": data.get("highestResistanceLevel", ""),
            "highest_diagnostic_level": data.get("highestDiagnosticImplicationLevel", ""),
            "highest_prognostic_level": data.get("highestPrognosticImplicationLevel", ""),
            "treatments": [
                {
                    "drugs": [d.get("drugName", "") for d in t.get("drugs", [])],
                    "level": t.get("level", ""),
                    "indication": t.get("levelAssociatedCancerType", {}).get("name", ""),
                }
                for t in data.get("treatments", [])
            ],
            "source": "oncokb",
        }
    except HTTPError as exc:
        return {"error": f"OncoKB HTTP {exc.code}: {exc.reason}", "source": "oncokb"}
    except Exception as exc:
        return {"error": str(exc), "source": "oncokb"}


def _civic_query_variant(gene: str, variant_name: str) -> dict[str, Any]:
    """Query CIViC GraphQL API for variant evidence."""
    query = """
    query($gene: String!, $variantName: String!) {
        variants(name: $variantName, geneName: $gene, first: 5) {
            nodes {
                id
                name
                gene { name }
                molecularProfiles {
                    nodes {
                        evidenceItems {
                            nodes {
                                id
                                status
                                evidenceType
                                evidenceLevel
                                evidenceDirection
                                significance
                                therapies { name }
                                disease { name }
                                source { citation }
                            }
                        }
                    }
                }
            }
        }
    }
    """
    payload = json.dumps({
        "query": query,
        "variables": {"gene": gene, "variantName": variant_name},
    }).encode()

    req = Request(
        _CIVIC_GRAPHQL,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )

    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())

        variants = data.get("data", {}).get("variants", {}).get("nodes", [])
        if not variants:
            return {"evidence_items": [], "source": "civic", "status": "not_found"}

        evidence_items: list[dict] = []
        for variant in variants:
            for mp in variant.get("molecularProfiles", {}).get("nodes", []):
                for ei in mp.get("evidenceItems", {}).get("nodes", []):
                    if ei.get("status") != "accepted":
                        continue
                    evidence_items.append({
                        "evidence_type": ei.get("evidenceType"),
                        "evidence_level": ei.get("evidenceLevel"),
                        "evidence_direction": ei.get("evidenceDirection"),
                        "significance": ei.get("significance"),
                        "therapies": [t.get("name") for t in ei.get("therapies", [])],
                        "disease": ei.get("disease", {}).get("name"),
                        "citation": ei.get("source", {}).get("citation"),
                    })

        return {
            "evidence_items": evidence_items,
            "source": "civic",
            "status": "found" if evidence_items else "no_accepted_evidence",
        }
    except Exception as exc:
        return {"error": str(exc), "source": "civic"}


@app.function(
    image=image_python_bio,
    secrets=[modal.Secret.from_name("oncokb-api-key")],
    volumes={
        MOUNT_WORKDIR: vol_workdir,
    },
    timeout=600,
)
def lookup_actionability(
    variants: list[dict],
    run_id: str,
    tumor_type: str | None = None,
) -> ToolResult:
    """Look up clinical actionability for somatic variants.

    Args:
        variants: List of dicts with keys: gene, protein_change.
            protein_change should be HGVS-like (e.g. "V600E", "p.V600E").
        run_id: Unique identifier for this run.
        tumor_type: OncoKB tumor type (e.g. "Melanoma", "NSCLC"). Optional.

    Returns:
        ToolResult with OncoKB and CIViC annotations per variant.
    """
    outdir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/oncokb_civic")
    oncokb_api_key = os.environ.get("ONCOKB_API_KEY", "")

    with ToolTimer() as timer:
        errors: list[str] = []
        warnings: list[str] = []
        results: list[dict] = []

        if not oncokb_api_key:
            warnings.append(
                "ONCOKB_API_KEY not set; OncoKB lookups will be skipped."
            )

        for v in variants:
            gene = v.get("gene", "")
            protein_change = v.get("protein_change", "")

            if not gene or not protein_change:
                warnings.append(f"Skipping variant with missing gene/protein_change: {v}")
                continue

            # Clean protein_change (remove "p." prefix for OncoKB)
            clean_change = protein_change.lstrip("p.")
            variant_result: dict[str, Any] = {
                "gene": gene,
                "protein_change": protein_change,
            }

            # OncoKB lookup
            if oncokb_api_key:
                oncokb = _oncokb_annotate_variant(
                    gene=gene,
                    protein_change=clean_change,
                    tumor_type=tumor_type,
                    api_key=oncokb_api_key,
                )
                variant_result["oncokb"] = oncokb
            else:
                variant_result["oncokb"] = {"skipped": True}

            # CIViC lookup
            civic = _civic_query_variant(gene=gene, variant_name=clean_change)
            variant_result["civic"] = civic

            # Consolidated actionability level
            oncokb_level = variant_result.get("oncokb", {}).get("highest_sensitive_level", "")
            civic_evidence = variant_result.get("civic", {}).get("evidence_items", [])
            civic_levels = [e.get("evidence_level") for e in civic_evidence if e.get("evidence_level")]

            variant_result["actionability_summary"] = {
                "oncokb_level": oncokb_level or None,
                "civic_best_level": min(civic_levels) if civic_levels else None,
                "has_therapeutic_evidence": bool(oncokb_level) or any(
                    e.get("evidence_type") == "PREDICTIVE" for e in civic_evidence
                ),
            }

            results.append(variant_result)

        # Write output
        output_json = str(outdir / "actionability_results.json")
        with open(output_json, "w") as fh:
            json.dump(results, fh, indent=2)

        output_paths = [output_json]

        # Build summary
        summary = {
            "total_queried": len(results),
            "with_therapeutic_evidence": sum(
                1 for r in results
                if r.get("actionability_summary", {}).get("has_therapeutic_evidence")
            ),
            "oncokb_oncogenic": sum(
                1 for r in results
                if r.get("oncokb", {}).get("oncogenic") in ("Oncogenic", "Likely Oncogenic")
            ),
            "results": results,
        }

        vol_workdir.commit()

    return ToolResult(
        tool_name="oncokb_civic",
        version="1.0",
        started_at=timer.started_at,
        completed_at=timer.completed_at,
        input_summary={
            "num_variants": len(variants),
            "tumor_type": tumor_type,
            "run_id": run_id,
            PROVENANCE_KEY: stamp(),
        },
        output_paths=output_paths,
        output_summary=summary,
        errors=errors,
        warnings=warnings,
    )
