"""OncoKB + CIViC — Somatic variant actionability lookup.

Consumes a VEP-annotated VCF, extracts (gene, protein_change) pairs from
the CSQ field, then queries OncoKB and CIViC for each unique alteration.

OncoKB requires an API key configured as Modal secret "oncokb-api-key";
CIViC is open.
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
from genes.infra.volumes import MOUNT_WORKDIR, vol_workdir
from genes.orchestrator.spec import Artifact, Criticality, Mode, ToolSpec, register
from genes.tools._base import ToolResult, ToolTimer, build_result, ensure_dir

_ONCOKB_BASE = "https://www.oncokb.org/api/v1"
_CIVIC_GRAPHQL = "https://civicdb.org/api/graphql"


def _oncokb_annotate_variant(
    gene: str, protein_change: str, tumor_type: str | None, api_key: str,
) -> dict[str, Any]:
    params: dict[str, str] = {"hugoSymbol": gene, "alteration": protein_change}
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
        }
    except HTTPError as exc:
        return {"error": f"OncoKB HTTP {exc.code}: {exc.reason}"}
    except Exception as exc:
        return {"error": str(exc)}


def _civic_query_variant(gene: str, variant_name: str) -> dict[str, Any]:
    query = """
    query($gene: String!, $variantName: String!) {
        variants(name: $variantName, geneName: $gene, first: 5) {
            nodes {
                id name gene { name }
                molecularProfiles { nodes { evidenceItems { nodes {
                    id status evidenceType evidenceLevel evidenceDirection
                    significance therapies { name } disease { name }
                    source { citation }
                } } } }
            }
        }
    }
    """
    payload = json.dumps({
        "query": query,
        "variables": {"gene": gene, "variantName": variant_name},
    }).encode()
    req = Request(
        _CIVIC_GRAPHQL, data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        variants = data.get("data", {}).get("variants", {}).get("nodes", [])
        if not variants:
            return {"evidence_items": [], "status": "not_found"}
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
            "status": "found" if evidence_items else "no_accepted_evidence",
        }
    except Exception as exc:
        return {"error": str(exc)}


def _extract_variants_from_vep_vcf(vcf_path: str) -> list[dict]:
    """Pull (gene, protein_change) pairs from a VEP-annotated VCF's CSQ field."""
    import gzip

    opener = gzip.open if vcf_path.endswith(".gz") else open
    csq_fields: list[str] | None = None
    out: dict[tuple[str, str], dict] = {}

    with opener(vcf_path, "rt") as fh:
        for line in fh:
            if line.startswith("##INFO=<ID=CSQ"):
                # Format: ##INFO=<ID=CSQ,...,Description="...Format: <fields>">
                if 'Format:' in line:
                    fmt = line.split('Format:', 1)[1].rsplit('"', 1)[0].strip()
                    csq_fields = fmt.split("|")
                continue
            if line.startswith("#") or not csq_fields:
                continue
            cols = line.rstrip().split("\t")
            if len(cols) < 8:
                continue
            info = cols[7]
            csq_blob = next(
                (kv[4:] for kv in info.split(";") if kv.startswith("CSQ=")),
                None,
            )
            if not csq_blob:
                continue
            sym_idx = csq_fields.index("SYMBOL") if "SYMBOL" in csq_fields else None
            amino_idx = csq_fields.index("Amino_acids") if "Amino_acids" in csq_fields else None
            prot_idx = csq_fields.index("Protein_position") if "Protein_position" in csq_fields else None
            if sym_idx is None or amino_idx is None or prot_idx is None:
                continue
            for csq in csq_blob.split(","):
                parts = csq.split("|")
                if len(parts) <= max(sym_idx, amino_idx, prot_idx):
                    continue
                gene = parts[sym_idx]
                aa = parts[amino_idx]
                ppos = parts[prot_idx]
                if not gene or "/" not in aa or not ppos:
                    continue
                ref_aa, alt_aa = aa.split("/", 1)
                # Use the start of the protein-position range for substitutions.
                start = ppos.split("-", 1)[0]
                change = f"{ref_aa}{start}{alt_aa}"
                key = (gene, change)
                if key not in out:
                    out[key] = {"gene": gene, "protein_change": change}
    return list(out.values())


@app.function(
    image=image_python_bio,
    secrets=[modal.Secret.from_name("oncokb-api-key")],
    volumes={MOUNT_WORKDIR: vol_workdir},
    timeout=1800,
)
def lookup_actionability(
    annotated_vcf: str,
    run_id: str,
    *,
    tumor_type: str | None = None,
) -> ToolResult:
    """Look up actionability for protein-changing variants in a VEP-annotated VCF."""
    outdir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/oncokb_civic")
    oncokb_api_key = os.environ.get("ONCOKB_API_KEY", "")

    with ToolTimer() as timer:
        errors: list[str] = []
        warnings: list[str] = []
        results: list[dict] = []

        if not oncokb_api_key:
            warnings.append("ONCOKB_API_KEY not set; OncoKB lookups will be skipped.")

        try:
            variants = _extract_variants_from_vep_vcf(annotated_vcf)
        except Exception as exc:
            errors.append(f"Failed to parse VEP-annotated VCF: {exc}")
            variants = []

        for v in variants:
            gene = v["gene"]
            change = v["protein_change"]
            entry: dict[str, Any] = {"gene": gene, "protein_change": change}

            if oncokb_api_key:
                entry["oncokb"] = _oncokb_annotate_variant(
                    gene=gene, protein_change=change,
                    tumor_type=tumor_type, api_key=oncokb_api_key,
                )
            else:
                entry["oncokb"] = {"skipped": True}

            entry["civic"] = _civic_query_variant(gene=gene, variant_name=change)

            oncokb_level = entry["oncokb"].get("highest_sensitive_level", "") or None
            civic_levels = [
                ei.get("evidence_level")
                for ei in entry["civic"].get("evidence_items", [])
                if ei.get("evidence_level")
            ]
            entry["actionability"] = {
                "oncokb_level": oncokb_level,
                "civic_best_level": min(civic_levels) if civic_levels else None,
                "has_therapeutic_evidence": bool(oncokb_level) or any(
                    ei.get("evidence_type") == "PREDICTIVE"
                    for ei in entry["civic"].get("evidence_items", [])
                ),
            }
            results.append(entry)

        output_json = str(outdir / "actionability_results.json")
        with open(output_json, "w") as fh:
            json.dump(results, fh, indent=2)
        vol_workdir.commit()

    return build_result(
        SPEC, timer,
        inputs={Artifact.ANNOTATED_VCF.value: annotated_vcf},
        payload={
            "tumor_type": tumor_type,
            "n_queried": len(results),
            "n_with_therapeutic_evidence": sum(
                1 for r in results if r["actionability"]["has_therapeutic_evidence"]
            ),
            "n_oncogenic": sum(
                1 for r in results
                if r["oncokb"].get("oncogenic") in ("Oncogenic", "Likely Oncogenic")
            ),
            "results": results[:200],
            "output_json": output_json,
        },
        errors=errors,
        warnings=warnings,
    )


SPEC = ToolSpec(
    name="oncokb_civic",
    version="1.0",
    modes=(Mode.SOMATIC,),
    consumes=(Artifact.ANNOTATED_VCF,),
    criticality=Criticality.STANDARD,
    timeout_s=1800,
    request_kwargs=("tumor_type",),
)
register(SPEC, lookup_actionability)
