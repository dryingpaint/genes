"""Trait and health variant lookups via GWAS Catalog + curated annotations.

Two-layer approach:
1. **GWAS Catalog** (primary): ~300K genome-wide significant associations from
   thousands of published studies. Matches sample rsIDs against the full catalog
   to find all trait/disease associations. This is the real database — not a
   hardcoded toy list.
2. **Curated highlights** (enrichment): A small set of well-characterized SNPs
   with genotype-specific plain-language interpretations. These are the "fun"
   23andMe-style results (eye color, caffeine, etc.) that people actually want
   to read. The curated layer adds human-readable color on top of the GWAS hits.
"""

from __future__ import annotations

import json
from pathlib import Path

from genes.app import app
from genes.infra.images import image_python_bio
from genes.infra.volumes import (
    MOUNT_CLINICAL,
    MOUNT_WORKDIR,
    vol_clinical,
    vol_workdir,
)
from genes.orchestrator.spec import Artifact, Criticality, Mode, ToolSpec, register
from genes.tools._base import ToolResult, ToolTimer, build_result, ensure_dir

# Paths to pre-built GWAS Catalog indices (JSON)
_GWAS_RSID_INDEX = f"{MOUNT_CLINICAL}/gwas_catalog/gwas_rsid_index.json"
_GWAS_POS_INDEX = f"{MOUNT_CLINICAL}/gwas_catalog/gwas_pos_index.json"

# Curated SNPs with genotype-specific interpretations.
# These overlay the GWAS Catalog with plain-language explanations.
# Only covers the "greatest hits" — GWAS Catalog covers everything else.
CURATED_VARIANTS = {
    # --- Fun traits ---
    # pos = "chr:pos" (GRCh38, no "chr" prefix) for position-based matching
    "rs12913832": {"trait": "Eye color", "category": "trait", "pos": "15:28365618",
                   "GG": "Very likely blue/gray eyes",
                   "AG": "Likely green/hazel eyes",
                   "AA": "Likely brown eyes"},
    "rs1805007": {"trait": "Red hair (MC1R)", "category": "trait", "pos": "16:89919709",
                  "CT": "Carrier for red hair",
                  "TT": "Likely red hair",
                  "CC": "Unlikely red hair"},
    "rs4988235": {"trait": "Lactose tolerance", "category": "trait", "pos": "2:135851076",
                  "AA": "Lactose tolerant (can digest dairy as adult)",
                  "GA": "Lactose tolerant (can digest dairy as adult)",
                  "GG": "Likely lactose intolerant (reduced dairy digestion)"},
    "rs762551": {"trait": "Caffeine metabolism (CYP1A2)", "category": "trait", "pos": "15:74749576",
                 "AA": "Fast caffeine metabolizer",
                 "AC": "Intermediate caffeine metabolizer",
                 "CC": "Slow caffeine metabolizer"},
    "rs713598": {"trait": "Bitter taste (PTC/PROP)", "category": "trait", "pos": "7:141972804",
                 "CC": "Strong bitter taste perception",
                 "CG": "Moderate bitter taste perception",
                 "GG": "Reduced bitter taste perception"},
    "rs671": {"trait": "Alcohol flush reaction (ALDH2)", "category": "trait", "pos": "12:111803962",
              "GA": "Alcohol flush (Asian glow) — reduced ALDH2 activity",
              "AA": "Strong alcohol flush — very low ALDH2 activity",
              "GG": "No alcohol flush — normal ALDH2"},
    "rs1815739": {"trait": "Muscle fiber type (ACTN3)", "category": "trait", "pos": "11:66560624",
                  "CC": "More fast-twitch muscle fibers (sprint/power advantage)",
                  "CT": "Mixed muscle fiber composition",
                  "TT": "More slow-twitch muscle fibers (endurance advantage)"},
    "rs4481887": {"trait": "Asparagus odor detection", "category": "trait", "pos": "1:248432503",
                  "AA": "Likely can smell asparagus in urine",
                  "GA": "May be able to smell asparagus in urine",
                  "GG": "Likely cannot smell asparagus in urine"},
    "rs17822931": {"trait": "Earwax type (ABCC11)", "category": "trait", "pos": "16:48224287",
                   "CC": "Wet earwax (typical in European/African descent)",
                   "CT": "Wet earwax (wet is dominant)",
                   "TT": "Dry earwax (common in East Asian descent)"},
    "rs72921001": {"trait": "Cilantro taste perception", "category": "trait", "pos": "11:22410534",
                   "CC": "Unlikely to perceive soapy taste",
                   "CA": "May perceive mild soapy taste",
                   "AA": "Likely perceives soapy taste in cilantro"},
    "rs1799971": {"trait": "Pain sensitivity (OPRM1)", "category": "trait", "pos": "6:154039662",
                  "AA": "Typical pain sensitivity",
                  "AG": "May have altered pain sensitivity and opioid response",
                  "GG": "Likely altered pain sensitivity — may need higher opioid doses"},

    # --- Health ---
    "rs429358": {"trait": "APOE (Alzheimer's risk component 1)", "category": "health", "pos": "19:44908684"},
    "rs7412": {"trait": "APOE (Alzheimer's risk component 2)", "category": "health", "pos": "19:44908822"},
    "rs7903146": {"trait": "Type 2 diabetes risk (TCF7L2)", "category": "health", "pos": "10:112998590",
                  "CC": "Typical T2D risk",
                  "CT": "Moderately increased T2D risk (~1.4x)",
                  "TT": "Increased T2D risk (~2x)"},
    "rs10490924": {"trait": "Macular degeneration risk (ARMS2)", "category": "health", "pos": "10:122454932",
                   "GG": "Typical AMD risk",
                   "GT": "Moderately increased AMD risk",
                   "TT": "Significantly increased AMD risk"},
    "rs6025": {"trait": "Factor V Leiden (blood clotting)", "category": "health", "pos": "1:169549811",
               "CT": "Heterozygous carrier — moderately increased clotting risk",
               "TT": "Homozygous — significantly increased clotting risk",
               "CC": "No Factor V Leiden variant"},
    "rs2187668": {"trait": "Celiac disease risk (HLA-DQ2.5)", "category": "health", "pos": "6:32628009",
                  "TT": "Lower celiac disease risk",
                  "TC": "Carrier of HLA-DQ2.5 — increased celiac risk",
                  "CC": "Homozygous HLA-DQ2.5 — significantly increased celiac risk"},

    # --- Carrier ---
    "rs80357914": {"trait": "BRCA1 5382insC (Ashkenazi founder)", "category": "carrier", "pos": "17:43071077",
                   "effect": "Significantly increased breast/ovarian cancer risk"},
    "rs80357713": {"trait": "BRCA1 185delAG (Ashkenazi founder)", "category": "carrier", "pos": "17:43094464",
                   "effect": "Significantly increased breast/ovarian cancer risk"},
    "rs75961395": {"trait": "CFTR ΔF508 (cystic fibrosis)", "category": "carrier", "pos": "7:117559590",
                   "effect": "Carrier for cystic fibrosis (most common CF mutation)"},
    "rs334": {"trait": "HBB (sickle cell)", "category": "carrier", "pos": "11:5227002",
              "TA": "Sickle cell trait carrier",
              "AA": "Sickle cell disease",
              "TT": "No sickle cell variant"},
}


def _load_gwas_indices() -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Load pre-built GWAS Catalog indices (rsID and position-based)."""
    rsid_index: dict[str, list[dict]] = {}
    pos_index: dict[str, list[dict]] = {}

    if Path(_GWAS_RSID_INDEX).exists():
        with open(_GWAS_RSID_INDEX) as fh:
            rsid_index = json.load(fh)
    if Path(_GWAS_POS_INDEX).exists():
        with open(_GWAS_POS_INDEX) as fh:
            pos_index = json.load(fh)

    return rsid_index, pos_index


def _categorize_gwas_trait(mapped_trait: str) -> str:
    """Rough categorization of GWAS mapped traits."""
    t = mapped_trait.lower()
    # Disease / health
    disease_keywords = [
        "disease", "disorder", "cancer", "carcinoma", "diabetes", "asthma",
        "obesity", "hypertension", "schizophrenia", "alzheimer", "parkinson",
        "stroke", "arthritis", "lupus", "sclerosis", "epilepsy", "depression",
        "anxiety", "autism", "adhd", "bipolar", "anemia", "fibrosis",
        "cardiovascular", "coronary", "atrial", "heart failure",
    ]
    if any(kw in t for kw in disease_keywords):
        return "health"

    # Measurements / biomarkers
    measurement_keywords = [
        "level", "concentration", "count", "ratio", "volume", "pressure",
        "cholesterol", "triglyceride", "glucose", "bmi", "body mass",
        "hemoglobin", "platelet", "creatinine", "urate",
    ]
    if any(kw in t for kw in measurement_keywords):
        return "biomarker"

    # Physical traits
    return "trait"


@app.function(
    image=image_python_bio,
    volumes={
        MOUNT_CLINICAL: vol_clinical,
        MOUNT_WORKDIR: vol_workdir,
    },
    timeout=1800,
)
def lookup_traits(
    vcf_path: str,
    run_id: str,
) -> ToolResult:
    """Look up trait and health variants using GWAS Catalog + curated annotations.

    Scans the VCF for all variants with rsIDs, then matches against:
    1. GWAS Catalog (~100K+ SNPs with genome-wide significant trait associations)
    2. Curated highlights (genotype-specific plain-language interpretations)
    """
    import pysam

    out_dir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/traits")
    warnings: list[str] = []
    errors: list[str] = []

    with ToolTimer() as timer:
        try:
            # Load GWAS indices (both rsID and position-based)
            gwas_rsid_index, gwas_pos_index = _load_gwas_indices()
            has_gwas = bool(gwas_rsid_index or gwas_pos_index)
            if not has_gwas:
                warnings.append(
                    "GWAS Catalog index not found. Run populate_gwas_catalog() to enable "
                    "comprehensive trait lookups. Falling back to curated variants only."
                )
            else:
                warnings.append(
                    f"Loaded GWAS indices: {len(gwas_rsid_index):,} rsIDs, "
                    f"{len(gwas_pos_index):,} positions"
                )

            vcf = pysam.VariantFile(vcf_path)

            # Scan VCF: collect rsIDs AND positions for matching
            sample_by_rsid: dict[str, str] = {}   # rsid -> genotype
            sample_by_pos: dict[str, str] = {}     # "chr:pos" -> genotype
            rsid_for_pos: dict[str, str] = {}       # "chr:pos" -> rsid (if available)
            n_total = 0
            n_with_rsid = 0

            for rec in vcf:
                n_total += 1
                sample = list(rec.samples.values())[0]
                alleles = sample.alleles
                if not alleles or not all(a is not None for a in alleles):
                    continue

                genotype = "".join(sorted(alleles))
                chrom_raw = rec.chrom.replace("chr", "")
                pos_key = f"{chrom_raw}:{rec.pos}"
                sample_by_pos[pos_key] = genotype

                rsid = rec.id
                if rsid and rsid != "." and rsid.startswith("rs"):
                    n_with_rsid += 1
                    sample_by_rsid[rsid] = genotype
                    rsid_for_pos[pos_key] = rsid

            vcf.close()

            # --- Layer 1: GWAS Catalog matches (position-based primary) ---
            gwas_hits: list[dict] = []
            gwas_traits_seen: set[str] = set()

            # Match by position (works for all VCFs)
            if gwas_pos_index:
                for pos_key, genotype in sample_by_pos.items():
                    associations = gwas_pos_index.get(pos_key)
                    if not associations:
                        continue
                    for assoc in associations:
                        trait = assoc["trait"]
                        key = f"{pos_key}:{trait}"
                        if key in gwas_traits_seen:
                            continue
                        gwas_traits_seen.add(key)

                        category = _categorize_gwas_trait(assoc.get("mapped_trait", trait))
                        rsid = assoc.get("rsid", rsid_for_pos.get(pos_key, pos_key))

                        gwas_hits.append({
                            "rsid": rsid,
                            "position": pos_key,
                            "genotype": genotype,
                            "trait": trait,
                            "category": category,
                            "gene": assoc.get("gene", ""),
                            "risk_allele": assoc.get("risk_allele", ""),
                            "pvalue": assoc.get("pvalue"),
                            "effect_size": assoc.get("or_beta", ""),
                        })

            # Also match by rsID for VCFs that have them
            if gwas_rsid_index and n_with_rsid > 0:
                for rsid, genotype in sample_by_rsid.items():
                    associations = gwas_rsid_index.get(rsid)
                    if not associations:
                        continue
                    for assoc in associations:
                        trait = assoc["trait"]
                        key = f"{rsid}:{trait}"
                        if key in gwas_traits_seen:
                            continue
                        gwas_traits_seen.add(key)

                        category = _categorize_gwas_trait(assoc.get("mapped_trait", trait))
                        gwas_hits.append({
                            "rsid": rsid,
                            "genotype": genotype,
                            "trait": trait,
                            "category": category,
                            "gene": assoc.get("gene", ""),
                            "risk_allele": assoc.get("risk_allele", ""),
                            "pvalue": assoc.get("pvalue"),
                            "effect_size": assoc.get("or_beta", ""),
                        })

            # Sort by p-value (most significant first)
            gwas_hits.sort(key=lambda x: x.get("pvalue") or 1.0)

            # --- Layer 2: Curated highlights (position-based + rsID) ---
            curated_results: list[dict] = []
            for rsid, info in CURATED_VARIANTS.items():
                # Try rsID first, then position
                genotype = sample_by_rsid.get(rsid)
                if genotype is None and info.get("pos"):
                    genotype = sample_by_pos.get(info["pos"])

                if genotype is None:
                    continue

                interpretation = info.get(genotype, info.get("effect", f"Genotype: {genotype}"))
                curated_results.append({
                    "rsid": rsid,
                    "trait": info["trait"],
                    "genotype": genotype,
                    "interpretation": interpretation,
                    "category": info["category"],
                })

            # Split curated by category
            traits = [r for r in curated_results if r["category"] == "trait"]
            health = [r for r in curated_results if r["category"] == "health"]
            carrier = [r for r in curated_results if r["category"] == "carrier"]

            # Split GWAS by category
            gwas_health = [h for h in gwas_hits if h["category"] == "health"]
            gwas_traits = [h for h in gwas_hits if h["category"] == "trait"]
            gwas_biomarkers = [h for h in gwas_hits if h["category"] == "biomarker"]

            # Write full results to disk
            output_json = str(out_dir / "trait_results.json")
            with open(output_json, "w") as fh:
                json.dump({
                    "curated": curated_results,
                    "gwas_hits": gwas_hits,
                }, fh, indent=2)

            vol_workdir.commit()

        except Exception as e:
            errors.append(str(e))
            traits, health, carrier = [], [], []
            gwas_hits, gwas_health, gwas_traits, gwas_biomarkers = [], [], [], []
            n_total, n_with_rsid = 0, 0
            sample_variants = {}
            output_json = ""

    return build_result(
        SPEC, timer,
        inputs={Artifact.VCF.value: vcf_path},
        payload={
            "n_variants_in_vcf": n_total,
            "n_with_rsid": n_with_rsid,
            "n_gwas_matches": len(gwas_hits),
            "n_curated_found": len(traits) + len(health) + len(carrier),
            "traits": traits,
            "health": health,
            "carrier": carrier,
            "gwas_disease_associations": gwas_health[:100],
            "gwas_trait_associations": gwas_traits[:100],
            "gwas_biomarker_associations": gwas_biomarkers[:50],
            "gwas_health_count": len(gwas_health),
            "gwas_trait_count": len(gwas_traits),
            "gwas_biomarker_count": len(gwas_biomarkers),
            "output_json": output_json or None,
        },
        errors=errors,
        warnings=warnings,
    )


SPEC = ToolSpec(
    name="traits",
    version="2.0",
    modes=(Mode.GERMLINE,),
    consumes=(Artifact.VCF,),
    criticality=Criticality.OPTIONAL,
    timeout_s=1800,
    reference_artifacts=("gwas_catalog",),
)
register(SPEC, lookup_traits)
