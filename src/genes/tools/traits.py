"""Known trait and health variant lookups.

This is how 23andMe generates most of their reports — simple lookups of
well-characterized SNPs against curated databases. No ML needed.
"""

from __future__ import annotations

from genes.app import app
from genes.infra.images import image_python_bio
from genes.infra.volumes import MOUNT_WORKDIR, vol_workdir
from genes.infra.provenance import PROVENANCE_KEY, stamp
from genes.tools._base import ToolResult, ToolTimer, ensure_dir

# Well-characterized trait/health SNPs with known effects.
# Format: rsid -> {chr, pos (GRCh38), ref, alt, trait, effect_allele, effect}
KNOWN_VARIANTS = {
    # --- Eye color (HIrisPlex) ---
    "rs12913832": {"chr": "chr15", "pos": 28365618, "ref": "A", "alt": "G",
                   "trait": "Eye color", "effect_allele": "G",
                   "GG": "Very likely blue/gray eyes",
                   "AG": "Likely green/hazel eyes",
                   "AA": "Likely brown eyes"},

    # --- Hair color ---
    "rs1805007": {"chr": "chr16", "pos": 89919709, "ref": "C", "alt": "T",
                  "trait": "Red hair", "effect_allele": "T",
                  "CT": "Carrier for red hair",
                  "TT": "Likely red hair",
                  "CC": "Unlikely red hair"},

    # --- Lactose intolerance ---
    "rs4988235": {"chr": "chr2", "pos": 135851076, "ref": "G", "alt": "A",
                  "trait": "Lactose tolerance", "effect_allele": "A",
                  "AA": "Lactose tolerant (can digest dairy as adult)",
                  "GA": "Lactose tolerant (can digest dairy as adult)",
                  "GG": "Likely lactose intolerant (reduced dairy digestion)"},

    # --- Caffeine metabolism ---
    "rs762551": {"chr": "chr15", "pos": 74749576, "ref": "A", "alt": "C",
                 "trait": "Caffeine metabolism", "effect_allele": "A",
                 "AA": "Fast caffeine metabolizer",
                 "AC": "Intermediate caffeine metabolizer",
                 "CC": "Slow caffeine metabolizer"},

    # --- Bitter taste perception ---
    "rs713598": {"chr": "chr7", "pos": 141972804, "ref": "C", "alt": "G",
                 "trait": "Bitter taste (PTC/PROP)", "effect_allele": "C",
                 "CC": "Strong bitter taste perception",
                 "CG": "Moderate bitter taste perception",
                 "GG": "Reduced bitter taste perception"},

    # --- Alcohol flush ---
    "rs671": {"chr": "chr12", "pos": 111803962, "ref": "G", "alt": "A",
              "trait": "Alcohol flush reaction", "effect_allele": "A",
              "GA": "Alcohol flush (Asian glow) — reduced ALDH2 activity",
              "AA": "Strong alcohol flush — very low ALDH2 activity",
              "GG": "No alcohol flush — normal ALDH2"},

    # --- Muscle composition ---
    "rs1815739": {"chr": "chr11", "pos": 66560624, "ref": "C", "alt": "T",
                  "trait": "Muscle fiber type (ACTN3)", "effect_allele": "T",
                  "CC": "More fast-twitch muscle fibers (sprint/power advantage)",
                  "CT": "Mixed muscle fiber composition",
                  "TT": "More slow-twitch muscle fibers (endurance advantage)"},

    # --- Asparagus smell ---
    "rs4481887": {"chr": "chr1", "pos": 248432503, "ref": "G", "alt": "A",
                  "trait": "Asparagus odor detection", "effect_allele": "A",
                  "AA": "Likely can smell asparagus in urine",
                  "GA": "May be able to smell asparagus in urine",
                  "GG": "Likely cannot smell asparagus in urine"},

    # --- HEALTH: APOE (Alzheimer's risk) ---
    "rs429358": {"chr": "chr19", "pos": 44908684, "ref": "T", "alt": "C",
                 "trait": "APOE (Alzheimer's risk component 1)", "effect_allele": "C",
                 "health": True},
    "rs7412": {"chr": "chr19", "pos": 44908822, "ref": "C", "alt": "T",
               "trait": "APOE (Alzheimer's risk component 2)", "effect_allele": "T",
               "health": True},

    # --- HEALTH: BRCA1 founder mutations ---
    "rs80357914": {"chr": "chr17", "pos": 43071077, "ref": "TCAA", "alt": "T",
                   "trait": "BRCA1 5382insC (Ashkenazi founder)", "effect_allele": "T",
                   "health": True, "carrier": True,
                   "effect": "Significantly increased breast/ovarian cancer risk"},
    "rs80357713": {"chr": "chr17", "pos": 43094464, "ref": "AG", "alt": "A",
                   "trait": "BRCA1 185delAG (Ashkenazi founder)", "effect_allele": "A",
                   "health": True, "carrier": True,
                   "effect": "Significantly increased breast/ovarian cancer risk"},

    # --- HEALTH: Factor V Leiden ---
    "rs6025": {"chr": "chr1", "pos": 169549811, "ref": "C", "alt": "T",
               "trait": "Factor V Leiden (blood clotting)", "effect_allele": "T",
               "health": True,
               "CT": "Heterozygous carrier — moderately increased clotting risk",
               "TT": "Homozygous — significantly increased clotting risk",
               "CC": "No Factor V Leiden variant"},

    # --- CARRIER: Cystic fibrosis ---
    "rs75961395": {"chr": "chr7", "pos": 117559590, "ref": "ATCT", "alt": "A",
                   "trait": "CFTR ΔF508 (cystic fibrosis)", "effect_allele": "A",
                   "health": True, "carrier": True,
                   "effect": "Carrier for cystic fibrosis (most common CF mutation)"},

    # --- CARRIER: Sickle cell ---
    "rs334": {"chr": "chr11", "pos": 5227002, "ref": "T", "alt": "A",
              "trait": "HBB (sickle cell)", "effect_allele": "A",
              "health": True, "carrier": True,
              "TA": "Sickle cell trait carrier",
              "AA": "Sickle cell disease",
              "TT": "No sickle cell variant"},
}


@app.function(
    image=image_python_bio,
    volumes={MOUNT_WORKDIR: vol_workdir},
    timeout=600,
)
def lookup_traits(
    vcf_path: str,
    run_id: str,
) -> ToolResult:
    """Look up well-characterized trait and health variants from a VCF."""
    import pysam

    out_dir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/traits")
    warnings = []
    errors = []

    with ToolTimer() as timer:
        try:
            vcf = pysam.VariantFile(vcf_path)
            results = {"traits": [], "health": [], "carrier": []}
            found = 0

            for rsid, info in KNOWN_VARIANTS.items():
                chrom = info["chr"]
                pos = info["pos"]

                # Try to fetch the variant
                genotype = None
                try:
                    for rec in vcf.fetch(chrom, pos - 1, pos):
                        if rec.pos == pos:
                            # Get genotype
                            sample = list(rec.samples.values())[0]
                            alleles = sample.alleles
                            if alleles:
                                genotype = "".join(sorted(alleles))
                            break
                except (ValueError, StopIteration):
                    pass

                if genotype is None:
                    continue

                found += 1
                # Look up the effect
                gt_key = genotype.replace(info["ref"], info["ref"]).replace(info["alt"], info["alt"])

                # Build readable genotype (e.g., "AG")
                readable_gt = genotype
                effect = info.get(readable_gt, info.get("effect", f"Genotype: {readable_gt}"))

                entry = {
                    "rsid": rsid,
                    "trait": info["trait"],
                    "genotype": readable_gt,
                    "interpretation": effect,
                    "position": f"{chrom}:{pos}",
                }

                if info.get("carrier"):
                    results["carrier"].append(entry)
                elif info.get("health"):
                    results["health"].append(entry)
                else:
                    results["traits"].append(entry)

            vcf.close()

        except Exception as e:
            errors.append(str(e))
            results = {"traits": [], "health": [], "carrier": []}
            found = 0

    return ToolResult(
        tool_name="traits",
        version="1.0",
        started_at=timer.started_at,
        completed_at=timer.completed_at,
        input_summary={
            "vcf_path": vcf_path,
            "run_id": run_id,
            PROVENANCE_KEY: stamp("reference_genome"),
        },
        output_paths=[],
        output_summary={
            "n_variants_checked": len(KNOWN_VARIANTS),
            "n_found": found,
            **results,
        },
        errors=errors,
        warnings=warnings,
    )
