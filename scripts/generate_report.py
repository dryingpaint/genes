"""Generate a plain-language genome analysis report from pipeline results.

Usage: python scripts/generate_report.py test_data/run_*.json
"""

import json
import sys
from pathlib import Path


def generate_report(result_path: str) -> str:
    with open(result_path) as f:
        data = json.load(f)

    tools = data.get("tool_results", {})
    lines = []

    lines.append("# Genome Analysis Report")
    lines.append("")
    lines.append(f"**Sample:** {Path(data.get('tool_results', {}).get('vep', {}).get('input_summary', {}).get('vcf_path', 'Unknown')).stem}")
    runtime = data.get("summary", {}).get("total_runtime_seconds")
    if runtime:
        lines.append(f"**Analysis time:** {runtime:.0f} seconds")
    succeeded = [n for n, t in tools.items() if not t.get("errors")]
    lines.append(f"**Tools completed:** {len(succeeded)}/{len(tools)}")
    lines.append("")

    # --- Variant Overview (VEP) ---
    vep = tools.get("vep", {}).get("output_summary", {})
    n_variants = vep.get("variant_count", 0)
    if n_variants:
        lines.append("## What's in your genome")
        lines.append("")
        lines.append(f"We analyzed **{n_variants:,} genetic variants** in your DNA. "
                      "These are positions where your genome differs from the human "
                      "reference sequence. Most of these differences are completely "
                      "normal — they're what make you genetically unique.")
        lines.append("")

    # --- Protein Impact (AlphaMissense) ---
    am = tools.get("alphamissense", {}).get("output_summary", {})
    n_scored = am.get("n_scores_found", 0)
    n_pathogenic = am.get("pathogenic_count", 0)
    n_benign = am.get("benign_count", 0)
    scored_variants = am.get("scored_variants", [])
    pathogenic_variants = [v for v in scored_variants
                           if v.get("classifications", {}).get("am_class") == "likely_pathogenic"]

    if n_scored:
        lines.append("## Protein-changing variants")
        lines.append("")
        lines.append(f"Of your {n_variants:,} variants, **{n_scored}** change the amino acid "
                      f"sequence of a protein. We scored each one using AlphaMissense, "
                      f"an AI model that predicts whether protein changes are harmful.")
        lines.append("")
        lines.append(f"- **{n_benign}** are predicted **benign** — they change a protein but "
                      "are unlikely to cause problems")
        lines.append(f"- **{n_pathogenic}** are flagged as **potentially harmful** — these "
                      "change proteins in ways that could affect their function")
        ambiguous = n_scored - n_benign - n_pathogenic
        if ambiguous:
            lines.append(f"- **{ambiguous}** are **uncertain** — the model can't confidently "
                          "classify them either way")
        lines.append("")

        if pathogenic_variants:
            lines.append("**Variants flagged for attention:**")
            lines.append("")
            for v in pathogenic_variants[:10]:
                score = v.get("scores", {}).get("am_pathogenicity", 0)
                gene = v.get("gene", "unknown gene")
                chrom = v.get("chrom", "")
                pos = v.get("pos", "")
                lines.append(f"- Position {chrom}:{pos} in gene {gene} — "
                              f"pathogenicity score {score:.2f}/1.00")
            lines.append("")
            lines.append("*These scores are computational predictions, not clinical diagnoses. "
                          "A score above 0.56 suggests the variant may be harmful, but "
                          "further evaluation by a genetic counselor is recommended before "
                          "taking any action.*")
            lines.append("")

    # --- Evolutionary Conservation (GPN-MSA) ---
    gpn = tools.get("gpn_msa", {}).get("output_summary", {})
    n_gpn_scored = gpn.get("n_scores_found", 0)
    n_deleterious = gpn.get("n_likely_deleterious", 0)

    if n_gpn_scored:
        lines.append("## Evolutionary conservation")
        lines.append("")
        lines.append(f"We compared **{n_gpn_scored:,}** of your variants against the genomes "
                      "of 89 other vertebrate species to see which positions are evolutionarily "
                      "conserved. Positions that have stayed the same across millions of years "
                      "of evolution are more likely to be important.")
        lines.append("")
        pct = (n_deleterious / n_gpn_scored * 100) if n_gpn_scored else 0
        lines.append(f"- **{n_deleterious:,}** variants ({pct:.1f}%) fall at highly conserved "
                      "positions, suggesting they could affect gene regulation or function")
        lines.append(f"- The remaining {n_gpn_scored - n_deleterious:,} variants are at "
                      "positions that vary naturally across species")
        lines.append("")
        lines.append("*A variant at a conserved position doesn't necessarily cause disease — "
                      "it means evolution has 'preferred' to keep that DNA sequence unchanged, "
                      "which suggests it may be functionally important.*")
        lines.append("")

    # --- Splicing (SpliceAI) ---
    sai = tools.get("spliceai", {}).get("output_summary", {})
    n_high_impact = sai.get("n_high_impact", 0)
    if "n_variants_queried" in sai:
        lines.append("## Splicing analysis")
        lines.append("")
        if n_high_impact > 0:
            lines.append(f"We found **{n_high_impact}** variants that may disrupt how your "
                          "genes are processed (spliced) into functional messages. Splicing "
                          "errors can prevent genes from producing the correct protein.")
        else:
            lines.append("We checked all your variants for potential splicing disruptions — "
                          "the process by which genes are processed into functional messages. "
                          "**No high-impact splice variants were detected.**")
        lines.append("")

    # --- Pharmacogenomics (PharmCAT) ---
    pgx = tools.get("pharmcat", {}).get("output_summary", {})
    n_genes = pgx.get("n_genes_called", 0)
    n_drugs = pgx.get("n_drug_recommendations", 0)
    diplotypes = pgx.get("diplotypes", [])

    lines.append("## Medication response")
    lines.append("")
    if n_genes > 0:
        lines.append(f"We identified your genetic variants in **{n_genes} genes** that affect "
                      "how your body processes medications. Based on your DNA:")
        lines.append("")
        for d in diplotypes[:10]:
            gene = d.get("gene", "")
            phenotype = d.get("phenotype", "")
            diplotype = d.get("diplotype", "")
            if gene and phenotype:
                lines.append(f"- **{gene}** ({diplotype}): You are a **{phenotype}**")
        lines.append("")
        if n_drugs > 0:
            lines.append(f"This affects your recommended dosing for **{n_drugs} medications**. "
                          "Share these results with your doctor or pharmacist.")
    else:
        lines.append("This analysis covers chromosome 22 only, which does not contain the "
                      "major pharmacogenomic genes (like CYP2D6, CYP2C19, CYP3A5). "
                      "A full-genome analysis would provide medication response predictions.")
    lines.append("")

    # --- Ancestry ---
    anc = tools.get("ancestry", {}).get("output_summary", {})
    pca = anc.get("pca_coordinates", {})
    admixture = anc.get("admixture", {})

    if pca:
        lines.append("## Genetic ancestry")
        lines.append("")
        if admixture:
            lines.append("Based on your DNA compared to global reference populations:")
            lines.append("")
            for pop, frac in sorted(admixture.items(), key=lambda x: -x[1]):
                lines.append(f"- {pop}: {frac*100:.1f}%")
        else:
            lines.append("We computed your genetic coordinates relative to 2,504 individuals "
                          "from the 1000 Genomes Project. The principal component analysis "
                          "positions you in genetic space relative to global populations.")
            lines.append("")
            lines.append("*Note: This analysis used chromosome 22 only. A full-genome analysis "
                          "with all chromosomes would give a more precise ancestry estimate.*")
        lines.append("")

    # --- Limitations ---
    lines.append("## Important limitations")
    lines.append("")
    lines.append("- This report is based on **computational predictions**, not clinical-grade "
                  "diagnostic tests. Results should be discussed with a healthcare provider "
                  "before making any medical decisions.")
    lines.append("- **AlphaMissense scores** predict whether protein changes are harmful based "
                  "on protein structure and evolution. They are not FDA-approved diagnostic tools.")
    lines.append("- **GPN-MSA conservation scores** reflect evolutionary constraint but do not "
                  "directly prove that a variant causes disease.")
    lines.append("- This analysis examined **one chromosome** (chr22). A full-genome analysis "
                  "would cover all 22 autosomes plus the sex chromosomes, providing a much "
                  "more complete picture.")
    lines.append("- Genetic risk is only one factor in health outcomes. Environment, lifestyle, "
                  "and other non-genetic factors play a major role in most conditions.")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/generate_report.py <result.json>")
        sys.exit(1)

    report = generate_report(sys.argv[1])
    print(report)

    # Also save to file
    out_path = sys.argv[1].replace("_results.json", "_report.md")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nSaved to: {out_path}", file=sys.stderr)
