"""SOTA baseline finder — research agent that identifies top-performing models.

Uses bioRxiv search and web search to find the current SOTA for each eval.
Designed to be run as a script (scripts/find_baselines.py) or imported.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from genbench.registry import get_eval, list_evals
from genbench.research.baselines_config import update_sota

# Search query templates per eval — maps eval name to search terms
SEARCH_QUERIES = {
    "clinvar": "ClinVar pathogenic variant classification benchmark SOTA 2024 2025 2026",
    "pgx": "pharmacogenomics star allele diplotype calling benchmark accuracy 2024 2025",
    "hla": "HLA typing accuracy benchmark WGS 2024 2025",
    "giab": "germline variant calling benchmark GIAB precision recall SOTA 2024 2025",
    "eqtl": "cis-eQTL prediction cross-individual GTEx benchmark SOTA 2024 2025",
    "caqtl": "chromatin accessibility QTL prediction DART-Eval benchmark 2024 2025",
    "sqtl": "splicing QTL prediction benchmark SpliceAI SOTA 2024 2025",
    "dms": "ProteinGym DMS zero-shot protein variant effect prediction SOTA 2024 2025",
    "brca1_sge": "BRCA1 saturation genome editing variant classification benchmark 2024 2025",
    "ukb_quantitative": "polygenic risk score PRS UK Biobank quantitative trait SOTA 2024 2025",
    "ukb_disease": "polygenic risk score PRS UK Biobank disease prediction SOTA 2024 2025",
    "hirisplex": "HIrisPlex-S externally visible characteristics prediction accuracy 2024",
    "portability": "cross-ancestry PRS portability PRS-CSx benchmark 2024 2025",
    "dgrp": "DGRP Drosophila genomic prediction GBLUP benchmark 2024 2025",
    "arabidopsis": "Arabidopsis 1001 genomes genomic prediction flowering time SOTA 2024 2025",
}

# Known baselines from BENCHMARKS.md (hardcoded as fallback)
KNOWN_BASELINES = {
    "clinvar": {
        "model": "AlphaMissense",
        "metric": "AUROC ~0.94 on ClinVar (unsupervised, no circularity)",
        "paper": "Cheng et al., Science 2023",
        "doi": "10.1126/science.adg7492",
    },
    "pgx": {
        "model": "Cyrius (CYP2D6) + PharmCAT (others)",
        "metric": "96.5–99.3% concordance on GeT-RM",
        "paper": "Chen et al., The Pharmacogenomics Journal 2021",
    },
    "hla": {
        "model": "HLA*LA",
        "metric": ">99% class I, >97% class II concordance",
        "paper": "Dilthey et al., Bioinformatics 2019",
    },
    "giab": {
        "model": "DeepVariant v1.6",
        "metric": "SNV F1 > 0.999, indel F1 > 0.995",
        "paper": "Poplin et al., Nature Biotechnology 2018",
    },
    "eqtl": {
        "model": "PrediXcan elastic-net",
        "metric": "Cross-individual r ≈ 0–0.1 (all models, Huang/Sasse 2023)",
        "paper": "Gamazon et al., Nature Genetics 2015",
    },
    "caqtl": {
        "model": "ChromBPNet ISM",
        "metric": "Wins DART-Eval counterfactual VEP",
        "paper": "Patel & Kundaje, NeurIPS 2024",
    },
    "sqtl": {
        "model": "Evo 2 / SpliceAI",
        "metric": "Evo 2 #1 on SpliceVarDB; SpliceAI Δ≥0.5 captures ~85% known splice-disrupting",
        "paper": "Jaganathan et al., Cell 2019 / Brixi et al., Science 2025",
    },
    "dms": {
        "model": "SaProt-1.3B",
        "metric": "Mean Spearman ρ ~0.48 on ProteinGym",
        "paper": "Su et al., ICLR 2024",
    },
    "brca1_sge": {
        "model": "Evo 2",
        "metric": ">90% accuracy on Findlay 2018 SGE",
        "paper": "Brixi et al., Science 2025",
    },
    "ukb_quantitative": {
        "model": "SBayesRC + covariates",
        "metric": "Height R² ~0.30–0.40 (EUR)",
        "paper": "Zheng et al., Nature Genetics 2024",
    },
    "ukb_disease": {
        "model": "SBayesRC PRS + clinical risk factors",
        "metric": "CAD AUC ~0.78–0.82 (PRS + covariates)",
        "paper": "Zheng et al., Nature Genetics 2024",
    },
    "hirisplex": {
        "model": "HIrisPlex-S",
        "metric": "Eye color AUC >0.90, hair ~0.80, skin ~0.75",
        "paper": "Chaitanya et al., FSI Genetics 2018",
    },
    "portability": {
        "model": "PRS-CSx",
        "metric": "1.2–1.8× improvement in AFR over EUR-only PRS",
        "paper": "Ruan et al., Nature Genetics 2022",
    },
    "dgrp": {
        "model": "GBLUP / BayesB",
        "metric": "CV R² ~0.3–0.7 per trait (approaches H²)",
        "paper": "Mackay et al., Nature 2012",
    },
    "arabidopsis": {
        "model": "GBLUP",
        "metric": "Flowering time R² ~0.4–0.7",
        "paper": "1001 Genomes Consortium, Cell 2016",
    },
}


def find_sota(eval_name: str, use_search: bool = True) -> dict[str, Any]:
    """Find the current SOTA for an eval.

    Args:
        eval_name: Registered eval name.
        use_search: If True, attempt web/bioRxiv search. If False, use known baselines only.

    Returns:
        Dict with model, metric, paper, updated fields.
    """
    # Start with known baseline
    result = KNOWN_BASELINES.get(eval_name, {}).copy()
    result["updated"] = str(date.today())
    result["source"] = "known_baseline"

    # TODO: integrate bioRxiv and web search for live SOTA discovery
    # When search is available:
    # 1. Build query from SEARCH_QUERIES[eval_name]
    # 2. Search bioRxiv for recent preprints
    # 3. Search web for benchmark leaderboards
    # 4. Parse results to extract model name + reported performance
    # 5. Compare against known baseline, keep the better one

    return result


def find_all_sota(use_search: bool = True) -> dict[str, dict]:
    """Find SOTA for all registered evals."""
    results = {}
    for eval_name in list_evals():
        print(f"  Researching: {eval_name}...")
        results[eval_name] = find_sota(eval_name, use_search=use_search)
    return results


def update_baselines_yaml(use_search: bool = True) -> None:
    """Run SOTA research and write results to baselines.yaml."""
    all_sota = find_all_sota(use_search=use_search)
    for eval_name, sota_info in all_sota.items():
        update_sota(eval_name, sota_info)
    print(f"\nUpdated baselines.yaml with {len(all_sota)} entries")
