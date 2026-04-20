"""Constants and configuration for the benchmark harness.

Values sourced from BENCHMARKS.md spec tables. Heritability estimates are for EUR ancestry
unless otherwise noted.
"""

from __future__ import annotations

# --- Volume mount paths (Modal) ---

REFERENCE_PATH = "/data/reference"
DATASETS_PATH = "/data/datasets"
RESULTS_PATH = "/data/results"
MODELS_PATH = "/data/models"

# --- Tier 0 failure thresholds ---
# Below these values, the pipeline is considered broken.

TIER0_GATES = {
    "clinvar.missense": {"metric": "auroc", "threshold": 0.85},
    "giab.snv": {"metric": "f1", "threshold": 0.99},
    "pgx.cyp2d6": {"metric": "accuracy", "threshold": 0.90},
    "hla.class_i": {"metric": "concordance", "threshold": 0.95},
}

# --- Heritability ceilings (h²_SNP for EUR) ---
# Used for ceiling-normalized metrics in Tier 2.

H2_SNP = {
    # Tier 2a: Quantitative traits
    "height": 0.80,
    "bmi": 0.40,
    "ldl_cholesterol": 0.50,
    "hdl_cholesterol": 0.45,
    "triglycerides": 0.35,
    "sbp": 0.30,
    "hba1c": 0.40,
    "egfr": 0.35,
    "fev1": 0.45,
    "grip_strength": 0.40,
    "heel_bmd": 0.50,
    "platelet_count": 0.55,
    "mcv": 0.60,
}

# --- Expected PRS R² at SOTA (EUR) ---

EXPECTED_PRS_R2 = {
    "height": (0.30, 0.40),
    "bmi": (0.08, 0.12),
    "ldl_cholesterol": (0.12, 0.18),
    "hdl_cholesterol": (0.10, 0.15),
    "triglycerides": (0.06, 0.10),
    "sbp": (0.05, 0.08),
    "hba1c": (0.08, 0.12),
    "egfr": (0.05, 0.08),
    "fev1": (0.08, 0.12),
    "grip_strength": (0.05, 0.08),
    "heel_bmd": (0.10, 0.15),
    "platelet_count": (0.15, 0.20),
    "mcv": (0.18, 0.25),
}

# --- Tier 1 ceilings ---

TIER1_CEILINGS = {
    "eqtl_median_cis_h2": 0.15,  # median gene cis-h²; cross-individual r ceiling ≈ sqrt(0.15) ≈ 0.39
    "dms_high_quality_rho": (0.70, 0.80),  # per-assay ceiling for high-quality assays
    "dms_aggregate_rho": (0.55, 0.65),  # across all assays including noisy ones
    "brca1_sge_ceiling": (0.95, 0.97),  # SGE assay noise floor
}

# --- Tier 3 ceilings (broad-sense H²) ---

TIER3_CEILINGS = {
    "dgrp": {
        "starvation_resistance": 0.60,
        "chill_coma_recovery": 0.50,
        "startle_response": 0.40,
        "lifespan": 0.35,
    },
    "arabidopsis": {
        "flowering_time": (0.60, 0.80),
        "metabolites": (0.30, 0.60),
        "morphological": (0.20, 0.50),
    },
}

# --- Ancestry portability expected ratios ---

PORTABILITY_RATIOS = {
    "AFR": (0.25, 0.50),
    "SAS": (0.50, 0.70),
    "EAS": (0.40, 0.60),
    "AMR": (0.40, 0.65),
}

# --- Dataset versions ---

DATASET_VERSIONS = {
    "clinvar": "2024-01-15",  # temporal split cutoff: 2023-12-31
    "giab": "v4.2.1",
    "giab_cmrg": "v1.0",
    "proteingym": "v1",
    "brca1_sge": "findlay2018",
    "dgrp": "dgrp2",
    "arabidopsis_genomes": "1001g",
    "arabidopsis_pheno": "arapheno",
    "gtex": "v8",
    "ukb": "wgs_500k",
}

# --- Anti-leakage defaults ---

KING_THRESHOLD = 0.0442  # No 2nd-degree or closer relatives across splits
TEMPORAL_CUTOFF = "2023-12-31"  # ClinVar train/test boundary
HOLDOUT_CHROMOSOMES = ("chr8", "chr21")
