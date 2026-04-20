"""Modal Volume definitions for reference data and working storage.

Split into domain-specific volumes so each tool only mounts what it needs,
reducing cold-start I/O. Total ~550 GB across all volumes.
"""

import modal

# GRCh38 reference genome + indices (BWA, samtools, minimap2)
# ~15 GB — needed by nearly all tools
vol_reference = modal.Volume.from_name("genes-reference", create_if_missing=True)

# Ensembl VEP cache + plugin data (homo_sapiens_merged/112_GRCh38)
# ~20 GB — VEP only
vol_vep = modal.Volume.from_name("genes-vep-cache", create_if_missing=True)

# Pre-computed variant effect scores
# AlphaMissense (~3.6 GB), SpliceAI (~60 GB), GPN-MSA (~10 GB), EVE (~2 GB)
# ~80 GB — lookup tools
vol_precomputed = modal.Volume.from_name("genes-precomputed", create_if_missing=True)

# Population genetics: gnomAD sites, LD matrices (SBayesRC/PRS-CSx), 1KG panels, PGS Catalog
# ~200 GB — PRS, ancestry, VEP gnomAD plugin
vol_popgen = modal.Volume.from_name("genes-popgen", create_if_missing=True)

# Exomiser data bundle (phenotype DB, variant DB, cross-species phenotype data)
# ~80 GB — Exomiser only
vol_exomiser = modal.Volume.from_name("genes-exomiser", create_if_missing=True)

# IMGT/HLA reference graph for HLA*LA
# ~30 GB — HLA typing only
vol_hla = modal.Volume.from_name("genes-hla-ref", create_if_missing=True)

# Clinical databases: ClinVar VCF, COSMIC, CIViC cache
# ~10 GB — VEP plugins, OncoKB/CIViC lookup
vol_clinical = modal.Volume.from_name("genes-clinical", create_if_missing=True)

# Deep learning model weights: DeepVariant, Borzoi, SaProt, ChromBPNet
# ~80 GB — GPU tools only
vol_models = modal.Volume.from_name("genes-models", create_if_missing=True)

# cfDNA references: Loyfer methylation atlas, ichorCNA PoN, Griffin TFBS lists
# ~60 GB — cfDNA tools only
vol_cfdna = modal.Volume.from_name("genes-cfdna-refs", create_if_missing=True)

# Per-run working directory for intermediate files and results
vol_workdir = modal.Volume.from_name("genes-workdir", create_if_missing=True)

# Benchmark truth sets: GIAB, GeT-RM, ClinVar temporal splits, ProteinGym, DGRP
# ~50 GB
vol_benchmarks = modal.Volume.from_name("genes-benchmark-data", create_if_missing=True)


# Mount path constants — use these in tool wrappers for consistency
MOUNT_REFERENCE = "/data/reference"
MOUNT_VEP = "/data/vep"
MOUNT_PRECOMPUTED = "/data/precomputed"
MOUNT_POPGEN = "/data/popgen"
MOUNT_EXOMISER = "/data/exomiser"
MOUNT_HLA = "/data/hla"
MOUNT_CLINICAL = "/data/clinical"
MOUNT_MODELS = "/data/models"
MOUNT_CFDNA = "/data/cfdna"
MOUNT_WORKDIR = "/work"
MOUNT_BENCHMARKS = "/data/benchmarks"
