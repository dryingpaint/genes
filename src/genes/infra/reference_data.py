"""One-time functions to populate Modal Volumes with reference data.

Each function is idempotent — it checks for existing files before downloading.
Run via: modal run src/genes/infra/reference_data.py

Total data: ~550 GB across all volumes. Functions are split so each can be
run independently and parallelized by chromosome where applicable.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from genes.app import app
from genes.infra.images import image_python_bio, image_vep, image_java
from genes.infra.volumes import (
    MOUNT_CLINICAL,
    MOUNT_POPGEN,
    MOUNT_PRECOMPUTED,
    MOUNT_REFERENCE,
    MOUNT_VEP,
    vol_clinical,
    vol_popgen,
    vol_precomputed,
    vol_reference,
    vol_vep,
)


def _run(cmd: str, check: bool = True) -> None:
    """Run a shell command, streaming output."""
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=check)


def _exists(path: str) -> bool:
    return os.path.exists(path)


# ---------------------------------------------------------------------------
# Reference genome (~15 GB)
# ---------------------------------------------------------------------------


@app.function(
    image=image_python_bio,
    volumes={MOUNT_REFERENCE: vol_reference},
    timeout=3600,
)
def populate_reference_genome():
    """Download GRCh38 reference genome + indices."""
    ref_fa = f"{MOUNT_REFERENCE}/GRCh38.fa"
    if _exists(ref_fa):
        print("Reference genome already present, skipping")
        return

    url = (
        "https://ftp.ensembl.org/pub/release-112/fasta/homo_sapiens/dna/"
        "Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
    )
    _run(f"wget -q -O {ref_fa}.gz {url}")
    _run(f"gunzip {ref_fa}.gz")
    _run(f"samtools faidx {ref_fa}")
    _run(f"samtools dict {ref_fa} -o {ref_fa.replace('.fa', '.dict')}")

    vol_reference.commit()
    print("Reference genome populated")


# ---------------------------------------------------------------------------
# VEP cache (~20 GB)
# ---------------------------------------------------------------------------


@app.function(
    image=image_vep,
    volumes={MOUNT_VEP: vol_vep},
    timeout=7200,
)
def populate_vep_cache():
    """Download Ensembl VEP cache for GRCh38 using vep_install (conda-installed VEP)."""
    cache_dir = f"{MOUNT_VEP}/homo_sapiens_merged/112_GRCh38"
    if _exists(cache_dir):
        print("VEP cache already present, skipping")
        return

    # vep_install is the conda-installed wrapper for INSTALL.pl
    _run(
        f"vep_install -a cf -s homo_sapiens_merged -y GRCh38"
        f" -c {MOUNT_VEP} --NO_UPDATE"
    )

    vol_vep.commit()
    print("VEP cache populated")


# ---------------------------------------------------------------------------
# Pre-computed variant scores (~80 GB)
# ---------------------------------------------------------------------------


@app.function(
    image=image_python_bio,
    volumes={MOUNT_PRECOMPUTED: vol_precomputed},
    timeout=14400,
)
def populate_alphamissense_scores():
    """Download AlphaMissense pre-computed scores (~3.6 GB)."""
    out = f"{MOUNT_PRECOMPUTED}/alphamissense"
    tsv = f"{out}/AlphaMissense_hg38.tsv.gz"
    if _exists(tsv):
        print("AlphaMissense scores already present, skipping")
        return

    os.makedirs(out, exist_ok=True)
    url = (
        "https://storage.googleapis.com/dm_alphamissense/"
        "AlphaMissense_hg38.tsv.gz"
    )
    _run(f"wget -q -O {tsv} {url}")
    _run(f"tabix -s 1 -b 2 -e 2 -S 1 {tsv}")

    vol_precomputed.commit()
    print("AlphaMissense scores populated")


@app.function(
    image=image_python_bio,
    volumes={MOUNT_PRECOMPUTED: vol_precomputed},
    timeout=28800,
)
def populate_spliceai_scores():
    """Download SpliceAI pre-computed scores (~60 GB).

    These are distributed via Illumina Basespace. For automated download,
    you may need to use the Basespace CLI or a direct URL if available.
    This function provides the structure; actual URLs may require authentication.
    """
    out = f"{MOUNT_PRECOMPUTED}/spliceai"
    vcf = f"{out}/spliceai_scores.raw.snv.hg38.vcf.gz"
    if _exists(vcf):
        print("SpliceAI scores already present, skipping")
        return

    os.makedirs(out, exist_ok=True)
    # SpliceAI pre-computed scores require Basespace download
    # Placeholder — user must download manually or via Basespace CLI
    print(
        "SpliceAI pre-computed scores must be downloaded from Illumina Basespace.\n"
        "1. Install: pip install basespace-cli\n"
        "2. Download SNV + indel score files for hg38\n"
        f"3. Place in {out}/\n"
        "4. Re-run this function to index with tabix"
    )

    # If files exist, index them
    if _exists(vcf):
        _run(f"tabix -p vcf {vcf}")
        vol_precomputed.commit()


@app.function(
    image=image_python_bio,
    volumes={MOUNT_PRECOMPUTED: vol_precomputed},
    timeout=28800,  # 8 hours — 37 GB download
)
def populate_gpn_msa_scores():
    """Download GPN-MSA pre-computed scores from Hugging Face (~37 GB)."""
    out = f"{MOUNT_PRECOMPUTED}/gpn_msa"
    scores_file = f"{out}/gpn_msa_scores_hg38.tsv.gz"
    if _exists(scores_file):
        print("GPN-MSA scores already present, skipping")
        return

    os.makedirs(out, exist_ok=True)
    base_url = "https://huggingface.co/datasets/songlab/gpn-msa-hg38-scores/resolve/main"
    _run(f"wget -q -O {scores_file} {base_url}/scores.tsv.bgz")
    _run(f"wget -q -O {scores_file}.tbi {base_url}/scores.tsv.bgz.tbi")

    vol_precomputed.commit()
    print("GPN-MSA scores populated")


# ---------------------------------------------------------------------------
# Clinical databases (~10 GB)
# ---------------------------------------------------------------------------


@app.function(
    image=image_python_bio,
    volumes={MOUNT_CLINICAL: vol_clinical},
    timeout=3600,
)
def populate_clinvar():
    """Download latest ClinVar VCF for GRCh38."""
    out = f"{MOUNT_CLINICAL}/clinvar"
    vcf = f"{out}/clinvar.vcf.gz"
    if _exists(vcf):
        print("ClinVar already present, skipping")
        return

    os.makedirs(out, exist_ok=True)
    url = (
        "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/"
        "clinvar.vcf.gz"
    )
    _run(f"wget -q -O {vcf} {url}")
    _run(f"wget -q -O {vcf}.tbi {url}.tbi")

    vol_clinical.commit()
    print("ClinVar populated")


# ---------------------------------------------------------------------------
# Population genetics (~200 GB total, download the essentials)
# ---------------------------------------------------------------------------


@app.function(
    image=image_python_bio,
    volumes={MOUNT_POPGEN: vol_popgen},
    timeout=14400,  # 4 hours
)
def populate_1kg_reference():
    """Download 1000 Genomes Phase 3 plink files for ancestry inference.

    Uses the PLINK2-formatted 1KG data from the official PLINK resources.
    """
    out = f"{MOUNT_POPGEN}/1kg"
    bed = f"{out}/all_phase3_GRCh38.bed"
    if _exists(bed):
        print("1KG reference panel already present, skipping")
        return

    os.makedirs(out, exist_ok=True)

    # Download 1KG Phase 3 in plink2 format (GRCh38 liftover)
    # These are available from multiple sources. Using the compact plink2 format.
    base = "https://www.dropbox.com/s"
    print(
        "1KG Phase 3 plink files for GRCh38 must be prepared manually:\n"
        "1. Download 1KG Phase 3 VCF from https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/\n"
        "2. Liftover to GRCh38 using CrossMap or picard LiftoverVcf\n"
        "3. Convert to plink: plink2 --vcf <lifted.vcf.gz> --make-bed --out all_phase3_GRCh38\n"
        f"4. Place files in {out}/\n"
        "\nAlternatively, download pre-made GRCh38 plink files from:\n"
        "  https://www.cog-genomics.org/plink/2.0/resources\n"
    )
    vol_popgen.commit()


# ---------------------------------------------------------------------------
# Entrypoint to run all population functions
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def populate_all():
    """Populate all reference data volumes. Run via: modal run src/genes/infra/reference_data.py"""
    print("Populating reference data volumes...")
    print("This may take several hours for the first run.\n")

    # Run in dependency order — reference genome first, then everything else in parallel
    populate_reference_genome.remote()
    print("Reference genome done.\n")

    # These can run in parallel — collect results individually so one failure
    # doesn't block the rest
    jobs = {
        "alphamissense": populate_alphamissense_scores.spawn(),
        "clinvar": populate_clinvar.spawn(),
        "spliceai": populate_spliceai_scores.spawn(),
        "gpn_msa": populate_gpn_msa_scores.spawn(),
        "vep_cache": populate_vep_cache.spawn(),
    }

    for name, handle in jobs.items():
        try:
            handle.get()
            print(f"  [{name}] done")
        except Exception as exc:
            print(f"  [{name}] FAILED: {exc}")

    print("\nReference data population complete.")
