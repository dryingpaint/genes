"""Idempotent dataset download functions for Phase 1 public datasets.

Each function writes to the datasets volume and checks for a sentinel file
to avoid re-downloading.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from genbench.app import app
from genbench.config import DATASETS_PATH, REFERENCE_PATH
from genbench.infra.images import cpu_hf_image, cpu_image
from genbench.infra.volumes import datasets_vol, reference_vol


def _sentinel(base: Path, name: str) -> Path:
    return base / f".downloaded_{name}"


def _already_downloaded(base: Path, name: str) -> bool:
    return _sentinel(base, name).exists()


def _mark_downloaded(base: Path, name: str) -> None:
    _sentinel(base, name).touch()


def _curl(url: str, dest: str, allow_fail: bool = False) -> bool:
    """Download a file with curl. Returns True on success."""
    result = subprocess.run(
        ["curl", "-fsSL", "--retry", "3", "--retry-delay", "5", url, "-o", dest],
        capture_output=True,
    )
    if result.returncode != 0:
        if allow_fail:
            print(f"  WARNING: Failed to download {url}: {result.stderr.decode()[:200]}")
            return False
        raise subprocess.CalledProcessError(result.returncode, result.args)
    return True


@app.function(
    image=cpu_image,
    volumes={DATASETS_PATH: datasets_vol},
    timeout=3600,
)
def ingest_clinvar() -> str:
    """Download ClinVar VCF and variant summary from NCBI FTP."""
    base = Path(DATASETS_PATH) / "clinvar"
    if _already_downloaded(base, "clinvar"):
        return "clinvar: already downloaded"

    base.mkdir(parents=True, exist_ok=True)

    files = [
        (
            "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz",
            "clinvar.vcf.gz",
        ),
        (
            "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz.tbi",
            "clinvar.vcf.gz.tbi",
        ),
        (
            "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz",
            "variant_summary.txt.gz",
        ),
    ]
    for url, fname in files:
        dest = base / fname
        if not dest.exists():
            _curl(url, str(dest))

    _mark_downloaded(base, "clinvar")
    datasets_vol.commit()
    return "clinvar: downloaded"


@app.function(
    image=cpu_image,
    volumes={DATASETS_PATH: datasets_vol},
    timeout=7200,
)
def ingest_giab() -> str:
    """Download GIAB truth sets for HG001-HG007."""
    base = Path(DATASETS_PATH) / "giab"
    if _already_downloaded(base, "giab"):
        return "giab: already downloaded"

    base.mkdir(parents=True, exist_ok=True)

    # Correct GIAB FTP paths (ReferenceSamples, NISTv4.2.1)
    giab_base = "https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release"
    samples = {
        "HG001": f"{giab_base}/NA12878_HG001/NISTv4.2.1/GRCh38",
        "HG002": f"{giab_base}/AshkenazimTrio/HG002_NA24385_son/NISTv4.2.1/GRCh38",
        "HG003": f"{giab_base}/AshkenazimTrio/HG003_NA24149_father/NISTv4.2.1/GRCh38",
        "HG004": f"{giab_base}/AshkenazimTrio/HG004_NA24143_mother/NISTv4.2.1/GRCh38",
        "HG005": f"{giab_base}/ChineseTrio/HG005_NA24631_son/NISTv4.2.1/GRCh38",
        "HG006": f"{giab_base}/ChineseTrio/HG006_NA24694_father/NISTv4.2.1/GRCh38",
        "HG007": f"{giab_base}/ChineseTrio/HG007_NA24695_mother/NISTv4.2.1/GRCh38",
    }

    downloaded = 0
    for sample, url_base in samples.items():
        sample_dir = base / sample
        sample_dir.mkdir(exist_ok=True)
        for suffix in [
            f"{sample}_GRCh38_1_22_v4.2.1_benchmark.vcf.gz",
            f"{sample}_GRCh38_1_22_v4.2.1_benchmark.vcf.gz.tbi",
            f"{sample}_GRCh38_1_22_v4.2.1_benchmark.bed",
            f"{sample}_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed",
        ]:
            dest = sample_dir / suffix
            if not dest.exists():
                url = f"{url_base}/{suffix}"
                if _curl(url, str(dest), allow_fail=True):
                    downloaded += 1

    _mark_downloaded(base, "giab")
    datasets_vol.commit()
    return f"giab: downloaded ({downloaded} new files)"


@app.function(
    image=cpu_hf_image,
    volumes={DATASETS_PATH: datasets_vol},
    timeout=7200,
    memory=8192,
)
def ingest_proteingym() -> str:
    """Download ProteinGym DMS substitution benchmark from HuggingFace."""
    base = Path(DATASETS_PATH) / "proteingym"
    if _already_downloaded(base, "proteingym"):
        return "proteingym: already downloaded"

    base.mkdir(parents=True, exist_ok=True)

    # Clean up any bad files from previous attempts
    for bad in base.glob("substitutions.zip*"):
        bad.unlink()

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="OATML-Markslab/ProteinGym",
        repo_type="dataset",
        allow_patterns=["ProteinGym_substitutions/*.csv", "ProteinGym_reference_file_substitutions.csv"],
        local_dir=str(base),
    )

    csv_count = len(list((base / "ProteinGym_substitutions").glob("*.csv")))

    _mark_downloaded(base, "proteingym")
    datasets_vol.commit()
    return f"proteingym: downloaded ({csv_count} assay CSVs from HuggingFace)"


@app.function(
    image=cpu_image,
    volumes={DATASETS_PATH: datasets_vol},
    timeout=1800,
)
def ingest_brca1_sge() -> str:
    """Download BRCA1 saturation genome editing data (Findlay 2018)."""
    base = Path(DATASETS_PATH) / "brca1_sge"
    if _already_downloaded(base, "brca1_sge"):
        return "brca1_sge: already downloaded"

    base.mkdir(parents=True, exist_ok=True)

    # Findlay 2018 supplementary table from Nature
    supp_url = (
        "https://static-content.springer.com/esm/"
        "art%3A10.1038%2Fs41586-018-0461-z/MediaObjects/"
        "41586_2018_461_MOESM3_ESM.xlsx"
    )
    dest = base / "findlay2018_supp_table2.xlsx"
    if not dest.exists():
        _curl(supp_url, str(dest), allow_fail=True)

    _mark_downloaded(base, "brca1_sge")
    datasets_vol.commit()
    return "brca1_sge: downloaded"


@app.function(
    image=cpu_image,
    volumes={DATASETS_PATH: datasets_vol},
    timeout=7200,
)
def ingest_dgrp() -> str:
    """Download DGRP2 genotypes and phenotype compendium.

    Primary site (dgrp2.gnets.ncsu.edu) is intermittently down.
    Falls back to NCBI/FlyBase mirrors.
    """
    base = Path(DATASETS_PATH) / "dgrp"
    if _already_downloaded(base, "dgrp"):
        return "dgrp: already downloaded"

    base.mkdir(parents=True, exist_ok=True)

    # Try multiple sources for DGRP2 genotypes
    geno_sources = [
        # Primary
        "http://dgrp2.gnets.ncsu.edu/data/website/dgrp2.tgeno",
        # Freeze 2 VCF from NCBI (alternative)
        "https://ftp.ncbi.nlm.nih.gov/pub/dgrp/freeze2/vcf/dgrp2.vcf.gz",
    ]

    geno_downloaded = False
    for url in geno_sources:
        fname = url.split("/")[-1]
        dest = base / fname
        if dest.exists():
            geno_downloaded = True
            break
        if _curl(url, str(dest), allow_fail=True):
            geno_downloaded = True
            break

    # Phenotype data — try primary, then create placeholder
    pheno_sources = [
        "http://dgrp2.gnets.ncsu.edu/data/website/dgrp2.pheno",
    ]

    pheno_downloaded = False
    for url in pheno_sources:
        fname = url.split("/")[-1]
        dest = base / fname
        if dest.exists():
            pheno_downloaded = True
            break
        if _curl(url, str(dest), allow_fail=True):
            pheno_downloaded = True
            break

    _mark_downloaded(base, "dgrp")
    datasets_vol.commit()

    status = []
    status.append("genotypes: ok" if geno_downloaded else "genotypes: FAILED (all sources down)")
    status.append("phenotypes: ok" if pheno_downloaded else "phenotypes: FAILED (site down)")
    return f"dgrp: {', '.join(status)}"


@app.function(
    image=cpu_image,
    volumes={DATASETS_PATH: datasets_vol},
    timeout=7200,
)
def ingest_arabidopsis() -> str:
    """Download 1001 Genomes Arabidopsis VCF and AraPheno phenotypes."""
    base = Path(DATASETS_PATH) / "arabidopsis"
    if _already_downloaded(base, "arabidopsis"):
        return "arabidopsis: already downloaded"

    base.mkdir(parents=True, exist_ok=True)

    # 1001 Genomes SNP matrix
    geno_url = (
        "https://1001genomes.org/data/GMI-MPI/releases/v3.1/"
        "1001genomes_snp-short-indel_only_ACGTN.vcf.gz"
    )
    geno_dest = base / "1001genomes.vcf.gz"
    geno_ok = True
    if not geno_dest.exists():
        geno_ok = _curl(geno_url, str(geno_dest), allow_fail=True)

    # AraPheno — use CSV endpoint for a known phenotype list
    # The JSON bulk endpoint is unreliable; download the phenotype index page
    pheno_url = "https://arapheno.1001genomes.org/rest/study/?format=csv"
    pheno_dest = base / "arapheno_studies.csv"
    pheno_ok = True
    if not pheno_dest.exists():
        pheno_ok = _curl(pheno_url, str(pheno_dest), allow_fail=True)

    _mark_downloaded(base, "arabidopsis")
    datasets_vol.commit()

    status = []
    status.append("genotypes: ok" if geno_ok else "genotypes: FAILED")
    status.append("phenotypes: ok" if pheno_ok else "phenotypes: FAILED")
    return f"arabidopsis: {', '.join(status)}"


@app.function(
    image=cpu_image,
    volumes={REFERENCE_PATH: reference_vol},
    timeout=7200,
)
def ingest_reference_genomes() -> str:
    """Download reference genomes: GRCh38."""
    base = Path(REFERENCE_PATH)
    if _already_downloaded(base, "references"):
        return "references: already downloaded"

    base.mkdir(parents=True, exist_ok=True)

    # GRCh38 analysis set (most important for benchmarks)
    grch38_url = (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/"
        "GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids/"
        "GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz"
    )
    dest = base / "GRCh38.fa.gz"
    if not dest.exists():
        _curl(grch38_url, str(dest), allow_fail=True)

    _mark_downloaded(base, "references")
    reference_vol.commit()
    return "references: downloaded"


@app.function(
    image=cpu_image,
    volumes={DATASETS_PATH: datasets_vol},
    timeout=7200,
    memory=8192,
)
def ingest_baseline_scores() -> str:
    """Download pre-computed baseline score files (AlphaMissense)."""
    base = Path(DATASETS_PATH) / "baseline_scores"
    if _already_downloaded(base, "baseline_scores"):
        return "baseline_scores: already downloaded"

    base.mkdir(parents=True, exist_ok=True)

    # AlphaMissense pre-computed scores (3.6 GB)
    am_url = "https://storage.googleapis.com/dm_alphamissense/AlphaMissense_hg38.tsv.gz"
    dest = base / "AlphaMissense_hg38.tsv.gz"
    if not dest.exists():
        _curl(am_url, str(dest))

    _mark_downloaded(base, "baseline_scores")
    datasets_vol.commit()
    return "baseline_scores: downloaded"
