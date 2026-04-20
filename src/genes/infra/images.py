"""Modal container image definitions.

Uses pre-built Docker images where available to avoid compiling from source.
Every image includes the local `genes` package via add_local_python_source.
"""

import modal


def _with_genes_src(image: modal.Image) -> modal.Image:
    """Add the local genes package to a Modal image."""
    return image.add_local_python_source("genes")


# --------------------------------------------------------------------------
# Image 1: Python bioinformatics (CPU)
# Most tools — lookups, scoring, Python-native tools
# --------------------------------------------------------------------------
image_python_bio = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("tabix", "bcftools", "samtools", "bedtools",
                 "libz-dev", "libbz2-dev", "liblzma-dev", "libcurl4-openssl-dev")
    .pip_install(
        "pydantic>=2.0", "polars>=1.0", "pysam>=0.22",
        "pandas>=2.2", "numpy>=1.26", "scipy>=1.12", "scikit-learn>=1.4",
    )
)

# --------------------------------------------------------------------------
# Image 2: VEP — use official Ensembl image
# --------------------------------------------------------------------------
# VEP image ships Perl + Python 2.7. We install Python 3.11 and our deps
# into it so the wrapper code can run alongside VEP's Perl.
image_vep = (
    modal.Image.from_registry("ensemblorg/ensembl-vep:release_112.0")
    .apt_install("python3", "python3-pip")
    .run_commands("python3 -m pip install --break-system-packages pydantic>=2.0 polars>=1.0")
)

# --------------------------------------------------------------------------
# Image 3: GATK/Mutect2 — use official Broad image (includes Java + GATK)
# PharmCAT and Exomiser added as JAR downloads
# --------------------------------------------------------------------------
# Java tools — debian base with JDK + pre-built JARs (no compilation)
image_java = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("openjdk-17-jre-headless", "wget", "unzip")
    .run_commands(
        # PharmCAT — single JAR download
        "wget -q https://github.com/PharmGKB/PharmCAT/releases/download/v2.13.0/pharmcat-2.13.0-all.jar"
        " -O /opt/pharmcat.jar",
        # GATK — zip download, no compilation
        "wget -q https://github.com/broadinstitute/gatk/releases/download/4.5.0.0/"
        "gatk-4.5.0.0.zip -O /tmp/gatk.zip"
        " && unzip -q /tmp/gatk.zip -d /opt/ && rm /tmp/gatk.zip",
    )
    .pip_install("pydantic>=2.0", "pysam>=0.22", "polars>=1.0")
)

# --------------------------------------------------------------------------
# Image 4: C++ tools — pre-compiled binaries only, no boost compilation
# plink2, ADMIXTURE, GCTB are all distributed as static binaries
# HLA*LA deferred to its own image if needed
# --------------------------------------------------------------------------
image_cpp_tools = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "unzip", "samtools", "bcftools", "tabix")
    .run_commands(
        # plink2 — pre-compiled static binary
        "wget -q https://s3.amazonaws.com/plink2-assets/alpha5/plink2_linux_x86_64_20240818.zip"
        " -O /tmp/plink2.zip && unzip -q /tmp/plink2.zip -d /usr/local/bin/ && rm /tmp/plink2.zip",
        # ADMIXTURE — pre-compiled binary
        "wget -q https://dalexander.github.io/admixture/binaries/admixture_linux-1.3.0.tar.gz"
        " -O /tmp/admixture.tar.gz && tar xzf /tmp/admixture.tar.gz -C /opt/"
        " && rm /tmp/admixture.tar.gz",
        # GCTB/SBayesRC — pre-compiled binary
        "wget -q https://cnsgenomics.com/software/gctb/download/gctb_2.05beta_Linux.zip"
        " -O /tmp/gctb.zip && unzip -q /tmp/gctb.zip -d /opt/ && rm /tmp/gctb.zip",
    )
    .pip_install("pydantic>=2.0", "pysam>=0.22", "pandas>=2.2", "numpy>=1.26", "polars>=1.0")
)

# --------------------------------------------------------------------------
# Image 5: GPU PyTorch — for SaProt, Borzoi, ChromBPNet, SpliceAI de novo
# --------------------------------------------------------------------------
image_gpu_torch = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0", "transformers>=4.40", "einops>=0.7",
        "biopython>=1.83", "pysam>=0.22", "pydantic>=2.0", "polars>=1.0",
    )
    .apt_install("samtools", "bcftools")
)

# --------------------------------------------------------------------------
# Image 6: DeepVariant — use official Google image
# --------------------------------------------------------------------------
image_gpu_jax = (
    modal.Image.from_registry("google/deepvariant:1.6.1-gpu")
    .pip_install("pydantic>=2.0", "polars>=1.0")
)

# --------------------------------------------------------------------------
# Image 7: R/Bioconductor — use Bioconductor base image (R pre-installed)
# --------------------------------------------------------------------------
# R image — only needed for cfDNA pipeline (ichorCNA). Deferred to avoid
# blocking germline/somatic pipelines. Use image_python_bio as fallback.
image_r = image_python_bio

# --------------------------------------------------------------------------
# Image 8: AnnotSV
# --------------------------------------------------------------------------
# AnnotSV image — only needed for SV annotation. Deferred to avoid blocking.
image_annotsv = image_python_bio

# ---------------------------------------------------------------------------
# Add local genes package to all images
# ---------------------------------------------------------------------------
image_python_bio = _with_genes_src(image_python_bio)
image_vep = _with_genes_src(image_vep)
image_java = _with_genes_src(image_java)
image_cpp_tools = _with_genes_src(image_cpp_tools)
image_gpu_torch = _with_genes_src(image_gpu_torch)
image_gpu_jax = _with_genes_src(image_gpu_jax)
image_r = _with_genes_src(image_r)
image_annotsv = _with_genes_src(image_annotsv)
