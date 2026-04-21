"""Modal container image definitions.

Strategy: base Python image for most tools, specialized images only where
a specific binary is required (VEP, Java, plink2).
"""

import modal


def _with_genes_src(image: modal.Image) -> modal.Image:
    """Add the local genes package to a Modal image."""
    return image.add_local_python_source("genes")


# --------------------------------------------------------------------------
# Base: Python 3.11 + bioinformatics CLI tools + common pip packages
# Used by most tools (lookups, scoring, Python-native tools)
# --------------------------------------------------------------------------
_base = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("tabix", "bcftools", "samtools", "bedtools", "wget", "unzip",
                 "libz-dev", "libbz2-dev", "liblzma-dev", "libcurl4-openssl-dev")
    .pip_install(
        "pydantic>=2.0", "polars>=1.0", "pysam>=0.22",
        "pandas>=2.2", "numpy>=1.26", "scipy>=1.12", "scikit-learn>=1.4",
    )
)

# --------------------------------------------------------------------------
# Java: base + JDK 17 + PharmCAT JAR + GATK
# For PharmCAT, Exomiser, Mutect2
# --------------------------------------------------------------------------
_java = (
    _base
    .apt_install("openjdk-17-jre-headless")
    .run_commands(
        "wget -q https://github.com/PharmGKB/PharmCAT/releases/download/v2.13.0/"
        "pharmcat-2.13.0-all.jar -O /opt/pharmcat.jar",
    )
)

# --------------------------------------------------------------------------
# C++ tools: base + plink2 static binary
# For PRS calculation and ancestry inference
# --------------------------------------------------------------------------
_cpp = (
    _base
    .run_commands(
        "wget -q https://s3.amazonaws.com/plink2-assets/alpha5/"
        "plink2_linux_x86_64_20240818.zip -O /tmp/plink2.zip"
        " && unzip -q /tmp/plink2.zip -d /usr/local/bin/ && rm /tmp/plink2.zip",
    )
)

# VEP: Install via conda (handles Bio::DB::HTS and all Perl deps cleanly)
_vep = (
    modal.Image.micromamba(python_version="3.11")
    .micromamba_install("ensembl-vep=112.0", "htslib", "samtools", "bcftools",
                        "pip", "pysam", "numpy", "pandas",
                        channels=["bioconda", "conda-forge"])
    .pip_install("pydantic>=2.0", "polars>=1.0")
)

# --------------------------------------------------------------------------
# Assign to named images used by tool wrappers
# --------------------------------------------------------------------------
image_python_bio = _base     # Lookups, scoring, Cyrius, Biolearn, etc.
image_vep = _vep             # VEP annotation
image_java = _java           # PharmCAT, Exomiser, GATK/Mutect2
image_cpp_tools = _cpp       # plink2 (PRS), ancestry
image_gpu_torch = _base      # TODO: restore with PyTorch + CUDA
image_gpu_jax = _base        # TODO: restore with DeepVariant
image_r = _base              # TODO: restore with R + ichorCNA
image_annotsv = _base        # TODO: restore with AnnotSV

# Add local genes package to all images
image_python_bio = _with_genes_src(image_python_bio)
image_vep = _with_genes_src(image_vep)
image_java = _with_genes_src(image_java)
image_cpp_tools = _with_genes_src(image_cpp_tools)
image_gpu_torch = _with_genes_src(image_gpu_torch)
image_gpu_jax = _with_genes_src(image_gpu_jax)
image_r = _with_genes_src(image_r)
image_annotsv = _with_genes_src(image_annotsv)
