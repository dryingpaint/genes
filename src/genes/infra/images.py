"""Modal container image definitions.

8 images grouped by runtime dependency family. Each tool uses the leanest
image that satisfies its dependencies, minimizing cold-start time.

Every image includes the local `genes` package via add_local_python_source
so that tool wrappers can import from genes.* inside Modal containers.
"""

import modal


def _with_genes_src(image: modal.Image) -> modal.Image:
    """Add the local genes package to a Modal image."""
    return image.add_local_python_source("genes")

# --------------------------------------------------------------------------
# Image 1: Python bioinformatics (CPU)
# Tools: Cyrius, Biolearn, SigProfiler, UXM, Griffin, arcasHLA, PRS-CSx,
#        GPN-MSA (lookup), EVE (lookup), EVEE (lookup)
# --------------------------------------------------------------------------
image_python_bio = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libz-dev",
        "libbz2-dev",
        "liblzma-dev",
        "libcurl4-openssl-dev",
        "tabix",
        "bcftools",
        "samtools",
        "bedtools",
    )
    .pip_install(
        "pydantic>=2.0",
        "polars>=1.0",
        "pysam>=0.22",
        "pandas>=2.2",
        "numpy>=1.26",
        "scipy>=1.12",
        "scikit-learn>=1.4",
        "cyrius>=1.1",
        "biolearn>=0.5",
        "SigProfilerMatrixGenerator>=1.2",
        "SigProfilerExtractor>=1.1",
        "SigProfilerAssignment>=0.1",
    )
)

# --------------------------------------------------------------------------
# Image 2: Ensembl VEP (Perl)
# Tools: VEP with all plugins
# --------------------------------------------------------------------------
image_vep = (
    modal.Image.from_registry("ensemblorg/ensembl-vep:release_112.0")
    .pip_install("pydantic>=2.0", "polars>=1.0")
)

# --------------------------------------------------------------------------
# Image 3: Java tools (JDK 17)
# Tools: PharmCAT, Exomiser, GATK4/Mutect2
# --------------------------------------------------------------------------
image_java = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("openjdk-17-jre-headless", "wget", "unzip")
    .run_commands(
        # PharmCAT v2.13
        "wget -q https://github.com/PharmGKB/PharmCAT/releases/download/v2.13.0/pharmcat-2.13.0-all.jar"
        " -O /opt/pharmcat.jar",
        # Exomiser v14
        "wget -q https://github.com/exomiser/Exomiser/releases/download/14.0.0/"
        "exomiser-cli-14.0.0-distribution.zip -O /tmp/exomiser.zip"
        " && unzip -q /tmp/exomiser.zip -d /opt/ && rm /tmp/exomiser.zip",
        # GATK 4.5
        "wget -q https://github.com/broadinstitute/gatk/releases/download/4.5.0.0/"
        "gatk-4.5.0.0.zip -O /tmp/gatk.zip"
        " && unzip -q /tmp/gatk.zip -d /opt/ && rm /tmp/gatk.zip",
    )
    .pip_install("pydantic>=2.0", "pysam>=0.22", "polars>=1.0")
)

# --------------------------------------------------------------------------
# Image 4: C++ compiled tools
# Tools: plink2, flashPCA2, ADMIXTURE, MSIsensor-pro, HLA*LA, GCTB/SBayesRC
# --------------------------------------------------------------------------
image_cpp_tools = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "wget",
        "unzip",
        "g++",
        "make",
        "cmake",
        "git",
        "zlib1g-dev",
        "libbz2-dev",
        "liblzma-dev",
        "libboost-all-dev",
        "samtools",
        "bwa",
        "libhts-dev",
        "perl",
        "libtext-levenshtein-perl",
    )
    .run_commands(
        # plink2
        "wget -q https://s3.amazonaws.com/plink2-assets/alpha5/plink2_linux_x86_64_20240818.zip"
        " -O /tmp/plink2.zip && unzip -q /tmp/plink2.zip -d /usr/local/bin/ && rm /tmp/plink2.zip",
        # ADMIXTURE
        "wget -q https://dalexander.github.io/admixture/binaries/admixture_linux-1.3.0.tar.gz"
        " -O /tmp/admixture.tar.gz && tar xzf /tmp/admixture.tar.gz -C /opt/"
        " && rm /tmp/admixture.tar.gz",
        # GCTB (SBayesRC)
        "wget -q https://cnsgenomics.com/software/gctb/download/gctb_2.05beta_Linux.zip"
        " -O /tmp/gctb.zip && unzip -q /tmp/gctb.zip -d /opt/ && rm /tmp/gctb.zip",
    )
    .pip_install("pydantic>=2.0", "pysam>=0.22", "pandas>=2.2", "numpy>=1.26", "polars>=1.0")
)

# --------------------------------------------------------------------------
# Image 5: GPU deep learning (PyTorch)
# Tools: SaProt, Borzoi/Flashzoi, Pangolin, SpliceAI de novo, ChromBPNet
# --------------------------------------------------------------------------
image_gpu_torch = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0",
        "transformers>=4.40",
        "einops>=0.7",
        "biopython>=1.83",
        "pysam>=0.22",
        "pydantic>=2.0",
        "polars>=1.0",
    )
    .apt_install("samtools", "bcftools")
)

# --------------------------------------------------------------------------
# Image 6: GPU deep learning (JAX/TF) — DeepVariant, DeepSomatic, AlphaMissense
# --------------------------------------------------------------------------
image_gpu_jax = (
    modal.Image.from_registry("google/deepvariant:1.6.1-gpu")
    .pip_install("pydantic>=2.0", "polars>=1.0")
)

# --------------------------------------------------------------------------
# Image 7: R tools — ichorCNA
# --------------------------------------------------------------------------
image_r = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("r-base", "r-base-dev", "libcurl4-openssl-dev", "libxml2-dev")
    .run_commands(
        'Rscript -e \'install.packages("BiocManager", repos="https://cloud.r-project.org")\'',
        'Rscript -e \'BiocManager::install(c("HMMcopy"))\'',
    )
    .pip_install("pydantic>=2.0", "rpy2>=3.5", "polars>=1.0")
)

# --------------------------------------------------------------------------
# Image 8: AnnotSV (Tcl + bedtools)
# --------------------------------------------------------------------------
image_annotsv = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("tcl", "wget", "bedtools")
    .run_commands(
        "wget -q https://github.com/lgmgeo/AnnotSV/archive/refs/tags/v3.4.2.tar.gz"
        " -O /tmp/annotsv.tar.gz"
        " && tar xzf /tmp/annotsv.tar.gz -C /opt/ && rm /tmp/annotsv.tar.gz",
    )
    .pip_install("pydantic>=2.0", "polars>=1.0")
)

# ---------------------------------------------------------------------------
# Add local genes package to all images so tool wrappers can import genes.*
# ---------------------------------------------------------------------------
image_python_bio = _with_genes_src(image_python_bio)
image_vep = _with_genes_src(image_vep)
image_java = _with_genes_src(image_java)
image_cpp_tools = _with_genes_src(image_cpp_tools)
image_gpu_torch = _with_genes_src(image_gpu_torch)
image_gpu_jax = _with_genes_src(image_gpu_jax)
image_r = _with_genes_src(image_r)
image_annotsv = _with_genes_src(image_annotsv)
