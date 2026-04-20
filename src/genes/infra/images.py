"""Modal container image definitions.

MVP: Uses a single Python image for all tools to avoid build failures from
transient network issues. Tool-specific images (VEP, GATK, DeepVariant)
will be restored once Modal's image cache is warm.
"""

import modal


def _with_genes_src(image: modal.Image) -> modal.Image:
    """Add the local genes package to a Modal image."""
    return image.add_local_python_source("genes")


# --------------------------------------------------------------------------
# Single base image for MVP — all tools use this
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

# All images point to the same base for now
image_python_bio = _base
image_vep = _base        # TODO: restore ensemblorg/ensembl-vep:release_112.0
image_java = _base       # TODO: restore with JDK + PharmCAT/GATK/Exomiser JARs
image_cpp_tools = _base  # TODO: restore with plink2/ADMIXTURE/GCTB binaries
image_gpu_torch = _base  # TODO: restore with PyTorch + CUDA
image_gpu_jax = _base    # TODO: restore with DeepVariant
image_r = _base          # TODO: restore with R + Bioconductor
image_annotsv = _base    # TODO: restore with AnnotSV

# Add local genes package to all images
image_python_bio = _with_genes_src(image_python_bio)
image_vep = _with_genes_src(image_vep)
image_java = _with_genes_src(image_java)
image_cpp_tools = _with_genes_src(image_cpp_tools)
image_gpu_torch = _with_genes_src(image_gpu_torch)
image_gpu_jax = _with_genes_src(image_gpu_jax)
image_r = _with_genes_src(image_r)
image_annotsv = _with_genes_src(image_annotsv)
