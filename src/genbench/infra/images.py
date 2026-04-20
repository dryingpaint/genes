"""Modal image definitions.

Three images with increasing weight:
- cpu_image: general compute, metrics, splits, PRS, GBLUP
- gpu_image: foundation model baselines (SaProt, Evo 2, Borzoi)
- bio_image: bioinformatics tools with large reference data (PharmCAT, HLA*LA, VEP)

Note: add_local_python_source must be LAST in the image chain.
Use _cpu_base / _gpu_base to extend images before the local source is added.
"""

import modal

# --- Base images (without local source — safe to extend) ---

_cpu_base = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "unzip", "tabix")
    .pip_install(
        "numpy>=1.26",
        "scipy>=1.12",
        "pandas>=2.2",
        "polars>=1.0",
        "scikit-learn>=1.4",
        "pyarrow>=15.0",
        "pydantic>=2.0",
        "pysam>=0.22",
    )
)

_gpu_base = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "git")
    .pip_install(
        "torch>=2.3",
        "transformers>=4.40",
        "safetensors",
        "numpy>=1.26",
        "scipy>=1.12",
        "pandas>=2.2",
        "polars>=1.0",
        "scikit-learn>=1.4",
        "pyarrow>=15.0",
        "pydantic>=2.0",
        "biopython>=1.83",
    )
)

# --- Final images (with local source — do NOT extend these with build steps) ---

cpu_image = _cpu_base.add_local_python_source("genbench")

gpu_image = _gpu_base.add_local_python_source("genbench")

bio_image = (
    _cpu_base
    .apt_install("default-jre", "perl", "cpanminus", "samtools", "bcftools")
    .run_commands(
        "curl -fsSL https://github.com/PharmGKB/PharmCAT/releases/download/v2.13.0/pharmcat-2.13.0-all.jar"
        " -o /opt/pharmcat.jar",
    )
    .pip_install("cyvcf2>=0.31")
    .add_local_python_source("genbench")
)

# --- Extended images for specific tasks ---

cpu_hf_image = (
    _cpu_base
    .pip_install("huggingface_hub")
    .add_local_python_source("genbench")
)
