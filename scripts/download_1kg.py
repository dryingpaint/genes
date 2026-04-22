"""Download 1000 Genomes Phase 3 data for ancestry inference.

Downloads chr22 plink files from the 1KG FTP (GRCh38 liftover available
from PLINK2 resources) and prepares them for the ancestry tool.

Run with: modal run scripts/download_1kg.py
"""

import modal

app = modal.App("genes-1kg-download")

vol_popgen = modal.Volume.from_name("genes-popgen", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "unzip", "plink2" if False else "wget")  # plink2 from apt not available
    .run_commands(
        "wget -q https://s3.amazonaws.com/plink2-assets/alpha5/"
        "plink2_linux_x86_64_20240818.zip -O /tmp/plink2.zip"
        " && unzip -q /tmp/plink2.zip -d /usr/local/bin/ && rm /tmp/plink2.zip",
    )
    .pip_install("pysam>=0.22")
)


@app.function(
    image=image,
    volumes={"/data/popgen": vol_popgen},
    timeout=14400,  # 4 hours
    cpu=4,
    memory=32768,
)
def download_1kg():
    """Download 1KG Phase 3 GRCh38 data and convert to plink format."""
    import os
    import subprocess

    out = "/data/popgen/1kg"
    bed = f"{out}/all_phase3_GRCh38.bed"

    if os.path.exists(bed):
        size = os.path.getsize(bed)
        print(f"1KG data already present ({size} bytes), skipping")
        return

    os.makedirs(out, exist_ok=True)

    # Download 1KG Phase 3 for chr22 as a starting point (full genome is ~30GB)
    # Using the IGSR GRCh38 VCFs
    base_url = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422.3202_phased"

    # Download chr22 VCF (~200MB) — enough for PCA-based ancestry inference
    chr22_vcf = f"{out}/1kGP_high_coverage_Illumina.chr22.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
    if not os.path.exists(chr22_vcf):
        print("Downloading 1KG chr22 VCF (~200 MB)...")
        url = f"{base_url}/1kGP_high_coverage_Illumina.chr22.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
        subprocess.run(["wget", "--progress=dot:mega", "-O", chr22_vcf, url], check=True)
        subprocess.run(["wget", "-q", "-O", f"{chr22_vcf}.tbi", f"{url}.tbi"], check=True)

    # Download population panel file
    panel = f"{out}/integrated_call_samples_v3.20130502.ALL.panel"
    if not os.path.exists(panel):
        print("Downloading population panel...")
        subprocess.run([
            "wget", "-q", "-O", panel,
            "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel",
        ], check=True)

    # Convert to plink2 binary format with common SNPs only (MAF > 5%)
    print("Converting to plink format...")
    subprocess.run([
        "plink2",
        "--vcf", chr22_vcf,
        "--maf", "0.05",
        "--snps-only",
        "--max-alleles", "2",
        "--make-bed",
        "--out", f"{out}/all_phase3_GRCh38",
        "--set-all-var-ids", "@:#:\\$r:\\$a",
        "--new-id-max-allele-len", "20",
        "--allow-extra-chr",
    ], check=True)

    vol_popgen.commit()

    # Report stats
    bim = f"{out}/all_phase3_GRCh38.bim"
    fam = f"{out}/all_phase3_GRCh38.fam"
    if os.path.exists(bim) and os.path.exists(fam):
        n_snps = sum(1 for _ in open(bim))
        n_samples = sum(1 for _ in open(fam))
        print(f"1KG reference panel: {n_samples} samples, {n_snps} SNPs")
    print("Done.")


@app.local_entrypoint()
def main():
    download_1kg.remote()
