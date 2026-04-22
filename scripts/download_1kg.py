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
    .apt_install("wget", "unzip", "tabix")
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

    # Download 1KG Phase 3 chr22 VCF (GRCh38)
    # Using the original Phase 3 GRCh38 liftover from IGSR
    chr22_vcf = f"{out}/chr22.vcf.gz"
    if not os.path.exists(chr22_vcf):
        print("Downloading 1KG chr22 VCF from NYGC high-coverage release...")
        # NYGC 1KG high-coverage GRCh38 (3,202 samples)
        url = (
            "http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
            "1000G_2504_high_coverage/working/20220422.3202_phased/"
            "1kGP_high_coverage_Illumina.chr22.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
        )
        # Try NYGC URL first, fall back to original Phase 3
        try:
            subprocess.run(["wget", "--progress=dot:mega", "-O", chr22_vcf, url], check=True, timeout=1800)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print("NYGC URL failed, trying original Phase 3...")
            url2 = (
                "http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
                "ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
            )
            subprocess.run(["wget", "--progress=dot:mega", "-O", chr22_vcf, url2], check=True, timeout=1800)
        # Index
        subprocess.run(["tabix", "-p", "vcf", chr22_vcf], check=False)

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
