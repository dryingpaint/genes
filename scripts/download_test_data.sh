#!/usr/bin/env bash
# Download GIAB HG002 test data for pipeline validation.
# Usage: bash scripts/download_test_data.sh

set -euo pipefail

DATA_DIR="test_data"
mkdir -p "$DATA_DIR"

echo "=== Downloading HG002 truth VCF (GRCh38, ~50 MB) ==="
BASE_URL="https://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/release/AshkenazimTrio/HG002_NA24385_son/latest/GRCh38"

if [ ! -f "$DATA_DIR/HG002.vcf.gz" ]; then
    wget -q --show-progress -O "$DATA_DIR/HG002.vcf.gz" \
        "$BASE_URL/HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
    wget -q --show-progress -O "$DATA_DIR/HG002.vcf.gz.tbi" \
        "$BASE_URL/HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz.tbi"
else
    echo "Already exists, skipping"
fi

echo ""
echo "=== Downloading high-confidence regions BED ==="
if [ ! -f "$DATA_DIR/HG002_confident.bed" ]; then
    wget -q --show-progress -O "$DATA_DIR/HG002_confident.bed" \
        "$BASE_URL/HG002_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed"
else
    echo "Already exists, skipping"
fi

echo ""
echo "=== Creating chr22 subset for fast iteration ==="
if command -v bcftools &> /dev/null; then
    if [ ! -f "$DATA_DIR/HG002_chr22.vcf.gz" ]; then
        bcftools view -r chr22 "$DATA_DIR/HG002.vcf.gz" -Oz -o "$DATA_DIR/HG002_chr22.vcf.gz"
        bcftools index -t "$DATA_DIR/HG002_chr22.vcf.gz"
        echo "Created chr22 subset"
    else
        echo "Already exists, skipping"
    fi
else
    echo "bcftools not found — skipping chr22 subset (install with: conda install bcftools)"
fi

echo ""
echo "=== Summary ==="
ls -lh "$DATA_DIR"/*.vcf.gz "$DATA_DIR"/*.bed 2>/dev/null || true
echo ""
echo "Done. Run the pipeline with:"
echo "  python scripts/run_single_sample.py --vcf test_data/HG002.vcf.gz"
echo "  python scripts/run_single_sample.py --vcf test_data/HG002_chr22.vcf.gz  # fast"
