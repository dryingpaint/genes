"""Debug: check why AlphaMissense lookup returns NaN on ClinVar variants."""

from genbench.app import app
from genbench.config import DATASETS_PATH
from genbench.infra.images import cpu_image
from genbench.infra.volumes import VOLUME_MOUNTS


@app.function(image=cpu_image, volumes=VOLUME_MOUNTS, timeout=600, memory=32768)
def debug() -> str:
    import gzip
    import pandas as pd
    from pathlib import Path

    # Load a few ClinVar variants
    clinvar_path = Path(DATASETS_PATH) / "clinvar" / "variant_summary.txt.gz"
    cv = pd.read_csv(clinvar_path, sep="\t", dtype={"Chromosome": str}, low_memory=False, nrows=1000)
    cv = cv[cv["Assembly"] == "GRCh38"]

    lines = []
    lines.append(f"ClinVar sample (GRCh38):")
    lines.append(f"  Chromosome values: {cv['Chromosome'].unique()[:5]}")
    lines.append(f"  PositionVCF sample: {cv['PositionVCF'].head(3).tolist()}")
    lines.append(f"  Type values: {cv['Type'].unique()}")

    # Check the AM TSV format
    am_path = Path(DATASETS_PATH) / "baseline_scores" / "AlphaMissense_hg38.tsv.gz"
    with gzip.open(am_path, "rt") as f:
        am_lines = []
        for line in f:
            if line.startswith("#"):
                continue
            am_lines.append(line.strip())
            if len(am_lines) >= 5:
                break

    lines.append(f"\nAlphaMissense TSV sample rows:")
    for l in am_lines:
        lines.append(f"  {l[:120]}")

    # Key question: does ClinVar use "7" or "chr7" for chromosomes?
    # And does AlphaMissense use "chr7" or "7"?
    lines.append(f"\nClinVar Chromosome format: '{cv['Chromosome'].iloc[0]}'")
    lines.append(f"AlphaMissense CHROM format: '{am_lines[0].split(chr(9))[0]}'")

    # Try a specific lookup
    # Find a ClinVar missense variant and check if it's in AM
    missense = cv[cv["Type"] == "single nucleotide variant"].head(5)
    lines.append(f"\nClinVar missense variants to look up:")
    for _, row in missense.iterrows():
        chrom = row["Chromosome"]
        pos = row["PositionVCF"]
        ref = row["ReferenceAlleleVCF"]
        alt = row["AlternateAlleleVCF"]
        lines.append(f"  {chrom}:{pos} {ref}>{alt}")

    return "\n".join(lines)


@app.local_entrypoint()
def main():
    print(debug.remote())
