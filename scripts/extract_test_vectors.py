"""Extract real scores from AlphaMissense TSV for test fixtures."""

from genbench.app import app
from genbench.config import DATASETS_PATH
from genbench.infra.images import cpu_image
from genbench.infra.volumes import VOLUME_MOUNTS


@app.function(
    image=cpu_image,
    volumes=VOLUME_MOUNTS,
    timeout=600,
    memory=32768,
)
def extract_scores() -> list[dict]:
    import gzip
    import pandas as pd
    from pathlib import Path

    tsv_path = Path(DATASETS_PATH) / "baseline_scores" / "AlphaMissense_hg38.tsv.gz"

    print("Loading AlphaMissense TSV...")
    with gzip.open(tsv_path, "rt") as f:
        for line in f:
            if line.startswith("#CHROM") or line.startswith("#chr"):
                header = line.strip().lstrip("#").split("\t")
                break
            elif not line.startswith("#"):
                header = ["CHROM", "POS", "REF", "ALT", "genome", "uniprot_id",
                          "transcript_id", "protein_variant", "am_pathogenicity", "am_class"]
                break

    df = pd.read_csv(tsv_path, sep="\t", comment="#", header=None, names=header,
                     dtype={"CHROM": str, "POS": int})
    print(f"Loaded {len(df)} variants")

    # Strategy: sample known pathogenic and benign variants directly from the TSV
    # Pick well-known cancer/disease genes and grab examples of each class
    results = []

    # Get TP53 hotspots (known pathogenic)
    tp53 = df[df["uniprot_id"] == "P04637"]
    tp53_path = tp53[tp53["am_class"] == "likely_pathogenic"].head(5)
    tp53_benign = tp53[tp53["am_class"] == "likely_benign"].head(5)

    # Get BRCA1 variants
    brca1 = df[df["uniprot_id"] == "P38398"]
    brca1_path = brca1[brca1["am_class"] == "likely_pathogenic"].head(5)
    brca1_benign = brca1[brca1["am_class"] == "likely_benign"].head(5)

    # Get CFTR variants
    cftr = df[df["uniprot_id"] == "P13569"]
    cftr_path = cftr[cftr["am_class"] == "likely_pathogenic"].head(3)
    cftr_benign = cftr[cftr["am_class"] == "likely_benign"].head(3)

    # Get MLH1 variants (Lynch syndrome)
    mlh1 = df[df["uniprot_id"] == "P40692"]
    mlh1_path = mlh1[mlh1["am_class"] == "likely_pathogenic"].head(2)
    mlh1_benign = mlh1[mlh1["am_class"] == "likely_benign"].head(2)

    for label, subset in [
        (1, tp53_path), (0, tp53_benign),
        (1, brca1_path), (0, brca1_benign),
        (1, cftr_path), (0, cftr_benign),
        (1, mlh1_path), (0, mlh1_benign),
    ]:
        for _, row in subset.iterrows():
            results.append({
                "chrom": row["CHROM"],
                "pos": int(row["POS"]),
                "ref": row["REF"],
                "alt": row["ALT"],
                "am_pathogenicity": float(row["am_pathogenicity"]),
                "am_class": row["am_class"],
                "protein_variant": row["protein_variant"],
                "uniprot_id": row["uniprot_id"],
                "label": label,
            })

    print(f"Extracted {len(results)} test vectors")
    return results


@app.local_entrypoint()
def main():
    results = extract_scores.remote()

    print("\nchrom,pos,ref,alt,am_pathogenicity,am_class,protein_variant,uniprot_id,label")
    for r in results:
        print(f"{r['chrom']},{r['pos']},{r['ref']},{r['alt']},"
              f"{r['am_pathogenicity']},{r['am_class']},"
              f"{r['protein_variant']},{r['uniprot_id']},{r['label']}")
