"""Extract real scores from AlphaMissense TSV for test fixtures.

Queries specific known variants and writes their actual scores to a fixture CSV.
Also extracts a small slice of the TSV for local testing.
"""

from genbench.app import app
from genbench.config import DATASETS_PATH
from genbench.infra.images import cpu_image
from genbench.infra.volumes import VOLUME_MOUNTS


# Variants to query — known pathogenic and benign from ClinVar
TEST_VARIANTS = [
    # BRCA1 known pathogenic missense
    ("chr17", 43094464, "G", "A"),  # p.C61G
    ("chr17", 43094464, "G", "T"),  # p.C61F
    ("chr17", 43082434, "C", "T"),  # p.R1699W
    ("chr17", 43091032, "T", "C"),  # p.M1775R region
    # BRCA1 known benign
    ("chr17", 43045714, "T", "C"),  # benign region
    # TP53 known pathogenic
    ("chr17", 7674220, "C", "T"),   # p.R248W (hotspot)
    ("chr17", 7674221, "C", "T"),   # p.R248Q (hotspot)
    ("chr17", 7675088, "C", "T"),   # p.R175H (hotspot)
    # CFTR
    ("chr7", 117559592, "G", "A"),  # near G542X
    # MLH1 (Lynch syndrome)
    ("chr3", 37050340, "C", "T"),   # missense region
]


@app.function(
    image=cpu_image,
    volumes=VOLUME_MOUNTS,
    timeout=600,
    memory=32768,
)
def extract_scores() -> list[dict]:
    import pandas as pd
    from pathlib import Path

    tsv_path = Path(DATASETS_PATH) / "baseline_scores" / "AlphaMissense_hg38.tsv.gz"
    if not tsv_path.exists():
        return [{"error": f"TSV not found at {tsv_path}"}]

    print("Loading AlphaMissense TSV (3.6GB)...")
    # The TSV has comment lines starting with ## and a header line starting with #CHROM
    # pandas comment="#" would skip the header too, so we read with no comment skipping
    # and handle the ## lines via skiprows or by reading the header separately
    import gzip
    with gzip.open(tsv_path, "rt") as f:
        # Skip ## comment lines, find the #CHROM header
        for line in f:
            if line.startswith("#CHROM") or line.startswith("#chr"):
                header = line.strip().lstrip("#").split("\t")
                break
            elif not line.startswith("#"):
                # No header found, first line is data
                header = ["CHROM", "POS", "REF", "ALT", "genome", "uniprot_id",
                          "transcript_id", "protein_variant", "am_pathogenicity", "am_class"]
                break

    print(f"Header: {header}")
    df = pd.read_csv(tsv_path, sep="\t", comment="#", header=None, names=header,
                     dtype={header[0]: str, header[1]: int})
    print(f"Loaded {len(df)} variants")
    print(f"Columns: {list(df.columns)}")
    print(f"Sample row: {dict(df.iloc[0])}")

    chrom_col = header[0]  # "CHROM" or whatever the first column is
    pos_col = header[1]
    ref_col = header[2]
    alt_col = header[3]
    score_col = "am_pathogenicity" if "am_pathogenicity" in header else header[8]
    class_col = "am_class" if "am_class" in header else header[9]

    # Build index for fast lookup
    print("Building lookup index...")
    df["_key"] = df[chrom_col] + ":" + df[pos_col].astype(str) + ":" + df[ref_col] + ":" + df[alt_col]
    indexed = df.set_index("_key")
    print(f"Index built, {len(indexed)} entries")

    results = []
    for chrom, pos, ref, alt in TEST_VARIANTS:
        key = f"{chrom}:{pos}:{ref}:{alt}"
        if key in indexed.index:
            row = indexed.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            results.append({
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "am_pathogenicity": float(row[score_col]),
                "am_class": str(row[class_col]),
                "protein_variant": str(row.get("protein_variant", "")),
                "uniprot_id": str(row.get("uniprot_id", "")),
                "found": True,
            })
        else:
            results.append({
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "found": False,
            })
            print(f"  NOT FOUND: {key}")

    return results


@app.local_entrypoint()
def main():
    results = extract_scores.remote()

    if results and "error" in results[0]:
        print(results[0]["error"])
        return

    print("\n# AlphaMissense Test Vectors (extracted from actual TSV)")
    print("chrom,pos,ref,alt,am_pathogenicity,am_class,protein_variant,uniprot_id")
    for r in results:
        if r["found"]:
            print(f"{r['chrom']},{r['pos']},{r['ref']},{r['alt']},"
                  f"{r['am_pathogenicity']},{r['am_class']},"
                  f"{r['protein_variant']},{r['uniprot_id']}")
        else:
            print(f"# NOT FOUND: {r['chrom']}:{r['pos']} {r['ref']}>{r['alt']}")
