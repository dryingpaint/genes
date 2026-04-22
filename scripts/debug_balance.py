"""Debug: check class balance after missense filtering + balancing."""

from genbench.app import app
from genbench.infra.images import cpu_image
from genbench.infra.volumes import VOLUME_MOUNTS


@app.function(image=cpu_image, volumes=VOLUME_MOUNTS, timeout=600, memory=32768)
def debug() -> str:
    from genbench.registry import get_eval

    ev = get_eval("clinvar")
    config = ev.get_config("alphamissense_balanced")
    df = ev.load_data(config)

    lines = []
    lines.append(f"Total variants after filtering + balancing: {len(df)}")
    lines.append(f"Label distribution: {df['label'].value_counts().to_dict()}")
    lines.append(f"Unique genes: {df['GeneSymbol'].nunique()}")

    # Check per-gene balance
    per_gene = df.groupby("GeneSymbol")["label"].value_counts().unstack(fill_value=0)
    lines.append(f"Genes with both classes: {(per_gene.min(axis=1) > 0).sum()}")
    lines.append(f"Sample per-gene counts:\n{per_gene.head(10)}")

    # Check score distribution
    lines.append(f"\nScore stats (from label):")
    lines.append(f"  Pathogenic (label=1): {(df['label']==1).sum()}")
    lines.append(f"  Benign (label=0): {(df['label']==0).sum()}")

    return "\n".join(lines)


@app.local_entrypoint()
def main():
    print(debug.remote())
