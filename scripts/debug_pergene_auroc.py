"""Check: does per-gene averaged AUROC match the paper's 0.94?"""

from genbench.app import app
from genbench.infra.images import cpu_image
from genbench.infra.volumes import VOLUME_MOUNTS


@app.function(image=cpu_image, volumes=VOLUME_MOUNTS, timeout=600, memory=32768)
def debug() -> str:
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from genbench.registry import get_eval, get_model

    ev = get_eval("clinvar")
    config = ev.get_config("alphamissense_balanced")
    df = ev.load_data(config)
    splits = ev.get_splits(df, config)
    test = splits["test"]

    inputs = ev.make_inputs(df, test)
    model = get_model("alphamissense")
    scores = model.predict(inputs)
    labels = ev.get_labels(df, test)

    # Global AUROC
    valid = ~np.isnan(scores)
    global_auc = roc_auc_score(labels[valid], scores[valid])

    # Per-gene AUROC (the paper's approach)
    gene_aucs = []
    for gene in test["GeneSymbol"].unique():
        mask = (test["GeneSymbol"] == gene).values & valid
        if mask.sum() < 10:
            continue
        yt = labels[mask]
        yp = scores[mask]
        if len(np.unique(yt)) < 2:
            continue
        gene_aucs.append(roc_auc_score(yt, yp))

    lines = []
    lines.append(f"Total test variants: {len(test)}")
    lines.append(f"Variants with AM score: {valid.sum()}")
    lines.append(f"Global AUROC: {global_auc:.4f}")
    lines.append(f"Per-gene AUROC (mean of {len(gene_aucs)} genes): {np.mean(gene_aucs):.4f}")
    lines.append(f"Per-gene AUROC (median): {np.median(gene_aucs):.4f}")
    lines.append(f"Per-gene AUROC distribution: min={np.min(gene_aucs):.3f} p25={np.percentile(gene_aucs,25):.3f} p75={np.percentile(gene_aucs,75):.3f} max={np.max(gene_aucs):.3f}")
    return "\n".join(lines)


@app.local_entrypoint()
def main():
    print(debug.remote())
