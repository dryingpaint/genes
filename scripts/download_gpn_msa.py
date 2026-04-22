"""Download GPN-MSA scores as a standalone Modal function.

The 37 GB download takes too long for the multi-function entrypoint.
Run with: modal run scripts/download_gpn_msa.py
"""

import modal

app = modal.App("genes-gpn-download")

vol = modal.Volume.from_name("genes-precomputed", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget")
)


@app.function(
    image=image,
    volumes={"/data/precomputed": vol},
    timeout=28800,  # 8 hours
)
def download_gpn_msa():
    import os
    import subprocess

    out = "/data/precomputed/gpn_msa"
    scores = f"{out}/gpn_msa_scores_hg38.tsv.gz"

    if os.path.exists(scores) and os.path.getsize(scores) > 30_000_000_000:
        print(f"GPN-MSA scores already present ({os.path.getsize(scores)} bytes), skipping")
        return

    os.makedirs(out, exist_ok=True)
    base = "https://huggingface.co/datasets/songlab/gpn-msa-hg38-scores/resolve/main"

    print("Downloading GPN-MSA scores (~37 GB)...")
    subprocess.run(
        ["wget", "--progress=dot:giga", "-O", scores, f"{base}/scores.tsv.bgz"],
        check=True,
    )
    print("Downloading index...")
    subprocess.run(
        ["wget", "-q", "-O", f"{scores}.tbi", f"{base}/scores.tsv.bgz.tbi"],
        check=True,
    )

    vol.commit()
    size_gb = os.path.getsize(scores) / 1e9
    print(f"GPN-MSA scores downloaded: {size_gb:.1f} GB")


@app.local_entrypoint()
def main():
    download_gpn_msa.remote()
    print("Done.")
