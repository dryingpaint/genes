"""Download all datasets to Modal volumes.

Usage: modal run scripts/ingest_all.py
"""

from genbench.app import app
from genbench.infra.data_ingest import (
    ingest_arabidopsis,
    ingest_baseline_scores,
    ingest_brca1_sge,
    ingest_clinvar,
    ingest_dgrp,
    ingest_giab,
    ingest_proteingym,
    ingest_reference_genomes,
)


@app.local_entrypoint()
def main():
    print("Starting data ingestion...\n")

    tasks = {
        # ClinVar: latest + archived version for AlphaMissense reproduction
        "clinvar_latest": ingest_clinvar.spawn(version="latest"),
        "clinvar_20230115": ingest_clinvar.spawn(version="20230115"),
        # ProteinGym: v1 (217 assays) + v0.1 (87 assays, SaProt's published eval)
        "proteingym_v1": ingest_proteingym.spawn(version="v1"),
        "proteingym_v0.1": ingest_proteingym.spawn(version="v0.1"),
        # Other datasets
        "giab": ingest_giab.spawn(),
        "brca1_sge": ingest_brca1_sge.spawn(),
        "dgrp": ingest_dgrp.spawn(),
        "arabidopsis": ingest_arabidopsis.spawn(),
        "references": ingest_reference_genomes.spawn(),
        "baseline_scores": ingest_baseline_scores.spawn(),
    }

    succeeded = 0
    failed = 0
    for name, future in tasks.items():
        try:
            result = future.get()
            print(f"  OK  {result}")
            succeeded += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1

    print(f"\nDone: {succeeded} succeeded, {failed} failed.")
