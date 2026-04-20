"""Download all Phase 1 datasets to Modal volumes.

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
    print("Starting Phase 1 data ingestion...\n")

    # Spawn all downloads in parallel
    tasks = {
        "clinvar": ingest_clinvar.spawn(),
        "giab": ingest_giab.spawn(),
        "proteingym": ingest_proteingym.spawn(),
        "brca1_sge": ingest_brca1_sge.spawn(),
        "dgrp": ingest_dgrp.spawn(),
        "arabidopsis": ingest_arabidopsis.spawn(),
        "references": ingest_reference_genomes.spawn(),
        "baseline_scores": ingest_baseline_scores.spawn(),
    }

    # Collect results, handling individual failures
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
