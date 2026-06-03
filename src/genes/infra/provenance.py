"""Provenance stamps for tool outputs.

Every ToolResult.input_summary should carry a provenance fingerprint of the
reference artifacts the tool consumed, so a result can be audited and
reproduced later. Reference artifacts here are the on-volume files populated
by reference_data.py — reference genome, VEP cache, AlphaMissense/SpliceAI/
GPN-MSA scores, ClinVar, GWAS catalog, model weights, etc.

For each artifact we record:
  - path: the canonical mount path
  - mtime_utc: the file's mtime (catches in-place updates)
  - size_bytes: the file size
  - sha16k: sha256 of the first 16 KB (catches the rare same-mtime-new-content
    case from volume snapshot restores; avoids hashing multi-GB files)

For directories (e.g. VEP cache, HLA reference) we summarize the latest child
mtime and file count instead. Missing artifacts are recorded as
{"status": "missing"} so a downstream audit can tell "tool ran without this
reference present" apart from "tool didn't claim to use this reference."
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from genes.infra.volumes import (
    MOUNT_CFDNA,
    MOUNT_CLINICAL,
    MOUNT_EXOMISER,
    MOUNT_HLA,
    MOUNT_MODELS,
    MOUNT_POPGEN,
    MOUNT_PRECOMPUTED,
    MOUNT_REFERENCE,
    MOUNT_VEP,
)

# Standard key under which tools stash their provenance dict in input_summary.
PROVENANCE_KEY = "provenance"


# Reference artifacts known to the pipeline. Keep this aligned with
# reference_data.py — every download function there should have an entry.
ARTIFACTS: dict[str, str] = {
    "reference_genome": f"{MOUNT_REFERENCE}/GRCh38.fa",
    "vep_cache": f"{MOUNT_VEP}/homo_sapiens_merged",
    "alphamissense_scores": f"{MOUNT_PRECOMPUTED}/alphamissense/AlphaMissense_hg38.tsv.gz",
    "alphamissense_plugin": f"{MOUNT_VEP}/plugins/AlphaMissense_hg38.tsv.gz",
    "spliceai_scores": f"{MOUNT_PRECOMPUTED}/spliceai/spliceai_scores.raw.snv.hg38.vcf.gz",
    "gpn_msa_scores": f"{MOUNT_PRECOMPUTED}/gpn_msa/gpn_msa_scores_hg38.tsv.gz",
    "eve_scores": f"{MOUNT_PRECOMPUTED}/eve",
    "evee_scores": f"{MOUNT_PRECOMPUTED}/evee/clinvar_evee_scores.tsv.gz",
    "clinvar": f"{MOUNT_CLINICAL}/clinvar/clinvar.vcf.gz",
    "gwas_catalog": f"{MOUNT_CLINICAL}/gwas_catalog/gwas_pos_index.json",
    "kg_panel": f"{MOUNT_POPGEN}/1kg",
    "pgs_catalog": f"{MOUNT_POPGEN}/pgs_catalog",
    "exomiser_data": MOUNT_EXOMISER,
    "hla_reference": MOUNT_HLA,
    "cfdna_refs": MOUNT_CFDNA,
    "deepvariant_model": f"{MOUNT_MODELS}/deepvariant",
}


def _sha16k(path: Path) -> str:
    """First 16 KB of the file, sha256, truncated to 16 hex chars."""
    try:
        with path.open("rb") as fh:
            return hashlib.sha256(fh.read(16 * 1024)).hexdigest()[:16]
    except OSError:
        return ""


def _fingerprint(path_str: str) -> dict[str, str] | None:
    p = Path(path_str)
    if not p.exists():
        return None
    if p.is_file():
        st = p.stat()
        return {
            "path": path_str,
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            "size_bytes": str(st.st_size),
            "sha16k": _sha16k(p),
        }
    try:
        children = [c for c in p.iterdir() if c.is_file()]
        if children:
            latest = max(c.stat().st_mtime for c in children)
        else:
            latest = p.stat().st_mtime
        n_files = sum(1 for c in p.rglob("*") if c.is_file())
    except OSError:
        return None
    return {
        "path": path_str,
        "latest_child_mtime_utc": datetime.fromtimestamp(latest, tz=timezone.utc).isoformat(),
        "n_files": str(n_files),
    }


def stamp(*artifact_keys: str) -> dict[str, dict[str, str]]:
    """Fingerprint each named artifact. Always returns a stamp for every key.

    A key unknown to ARTIFACTS records {"status": "unknown_artifact"} — that
    way typos in tool wrappers are visible in the result instead of silently
    dropping provenance. Known-but-missing files record
    {"status": "missing", "path": ...}.
    """
    out: dict[str, dict[str, str]] = {}
    for key in artifact_keys:
        if key not in ARTIFACTS:
            out[key] = {"status": "unknown_artifact"}
            continue
        fp = _fingerprint(ARTIFACTS[key])
        out[key] = fp if fp is not None else {"status": "missing", "path": ARTIFACTS[key]}
    return out
