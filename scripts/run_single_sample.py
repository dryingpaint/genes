"""Run the full germline pipeline on a single WGS sample.

Usage:
    # Full genome (all chromosomes)
    modal run scripts/run_single_sample.py --vcf test_data/HG002.vcf.gz

    # Fast iteration (chr22 only)
    modal run scripts/run_single_sample.py --vcf test_data/HG002_chr22.vcf.gz

    # With Mendelian prioritization (requires HPO terms)
    modal run scripts/run_single_sample.py --vcf test_data/HG002.vcf.gz \
        --hpo HP:0001250,HP:0001263

    # With PRS for specific traits
    modal run scripts/run_single_sample.py --vcf test_data/HG002.vcf.gz \
        --prs-traits height,CAD,T2D

This script:
1. Uploads the VCF to the Modal workdir volume
2. Runs the germline VCF pipeline (VEP → parallel scoring/PGx/PRS/ancestry)
3. Prints structured results
4. Optionally writes a JSON report to disk
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import modal

from genes.app import app
from genes.infra.images import image_python_bio
from genes.infra.volumes import MOUNT_WORKDIR, vol_workdir
from genes.orchestrator.models import AnalysisRequest, InputType

# Import all tool modules so their @app.function decorators register with the app
import genes.tools.vep
import genes.tools.alphamissense
import genes.tools.spliceai
import genes.tools.gpn_msa
import genes.tools.pharmcat
import genes.tools.prs
import genes.tools.ancestry
import genes.tools.deepvariant
import genes.tools.eve
import genes.tools.evee
import genes.tools.cyrius
import genes.tools.hla
import genes.tools.exomiser
import genes.tools.mutect2
import genes.tools.sigprofiler
import genes.tools.msisensor
import genes.tools.oncokb_civic
import genes.tools.annotsv
import genes.tools.classifycnv
import genes.tools.cfdna
import genes.tools.biolearn
import genes.tools.traits


@app.function(
    image=image_python_bio,
    volumes={MOUNT_WORKDIR: vol_workdir},
    timeout=300,
)
def upload_vcf(local_content: bytes, run_id: str, filename: str) -> str:
    """Upload VCF bytes to the workdir volume. Returns the remote path."""
    import os

    remote_dir = f"{MOUNT_WORKDIR}/{run_id}/input"
    os.makedirs(remote_dir, exist_ok=True)
    remote_path = f"{remote_dir}/{filename}"
    with open(remote_path, "wb") as f:
        f.write(local_content)

    # Also upload index if it was included
    vol_workdir.commit()
    return remote_path


@app.function(
    image=image_python_bio,
    volumes={MOUNT_WORKDIR: vol_workdir},
    timeout=14400,  # 4 hours max for full genome
)
def run_germline_analysis(
    vcf_remote_path: str,
    run_id: str,
    hpo_terms: list[str] | None = None,
    prs_traits: list[str] | None = None,
) -> dict:
    """Run the full germline VCF pipeline on Modal and return results as dict."""
    from genes.orchestrator.dispatcher import run_pipeline
    from genes.orchestrator.models import AnalysisRequest, InputType

    request = AnalysisRequest(
        run_id=run_id,
        input_type=InputType.GERMLINE_VCF,
        input_paths=[vcf_remote_path],
        hpo_terms=hpo_terms,
        prs_traits=prs_traits,
    )

    result = run_pipeline(request)
    result_dict = result.model_dump(mode="json")

    # Save results to volume for detached mode retrieval
    import os
    results_dir = f"{MOUNT_WORKDIR}/{run_id}"
    os.makedirs(results_dir, exist_ok=True)
    results_path = f"{results_dir}/results.json"
    with open(results_path, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)
    vol_workdir.commit()

    return result_dict


@app.function(
    image=image_python_bio,
    volumes={MOUNT_WORKDIR: vol_workdir},
    timeout=14400,
)
@modal.fastapi_endpoint(method="POST")
def run_pipeline_endpoint(item: dict):
    """Trigger pipeline via HTTP POST. Runs entirely on Modal — no local heartbeat needed.

    Usage:
        # Upload VCF first, then trigger:
        curl -X POST https://<app-url>/run_pipeline_endpoint \\
            -H "Content-Type: application/json" \\
            -d '{"vcf_path": "/work/<run_id>/input/file.vcf.gz", "run_id": "run_xxx"}'

        # Or trigger with a VCF already on the volume from a previous run:
        modal run scripts/run_single_sample.py --vcf test_data/HG002_full.vcf.gz --detach
    """
    from genes.orchestrator.dispatcher import run_pipeline
    from genes.orchestrator.models import AnalysisRequest, InputType

    vcf_path = item.get("vcf_path")
    run_id = item.get("run_id")
    hpo_terms = item.get("hpo_terms")
    prs_traits = item.get("prs_traits")

    if not vcf_path or not run_id:
        return {"error": "vcf_path and run_id required"}

    request = AnalysisRequest(
        run_id=run_id,
        input_type=InputType.GERMLINE_VCF,
        input_paths=[vcf_path],
        hpo_terms=hpo_terms,
        prs_traits=prs_traits,
    )

    result = run_pipeline(request)
    result_dict = result.model_dump(mode="json")

    # Save results to volume
    import json
    output_path = f"{MOUNT_WORKDIR}/{run_id}/results.json"
    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)
    vol_workdir.commit()

    return {
        "run_id": run_id,
        "status": "completed",
        "tools_succeeded": result_dict.get("summary", {}).get("tools_succeeded", []),
        "tools_failed": result_dict.get("summary", {}).get("tools_failed", []),
        "runtime_seconds": result_dict.get("summary", {}).get("total_runtime_seconds"),
        "results_path": output_path,
    }


@app.local_entrypoint()
def main(
    vcf: str,
    hpo: str = "",
    prs_traits: str = "",
    output: str = "",
    detach: bool = False,
):
    """Entry point: modal run scripts/run_single_sample.py --vcf path/to/file.vcf.gz

    With --detach, uploads VCF then triggers the pipeline as a spawned remote function.
    Your laptop can disconnect — the pipeline runs entirely on Modal.
    """
    vcf_path = Path(vcf)
    if not vcf_path.exists():
        print(f"Error: VCF not found: {vcf_path}")
        sys.exit(1)

    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    print(f"Run ID: {run_id}")
    print(f"Input:  {vcf_path} ({vcf_path.stat().st_size / 1e6:.1f} MB)")

    # Parse optional args
    hpo_terms = [h.strip() for h in hpo.split(",") if h.strip()] or None
    trait_list = [t.strip() for t in prs_traits.split(",") if t.strip()] or None

    if hpo_terms:
        print(f"HPO:    {hpo_terms}")
    if trait_list:
        print(f"PRS:    {trait_list}")
    print()

    # Step 1: Upload VCF to Modal volume
    print("Uploading VCF to Modal...")
    vcf_bytes = vcf_path.read_bytes()
    remote_path = upload_vcf.remote(vcf_bytes, run_id, vcf_path.name)

    # Also upload index if it exists
    tbi_path = Path(f"{vcf_path}.tbi")
    if tbi_path.exists():
        upload_vcf.remote(tbi_path.read_bytes(), run_id, tbi_path.name)

    print(f"Uploaded to: {remote_path}")
    print()

    if detach:
        # Fire-and-forget: spawn the pipeline on Modal and exit.
        # The pipeline runs entirely server-side — laptop can disconnect.
        handle = run_germline_analysis.spawn(
            remote_path, run_id, hpo_terms, trait_list
        )
        print("Pipeline spawned in detached mode!")
        print(f"  Run ID:    {run_id}")
        print(f"  Function:  run_germline_analysis")
        print(f"  Results:   Will be saved to /work/{run_id}/results.json on the volume")
        print()
        print("Your laptop can disconnect. To check results later:")
        print(f"  modal volume get genes-workdir {run_id}/results.json")
        print(f"  # or download: modal volume get genes-workdir {run_id}/ ./results/")
        return

    # Step 2: Run the pipeline (attached — requires stable connection)
    print("Running germline VCF pipeline...")
    print("  VEP annotation → [AlphaMissense, SpliceAI, GPN-MSA, PharmCAT, PRS, Ancestry] parallel")
    if hpo_terms:
        print("  + Exomiser (Mendelian prioritization)")
    print()

    result_dict = run_germline_analysis.remote(
        remote_path, run_id, hpo_terms, trait_list
    )

    # Step 3: Print results
    print("=" * 70)
    print(f"PIPELINE COMPLETE — Run {run_id}")
    print("=" * 70)
    print()

    tool_results = result_dict.get("tool_results", {})
    by_status: dict[str, list[str]] = {}
    for name, tr in tool_results.items():
        by_status.setdefault(tr.get("status", "unknown"), []).append(name)

    print(f"Pipeline status: {result_dict.get('status') or 'unknown'}")
    for status in ("ok", "partial", "failed", "timeout", "skipped"):
        if by_status.get(status):
            print(f"  {status:>8}: {', '.join(by_status[status])}")
    print()

    for tool_name, tr in tool_results.items():
        status = (tr.get("status") or "unknown").upper()
        runtime_s = ""
        if tr.get("started_at") and tr.get("completed_at"):
            from datetime import datetime as dt
            try:
                t0 = dt.fromisoformat(tr["started_at"])
                t1 = dt.fromisoformat(tr["completed_at"])
                runtime_s = f" ({(t1 - t0).total_seconds():.1f}s)"
            except (ValueError, TypeError):
                pass

        print(f"  [{status}] {tool_name} v{tr.get('version', '?')}{runtime_s}")

        for art, path in (tr.get("output_paths") or {}).items():
            print(f"         produced {art}: {path}")

        payload = tr.get("payload", {})
        for k, v in list(payload.items())[:5]:
            if isinstance(v, dict):
                print(f"         {k}:")
                for k2, v2 in list(v.items())[:3]:
                    print(f"           {k2}: {v2}")
            elif isinstance(v, list):
                print(f"         {k}: {len(v)} items")
            else:
                print(f"         {k}: {v}")

        for err in tr.get("errors", []) or []:
            print(f"         ERROR: {err}")
        for w in (tr.get("warnings", []) or [])[:3]:
            print(f"         WARN: {w}")
        print()

    if result_dict.get("pipeline_errors"):
        print("PIPELINE ERRORS:")
        for err in result_dict["pipeline_errors"]:
            print(f"  - {err}")
        print()

    # Step 4: Write JSON report
    output_path = output or f"test_data/{run_id}_results.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)
    print(f"Full results written to: {output_path}")
