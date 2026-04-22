"""Deployed Modal app — persistent endpoints that don't require a local client.

Deploy with: modal deploy src/genes/deploy.py
Trigger via: curl -X POST https://<app-url>/run -d '{"vcf_path": "...", "run_id": "..."}'

This solves the heartbeat timeout issue for long-running jobs (e.g., VEP on 4M variants).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import modal

from genes.app import app
from genes.infra.images import image_python_bio
from genes.infra.volumes import MOUNT_WORKDIR, vol_workdir

# Import all tool modules to register their @app.function decorators
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


@app.function(
    image=image_python_bio,
    volumes={MOUNT_WORKDIR: vol_workdir},
    timeout=14400,  # 4 hours — enough for VEP on full WGS
)
def run_analysis(
    vcf_path: str,
    run_id: str | None = None,
    hpo_terms: list[str] | None = None,
    prs_traits: list[str] | None = None,
) -> dict:
    """Run the full germline pipeline. Called via web_endpoint or .remote()."""
    from genes.orchestrator.dispatcher import run_pipeline
    from genes.orchestrator.models import AnalysisRequest, InputType

    if run_id is None:
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"

    request = AnalysisRequest(
        run_id=run_id,
        input_type=InputType.GERMLINE_VCF,
        input_paths=[vcf_path],
        hpo_terms=hpo_terms,
        prs_traits=prs_traits,
    )

    result = run_pipeline(request)
    return result.model_dump(mode="json")


@app.function(
    image=image_python_bio,
    volumes={MOUNT_WORKDIR: vol_workdir},
    timeout=300,
)
@modal.web_endpoint(method="POST")
def trigger_run(item: dict) -> dict:
    """HTTP endpoint to trigger a pipeline run.

    POST body: {"vcf_path": "/work/...", "run_id": "optional", "hpo_terms": [], "prs_traits": []}
    Returns: {"status": "started", "run_id": "..."}

    The actual analysis runs asynchronously via .spawn().
    """
    import modal

    vcf_path = item.get("vcf_path")
    if not vcf_path:
        return {"error": "vcf_path is required"}

    run_id = item.get("run_id") or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    hpo_terms = item.get("hpo_terms")
    prs_traits = item.get("prs_traits")

    # Spawn the analysis — returns immediately, runs in background
    call = run_analysis.spawn(vcf_path, run_id, hpo_terms, prs_traits)

    return {
        "status": "started",
        "run_id": run_id,
        "call_id": call.object_id,
    }


@app.function(
    image=image_python_bio,
    volumes={MOUNT_WORKDIR: vol_workdir},
    timeout=300,
)
@modal.web_endpoint(method="GET")
def get_result(run_id: str) -> dict:
    """Check if a run completed and return results.

    GET /result?run_id=run_20260421_...
    """
    import os

    result_path = f"{MOUNT_WORKDIR}/{run_id}/result.json"
    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)
    return {"status": "running_or_not_found", "run_id": run_id}
