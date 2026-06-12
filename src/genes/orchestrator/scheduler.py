"""Stage-based DAG scheduler.

The scheduler is the single source of execution truth. It:

  1. Imports every tool module so that ToolSpecs land in REGISTRY.
  2. Maps the AnalysisRequest's input_type to a (Mode, initial artifact set).
  3. Filters tools by mode and runtime gate.
  4. Loops:
       - find all tools whose consumes are present in the artifact pool;
       - spawn them in parallel as Modal FunctionCalls;
       - await each with its declared timeout;
       - merge produced artifacts into the pool;
     until either no tools remain or no tool can advance.
  5. Tools that never run because their inputs were never produced are
     recorded as SKIPPED, never silently dropped.

There is no per-pipeline routing. The DAG is whatever the specs describe.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from genes.orchestrator.models import AnalysisRequest, InputType, PipelineResult
from genes.orchestrator.spec import REGISTRY, Artifact, Mode, Status, ToolSpec
from genes.tools._base import ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# input_type -> (Mode, initial artifact materialization)
# ---------------------------------------------------------------------------


def _bootstrap(request: AnalysisRequest) -> tuple[Mode, dict[Artifact, str]]:
    paths = request.input_paths
    it = InputType(request.input_type)

    if it == InputType.GERMLINE_VCF:
        return Mode.GERMLINE, {Artifact.VCF: paths[0]}
    if it == InputType.GERMLINE_BAM:
        return Mode.GERMLINE, {Artifact.BAM: paths[0]}
    if it == InputType.GERMLINE_FASTQ:
        # No aligner registered yet — surface as a pipeline-level error.
        return Mode.GERMLINE, {}
    if it == InputType.TUMOR_NORMAL_BAM:
        return Mode.SOMATIC, {Artifact.TUMOR_BAM: paths[0], Artifact.NORMAL_BAM: paths[1]}
    if it == InputType.TUMOR_ONLY_BAM:
        return Mode.SOMATIC, {Artifact.TUMOR_BAM: paths[0]}
    if it == InputType.SOMATIC_VCF:
        return Mode.SOMATIC, {Artifact.VCF: paths[0]}
    if it == InputType.CFDNA_BAM:
        initial: dict[Artifact, str] = {Artifact.CFDNA_BAM: paths[0]}
        if len(paths) > 1 and "bisulfite" in paths[1].lower():
            initial[Artifact.CFDNA_BISULFITE_BAM] = paths[1]
        return Mode.CFDNA, initial
    if it == InputType.METHYLATION_ARRAY:
        return Mode.METHYLATION, {Artifact.METHYLATION_MATRIX: paths[0]}

    raise ValueError(f"Unsupported input type: {it}")


# ---------------------------------------------------------------------------
# Reachability — which tools can possibly run
# ---------------------------------------------------------------------------


def _reachable(
    tools: dict[str, tuple[ToolSpec, Any]],
    initial_artifacts: set[Artifact],
) -> dict[str, tuple[ToolSpec, Any]]:
    """Return only the tools whose required inputs are reachable.

    An artifact is reachable iff it's in the initial set or it's produced
    by some reachable tool. A tool's required inputs must all be reachable
    for the tool itself to be applicable. Iterate to a fixed point.
    """
    artifacts = set(initial_artifacts)
    applicable: dict[str, tuple[ToolSpec, Any]] = {}
    changed = True
    while changed:
        changed = False
        for name, (spec, fn) in tools.items():
            if name in applicable:
                continue
            if all(c in artifacts for c in spec.consumes):
                applicable[name] = (spec, fn)
                for produced in spec.produces:
                    if produced not in artifacts:
                        artifacts.add(produced)
                        changed = True
    return applicable


# ---------------------------------------------------------------------------
# Tool invocation
# ---------------------------------------------------------------------------


def _spawn(spec: ToolSpec, fn: Any, artifacts: dict[Artifact, str], request: AnalysisRequest) -> Any:
    """Build the kwargs the spec asks for and spawn the Modal function."""
    primary = artifacts[spec.consumes[0]]  # every spec has exactly one primary input
    kwargs: dict[str, Any] = {}
    for opt in spec.optional_consumes:
        if opt in artifacts:
            kwargs[opt.value] = artifacts[opt]
    for key in spec.request_kwargs:
        val = getattr(request, key, None)
        if val is not None:
            kwargs[key] = val
    return fn.spawn(primary, request.run_id, **kwargs)


def _await(handle: Any, spec: ToolSpec) -> ToolResult:
    """Await a single spawned call, mapping timeouts/exceptions to ToolResult."""
    try:
        return handle.get(timeout=spec.timeout_s)
    except TimeoutError:
        logger.warning("Tool %s timed out after %ds", spec.name, spec.timeout_s)
        return _synthetic(spec, Status.TIMEOUT, f"Timed out after {spec.timeout_s}s")
    except Exception as exc:
        logger.exception("Tool %s failed during execution", spec.name)
        return _synthetic(spec, Status.FAILED, str(exc))


def _synthetic(spec: ToolSpec, status: Status, reason: str) -> ToolResult:
    """Build a ToolResult for tools that never produced one."""
    now = datetime.now(timezone.utc)
    return ToolResult(
        tool_name=spec.name,
        version=spec.version,
        status=status,
        criticality=spec.criticality,
        started_at=now,
        completed_at=now,
        errors=[reason] if status != Status.SKIPPED else [],
        warnings=[reason] if status == Status.SKIPPED else [],
    )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run_pipeline(request: AnalysisRequest) -> PipelineResult:
    """Discover applicable tools, run the DAG, return aggregated result."""
    # Importing the tools package populates REGISTRY via each tool's
    # register() call.
    import genes.tools  # noqa: F401

    mode, artifacts = _bootstrap(request)
    result = PipelineResult(run_id=request.run_id, mode=mode, input_type=request.input_type)

    if not artifacts:
        result.pipeline_errors.append(
            f"No initial artifacts derivable from input_type={request.input_type}"
        )
        result.finalize()
        return result

    mode_tools = {
        name: (spec, fn)
        for name, (spec, fn) in REGISTRY.items()
        if mode in spec.modes and spec.gate(request)
    }
    # Reachability pass: prune tools whose required inputs can never be
    # produced. This is how we distinguish "user passed a pre-called VCF so
    # DeepVariant is N/A" from "DeepVariant was supposed to run but couldn't."
    applicable = _reachable(mode_tools, set(artifacts.keys()))

    if not applicable:
        result.pipeline_errors.append(f"No tools registered for mode={mode}")
        result.finalize()
        return result

    remaining = dict(applicable)
    logger.info(
        "Starting %s pipeline for run_id=%s, %d tools applicable",
        mode.value, request.run_id, len(remaining),
    )

    while remaining:
        ready = [
            (name, spec, fn)
            for name, (spec, fn) in remaining.items()
            if all(c in artifacts for c in spec.consumes)
        ]
        if not ready:
            # Everything left has unmet deps — upstream skipped/failed.
            for name, (spec, _) in remaining.items():
                missing = [c.value for c in spec.consumes if c not in artifacts]
                result.tool_results[name] = _synthetic(
                    spec, Status.SKIPPED,
                    f"required artifacts not produced: {missing}",
                )
            break

        # Spawn this stage in parallel.
        handles: dict[str, tuple[ToolSpec, Any]] = {}
        for name, spec, fn in ready:
            try:
                handles[name] = (spec, _spawn(spec, fn, artifacts, request))
            except Exception as exc:
                logger.exception("Failed to spawn %s", name)
                result.tool_results[name] = _synthetic(spec, Status.FAILED, f"spawn failed: {exc}")
            del remaining[name]

        # Await this stage's results.
        for name, (spec, handle) in handles.items():
            tr = _await(handle, spec)
            result.tool_results[name] = tr
            if tr.status in (Status.OK, Status.PARTIAL):
                for produced in spec.produces:
                    if produced.value in tr.output_paths:
                        artifacts[produced] = tr.output_paths[produced.value]

    result.finalize()
    logger.info(
        "Pipeline complete for run_id=%s: status=%s, %d tools run",
        request.run_id, result.status.value, len(result.tool_results),
    )
    return result
