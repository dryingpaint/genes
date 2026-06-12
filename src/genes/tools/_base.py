"""Shared types and helpers for tool wrappers.

Each tool returns a ToolResult. The orchestrator only ever reads ToolResult
fields — never tool internals — so this is the entire contract.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from genes.infra.provenance import stamp
from genes.orchestrator.spec import Criticality, Status, ToolSpec


class ToolResult(BaseModel):
    """Canonical output of every tool wrapper."""

    tool_name: str
    version: str
    status: Status
    criticality: Criticality
    started_at: datetime
    completed_at: datetime | None = None

    # Paths consumed by this run (artifact-name -> path), captured for audit.
    inputs: dict[str, str] = Field(default_factory=dict)
    # Paths produced by this run (artifact-name -> path). The scheduler
    # reads these and adds them to the artifact pool for downstream tools.
    output_paths: dict[str, str] = Field(default_factory=dict)
    # Tool-specific structured result. Report renderers read this.
    payload: dict[str, Any] = Field(default_factory=dict)
    # Reference-data fingerprints; populated automatically by build_result.
    provenance: dict[str, dict[str, str]] = Field(default_factory=dict)

    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def runtime_s(self) -> float | None:
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()


class VariantScore(BaseModel):
    """A single variant with scores from one or more predictors."""

    chrom: str
    pos: int
    ref: str
    alt: str
    gene: str | None = None
    consequence: str | None = None
    scores: dict[str, float | str | None] = Field(default_factory=dict)
    classifications: dict[str, str] = Field(default_factory=dict)
    explanations: dict[str, str] = Field(default_factory=dict)


class ToolTimer:
    """Context manager for timing tool execution."""

    def __init__(self) -> None:
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None

    def __enter__(self) -> ToolTimer:
        self.started_at = datetime.now(timezone.utc)
        return self

    def __exit__(self, *args: Any) -> None:
        self.completed_at = datetime.now(timezone.utc)


def build_result(
    spec: ToolSpec,
    timer: ToolTimer,
    *,
    inputs: dict[str, str] | None = None,
    output_paths: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
    status: Status | None = None,
) -> ToolResult:
    """Standard ToolResult builder.

    Status defaults to OK / PARTIAL / FAILED based on whether warnings or
    errors were recorded — pass `status=` explicitly to override (used by
    the scheduler for TIMEOUT and SKIPPED).
    """
    errors = errors or []
    warnings = warnings or []
    if status is None:
        if errors:
            status = Status.FAILED
        elif warnings:
            status = Status.PARTIAL
        else:
            status = Status.OK
    return ToolResult(
        tool_name=spec.name,
        version=spec.version,
        status=status,
        criticality=spec.criticality,
        started_at=timer.started_at or datetime.now(timezone.utc),
        completed_at=timer.completed_at or datetime.now(timezone.utc),
        inputs=inputs or {},
        output_paths=output_paths or {},
        payload=payload or {},
        provenance=stamp(*spec.reference_artifacts),
        errors=errors,
        warnings=warnings,
    )


def run_cmd(cmd: list[str], *, check: bool = True, timeout: int = 3600) -> subprocess.CompletedProcess:
    """Run a shell command with standard error handling."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=check,
        timeout=timeout,
    )


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist, return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
