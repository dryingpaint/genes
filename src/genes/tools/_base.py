"""Shared types and helpers for all tool wrappers.

Every tool returns a ToolResult. This is the universal contract between
tool wrappers and the orchestrator/Claude integration layer.
"""

from __future__ import annotations

import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """The canonical output of every tool wrapper."""

    tool_name: str
    version: str
    started_at: datetime
    completed_at: datetime
    input_summary: dict[str, Any] = Field(default_factory=dict)
    output_paths: list[str] = Field(default_factory=list)
    output_summary: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    @property
    def runtime_seconds(self) -> float:
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
