"""MSIsensor-pro — Microsatellite instability detection.

Runs msisensor-pro to detect microsatellite instability (MSI) from
tumor BAM with optional matched normal.
"""

from __future__ import annotations

from pathlib import Path

from genes.app import app
from genes.infra.images import image_cpp_tools
from genes.infra.volumes import (
    MOUNT_REFERENCE,
    MOUNT_WORKDIR,
    vol_reference,
    vol_workdir,
)
from genes.infra.provenance import PROVENANCE_KEY, stamp
from genes.tools._base import ToolResult, ToolTimer, ensure_dir, run_cmd

_REFERENCE_FASTA = f"{MOUNT_REFERENCE}/GRCh38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna"
_MS_SITES_LIST = f"{MOUNT_REFERENCE}/GRCh38/microsatellites.list"


@app.function(
    image=image_cpp_tools,
    volumes={
        MOUNT_REFERENCE: vol_reference,
        MOUNT_WORKDIR: vol_workdir,
    },
    timeout=3600,
    cpu=8,
)
def detect_msi(
    tumor_bam: str,
    run_id: str,
    normal_bam: str | None = None,
    min_coverage: int = 20,
    threads: int = 8,
) -> ToolResult:
    """Detect microsatellite instability from tumor +/- normal BAM.

    Args:
        tumor_bam: Path to the tumor BAM file (indexed).
        run_id: Unique identifier for this run.
        normal_bam: Path to the matched normal BAM (optional; uses pro mode if absent).
        min_coverage: Minimum coverage for microsatellite loci (default 20).
        threads: Number of threads.

    Returns:
        ToolResult with MSI score and status (MSI-H / MSS / MSI-L).
    """
    outdir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/msisensor")
    output_prefix = str(outdir / "msisensor")

    with ToolTimer() as timer:
        errors: list[str] = []
        warnings: list[str] = []
        summary: dict = {}
        output_paths: list[str] = []

        if not Path(tumor_bam).exists():
            errors.append(f"Tumor BAM not found: {tumor_bam}")
        if normal_bam and not Path(normal_bam).exists():
            errors.append(f"Normal BAM not found: {normal_bam}")

        # Step 1: Scan microsatellite sites if list doesn't exist
        sites_list = _MS_SITES_LIST
        if not Path(sites_list).exists():
            sites_list = str(outdir / "microsatellites.list")
            if not errors:
                try:
                    run_cmd([
                        "msisensor-pro", "scan",
                        "-d", _REFERENCE_FASTA,
                        "-o", sites_list,
                    ], timeout=1800)
                except Exception as exc:
                    errors.append(f"msisensor-pro scan failed: {exc}")

        # Step 2: MSI detection
        if not errors:
            if normal_bam:
                # Paired tumor-normal mode
                cmd = [
                    "msisensor-pro", "msi",
                    "-d", sites_list,
                    "-t", tumor_bam,
                    "-n", normal_bam,
                    "-o", output_prefix,
                    "-b", str(threads),
                    "-c", str(min_coverage),
                ]
            else:
                # Tumor-only (pro) mode
                cmd = [
                    "msisensor-pro", "pro",
                    "-d", sites_list,
                    "-t", tumor_bam,
                    "-o", output_prefix,
                    "-b", str(threads),
                    "-c", str(min_coverage),
                ]

            try:
                run_cmd(cmd, timeout=3000)
            except Exception as exc:
                errors.append(f"msisensor-pro detection failed: {exc}")

        # Parse results
        main_output = Path(output_prefix)
        if main_output.exists():
            output_paths.append(str(main_output))
            try:
                with open(main_output) as fh:
                    lines = fh.readlines()
                if len(lines) >= 2:
                    # Format: Total_Number_of_Sites  Number_of_Somatic_Sites  %
                    parts = lines[1].strip().split("\t")
                    if len(parts) >= 3:
                        summary["total_sites"] = int(parts[0])
                        summary["unstable_sites"] = int(parts[1])
                        msi_score = float(parts[2])
                        summary["msi_score"] = round(msi_score, 2)

                        # Classification thresholds (common convention)
                        if msi_score >= 20.0:
                            summary["msi_status"] = "MSI-H"
                        elif msi_score >= 10.0:
                            summary["msi_status"] = "MSI-L"
                        else:
                            summary["msi_status"] = "MSS"
            except Exception as exc:
                warnings.append(f"Failed to parse msisensor-pro output: {exc}")

        # Collect additional output files
        for suffix in ["_dis", "_germline", "_somatic"]:
            p = Path(f"{output_prefix}{suffix}")
            if p.exists():
                output_paths.append(str(p))

        summary["mode"] = "paired" if normal_bam else "tumor_only"

        vol_workdir.commit()

    return ToolResult(
        tool_name="msisensor_pro",
        version="1.2.0",
        started_at=timer.started_at,
        completed_at=timer.completed_at,
        input_summary={
            "tumor_bam": tumor_bam,
            "normal_bam": normal_bam,
            "run_id": run_id,
            "min_coverage": min_coverage,
            "mode": "paired" if normal_bam else "tumor_only",
            PROVENANCE_KEY: stamp("reference_genome"),
        },
        output_paths=output_paths,
        output_summary=summary,
        errors=errors,
        warnings=warnings,
    )
