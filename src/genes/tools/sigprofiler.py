"""SigProfiler — Mutational signature analysis.

Runs SigProfilerMatrixGenerator to build the mutation count matrix from
a somatic VCF, then SigProfilerAssignment to decompose the spectrum
into known COSMIC mutational signatures.
"""

from __future__ import annotations

import json
from pathlib import Path

from genes.app import app
from genes.infra.images import image_python_bio
from genes.infra.volumes import (
    MOUNT_REFERENCE,
    MOUNT_WORKDIR,
    vol_reference,
    vol_workdir,
)
from genes.tools._base import ToolResult, ToolTimer, ensure_dir

_REFERENCE_GENOME = "GRCh38"


@app.function(
    image=image_python_bio,
    volumes={
        MOUNT_REFERENCE: vol_reference,
        MOUNT_WORKDIR: vol_workdir,
    },
    timeout=3600,
    memory=16384,
)
def analyze_signatures(
    vcf_path: str,
    run_id: str,
    sample_name: str = "sample",
    genome: str = "GRCh38",
    context_types: list[str] | None = None,
) -> ToolResult:
    """Analyze mutational signatures from a somatic VCF.

    Args:
        vcf_path: Path to the somatic VCF (PASS variants only recommended).
        run_id: Unique identifier for this run.
        sample_name: Name for the sample in output matrices.
        genome: Reference genome (default "GRCh38").
        context_types: Mutation context types to generate (default ["96", "DINUC", "ID"]).

    Returns:
        ToolResult with signature decomposition and exposures.
    """
    if context_types is None:
        context_types = ["96", "DINUC", "ID"]

    outdir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/sigprofiler")
    vcf_dir = ensure_dir(outdir / "input")

    with ToolTimer() as timer:
        errors: list[str] = []
        warnings: list[str] = []
        summary: dict = {}
        output_paths: list[str] = []

        if not Path(vcf_path).exists():
            errors.append(f"VCF file not found: {vcf_path}")

        if not errors:
            # SigProfilerMatrixGenerator expects VCFs in a directory
            import shutil
            dest_vcf = vcf_dir / f"{sample_name}.vcf"
            shutil.copy2(vcf_path, str(dest_vcf))

        # Step 1: Generate mutation matrices
        if not errors:
            try:
                from SigProfilerMatrixGenerator.scripts import (
                    SigProfilerMatrixGeneratorFunc as matGen,
                )

                matrices = matGen.SigProfilerMatrixGeneratorFunc(
                    project=sample_name,
                    genome=genome,
                    vcfFiles=str(vcf_dir),
                    plot=True,
                    exome=False,
                )
            except Exception as exc:
                errors.append(f"SigProfilerMatrixGenerator failed: {exc}")

        # Collect matrix output paths
        matrix_dir = outdir / "input" / "output" / "SBS"
        if matrix_dir.exists():
            for f in matrix_dir.iterdir():
                if f.suffix in (".txt", ".pdf", ".png"):
                    output_paths.append(str(f))

        # Step 2: Decompose into COSMIC signatures
        if not errors:
            try:
                from SigProfilerAssignment import Analyzer as Analyze

                sbs_matrix = matrix_dir / f"{sample_name}.SBS96.all"
                if not sbs_matrix.exists():
                    # Try alternative naming
                    candidates = list(matrix_dir.glob("*.SBS96.*"))
                    sbs_matrix = candidates[0] if candidates else None

                if sbs_matrix and sbs_matrix.exists():
                    assignment_outdir = ensure_dir(outdir / "assignment")
                    Analyze.cosmic_fit(
                        samples=str(sbs_matrix),
                        output=str(assignment_outdir),
                        input_type="matrix",
                        genome_build=genome,
                        signature_database=None,  # use default COSMIC v3.4
                    )

                    # Parse assignment results
                    activities_file = assignment_outdir / "Assignment_Solution" / "Activities" / "Assignment_Solution_Activities.txt"
                    if activities_file.exists():
                        output_paths.append(str(activities_file))
                        import pandas as pd
                        df = pd.read_csv(str(activities_file), sep="\t", index_col=0)
                        if sample_name in df.columns:
                            exposures = df[sample_name].to_dict()
                        else:
                            exposures = df.iloc[:, 0].to_dict()
                        # Filter to non-zero signatures
                        active_sigs = {
                            sig: int(count) for sig, count in exposures.items()
                            if count > 0
                        }
                        summary["active_signatures"] = active_sigs
                        summary["total_mutations_assigned"] = sum(active_sigs.values())
                        summary["num_active_signatures"] = len(active_sigs)

                        # Top 5 signatures
                        sorted_sigs = sorted(
                            active_sigs.items(), key=lambda x: x[1], reverse=True
                        )
                        summary["top_signatures"] = [
                            {"signature": sig, "mutations": count}
                            for sig, count in sorted_sigs[:5]
                        ]
                else:
                    warnings.append("SBS96 matrix not found; skipping signature assignment.")

            except Exception as exc:
                errors.append(f"SigProfilerAssignment failed: {exc}")

        # Write summary JSON
        summary_path = str(outdir / "signature_summary.json")
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2)
        output_paths.append(summary_path)

        vol_workdir.commit()

    return ToolResult(
        tool_name="sigprofiler",
        version="1.2",
        started_at=timer.started_at,
        completed_at=timer.completed_at,
        input_summary={
            "vcf_path": vcf_path,
            "run_id": run_id,
            "sample_name": sample_name,
            "genome": genome,
        },
        output_paths=output_paths,
        output_summary=summary,
        errors=errors,
        warnings=warnings,
    )
