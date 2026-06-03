"""Biolearn epigenetic clock analysis.

Runs the Biolearn library to compute multiple epigenetic aging clocks from
DNA methylation beta-value matrices (e.g., Illumina 450K/EPIC arrays).

Supported clocks: Horvath, PhenoAge, GrimAge2, DunedinPACE, PC-Clocks.
"""

from __future__ import annotations

from pathlib import Path

from genes.app import app
from genes.infra.images import image_python_bio
from genes.infra.volumes import MOUNT_WORKDIR, vol_workdir
from genes.infra.provenance import PROVENANCE_KEY, stamp
from genes.tools._base import ToolResult, ToolTimer, ensure_dir

# All clocks to run by default
DEFAULT_CLOCKS = [
    "Horvath",
    "PhenoAge",
    "GrimAge2",
    "DunedinPACE",
    "PCHorvath1",
    "PCHorvath2",
    "PCPhenoAge",
    "PCGrimAge",
]


@app.function(
    image=image_python_bio,
    volumes={MOUNT_WORKDIR: vol_workdir},
    timeout=3600,
    cpu=4,
    memory=16384,
)
def compute_clocks(
    methylation_path: str,
    run_id: str,
    *,
    chronological_age: float | None = None,
    sex: str | None = None,
    clocks: list[str] | None = None,
) -> ToolResult:
    """Compute epigenetic clocks from a methylation beta-value matrix.

    Args:
        methylation_path: Path to a CSV/TSV with CpG rows x sample columns
            containing beta values (0-1). First column is CpG probe ID.
        run_id: Unique run identifier.
        chronological_age: Known chronological age (needed for age-acceleration).
        sex: Biological sex ('M'/'F') for sex-adjusted clocks.
        clocks: List of clock names to run. Defaults to all supported clocks.
    """
    outdir = ensure_dir(f"{MOUNT_WORKDIR}/{run_id}/biolearn")
    clock_list = clocks or DEFAULT_CLOCKS

    with ToolTimer() as timer:
        errors: list[str] = []
        warnings: list[str] = []
        summary: dict = {}
        output_paths: list[str] = []

        if not Path(methylation_path).exists():
            errors.append(f"Methylation file not found: {methylation_path}")

        if not errors:
            try:
                import pandas as pd
                from biolearn.model_gallery import ModelGallery

                gallery = ModelGallery()

                # Load methylation matrix
                meth_ext = Path(methylation_path).suffix.lower()
                sep = "\t" if meth_ext in {".tsv", ".txt"} else ","
                df = pd.read_csv(methylation_path, sep=sep, index_col=0)

                # Build metadata DataFrame expected by Biolearn
                sample_cols = df.columns.tolist()
                metadata = pd.DataFrame({"sample": sample_cols})
                if chronological_age is not None:
                    metadata["age"] = chronological_age
                if sex is not None:
                    metadata["sex"] = 1 if sex.upper() == "M" else 0
                metadata = metadata.set_index("sample")

                clock_results: dict[str, dict] = {}
                for clock_name in clock_list:
                    try:
                        model = gallery.get(clock_name)
                        result = model.predict(df, metadata)
                        # result is a Series or DataFrame with predicted ages
                        predictions = result.to_dict() if hasattr(result, "to_dict") else {}
                        clock_results[clock_name] = {
                            "predicted_ages": predictions,
                        }
                        if chronological_age is not None:
                            for sample, pred_age in predictions.items():
                                if isinstance(pred_age, (int, float)):
                                    clock_results[clock_name]["age_acceleration"] = (
                                        round(pred_age - chronological_age, 2)
                                    )
                    except Exception as exc:
                        warnings.append(f"Clock {clock_name} failed: {exc}")

                summary["clocks"] = clock_results
                summary["num_probes"] = len(df)
                summary["num_samples"] = len(sample_cols)

                # Write results to JSON
                import json

                results_path = outdir / f"{run_id}.biolearn_clocks.json"
                with open(results_path, "w") as fh:
                    json.dump(clock_results, fh, indent=2, default=str)
                output_paths.append(str(results_path))

            except ImportError as exc:
                errors.append(f"Biolearn import error: {exc}")
            except Exception as exc:
                errors.append(f"Biolearn analysis failed: {exc}")

        vol_workdir.commit()

    return ToolResult(
        tool_name="biolearn",
        version="0.5.0",
        started_at=timer.started_at,
        completed_at=timer.completed_at,
        input_summary={
            "methylation_path": methylation_path,
            "run_id": run_id,
            "chronological_age": chronological_age,
            "sex": sex,
            "clocks": clock_list,
            PROVENANCE_KEY: stamp(),
        },
        output_paths=output_paths,
        output_summary=summary,
        errors=errors,
        warnings=warnings,
    )
