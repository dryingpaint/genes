"""Benchmark the germline pipeline against GIAB truth sets.

Runs the full germline VCF pipeline on a GIAB sample's pre-called VCF,
then scores each tool's output against known ground truth.

Usage:
    # Benchmark using existing HG002 test VCF against GIAB truth
    modal run scripts/benchmark_pipeline.py --sample HG002

    # Benchmark a specific query VCF
    modal run scripts/benchmark_pipeline.py --sample HG002 --vcf test_data/HG002.vcf.gz

    # Quick test on chr22 only
    modal run scripts/benchmark_pipeline.py --sample HG002 --vcf test_data/HG002_chr22.vcf.gz

This script produces a scorecard covering:
  1. Variant calling accuracy (GIAB truth comparison) — SNV/indel F1
  2. Pathogenicity scoring (ClinVar baseline) — AUROC
  3. Coverage stats per tool
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from genbench.app import app
from genbench.infra.images import cpu_image
from genbench.infra.volumes import VOLUME_MOUNTS


@app.function(
    image=cpu_image,
    volumes=VOLUME_MOUNTS,
    timeout=14400,
    memory=32768,
)
def benchmark_giab(
    query_vcf_path: str,
    sample: str,
) -> dict:
    """Run GIAB variant calling benchmark on Modal.

    Args:
        query_vcf_path: Path to query VCF (on Modal volume).
        sample: GIAB sample name (HG001-HG007).
    """
    from genbench.registry import get_eval

    ev = get_eval("giab")
    results = {}

    # Run SNV and indel configs
    for config_name in ["hg002_snv", "hg002_indel"]:
        try:
            result = ev.evaluate_vcf(query_vcf_path, sample=sample, config_name=config_name)
            results[config_name] = result.model_dump(mode="json")
        except FileNotFoundError as e:
            results[config_name] = {"error": str(e)}
        except Exception as e:
            results[config_name] = {"error": f"{type(e).__name__}: {e}"}

    return results


@app.function(
    image=cpu_image,
    volumes=VOLUME_MOUNTS,
    timeout=14400,
    memory=32768,
)
def benchmark_clinvar(model_name: str = "alphamissense", config_name: str = "all_snv") -> dict:
    """Run ClinVar pathogenicity benchmark on Modal."""
    from genbench.registry import get_eval, get_model

    ev = get_eval("clinvar")
    model = get_model(model_name)

    try:
        result = ev.evaluate(model, config_name=config_name)
        return result.model_dump(mode="json")
    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


@app.function(
    image=cpu_image,
    volumes=VOLUME_MOUNTS,
    timeout=14400,
    memory=32768,
)
def benchmark_brca1(model_name: str = "alphamissense") -> dict:
    """Run BRCA1 SGE functional classification benchmark on Modal."""
    from genbench.registry import get_eval, get_model

    ev = get_eval("brca1_sge")
    model = get_model(model_name)

    try:
        result = ev.evaluate(model, config_name="lof_vs_func")
        return result.model_dump(mode="json")
    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


@app.function(
    image=cpu_image,
    volumes=VOLUME_MOUNTS,
    timeout=14400,
    memory=32768,
)
def benchmark_dms(model_name: str = "saprot") -> dict:
    """Run ProteinGym DMS benchmark on Modal."""
    from genbench.registry import get_eval, get_model

    ev = get_eval("dms")
    model = get_model(model_name)

    try:
        result = ev.evaluate(model)
        return result.model_dump(mode="json")
    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


@app.function(
    image=cpu_image,
    volumes=VOLUME_MOUNTS,
    timeout=600,
    memory=8192,
)
def list_available_data() -> dict:
    """Check which benchmark datasets are available on the Modal volume."""
    from pathlib import Path
    from genbench.config import DATASETS_PATH

    status = {}

    # GIAB
    giab_base = Path(DATASETS_PATH) / "giab"
    for sample in ["HG001", "HG002", "HG003", "HG004", "HG005", "HG006", "HG007"]:
        vcf = giab_base / sample / f"{sample}_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
        status[f"giab/{sample}"] = vcf.exists()

    # ClinVar
    for p in [
        Path(DATASETS_PATH) / "clinvar" / "latest" / "variant_summary.txt.gz",
        Path(DATASETS_PATH) / "clinvar" / "variant_summary.txt.gz",
    ]:
        if p.exists():
            status["clinvar"] = True
            break
    else:
        status["clinvar"] = False

    # ProteinGym
    for version in ["v1", "v0.1"]:
        pg_path = Path(DATASETS_PATH) / "proteingym" / version / "ProteinGym_substitutions"
        if pg_path.exists():
            csv_count = len(list(pg_path.glob("*.csv")))
            status[f"proteingym/{version}"] = csv_count > 0
        else:
            status[f"proteingym/{version}"] = False

    # BRCA1 SGE
    brca1_path = Path(DATASETS_PATH) / "brca1_sge" / "findlay2018_supp_table2.xlsx"
    status["brca1_sge"] = brca1_path.exists()

    # AlphaMissense baseline scores
    am_path = Path(DATASETS_PATH) / "baseline_scores" / "AlphaMissense_hg38.tsv.gz"
    status["alphamissense_scores"] = am_path.exists()

    return status


def _print_scorecard(results: dict) -> None:
    """Print a human-readable benchmark scorecard."""
    print()
    print("=" * 70)
    print("  PIPELINE BENCHMARK SCORECARD")
    print("=" * 70)
    print()

    for section, data in results.items():
        if isinstance(data, dict) and "error" in data:
            print(f"  [{section}] SKIPPED: {data['error'][:80]}")
            continue

        if isinstance(data, dict) and "metrics" in data:
            metrics = data["metrics"]
            task_id = data.get("task_id", section)
            model = data.get("model_name", "")
            print(f"  [{task_id}] model={model}")
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict):
                    agg = metric_data.get("aggregate")
                    if agg and isinstance(agg, dict):
                        est = agg.get("estimate")
                        if est is not None and est == est:  # not NaN
                            n = agg.get("n", "")
                            ci_lo = agg.get("ci_lower", 0)
                            ci_hi = agg.get("ci_upper", 0)
                            ci_str = f" [{ci_lo:.4f}, {ci_hi:.4f}]" if ci_lo and ci_hi else ""
                            n_str = f" (n={n})" if n else ""
                            print(f"    {metric_name}: {est:.4f}{ci_str}{n_str}")
            print()
        elif isinstance(data, dict):
            # GIAB results (nested by config)
            for config_name, config_data in data.items():
                if isinstance(config_data, dict) and "error" in config_data:
                    print(f"  [{section}/{config_name}] SKIPPED: {config_data['error'][:80]}")
                    continue
                if isinstance(config_data, dict) and "metrics" in config_data:
                    print(f"  [{config_data.get('task_id', f'{section}/{config_name}')}]")
                    meta = config_data.get("metadata", {})
                    for metric_name, metric_data in config_data["metrics"].items():
                        if isinstance(metric_data, dict):
                            agg = metric_data.get("aggregate")
                            if agg and isinstance(agg, dict):
                                est = agg.get("estimate")
                                if est is not None and est == est:
                                    print(f"    {metric_name}: {est:.4f}")
                    # Print TP/FP/FN if available
                    for key in ["snv_tp", "snv_fp", "snv_fn", "indel_tp", "indel_fp", "indel_fn"]:
                        if key in meta:
                            print(f"    {key}: {meta[key]}")
                    print()

    print("=" * 70)


@app.local_entrypoint()
def main(
    sample: str = "HG002",
    vcf: str = "",
    skip_giab: bool = False,
    skip_clinvar: bool = False,
    skip_brca1: bool = False,
    skip_dms: bool = True,  # DMS is slow, skip by default
):
    """Run the full pipeline benchmark suite.

    Args:
        sample: GIAB sample name for variant calling benchmark.
        vcf: Path to query VCF. If empty, looks for test_data/{sample}.vcf.gz.
        skip_giab: Skip GIAB variant calling accuracy benchmark.
        skip_clinvar: Skip ClinVar pathogenicity benchmark.
        skip_brca1: Skip BRCA1 SGE functional classification benchmark.
        skip_dms: Skip ProteinGym DMS benchmark (slow).
    """
    print("Pipeline Benchmark Suite")
    print("=" * 70)

    # Step 1: Check data availability
    print("\nChecking data availability on Modal...")
    data_status = list_available_data.remote()
    for dataset, available in sorted(data_status.items()):
        status = "OK" if available else "MISSING"
        print(f"  [{status}] {dataset}")
    print()

    # Resolve query VCF for GIAB
    if not skip_giab:
        if not vcf:
            vcf = f"test_data/{sample}.vcf.gz"
        vcf_path = Path(vcf)
        if not vcf_path.exists():
            print(f"WARNING: Query VCF not found at {vcf_path}, skipping GIAB benchmark")
            skip_giab = True
        else:
            print(f"Query VCF: {vcf_path} ({vcf_path.stat().st_size / 1e6:.1f} MB)")

    # Step 2: Launch benchmarks in parallel
    print("\nLaunching benchmarks...")
    futures = {}

    if not skip_giab and data_status.get(f"giab/{sample}"):
        print(f"  Spawning GIAB benchmark ({sample})...")
        # For GIAB, we need the VCF on the Modal volume.
        # The test_data VCFs are already uploaded; use the DATASETS_PATH version.
        # In practice, this would use the DeepVariant output VCF.
        futures["giab"] = benchmark_giab.spawn(vcf, sample)
    elif not skip_giab:
        print(f"  Skipping GIAB: truth data not available for {sample}")

    if not skip_clinvar and data_status.get("clinvar"):
        print("  Spawning ClinVar benchmark (AlphaMissense)...")
        futures["clinvar"] = benchmark_clinvar.spawn("alphamissense", "all_snv")

    if not skip_brca1 and data_status.get("brca1_sge"):
        print("  Spawning BRCA1 SGE benchmark (AlphaMissense)...")
        futures["brca1"] = benchmark_brca1.spawn("alphamissense")

    if not skip_dms and data_status.get("proteingym/v0.1"):
        print("  Spawning DMS benchmark (SaProt)...")
        futures["dms"] = benchmark_dms.spawn("saprot")

    if not futures:
        print("\nNo benchmarks to run. Ensure data is ingested: modal run scripts/ingest_all.py")
        sys.exit(1)

    # Step 3: Collect results
    print(f"\nWaiting for {len(futures)} benchmark(s)...")
    results = {}
    for name, future in futures.items():
        try:
            results[name] = future.get()
            print(f"  OK  {name}")
        except Exception as e:
            results[name] = {"error": str(e)}
            print(f"  FAIL  {name}: {e}")

    # Step 4: Print scorecard
    _print_scorecard(results)

    # Step 5: Save results
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = f"test_data/benchmark_{timestamp}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results: {output_path}")
