"""Runner — the only way to execute evals.

run_eval("clinvar", model="alphamissense")   # one eval, one model
run_eval("clinvar")                          # one eval, all default baselines
run_all(model=my_model)                      # all available evals, one model
run_all()                                    # all available evals, all default baselines
"""

from __future__ import annotations

from genbench.registry import get_eval, get_model, list_evals
from genbench.model import Model
from genbench.reporting.results import save_results
from genbench.types import BenchmarkResult


def run_eval(
    eval_name: str,
    model: Model | str | None = None,
) -> list[BenchmarkResult]:
    """Run one eval.

    Args:
        eval_name: Registered eval name (e.g., "clinvar", "dms").
        model: A Model instance, a registered model name (str), or None.
            If None, runs all default baselines for this eval.

    Returns:
        List of BenchmarkResult (one per model run).
    """
    ev = get_eval(eval_name)

    if model is not None:
        if isinstance(model, str):
            model = get_model(model)
        results = [ev.evaluate(model)]
    else:
        results = []
        for baseline_name in ev.default_baselines:
            try:
                baseline = get_model(baseline_name)
                result = ev.evaluate(baseline)
                results.append(result)
                print(f"  {eval_name} + {baseline_name}: done")
            except Exception as e:
                print(f"  {eval_name} + {baseline_name}: FAILED ({e})")

    save_results(results)
    return results


def run_all(
    model: Model | str | None = None,
    only_available: bool = True,
) -> list[BenchmarkResult]:
    """Run all evals.

    Args:
        model: If provided, run this model on every eval. If None, run
            each eval's default baselines.
        only_available: Skip evals whose data isn't available.

    Returns:
        All BenchmarkResults across all evals.
    """
    all_results: list[BenchmarkResult] = []

    for eval_name in list_evals():
        ev = get_eval(eval_name)

        if only_available and not ev.is_available():
            print(f"  {eval_name}: skipped (data not available)")
            continue

        print(f"Running eval: {eval_name}")
        try:
            results = run_eval(eval_name, model=model)
            all_results.extend(results)
        except Exception as e:
            print(f"  {eval_name}: FAILED ({e})")

    return all_results
