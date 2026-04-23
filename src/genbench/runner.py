"""Runner — the only way to execute evals.

run_eval("clinvar", model="alphamissense", config="alphamissense_balanced")
run_eval("clinvar")                          # default config, all default baselines
run_all(model=my_model)                      # all available evals, one model

With visualization and tracking:
run_eval("clinvar", model="alphamissense", save_predictions=True)
"""

from __future__ import annotations

from genbench.eval import EvalResult
from genbench.registry import get_eval, get_model, list_evals
from genbench.model import Model
from genbench.reporting.results import save_results
from genbench.types import BenchmarkResult


def run_eval(
    eval_name: str,
    model: Model | str | None = None,
    config: str | None = None,
    save_predictions: bool = False,
    output_dir: str | None = None,
) -> list[BenchmarkResult]:
    """Run one eval.

    Args:
        eval_name: Registered eval name (e.g., "clinvar", "dms").
        model: A Model instance, a registered model name (str), or None.
            If None, runs all default baselines.
        config: Config name (e.g., "alphamissense_balanced"). Defaults to eval's default.
        save_predictions: If True, saves y_true/y_pred and generates plots.
        output_dir: Directory for predictions/plots. Defaults to results/{eval}/{model}/.
    """
    ev = get_eval(eval_name)

    if model is not None:
        if isinstance(model, str):
            model = get_model(model)
        result = ev.evaluate(
            model, config_name=config,
            save_predictions=save_predictions,
            output_dir=output_dir,
        )
        results = [_unwrap(result)]
        eval_results = [result] if isinstance(result, EvalResult) else []
    else:
        results = []
        eval_results = []
        for baseline_name in ev.default_baselines:
            try:
                baseline = get_model(baseline_name)
                result = ev.evaluate(
                    baseline, config_name=config,
                    save_predictions=save_predictions,
                    output_dir=output_dir,
                )
                results.append(_unwrap(result))
                if isinstance(result, EvalResult):
                    eval_results.append(result)
                print(f"  {eval_name} + {baseline_name}: done")
            except Exception as e:
                print(f"  {eval_name} + {baseline_name}: FAILED ({e})")

    # Save to Modal volume (existing behavior)
    save_results(results)

    # Save to local store if available
    _record_to_store(results, eval_results)

    # Print plot paths if any were generated
    for er in eval_results:
        if er.plots_dir:
            print(f"  Plots saved to: {er.plots_dir}")
        if er.predictions_path:
            print(f"  Predictions saved to: {er.predictions_path}")

    return results


def run_all(
    model: Model | str | None = None,
    config: str | None = None,
    only_available: bool = True,
    save_predictions: bool = False,
) -> list[BenchmarkResult]:
    """Run all evals."""
    all_results: list[BenchmarkResult] = []

    for eval_name in list_evals():
        ev = get_eval(eval_name)

        if only_available and not ev.is_available():
            print(f"  {eval_name}: skipped (data not available)")
            continue

        print(f"Running eval: {eval_name}")
        try:
            results = run_eval(
                eval_name, model=model, config=config,
                save_predictions=save_predictions,
            )
            all_results.extend(results)
        except Exception as e:
            print(f"  {eval_name}: FAILED ({e})")

    return all_results


def _unwrap(result: BenchmarkResult | EvalResult) -> BenchmarkResult:
    """Extract BenchmarkResult from EvalResult if needed."""
    if isinstance(result, EvalResult):
        return result.benchmark
    return result


def _record_to_store(
    results: list[BenchmarkResult],
    eval_results: list[EvalResult],
) -> None:
    """Record results to local store for tracking."""
    try:
        from genbench.reporting.store import ResultStore
        store = ResultStore()

        pred_paths = {er.benchmark.task_id: er.predictions_path for er in eval_results}

        for result in results:
            store.record(result, predictions_path=pred_paths.get(result.task_id))
    except Exception:
        pass  # Don't fail the eval if store is unavailable
