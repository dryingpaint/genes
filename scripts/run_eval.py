"""Run evals on Modal.

Usage:
    modal run scripts/run_eval.py --eval clinvar
    modal run scripts/run_eval.py --eval clinvar --model alphamissense
    modal run scripts/run_eval.py --all
"""

from genbench.app import app
from genbench.infra.images import cpu_image
from genbench.infra.volumes import VOLUME_MOUNTS


@app.function(
    image=cpu_image,
    volumes=VOLUME_MOUNTS,
    timeout=28800,
    memory=16384,
)
def remote_run(eval_name: str | None, model_name: str | None, run_all: bool) -> dict:
    from genbench.registry import list_evals, list_models
    from genbench.runner import run_all as _run_all
    from genbench.runner import run_eval as _run_eval
    from genbench.reporting.templates import render_report

    print(f"Available evals: {list_evals()}")
    print(f"Available models: {list_models()}")
    print()

    if run_all:
        results = _run_all(model=model_name)
    elif eval_name:
        print(f"Running eval: {eval_name}" + (f" with model: {model_name}" if model_name else ""))
        results = _run_eval(eval_name, model=model_name)
    else:
        return {"error": "Specify --eval <name> or --all"}

    report = render_report(results)

    return {
        "n_results": len(results),
        "report": report,
        "results": [
            {
                "task_id": r.task_id,
                "model": r.model_name,
                "metrics": {
                    k: v.aggregate.estimate if v.aggregate else None
                    for k, v in r.metrics.items()
                },
            }
            for r in results
        ],
    }


@app.local_entrypoint()
def main(
    eval: str = "",
    model: str = "",
    all: bool = False,
):
    result = remote_run.remote(
        eval_name=eval or None,
        model_name=model or None,
        run_all=all,
    )

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n{'=' * 80}")
    print(result["report"])
    print(f"{'=' * 80}")
    print(f"\n{result['n_results']} result(s)")

    for r in result.get("results", []):
        metrics_str = ", ".join(f"{k}={v:.4f}" if v else f"{k}=N/A" for k, v in r["metrics"].items())
        print(f"  {r['task_id']} + {r['model']}: {metrics_str}")
