"""Run evals on Modal.

Usage:
    modal run scripts/run_eval.py --eval clinvar --model alphamissense --config alphamissense_balanced
    modal run scripts/run_eval.py --eval clinvar
    modal run scripts/run_eval.py --all
"""

from genbench.app import app
from genbench.infra.images import cpu_image
from genbench.infra.volumes import VOLUME_MOUNTS


@app.function(
    image=cpu_image,
    volumes=VOLUME_MOUNTS,
    timeout=28800,
    memory=32768,
)
def remote_run(
    eval_name: str | None,
    model_name: str | None,
    config_name: str | None,
    run_all: bool,
) -> dict:
    from genbench.registry import list_evals, list_models, get_eval
    from genbench.runner import run_all as _run_all
    from genbench.runner import run_eval as _run_eval
    from genbench.reporting.templates import render_report

    print(f"Available evals: {list_evals()}")
    print(f"Available models: {list_models()}")

    if eval_name:
        ev = get_eval(eval_name)
        print(f"Configs for {eval_name}: {ev.list_configs()}")
    print()

    if run_all:
        results = _run_all(model=model_name, config=config_name)
    elif eval_name:
        config_str = f" config={config_name}" if config_name else ""
        model_str = f" model={model_name}" if model_name else ""
        print(f"Running: {eval_name}{config_str}{model_str}")
        results = _run_eval(eval_name, model=model_name, config=config_name)
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
                "config": r.metadata.get("config", ""),
                "paper": r.metadata.get("paper", ""),
                "metrics": {
                    k: v.aggregate.estimate if v.aggregate else None
                    for k, v in r.metrics.items()
                },
                "expected": r.metadata.get("expected_baselines", {}),
            }
            for r in results
        ],
    }


@app.local_entrypoint()
def main(
    eval: str = "",
    model: str = "",
    config: str = "",
    all: bool = False,
):
    result = remote_run.remote(
        eval_name=eval or None,
        model_name=model or None,
        config_name=config or None,
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
        metrics_str = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) and v == v else f"{k}=N/A"
            for k, v in r["metrics"].items()
        )
        config_str = f" [{r['config']}]" if r.get("config") else ""
        paper_str = f" ({r['paper']})" if r.get("paper") else ""
        print(f"  {r['task_id']} + {r['model']}{config_str}: {metrics_str}{paper_str}")

        if r.get("expected"):
            for model_name, expected in r["expected"].items():
                if model_name == r["model"]:
                    for metric, val in expected.items():
                        print(f"    ^ expected {metric}={val:.4f}")
