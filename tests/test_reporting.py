"""Tests for the reporting infrastructure: store, plots, and eval result tracking."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from genbench.eval import EvalResult
from genbench.reporting.store import ResultStore, EvalRun
from genbench.types import (
    AncestryStratifiedMetric,
    BenchmarkResult,
    LeakageReport,
    MetricValue,
    SplitType,
)


def _make_result(
    task_id: str = "clinvar/all_snv",
    model_name: str = "alphamissense",
    auroc: float = 0.85,
    config: str = "all_snv",
) -> BenchmarkResult:
    return BenchmarkResult(
        task_id=task_id,
        model_name=model_name,
        split_type=SplitType.TEMPORAL,
        leakage_report=LeakageReport(),
        metrics={
            "auroc": AncestryStratifiedMetric(
                aggregate=MetricValue(estimate=auroc, ci_lower=0.83, ci_upper=0.87, n=1000)
            )
        },
        n_samples={"ALL": 1000},
        metadata={"config": config, "tier": 0},
    )


class TestResultStore:
    def test_record_and_list(self, tmp_path):
        store = ResultStore(tmp_path / "test.jsonl")
        store.record(_make_result())
        runs = store.list_runs()
        assert len(runs) == 1
        assert runs[0].task_id == "clinvar/all_snv"
        assert runs[0].model_name == "alphamissense"

    def test_filter_by_eval(self, tmp_path):
        store = ResultStore(tmp_path / "test.jsonl")
        store.record(_make_result(task_id="clinvar/all_snv"))
        store.record(_make_result(task_id="dms/substitutions_v1", model_name="saprot"))

        runs = store.list_runs(eval_name="clinvar")
        assert len(runs) == 1
        assert runs[0].task_id == "clinvar/all_snv"

    def test_filter_by_model(self, tmp_path):
        store = ResultStore(tmp_path / "test.jsonl")
        store.record(_make_result(model_name="alphamissense"))
        store.record(_make_result(model_name="null", auroc=0.50))

        runs = store.list_runs(model_name="null")
        assert len(runs) == 1
        assert runs[0].model_name == "null"

    def test_latest(self, tmp_path):
        store = ResultStore(tmp_path / "test.jsonl")
        store.record(_make_result(auroc=0.80))
        store.record(_make_result(auroc=0.85))
        store.record(_make_result(auroc=0.90))

        latest = store.latest("clinvar", "alphamissense")
        assert latest is not None
        assert latest.metrics["auroc"] == pytest.approx(0.90)

    def test_compare(self, tmp_path):
        store = ResultStore(tmp_path / "test.jsonl")
        store.record(_make_result(model_name="alphamissense", auroc=0.85))
        store.record(_make_result(model_name="null", auroc=0.50))
        store.record(_make_result(model_name="evo2", auroc=0.90))

        comparison = store.compare("clinvar")
        assert len(comparison) == 3
        models = {r.model_name for r in comparison}
        assert models == {"alphamissense", "null", "evo2"}

    def test_history(self, tmp_path):
        store = ResultStore(tmp_path / "test.jsonl")
        store.record(_make_result(auroc=0.80))
        store.record(_make_result(auroc=0.85))
        store.record(_make_result(auroc=0.90))

        history = store.history("clinvar", "alphamissense", "auroc")
        assert len(history) == 3
        values = [h[1] for h in history]
        assert values == [0.80, 0.85, 0.90]

    def test_summary_table(self, tmp_path):
        store = ResultStore(tmp_path / "test.jsonl")
        store.record(_make_result())
        table = store.summary_table()
        assert "clinvar/all_snv" in table
        assert "alphamissense" in table

    def test_empty_store(self, tmp_path):
        store = ResultStore(tmp_path / "empty.jsonl")
        assert store.list_runs() == []
        assert store.latest("clinvar", "am") is None
        assert store.summary_table() == "No results stored."

    def test_predictions_path(self, tmp_path):
        store = ResultStore(tmp_path / "test.jsonl")
        store.record(_make_result(), predictions_path="/data/pred.npz")
        runs = store.list_runs()
        assert runs[0].predictions_path == "/data/pred.npz"


class TestEvalRun:
    def test_summary_line(self):
        data = {
            "task_id": "clinvar/all_snv",
            "model_name": "alphamissense",
            "timestamp": "2026-04-23T12:00:00Z",
            "metrics": {
                "auroc": {"aggregate": {"estimate": 0.85, "ci_lower": 0.83, "ci_upper": 0.87, "n": 1000}}
            },
            "metadata": {"config": "all_snv"},
        }
        run = EvalRun(data)
        line = run.summary_line()
        assert "clinvar/all_snv" in line
        assert "alphamissense" in line
        assert "0.8500" in line


class TestEvalResult:
    def test_save_predictions(self, tmp_path):
        br = _make_result()
        er = EvalResult(
            benchmark=br,
            y_true=np.array([0, 1, 1, 0]),
            y_pred=np.array([0.1, 0.9, 0.8, 0.2]),
        )
        path = er.save_predictions(str(tmp_path))
        assert Path(path).exists()

        # Verify saved data
        data = np.load(path)
        np.testing.assert_array_equal(data["y_true"], er.y_true)
        np.testing.assert_array_equal(data["y_pred"], er.y_pred)

    def test_generate_plots(self, tmp_path):
        br = _make_result()
        er = EvalResult(
            benchmark=br,
            y_true=np.concatenate([np.ones(50), np.zeros(50)]),
            y_pred=np.concatenate([
                np.random.default_rng(42).beta(5, 2, 50),
                np.random.default_rng(42).beta(2, 5, 50),
            ]),
        )
        plots = er.generate_plots(str(tmp_path))
        assert len(plots) >= 1  # at least classification plot
        for p in plots:
            assert Path(p).exists()

    def test_empty_predictions_no_plots(self, tmp_path):
        br = _make_result()
        er = EvalResult(benchmark=br)
        plots = er.generate_plots(str(tmp_path))
        assert plots == []


class TestPlots:
    """Test plot functions don't crash with various inputs."""

    def test_classification_plot(self, tmp_path):
        from genbench.reporting.plots import plot_classification
        rng = np.random.default_rng(42)
        y_true = np.concatenate([np.ones(100), np.zeros(100)])
        y_pred = np.concatenate([rng.beta(5, 2, 100), rng.beta(2, 5, 100)])
        fig = plot_classification(y_true, y_pred, save_path=str(tmp_path / "roc.png"))
        assert (tmp_path / "roc.png").exists()
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_regression_plot(self, tmp_path):
        from genbench.reporting.plots import plot_regression
        rng = np.random.default_rng(42)
        y_true = rng.normal(0, 1, 200)
        y_pred = y_true * 0.7 + rng.normal(0, 0.3, 200)
        fig = plot_regression(y_true, y_pred, save_path=str(tmp_path / "reg.png"))
        assert (tmp_path / "reg.png").exists()
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_per_assay_rho(self, tmp_path):
        from genbench.reporting.plots import plot_per_assay_rho
        rhos = np.random.default_rng(42).beta(5, 3, 30).tolist()
        fig = plot_per_assay_rho(rhos, save_path=str(tmp_path / "rho.png"))
        assert (tmp_path / "rho.png").exists()
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_input_summary_binary(self, tmp_path):
        from genbench.reporting.plots import plot_input_summary
        labels = np.concatenate([np.ones(80), np.zeros(120)])
        fig = plot_input_summary(labels, save_path=str(tmp_path / "inputs.png"))
        assert (tmp_path / "inputs.png").exists()
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_input_summary_continuous(self, tmp_path):
        from genbench.reporting.plots import plot_input_summary
        labels = np.random.default_rng(42).normal(0, 1, 200)
        fig = plot_input_summary(labels, save_path=str(tmp_path / "inputs_cont.png"))
        assert (tmp_path / "inputs_cont.png").exists()
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_model_comparison(self, tmp_path):
        from genbench.reporting.plots import plot_model_comparison
        results = [
            {"model_name": "alphamissense", "metrics": {"auroc": 0.85}},
            {"model_name": "null", "metrics": {"auroc": 0.50}},
            {"model_name": "evo2", "metrics": {"auroc": 0.90}},
        ]
        fig = plot_model_comparison(results, save_path=str(tmp_path / "compare.png"))
        assert (tmp_path / "compare.png").exists()
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_variant_calling_plot(self, tmp_path):
        from genbench.reporting.plots import plot_variant_calling
        metrics = {
            "snv": {"precision": 0.999, "recall": 0.998, "f1": 0.999},
            "indel": {"precision": 0.990, "recall": 0.985, "f1": 0.988},
        }
        fig = plot_variant_calling(metrics, save_path=str(tmp_path / "vc.png"))
        assert (tmp_path / "vc.png").exists()
        import matplotlib.pyplot as plt
        plt.close("all")
