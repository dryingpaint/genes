"""Test the eval/model framework itself — registry, protocol, runner logic."""

import numpy as np
import pytest

from genbench.eval import Eval
from genbench.model import FunctionModel, Model
from genbench.registry import get_eval, get_model, list_evals, list_models
from genbench.types import AncestryStratifiedMetric, LeakageReport, MetricValue, SplitType


class TestRegistry:
    def test_all_15_evals_registered(self):
        evals = list_evals()
        assert len(evals) == 15
        assert "clinvar" in evals
        assert "dms" in evals
        assert "dgrp" in evals

    def test_all_models_registered(self):
        models = list_models()
        assert "null" in models
        assert "alphamissense" in models
        assert "saprot" in models
        assert len(models) >= 7

    def test_get_eval_returns_eval_instance(self):
        ev = get_eval("clinvar")
        assert isinstance(ev, Eval)
        assert ev.name == "clinvar"

    def test_get_model_returns_model(self):
        m = get_model("null")
        assert isinstance(m, Model)
        assert m.name == "null"

    def test_unknown_eval_raises(self):
        with pytest.raises(KeyError, match="Unknown eval"):
            get_eval("nonexistent_eval")

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError, match="Unknown model"):
            get_model("nonexistent_model")


class TestModelProtocol:
    def test_function_model(self):
        fn = lambda inputs: np.ones(inputs["n"])
        m = FunctionModel("test", fn)
        assert m.name == "test"
        result = m.predict({"n": 5})
        assert len(result) == 5
        assert all(result == 1.0)

    def test_null_model_returns_correct_length(self):
        m = get_model("null")
        result = m.predict({"chroms": ["chr1"] * 10, "positions": list(range(10)),
                            "refs": ["A"] * 10, "alts": ["T"] * 10})
        assert len(result) == 10
        assert all(0 <= x <= 1 for x in result)

    def test_null_model_is_deterministic(self):
        m = get_model("null")
        r1 = m.predict({"n": 100})
        r2 = m.predict({"n": 100})
        np.testing.assert_array_equal(r1, r2)


class TestEvalProperties:
    """Verify all registered evals have required properties."""

    @pytest.fixture(params=list_evals())
    def eval_instance(self, request):
        return get_eval(request.param)

    def test_has_name(self, eval_instance):
        assert isinstance(eval_instance.name, str)
        assert len(eval_instance.name) > 0

    def test_has_description(self, eval_instance):
        assert isinstance(eval_instance.description, str)
        assert len(eval_instance.description) > 0

    def test_has_tier(self, eval_instance):
        assert eval_instance.tier in (0, 1, 2, 3)

    def test_has_split_type(self, eval_instance):
        assert isinstance(eval_instance.split_type, SplitType)

    def test_has_default_baselines(self, eval_instance):
        assert isinstance(eval_instance.default_baselines, list)
        assert len(eval_instance.default_baselines) > 0
        assert "null" in eval_instance.default_baselines

    def test_is_available_returns_bool(self, eval_instance):
        result = eval_instance.is_available()
        assert isinstance(result, bool)
