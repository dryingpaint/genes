"""Eval and model registries with auto-discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genbench.eval import Eval
    from genbench.model import Model

_EVAL_REGISTRY: dict[str, type] = {}
_MODEL_REGISTRY: dict[str, type] = {}


def register_eval(name: str):
    """Decorator: @register_eval("clinvar")"""

    def wrapper(cls):
        _EVAL_REGISTRY[name] = cls
        return cls

    return wrapper


def register_model(name: str):
    """Decorator: @register_model("alphamissense")"""

    def wrapper(cls):
        _MODEL_REGISTRY[name] = cls
        return cls

    return wrapper


def get_eval(name: str) -> Eval:
    _ensure_evals_discovered()
    if name not in _EVAL_REGISTRY:
        available = ", ".join(sorted(_EVAL_REGISTRY.keys()))
        raise KeyError(f"Unknown eval '{name}'. Available: {available}")
    return _EVAL_REGISTRY[name]()


def get_model(name: str) -> Model:
    _ensure_models_discovered()
    if name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return _MODEL_REGISTRY[name]()


def list_evals() -> list[str]:
    _ensure_evals_discovered()
    return sorted(_EVAL_REGISTRY.keys())


def list_models() -> list[str]:
    _ensure_models_discovered()
    return sorted(_MODEL_REGISTRY.keys())


_evals_discovered = False
_models_discovered = False


def _ensure_evals_discovered():
    global _evals_discovered
    if _evals_discovered:
        return
    import importlib
    import pkgutil

    import genbench.evals as pkg

    for _, mod_name, _ in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"genbench.evals.{mod_name}")
    _evals_discovered = True


def _ensure_models_discovered():
    global _models_discovered
    if _models_discovered:
        return
    import importlib
    import pkgutil

    import genbench.models as pkg

    for _, mod_name, _ in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"genbench.models.{mod_name}")
    _models_discovered = True
