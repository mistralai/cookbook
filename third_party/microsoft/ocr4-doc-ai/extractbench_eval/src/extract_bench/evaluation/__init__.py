"""Evaluation package exports."""

__all__ = ["EvaluationRunner"]


def __getattr__(name: str) -> object:  # pragma: no cover - simple lazy import shim.
    if name == "EvaluationRunner":
        from .runner import EvaluationRunner

        return EvaluationRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
