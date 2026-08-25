"""Product-specific evaluators."""

from extract_bench.evaluation.evaluators.base import BaseEvaluator

__all__ = ["BaseEvaluator", "ExtractEvaluator", "ParseEvaluator"]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name == "ExtractEvaluator":
        from extract_bench.evaluation.evaluators.extract import ExtractEvaluator

        return ExtractEvaluator
    if name == "ParseEvaluator":
        from extract_bench.evaluation.evaluators.parse import ParseEvaluator

        return ParseEvaluator
    raise AttributeError(name)
