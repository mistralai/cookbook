"""Registry for layout label mapper strategies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from extract_bench.evaluation.layout_adapters.registry import resolve_layout_provider_name
from extract_bench.evaluation.layout_label_mappers.base import (
    LayoutLabelMapper,
    MappingContext,
)
from extract_bench.layout_label_mapping import detect_llamaparse_label_version
from extract_bench.schemas.layout_detection_output import LayoutDetectionModel, LayoutOutput
from extract_bench.schemas.pipeline_io import InferenceResult


@dataclass(frozen=True)
class _LayoutLabelMapperRegistration:
    keys: tuple[str, ...]
    priority: int
    mapper_cls: type[LayoutLabelMapper]


_LAYOUT_LABEL_MAPPER_REGISTRY: list[_LayoutLabelMapperRegistration] = []
_DEFAULT_LAYOUT_MAPPER_KEY = "__default__"


def register_layout_label_mapper(
    *mapper_keys: str,
    priority: int = 0,
) -> Callable[[type[LayoutLabelMapper]], type[LayoutLabelMapper]]:
    """Register a layout label mapper for one or more keys."""
    if not mapper_keys:
        raise ValueError("register_layout_label_mapper requires at least one key")

    def decorator(cls: type[LayoutLabelMapper]) -> type[LayoutLabelMapper]:
        existing_keys = {key for entry in _LAYOUT_LABEL_MAPPER_REGISTRY for key in entry.keys}
        for key in mapper_keys:
            if key in existing_keys:
                raise ValueError(f"Layout label mapper already registered for key '{key}'")

        _LAYOUT_LABEL_MAPPER_REGISTRY.append(
            _LayoutLabelMapperRegistration(
                keys=tuple(mapper_keys),
                priority=priority,
                mapper_cls=cls,
            )
        )
        return cls

    return decorator


def list_layout_label_mappers() -> list[str]:
    """List registered mapper keys."""
    keys = {key for entry in _LAYOUT_LABEL_MAPPER_REGISTRY for key in entry.keys}
    return sorted(keys)


def build_mapping_context(
    inference_result: InferenceResult,
    layout_output: LayoutOutput,
    *,
    test_case_ontology: str | None = None,
    cli_ontology: str | None = None,
) -> MappingContext:
    """Build mapping context for label mapper resolution."""
    raw_label_version = None
    if layout_output.model == LayoutDetectionModel.LLAMAPARSE:
        labels = [pred.label for pred in layout_output.predictions if pred.label]
        raw_label_version = detect_llamaparse_label_version(labels)

    return MappingContext(
        provider_name=resolve_layout_provider_name(inference_result),
        pipeline_name=inference_result.pipeline_name,
        model=layout_output.model,
        raw_output=inference_result.raw_output,
        layout_output=layout_output,
        raw_label_version=raw_label_version,
        test_case_ontology=test_case_ontology,
        cli_ontology=cli_ontology,
    )


def resolve_layout_label_mapper(context: MappingContext) -> LayoutLabelMapper:
    """Resolve mapper by provider key/model key first, matcher fallback second."""
    candidate_keys: list[str] = [f"model:{context.model.value}"]
    if context.provider_name:
        candidate_keys.insert(0, context.provider_name)

    keyed_matches: list[_LayoutLabelMapperRegistration] = []
    for entry in _LAYOUT_LABEL_MAPPER_REGISTRY:
        if any(key in entry.keys for key in candidate_keys):
            keyed_matches.append(entry)

    if keyed_matches:
        chosen = sorted(keyed_matches, key=lambda entry: entry.priority, reverse=True)[0]
        return chosen.mapper_cls()

    matcher_matches: list[_LayoutLabelMapperRegistration] = []
    for entry in _LAYOUT_LABEL_MAPPER_REGISTRY:
        if _DEFAULT_LAYOUT_MAPPER_KEY in entry.keys:
            continue
        if entry.mapper_cls.matches(context):
            matcher_matches.append(entry)

    if matcher_matches:
        chosen = sorted(matcher_matches, key=lambda entry: entry.priority, reverse=True)[0]
        return chosen.mapper_cls()

    default_matches = [entry for entry in _LAYOUT_LABEL_MAPPER_REGISTRY if _DEFAULT_LAYOUT_MAPPER_KEY in entry.keys]
    if default_matches:
        chosen = sorted(default_matches, key=lambda entry: entry.priority, reverse=True)[0]
        return chosen.mapper_cls()

    keys = ", ".join(candidate_keys)
    available = ", ".join(list_layout_label_mappers())
    raise ValueError(f"No layout label mapper found for keys [{keys}]. Available mapper keys: {available}")
