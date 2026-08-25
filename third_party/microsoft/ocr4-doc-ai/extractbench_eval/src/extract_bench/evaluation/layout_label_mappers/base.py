"""Base abstractions for layout label mappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from extract_bench.layout_label_mapping import map_canonical_label_to_target_ontology
from extract_bench.schemas.layout_detection_output import (
    LayoutDetectionModel,
    LayoutOutput,
    LayoutPrediction,
)
from extract_bench.schemas.layout_ontology import CanonicalLabel


@dataclass(frozen=True)
class MappingContext:
    """Context used to resolve provider/model-specific label mappers."""

    provider_name: str | None
    pipeline_name: str
    model: LayoutDetectionModel
    raw_output: dict[str, Any]
    layout_output: LayoutOutput
    raw_label_version: str | None = None
    test_case_ontology: str | None = None
    cli_ontology: str | None = None


class LayoutLabelMapper(ABC):
    """Strategy interface for mapping raw provider labels to evaluation ontologies."""

    @classmethod
    def get_mapper_keys(cls) -> tuple[str, ...]:
        """Mapper registry keys this class supports."""
        return ()

    @classmethod
    def matches(cls, context: MappingContext) -> bool:
        """Optional context-based fallback matcher."""
        del context
        return False

    @abstractmethod
    def to_canonical(
        self,
        label: str,
        prediction: LayoutPrediction,
        context: MappingContext,
    ) -> CanonicalLabel:
        """Map raw provider label to canonical ontology label."""

    def should_include_prediction(
        self,
        prediction: LayoutPrediction,
        context: MappingContext,
    ) -> bool:
        """Optional prediction-level filtering hook."""
        del prediction, context
        return True

    def to_target_ontology(self, canonical: CanonicalLabel, target_ontology: str) -> str:
        """Map canonical label to requested evaluation ontology."""
        return map_canonical_label_to_target_ontology(canonical, target_ontology)
