"""Layout label mapper interfaces, registry, and concrete mappings."""

# Ensure mapper classes are registered via import side-effects.
from . import mappers as _mappers  # noqa: F401
from .base import LayoutLabelMapper, MappingContext
from .projection import project_layout_predictions
from .registry import (
    build_mapping_context,
    list_layout_label_mappers,
    register_layout_label_mapper,
    resolve_layout_label_mapper,
)

__all__ = [
    "LayoutLabelMapper",
    "MappingContext",
    "build_mapping_context",
    "list_layout_label_mappers",
    "project_layout_predictions",
    "register_layout_label_mapper",
    "resolve_layout_label_mapper",
]
