"""Layout adapter registry and provider-specific adapter bindings."""

from .base import LayoutAdapter
from .registry import (
    create_layout_adapter_for_result,
    list_layout_adapters,
    register_layout_adapter,
    resolve_layout_provider_name,
)

__all__ = [
    "LayoutAdapter",
    "create_layout_adapter_for_result",
    "list_layout_adapters",
    "register_layout_adapter",
    "resolve_layout_provider_name",
]
