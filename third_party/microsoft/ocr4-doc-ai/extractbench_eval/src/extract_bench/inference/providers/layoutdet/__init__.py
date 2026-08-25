"""Layout detection providers imported lazily for registry side effects."""

import importlib
import logging

logger = logging.getLogger(__name__)

_PROVIDER_MODULES = [
    "chandra",
    "docling",
    "dots_ocr",
    "layout_v3",
    "layout_v3_byoc",
    "paddle",
    "qwen3vl",
    "surya",
    "yolo",
]

for _mod in _PROVIDER_MODULES:
    try:
        importlib.import_module(f"extract_bench.inference.providers.layoutdet.{_mod}")
    except ImportError as exc:
        # A missing third-party SDK is expected on a core-only install; a
        # failure originating inside this package is a packaging bug and must
        # be loud, not silently skipped.
        if (exc.name or "").startswith("extract_bench"):
            logger.warning("Layout provider %s failed to import: %s", _mod, exc)
        else:
            logger.debug("Skipping layout provider %s (missing dependency)", _mod)

__all__ = _PROVIDER_MODULES
