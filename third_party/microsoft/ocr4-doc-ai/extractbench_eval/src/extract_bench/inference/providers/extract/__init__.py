"""Extract providers imported for registry side effects.

Imports are best-effort so a core ExtractBench install does not require
optional provider SDKs such as ``extend-ai``; failures that originate inside
this package are logged loudly instead of being skipped.
"""

import importlib
import logging

logger = logging.getLogger(__name__)

_PROVIDER_MODULES = [
    "anthropic_direct",
    "azure_content_understanding",
    "claude_code_extract",
    "codex_code_extract",
    "codex_deepseek_extract",
    "collaborative_direct",
    "datalab",
    "deepseek_extract",
    "extend",
    "gemini_direct",
    "glm_extract",
    "landingai",
    "lift",
    "llamaextract_v2_api",
    "nuextract3_extract",
    "openai_responses",
    "pulse",
    "reducto",
    "table_codegen_extract",
    "text_only",
    "vllm_extract",
    "mistral_ocr4",
]

for _mod in _PROVIDER_MODULES:
    try:
        importlib.import_module(f"extract_bench.inference.providers.extract.{_mod}")
    except ImportError as exc:
        # A missing third-party SDK is expected on a core-only install; a
        # failure originating inside this package is a packaging bug and must
        # be loud, not silently skipped.
        if (exc.name or "").startswith("extract_bench"):
            logger.warning("Extract provider %s failed to import: %s", _mod, exc)
        else:
            logger.debug("Skipping extract provider %s (missing dependency)", _mod)
