"""Configuration for LLM normalization of chart evaluation metrics."""

from __future__ import annotations

import os
from enum import StrEnum


class NormalizationMode(StrEnum):
    """LLM normalization mode, controlled by EXTRACT_BENCH_LLM_NORMALIZATION env var."""

    OFF = "off"
    JUDGE = "judge"


# Anthropic model used for LLM-as-judge normalization
JUDGE_MODEL = "claude-haiku-4-5-20251001"

# Confidence threshold for accepting an LLM label match
LABEL_CONFIDENCE_THRESHOLD = 0.7

# Numeric tolerance for value comparison (relative, e.g. 0.02 = 2%)
VALUE_RELATIVE_TOLERANCE = 0.02

# Maximum number of value pairs to send per API call
VALUE_BATCH_SIZE = 30

# Safety limit: maximum API calls per normalizer instance.
# Prevents runaway costs on unexpectedly large datasets.
# A typical charts_core run (49 test cases) needs ~65 calls.
MAX_API_CALLS_PER_NORMALIZER = 500


def get_normalization_mode() -> NormalizationMode:
    """Read EXTRACT_BENCH_LLM_NORMALIZATION env var and return the mode.

    Returns NormalizationMode.OFF unless the mode is explicitly requested.
    Anything unrecognized is OFF too: this switch spends money on judge calls,
    so it has to be asked for by name rather than fallen into by typo.
    """
    raw = os.environ.get("EXTRACT_BENCH_LLM_NORMALIZATION", "off").strip().lower()
    try:
        return NormalizationMode(raw)
    except ValueError:
        return NormalizationMode.OFF


def get_anthropic_api_key() -> str | None:
    """Return ANTHROPIC_API_KEY from environment, or None if not set."""
    return os.environ.get("ANTHROPIC_API_KEY")
