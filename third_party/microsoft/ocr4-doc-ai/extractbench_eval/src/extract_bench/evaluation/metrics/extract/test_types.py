"""Test type definitions for extract evaluation."""

from enum import Enum


class ExtractTestType(str, Enum):
    """Test types for extract evaluation."""

    ARRAY_LENGTH = "array_length"
    ARRAY_HEAD = "array_head"
    ARRAY_TAIL = "array_tail"
