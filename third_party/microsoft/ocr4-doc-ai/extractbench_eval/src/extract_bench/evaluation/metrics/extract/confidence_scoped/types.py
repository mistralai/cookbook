from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ArrayAlignment = dict[int, tuple[int, float, str]]


@dataclass
class EvaluationContext:
    """Per-document evaluation state: alignment cache (by path and
    ``pred=>gt`` pair) plus exclusive-claim sets so a GT leaf/element
    validates at most one emitted field. Claims are granted greedily in
    confidence-descending order — with duplicate predictions, the
    highest-confidence citation wins the credit (ties: citation order), so
    the surplus False lands on the copy the calibration signal should
    penalize."""

    alignments: dict[str, ArrayAlignment] = field(default_factory=dict)
    claimed_grouped: dict[str, set[str]] = field(default_factory=dict)
    claimed_scalar: dict[str, set[int]] = field(default_factory=dict)


@dataclass(frozen=True)
class EmittedField:
    path: str
    value: Any
    confidence: float
    cited_text: str


@dataclass(frozen=True)
class MatchResult:
    correct: bool
    expected_path: str
    expected_value: Any
    comparison_mode: str
    alignment_mode: str
    alignment_score: float


@dataclass(frozen=True)
class ConfidenceFieldRow:
    """Dashboard/evaluator row for one confidence-bearing emitted field."""

    field_path: str
    field_pattern: str
    predicted_path: str | None
    expected_value: Any
    predicted_value: Any
    confidence: float | None
    correct: bool
    comparison_score: float
    comparison_mode: str
    verified: bool
    alignment_mode: str = ""
    alignment_score: float = 0.0
    cited_text: str = ""
    schema_valid: bool = True
    matched_gt_path: str = ""
