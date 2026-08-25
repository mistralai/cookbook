"""Answer comparison metric for QA evaluation."""

import re
from typing import Any

from extract_bench.schemas.evaluation import MetricValue


class AnswerComparisonMetric:
    """Metric for comparing predicted answers with expected answers."""

    def compare(
        self,
        predicted: str,
        expected: str,
        question_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> MetricValue:
        """
        Compare predicted answer with expected answer.

        :param predicted: Predicted answer from LLM
        :param expected: Expected answer from test case
        :param question_type: Type of question ("single_choice", "multiple_choice", "numerical")
        :param metadata: Optional metadata (tolerance, options, etc.)
        :return: MetricValue with pass/fail and metadata
        """
        if question_type == "single_choice":
            return self._compare_single_choice(predicted, expected, metadata)
        elif question_type == "multiple_choice":
            return self._compare_multiple_choice(predicted, expected, metadata)
        elif question_type == "numerical":
            return self._compare_numerical(predicted, expected, metadata)
        elif question_type == "free_text":
            return self._compare_free_text(predicted, expected, metadata)
        else:
            return MetricValue(
                metric_name="qa_answer_match",
                value=0.0,
                metadata={
                    "passed": False,
                    "predicted": predicted,
                    "expected": expected,
                    "error": f"Unknown question type: {question_type}",
                },
            )

    def _compare_single_choice(self, predicted: str, expected: str, metadata: dict[str, Any] | None) -> MetricValue:
        """Compare single choice answers."""
        # Normalize both answers
        pred_normalized = self._normalize_answer(predicted)
        exp_normalized = self._normalize_answer(expected)

        # Try exact match first
        if pred_normalized == exp_normalized:
            return MetricValue(
                metric_name="qa_answer_match",
                value=1.0,
                metadata={
                    "passed": True,
                    "predicted": predicted,
                    "expected": expected,
                    "question_type": "single_choice",
                },
            )

        # Try extracting letter from predicted answer
        pred_letter = self._extract_letter(predicted)
        exp_letter = self._extract_letter(expected)

        if pred_letter and exp_letter and pred_letter == exp_letter:
            return MetricValue(
                metric_name="qa_answer_match",
                value=1.0,
                metadata={
                    "passed": True,
                    "predicted": predicted,
                    "expected": expected,
                    "question_type": "single_choice",
                    "matched_letter": pred_letter,
                },
            )

        # Case-insensitive comparison
        if pred_normalized.lower() == exp_normalized.lower():
            return MetricValue(
                metric_name="qa_answer_match",
                value=1.0,
                metadata={
                    "passed": True,
                    "predicted": predicted,
                    "expected": expected,
                    "question_type": "single_choice",
                },
            )

        return MetricValue(
            metric_name="qa_answer_match",
            value=0.0,
            metadata={
                "passed": False,
                "predicted": predicted,
                "expected": expected,
                "question_type": "single_choice",
            },
        )

    def _compare_multiple_choice(self, predicted: str, expected: str, metadata: dict[str, Any] | None) -> MetricValue:
        """Compare multiple choice answers."""
        # Parse answers into sets (order-independent)
        pred_set = self._parse_multiple_choice(predicted)
        exp_set = self._parse_multiple_choice(expected)

        # Compare sets
        passed = pred_set == exp_set
        value = 1.0 if passed else 0.0

        return MetricValue(
            metric_name="qa_answer_match",
            value=value,
            metadata={
                "passed": passed,
                "predicted": predicted,
                "expected": expected,
                "predicted_set": sorted(pred_set),
                "expected_set": sorted(exp_set),
                "question_type": "multiple_choice",
            },
        )

    def _compare_numerical(self, predicted: str, expected: str, metadata: dict[str, Any] | None) -> MetricValue:
        """Compare numerical answers with optional tolerance."""
        # Extract numbers from strings
        pred_num = self._extract_number(predicted)
        exp_num = self._extract_number(expected)

        if pred_num is None or exp_num is None:
            return MetricValue(
                metric_name="qa_answer_match",
                value=0.0,
                metadata={
                    "passed": False,
                    "predicted": predicted,
                    "expected": expected,
                    "error": "Could not extract numbers from answers",
                    "question_type": "numerical",
                },
            )

        # Get tolerance from metadata
        tolerance = 0.0
        if metadata:
            tolerance_val = metadata.get("tolerance")
            if tolerance_val is not None:
                try:
                    tolerance = float(tolerance_val)
                except (ValueError, TypeError):
                    pass

        # Compare with tolerance
        diff = abs(pred_num - exp_num)
        passed = diff <= tolerance
        value = 1.0 if passed else 0.0

        return MetricValue(
            metric_name="qa_answer_match",
            value=value,
            metadata={
                "passed": passed,
                "predicted": predicted,
                "expected": expected,
                "predicted_number": pred_num,
                "expected_number": exp_num,
                "difference": diff,
                "tolerance": tolerance,
                "question_type": "numerical",
            },
        )

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer string for comparison, matching official FinMME format."""
        # Use the same normalization as official FinMME eval
        normalized = (
            answer.replace("**", "")
            .replace(":", "")
            .replace("$\\boxed{", "")
            .replace("}$", "")
            .replace("\\$", "")
            .replace("$", "")
            .replace("{", "")
            .replace("\\boxed", "")
        )
        return normalized.strip()

    def _extract_letter(self, answer: str) -> str | None:
        """Extract letter code (A, B, C, etc.) from answer."""
        # Look for single letter at start or in parentheses
        match = re.search(r"\b([A-Z])\b", answer.upper())
        if match:
            return match.group(1)
        return None

    def _parse_multiple_choice(self, answer: str) -> set[str]:
        """
        Parse multiple choice answer into set of letters.

        Matches the official FinMME eval logic: extract any character
        that's a valid choice letter (A-Z).
        """
        # Normalize answer
        normalized = self._normalize_answer(answer.upper())

        # Extract any character that's a valid choice letter (A-Z)
        # This matches the official FinMME eval script logic
        valid_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        letters = {c for c in normalized if c in valid_letters}

        return letters

    def _compare_free_text(self, predicted: str, expected: str, metadata: dict[str, Any] | None) -> MetricValue:
        """Compare free-text answers with case-insensitive exact match."""
        pred_normalized = predicted.strip().lower()
        exp_normalized = expected.strip().lower()

        if "," in exp_normalized:
            pred_set = {s.strip() for s in pred_normalized.split(",")}
            exp_set = {s.strip() for s in exp_normalized.split(",")}
            passed = pred_set == exp_set
        else:
            passed = pred_normalized == exp_normalized

        return MetricValue(
            metric_name="qa_answer_match",
            value=1.0 if passed else 0.0,
            metadata={
                "passed": passed,
                "predicted": predicted,
                "expected": expected,
                "question_type": "free_text",
            },
        )

    def _extract_number(self, text: str) -> float | None:
        """Extract number from text string."""
        # Remove common prefixes
        text = re.sub(
            r"^(answer|answer:|the answer is|the answer:)\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = text.strip()

        # Try to find number (including decimals, negatives, scientific notation)
        # Match numbers with optional commas, decimals, negatives
        pattern = r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?"
        match = re.search(pattern, text)
        if match:
            # Remove commas before parsing
            num_str = match.group(0).replace(",", "")
            try:
                return float(num_str)
            except ValueError:
                pass

        return None
