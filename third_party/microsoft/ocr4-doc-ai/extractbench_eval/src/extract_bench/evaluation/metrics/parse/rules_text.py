"""Text presence, order, and baseline test rules."""

import re
from typing import cast

from fuzzysearch import find_near_matches
from rapidfuzz import fuzz

from extract_bench.evaluation.metrics.parse.rules_base import (
    ParseTestRule,
    _strip_and_replace_latex,
)
from extract_bench.evaluation.metrics.parse.test_types import ParseRuleType
from extract_bench.evaluation.metrics.parse.utils import normalize_text, normalize_text_light
from extract_bench.test_cases.parse_rule_schemas import (
    ParseBaselineRule,
    ParseOrderRule,
    ParsePresenceRule,
)


class TextPresenceRule(ParseTestRule):
    """Test rule for text presence/absence."""

    def __init__(self, rule_data: ParsePresenceRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParsePresenceRule, self._rule_data)

        if self.type not in {ParseRuleType.PRESENT.value, ParseRuleType.ABSENT.value}:
            raise ValueError(f"Invalid type for TextPresenceRule: {self.type}")

        # Check if we should use light normalization that preserves formatting tags
        self.keep_formatting = rule_data.keep_formatting_text_normalisation
        normalize_fn = normalize_text_light if self.keep_formatting else normalize_text

        self.text = normalize_fn(rule_data.text)
        if not self.text.strip():
            raise ValueError("Text field cannot be empty")

        self.case_sensitive = rule_data.case_sensitive
        self.first_n = rule_data.first_n
        self.last_n = rule_data.last_n
        self.count = rule_data.count

        if self.count is not None:
            if not isinstance(self.count, int):
                raise ValueError("Count field must be an integer when provided")
            if self.count < 0:
                raise ValueError("Count field cannot be negative")

    def _count_non_overlapping_fuzzy_matches(self, query: str, content: str) -> int:
        """Count non-overlapping fuzzy matches for query in content.

        We keep matches non-overlapping to avoid over-counting near-duplicate
        windows for the same textual occurrence.
        """
        max_distance = min(self.max_diffs, 15)
        matches = sorted(
            find_near_matches(query, content, max_l_dist=max_distance),
            key=lambda match: (match.start, match.end),
        )

        non_overlapping_count = 0
        last_end = -1
        for match in matches:
            if match.start >= last_end:
                non_overlapping_count += 1
                last_end = match.end
        return non_overlapping_count

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        """Check if text is present or absent in markdown."""
        reference_query = self.text

        # When keep_formatting is enabled, we must re-normalize content with light normalization
        # since the pre-normalized content from the metric uses standard normalization
        if self.keep_formatting:
            # Always use light normalization when testing formatting
            normalized_content = normalize_text_light(md_content)
            # Backward compatibility: standard normalize_text lowercases by default,
            # so keep-formatting mode mirrors that behavior unless case sensitivity
            # is explicitly disabled below.
            reference_query = reference_query.lower()
            normalized_content = normalized_content.lower()
        elif normalized_content is None:
            # Use pre-normalized content if provided, otherwise normalize
            normalized_content = normalize_text(md_content)

        if not self.case_sensitive:
            reference_query = reference_query.lower()
            normalized_content = normalized_content.lower()

        # Apply first_n/last_n if specified
        if self.first_n and self.last_n:
            normalized_content = normalized_content[: self.first_n] + normalized_content[-self.last_n :]
        elif self.first_n:
            normalized_content = normalized_content[: self.first_n]
        elif self.last_n:
            normalized_content = normalized_content[-self.last_n :]

        # Threshold for fuzzy matching derived from max_diffs.
        # Floor at 0.7 so short queries (e.g. 2-3 chars) don't produce
        # wildly permissive thresholds that match almost anything.
        raw_threshold = 1.0 - (self.max_diffs / (len(reference_query) if len(reference_query) > 0 else 1))
        threshold = max(0.7, min(1.0, raw_threshold))
        best_ratio = fuzz.partial_ratio(reference_query, normalized_content) / 100.0

        if self.type == ParseRuleType.PRESENT.value:
            # Backward compatibility: count=None or count=0 keeps legacy behavior
            # (presence check only, regardless of number of occurrences).
            if self.count not in {None, 0}:
                if self.max_diffs == 0:
                    actual_count = normalized_content.count(reference_query)
                else:
                    actual_count = self._count_non_overlapping_fuzzy_matches(reference_query, normalized_content)

                if actual_count == self.count:
                    return True, ""
                msg = f"Expected '{reference_query[:40]}...' exactly {self.count} time(s), but found {actual_count}"
                return False, msg

            if best_ratio >= threshold:
                return True, ""
            else:
                msg = (
                    f"Expected '{reference_query[:40]}...' with threshold {threshold} "
                    f"but best match ratio was {best_ratio:.3f}"
                )
                return False, msg
        else:  # ABSENT
            if best_ratio < threshold:
                return True, ""
            else:
                msg = (
                    f"Expected absence of '{reference_query[:40]}...' with threshold {threshold} "
                    f"but best match ratio was {best_ratio:.3f}"
                )
                return False, msg


class BaselineRule(ParseTestRule):
    """Test rule for baseline quality checks (blank pages, repeats, character sets)."""

    def __init__(self, rule_data: ParseBaselineRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseBaselineRule, self._rule_data)

        self.max_length = rule_data.max_length
        self.max_length_skips_image_alt_tags = rule_data.max_length_skips_image_alt_tags
        self.max_repeats = rule_data.max_repeats
        self.check_disallowed_characters = rule_data.check_disallowed_characters

    def run(self, content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        """Run baseline quality checks."""
        base_content_len = len("".join(c for c in content if c.isalnum()).strip())

        # Blank page check
        if self.max_length is not None:
            if self.max_length_skips_image_alt_tags:
                # Remove markdown image tags
                content_for_length_check = re.sub(r"!\[.*?\]\(.*?\)", "", content)
                base_content_len = len("".join(c for c in content_for_length_check if c.isalnum()).strip())

            if base_content_len > self.max_length:
                return (
                    False,
                    f"{base_content_len} characters were output for a page we expected to be blank",
                )
            else:
                return True, ""

        # Check for empty content
        if base_content_len == 0:
            return False, "The text contains no alpha numeric characters"

        # Check for excessive repetition using sampled windows across the
        # document.  Sampling keeps this O(1) even on very large documents.
        if len(content) > 50:
            # Sample up to 5 windows of 200 chars evenly spaced through the doc
            sample_size = min(200, len(content))
            num_samples = min(5, max(1, len(content) // sample_size))
            step = max(1, (len(content) - sample_size) // max(1, num_samples - 1)) if num_samples > 1 else 0
            repetitive_samples = 0
            for i in range(num_samples):
                start = i * step
                window = content[start : start + sample_size]
                if len(set(window)) < 3:
                    repetitive_samples += 1
            # Fail if majority of samples are repetitive
            if repetitive_samples > num_samples // 2:
                return False, "Text appears to be excessively repetitive"

        # Check for disallowed characters (CJK, emoji, etc.)
        if self.check_disallowed_characters:
            pattern = re.compile(
                r"["
                r"\u4e00-\u9FFF"  # CJK Unified Ideographs
                r"\u3040-\u309F"  # Hiragana
                r"\u30A0-\u30FF"  # Katakana
                r"\U0001F600-\U0001F64F"  # Emoticons
                r"\U0001F300-\U0001F5FF"  # Miscellaneous Symbols
                r"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
                r"\U0001F1E0-\U0001F1FF"  # Regional Indicator Symbols
                r"]",
                flags=re.UNICODE,
            )
            matches = pattern.findall(content)
            if matches:
                return False, f"Text contains disallowed characters: {matches[:5]}"

        return True, ""


class TextOrderRule(ParseTestRule):
    """Test rule to verify that one text appears before another."""

    def __init__(self, rule_data: ParseOrderRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseOrderRule, self._rule_data)

        if self.type != ParseRuleType.ORDER.value:
            raise ValueError(f"Invalid type for TextOrderRule: {self.type}")

        # Check if we should use light normalization that preserves formatting tags
        self.keep_formatting = rule_data.keep_formatting_text_normalisation
        normalize_fn = normalize_text_light if self.keep_formatting else normalize_text

        # Canonicalize LaTeX in query strings so "$...$" and "LATEX" authored
        # expectations are matched consistently against content.
        before_for_match = _strip_and_replace_latex(rule_data.before)
        after_for_match = _strip_and_replace_latex(rule_data.after)

        # Import SentenceBagRule lazily to access _MULTI_DOT_PATTERN
        from extract_bench.evaluation.metrics.parse.rules_bag import SentenceBagRule

        self.before = re.sub(r" +", " ", SentenceBagRule._MULTI_DOT_PATTERN.sub(" ", normalize_fn(before_for_match)))
        self.after = re.sub(r" +", " ", SentenceBagRule._MULTI_DOT_PATTERN.sub(" ", normalize_fn(after_for_match)))
        if not self.before.strip():
            raise ValueError("Before field cannot be empty")
        if not self.after.strip():
            raise ValueError("After field cannot be empty")
        if self.max_diffs > len(self.before) // 2 or self.max_diffs > len(self.after) // 2:
            raise ValueError("Max diffs is too large for this test, greater than 50% of the search string")

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        """Check if 'before' text appears before 'after' text.

        When multiple instances exist, we check that the FIRST occurrence of 'before'
        appears before the LAST occurrence of 'after'. This handles cases where the same
        text may appear multiple times in the document.
        """
        from extract_bench.evaluation.metrics.parse.rules_bag import SentenceBagRule

        # Order matching must canonicalize LaTeX consistently between rule text and content.
        # Re-normalize from raw markdown so inline math ($...$) becomes a stable LATEX token.
        content_for_match = _strip_and_replace_latex(md_content)
        if self.keep_formatting:
            normalized_content = normalize_text_light(content_for_match)
        else:
            normalized_content = normalize_text(content_for_match)
        normalized_content = re.sub(r" +", " ", SentenceBagRule._MULTI_DOT_PATTERN.sub(" ", normalized_content))

        # OPTIMIZATION: Try exact match first (O(n) vs O(n*m) for fuzzy)
        # This provides ~10-100x speedup for rules that match exactly
        # Use find() for FIRST occurrence of before, rfind() for LAST occurrence of after
        before_pos = normalized_content.find(self.before)
        after_pos = normalized_content.rfind(self.after)

        if before_pos != -1 and after_pos != -1 and before_pos < after_pos:
            # Fast path: both found exactly and in correct order
            return True, ""

        # Slow path: fall back to fuzzy matching
        # instead of max_diffs = 2, use a relative distance
        # here as the annotation fail quite a lot on typo / unicode chars
        # Cap at 30 to avoid combinatorial explosion in fuzzysearch for long strings
        # (e.g. 7k pattern on 30k text with max_l_dist=350 is ~73B operations)
        before_max_dist = min(max(len(self.before) // 20, self.max_diffs), 15)
        after_max_dist = min(max(len(self.after) // 20, self.max_diffs), 15)

        # Only do expensive fuzzy search if exact match failed
        if before_pos == -1:
            before_matches = find_near_matches(self.before, normalized_content, max_l_dist=before_max_dist)
        else:
            # Create a fake match object for the exact match (first occurrence)
            before_matches = [type("Match", (), {"start": before_pos, "end": before_pos + len(self.before)})()]

        if after_pos == -1:
            after_matches = find_near_matches(self.after, normalized_content, max_l_dist=after_max_dist)
        else:
            # Create a fake match object for the exact match (last occurrence)
            after_matches = [type("Match", (), {"start": after_pos, "end": after_pos + len(self.after)})()]

        if not before_matches:
            return (
                False,
                f"'before' text '{self.before[:40]}...' not found with max_l_dist {before_max_dist}",
            )
        if not after_matches:
            return (
                False,
                f"'after' text '{self.after[:40]}...' not found with max_l_dist {after_max_dist}",
            )

        # Get FIRST occurrence of before (earliest start position)
        first_before_match = min(before_matches, key=lambda m: m.start)
        # Get LAST occurrence of after (latest start position)
        last_after_match = max(after_matches, key=lambda m: m.start)

        if first_before_match.start < last_after_match.start:
            return True, ""

        return (
            False,
            f"Could not find a location where '{self.before}...' appears before '{self.after}...'. "
            f"First 'before' at position {first_before_match.start}, "
            f"last 'after' at position {last_after_match.start}.",
        )
