"""Sentence, word, and digit bag test rules."""

import hashlib
import logging
import re
from collections import Counter
from typing import cast

from extract_bench.evaluation.metrics.parse.rules_base import (
    ParseTestRule,
    SentenceBagRuleData,
    WordBagRuleData,
    _augment_with_table_cell_text,
    _strip_and_replace_latex,
    _strip_fenced_code_blocks,
    _strip_html_tables_and_content,
    _unescape_html_entities,
)
from extract_bench.evaluation.metrics.parse.test_types import ParseRuleType
from extract_bench.evaluation.metrics.parse.utils import normalize_text
from extract_bench.test_cases.parse_rule_schemas import (
    ParseBagOfDigitPercentRule,
    ParseExtraContentRule,
    ParseMissingSentencePercentRule,
    ParseMissingSentenceRule,
    ParseMissingSpecificSentenceRule,
    ParseMissingSpecificWordRule,
    ParseMissingWordPercentRule,
    ParseMissingWordRule,
    ParseTooManySentenceOccurrencePercentRule,
    ParseTooManySentenceOccurrenceRule,
    ParseTooManyWordOccurrencePercentRule,
    ParseTooManyWordOccurrenceRule,
    ParseUnexpectedSentencePercentRule,
    ParseUnexpectedSentenceRule,
    ParseUnexpectedWordPercentRule,
    ParseUnexpectedWordRule,
)

logger = logging.getLogger(__name__)

# Matches HTML tags including their attributes (e.g. <td colspan="2">)
_HTML_TAG_WITH_ATTRS_PATTERN = re.compile(r"<[^>]+>")

# CJK Unicode ranges for character-level word splitting
_CJK_RANGES = (
    ("\u4e00", "\u9fff"),  # CJK Unified Ideographs
    ("\u3400", "\u4dbf"),  # CJK Extension A
    ("\uf900", "\ufaff"),  # CJK Compatibility Ideographs
    ("\u3040", "\u309f"),  # Hiragana
    ("\u30a0", "\u30ff"),  # Katakana
    ("\uac00", "\ud7af"),  # Hangul Syllables
)


def _is_cjk_char(ch: str) -> bool:
    """Return True if *ch* is a CJK/Japanese/Korean character."""
    return any(lo <= ch <= hi for lo, hi in _CJK_RANGES)


# Unicode-aware word tokenization: matches sequences of Unicode letters
# and/or digits (using Unicode categories L and N), properly handling
# accented characters, CJK, etc.
# Aligned with JS annotation tool which uses /[\p{L}\p{N}]+/gu — consecutive
# CJK characters stay grouped as a single token.
_UNICODE_WORD_PATTERN = re.compile(r"[\w]+", re.UNICODE)


def _tokenize_unicode_words(text: str, min_length: int = 2) -> list[str]:
    """Tokenize *text* into words using Unicode-aware rules.

    - Latin/accented/Cyrillic text is split on word boundaries as usual.
    - Consecutive CJK characters are kept together as a single token,
      matching the JS annotation tool's ``/[\\p{L}\\p{N}]+/gu`` behaviour.
    - Words shorter than *min_length* are discarded (CJK single characters
      always pass regardless of *min_length*).
    """
    raw_tokens = _UNICODE_WORD_PATTERN.findall(text)
    words: list[str] = []
    for token in raw_tokens:
        has_cjk = any(_is_cjk_char(ch) for ch in token)
        if has_cjk:
            # Group consecutive CJK characters together; non-CJK runs are
            # separate tokens.  E.g. "検査abc結果" → ["検査", "abc", "結果"]
            buf: list[str] = []
            in_cjk = False
            for ch in token:
                is_cjk = _is_cjk_char(ch)
                if is_cjk != in_cjk and buf:
                    run = "".join(buf)
                    if in_cjk or len(run) >= min_length:
                        words.append(run)
                    buf = []
                buf.append(ch)
                in_cjk = is_cjk
            if buf:
                run = "".join(buf)
                if in_cjk or len(run) >= min_length:
                    words.append(run)
        else:
            if len(token) >= min_length:
                words.append(token)
    return words


def _word_boundary_count(word: str, text: str) -> int:
    """Count word-boundary-delimited occurrences of *word* in *text*.

    Uses Unicode-aware boundaries: for CJK words (single or multi-char),
    count raw substring occurrences.  For Latin words, uses ``\\b``.
    """
    if any(_is_cjk_char(ch) for ch in word):
        # CJK: count raw substring occurrences (no word boundary concept)
        return text.count(word)
    return len(re.findall(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text, re.UNICODE))


class SentenceBagRule(ParseTestRule):
    """Shared utilities for sentence-bag parse rules.

    ⚠️  SYNC: Splitting & boundary logic must stay aligned with the JS annotation tool in
    text_annotation_tools/toBagOfSentences.js (markdownToBagOfSentences).
    If you change orune, update the other.
    """

    MIN_SENTENCE_LENGTH = 7
    _LEADING_MARKDOWN_PATTERN = re.compile(r"^(?:\s*(?:#{1,6}\s+|>\s+|[*+-]\s+|\d+[.)]\s+))+")
    _HTML_TAG_PATTERN = re.compile(r"</?[^>]+>")
    _AUTOLINK_PATTERN = re.compile(r"<((?:https?://|mailto:)[^>\s]+|[^>@\s]+@[^>@\s]+\.[^>@\s]+)>", re.IGNORECASE)
    _MULTI_DOT_PATTERN = re.compile(r"\.(?:\s*\.)+")
    # Split on newlines, non-numeric periods, runs of !/?  **and** CJK / Asian
    # sentence separators.  Some (。，、！？；) may already have been normalised
    # to ASCII by normalize_text, but we keep the originals for safety.
    #   。 \u3002  、 \u3001  ， \uFF0C  ！ \uFF01  ？ \uFF1F  ； \uFF1B
    #   … \u2026  ‥ \u2025  ⋯ \u22EF
    _SENTENCE_SPLIT_PATTERN = re.compile(
        r"\n+|(?<!\d)\.(?!\d)|[!?]+|[\u3002\u3001\uFF0C\uFF01\uFF1F\uFF1B\u2026\u2025\u22EF]+"
    )
    _BOUNDARY_PUNCT_PATTERN = re.compile(r"^[^\w]+|[^\w]+$")
    # Matches markdown image tags: ![alt text](url) — stripped entirely so alt text
    # does not produce spurious sentence/word matches.
    _MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]*\)")

    def __init__(self, rule_data: SentenceBagRuleData | dict, expected_type: str):
        super().__init__(rule_data)
        rule_data = cast(SentenceBagRuleData, self._rule_data)

        if self.type != expected_type:
            raise ValueError(f"Invalid type for {self.__class__.__name__}: {self.type}")

        bag_of_sentence = rule_data.bag_of_sentence
        logger.debug(
            "Initializing %s with bag_of_sentence type=%s size=%s",
            self.__class__.__name__,
            type(bag_of_sentence).__name__,
            len(bag_of_sentence) if isinstance(bag_of_sentence, dict) else None,
        )

        if not isinstance(bag_of_sentence, dict) or not bag_of_sentence:
            raise ValueError("bag_of_sentence must be a non-empty dictionary")

        self.sentence_bag: Counter[str] = Counter()
        for sentence, occurrences in bag_of_sentence.items():
            if not isinstance(sentence, str) or not sentence.strip():
                raise ValueError("bag_of_sentence keys must be non-empty strings")
            if not isinstance(occurrences, int):
                raise ValueError("bag_of_sentence values must be integers")
            if occurrences < 0:
                raise ValueError("bag_of_sentence values cannot be negative")

            normalized_sentence = self._normalize_sentence_fragment(sentence)
            if not normalized_sentence:
                # Skip noisy fragments (e.g., dot leaders, formatting-only tokens, very short text)
                # so large heterogeneous datasets do not fail rule initialization on one bad entry.
                logger.debug(
                    ("Skipping bag_of_sentence entry after normalization: original=%r occurrences=%d"),
                    sentence,
                    occurrences,
                )
                continue
            self.sentence_bag[normalized_sentence] += occurrences

        if not self.sentence_bag:
            logger.warning(
                ("%s has no valid sentence after normalization; executing rule with empty bag_of_sentence"),
                self.__class__.__name__,
            )

    @staticmethod
    def _normalize_sentence_fragment(text: str) -> str:
        """Normalize sentence fragments for stable matching across layouts."""
        normalized_sentence = normalize_text(text).strip().strip(".")
        normalized_sentence = SentenceBagRule._MULTI_DOT_PATTERN.sub(" ", normalized_sentence)
        # Decode entities before stripping tags so encoded markup (&lt;h1&gt;) is treated
        # the same as raw markup (<h1>) and plain symbols (&lt;) normalize to '<'.
        unescaped_sentence = _unescape_html_entities(normalized_sentence)
        if unescaped_sentence != normalized_sentence:
            logger.debug("Decoded HTML entities in sentence fragment during normalization")
        normalized_sentence = unescaped_sentence
        normalized_sentence = SentenceBagRule._AUTOLINK_PATTERN.sub(r"\1", normalized_sentence)
        normalized_sentence = SentenceBagRule._HTML_TAG_PATTERN.sub(" ", normalized_sentence)
        normalized_sentence = SentenceBagRule._LEADING_MARKDOWN_PATTERN.sub("", normalized_sentence).strip()
        normalized_sentence = SentenceBagRule._BOUNDARY_PUNCT_PATTERN.sub("", normalized_sentence).strip()
        normalized_sentence = re.sub(r"\s+", " ", normalized_sentence)
        if len(normalized_sentence) < SentenceBagRule.MIN_SENTENCE_LENGTH:
            return ""
        return normalized_sentence

    @staticmethod
    def _normalize_full_text(md_content: str) -> str:
        """Normalize full markdown content for substring matching.

        Applies the same transformations used by ``_normalize_sentence_fragment``
        (normalize_text, HTML entity decoding, HTML tag stripping, multi-dot
        collapsing, whitespace collapse) but on the full document so that
        substring searches are compatible with the per-fragment normalization
        applied to ``bag_of_sentence`` keys during ``__init__``.
        """
        md_content = _strip_fenced_code_blocks(md_content)
        md_content = _strip_and_replace_latex(md_content)
        md_content = SentenceBagRule._MARKDOWN_IMAGE_PATTERN.sub(" ", md_content)
        md_content = _augment_with_table_cell_text(md_content)
        md_content = normalize_text(md_content)
        md_content = SentenceBagRule._MULTI_DOT_PATTERN.sub(" ", md_content)
        md_content = _unescape_html_entities(md_content)
        md_content = SentenceBagRule._AUTOLINK_PATTERN.sub(r"\1", md_content)
        md_content = SentenceBagRule._HTML_TAG_PATTERN.sub(" ", md_content)
        md_content = re.sub(r"\s+", " ", md_content)
        return md_content

    @staticmethod
    def _count_sentence_in_full_text(sentence: str, full_text: str) -> int:
        """Count non-overlapping occurrences of *sentence* as a substring of *full_text*.

        This is the substring-matching counterpart to bag-based lookup.  It
        handles cases where sentence boundary splitting produces different
        fragments than the annotated bag keys (e.g. abbreviations, line breaks).

        Uses word-boundary anchors for short sentences (< 20 chars) to avoid
        false positives where e.g. "the" matches inside "then" or "other".
        Longer sentences are unlikely to produce false substring hits, so
        plain ``str.count`` is used for performance.
        """
        if len(sentence) < 20:
            return len(re.findall(r"(?<!\w)" + re.escape(sentence) + r"(?!\w)", full_text))
        return full_text.count(sentence)

    def _extract_normalized_sentences(self, md_content: str, include_table_cells: bool = False) -> Counter[str]:
        """Split by sentence boundaries and return normalized occurrence counts."""
        return SentenceBagRule._extract_normalized_sentences_static(md_content, include_table_cells=include_table_cells)

    @staticmethod
    def _merge_short_chunks(chunks: list[str]) -> list[str]:
        """Merge chunks shorter than MIN_SENTENCE_LENGTH with the next chunk.

        This prevents short but valid sentence fragments (e.g. "Fig 1",
        "See A") from being silently dropped during sentence extraction.
        """
        merged: list[str] = []
        carry = ""
        for chunk in chunks:
            text = chunk.strip()
            if not text:
                continue
            if carry:
                text = carry + " " + text
                carry = ""
            # Check normalized length to decide if merge is needed
            normalized = SentenceBagRule._normalize_sentence_fragment(text)
            if not normalized and len(text.strip()) > 0:
                # Too short after normalization — carry forward to merge with next
                carry = text
            else:
                merged.append(text)
        # If there's a leftover carry, append to last entry or add standalone
        if carry:
            if merged:
                merged[-1] = merged[-1] + " " + carry
            else:
                merged.append(carry)
        return merged

    @staticmethod
    def _extract_normalized_sentences_static(md_content: str, include_table_cells: bool = False) -> Counter[str]:
        """Split by sentence boundaries and return normalized occurrence counts.

        Fenced code blocks (mermaid, description) and LaTeX are stripped before
        sentence extraction. Short chunks (< MIN_SENTENCE_LENGTH after
        normalization) are merged with the next chunk instead of being dropped.
        """
        md_content = _strip_fenced_code_blocks(md_content)
        md_content = _strip_and_replace_latex(md_content)
        md_content = SentenceBagRule._MARKDOWN_IMAGE_PATTERN.sub(" ", md_content)
        if include_table_cells:
            md_content = _augment_with_table_cell_text(md_content)
        else:
            md_content = _strip_html_tables_and_content(md_content)
        md_content = SentenceBagRule._MULTI_DOT_PATTERN.sub(" ", md_content)
        sentence_chunks = SentenceBagRule._SENTENCE_SPLIT_PATTERN.split(md_content)
        # Merge short chunks with the next one to avoid losing short fragments
        sentence_chunks = SentenceBagRule._merge_short_chunks(sentence_chunks)
        sentence_counter: Counter[str] = Counter()
        for chunk in sentence_chunks:
            normalized_sentence = SentenceBagRule._normalize_sentence_fragment(chunk)
            if normalized_sentence:
                sentence_counter[normalized_sentence] += 1
        return sentence_counter

    @staticmethod
    def _format_sentence_debug(sentence: str, edge_chars: int = 48) -> str:
        """Return a compact but unique sentence descriptor for failure messages.

        Why: sentence previews that only show the first characters can hide differences
        in long strings (e.g., trailing tokens or punctuation), leading to confusing
        diagnostics where a sentence appears both missing and unexpected.
        """
        sentence_hash = hashlib.sha1(sentence.encode("utf-8")).hexdigest()[:10]
        if len(sentence) <= edge_chars * 2:
            compact = sentence
        else:
            compact = f"{sentence[:edge_chars]}…{sentence[-edge_chars:]}"
        return f"{compact!r} [len={len(sentence)}, sha1={sentence_hash}]"


class UnexpectedSentenceRule(SentenceBagRule):
    """Fail when output contains sentence fragments not listed in bag_of_sentence.

    An actual sentence is considered expected if it appears in the bag
    (exact match) OR if it is a substring of any bag entry (handles
    sentence-boundary misalignment where the actual fragment is a
    sub-piece of an expected sentence).
    """

    def __init__(self, rule_data: ParseUnexpectedSentenceRule):
        super().__init__(rule_data, ParseRuleType.UNEXPECTED_SENTENCE.value)
        # Pre-build a concatenated reference text from bag keys for substring fallback
        self._bag_full_text: str = " ".join(self.sentence_bag.keys())

    def _is_expected(self, sentence: str) -> bool:
        """Return True if *sentence* is in the bag or is a substring of any bag entry."""
        if sentence in self.sentence_bag:
            return True
        if sentence in self._bag_full_text:
            return True
        return False

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        actual_sentence_bag = self._extract_normalized_sentences(md_content)

        unexpected = [
            (sentence, actual_count)
            for sentence, actual_count in actual_sentence_bag.items()
            if not self._is_expected(sentence)
        ]
        if not unexpected:
            return True, ""

        unexpected.sort(key=lambda item: (-item[1], item[0]))
        preview = "; ".join(f"{self._format_sentence_debug(sentence)} ({count}x)" for sentence, count in unexpected)
        return False, f"Found unexpected sentence(s): {preview}"


class UnexpectedSentencePercentRule(SentenceBagRule):
    """Score unexpected-sentence compliance in [0, 1].

    1.0 means all observed sentence fragments are in `bag_of_sentence`.
    0.0 means no observed sentence fragment is in `bag_of_sentence`.

    When ``original_md`` is provided in the rule data, a sentence that is
    not in ``bag_of_sentence`` is still considered expected if it appears
    as a substring of the normalized original markdown.  This avoids false
    positives caused by sentence-boundary misalignment or line breaks.
    """

    def __init__(self, rule_data: ParseUnexpectedSentencePercentRule):
        super().__init__(rule_data, ParseRuleType.UNEXPECTED_SENTENCE_PERCENT.value)
        rule_data = cast(ParseUnexpectedSentencePercentRule, self._rule_data)
        original_md = rule_data.original_md
        if original_md is not None:
            self._normalized_original_md: str | None = SentenceBagRule._normalize_full_text(original_md)
        else:
            self._normalized_original_md = None
        # Pre-build a concatenated reference text from bag keys for substring fallback
        self._bag_full_text: str = " ".join(self.sentence_bag.keys())

    def _is_expected(self, sentence: str) -> bool:
        """Return True if *sentence* is in the bag, is a substring of a bag entry, or found in the original MD."""
        if sentence in self.sentence_bag:
            return True
        if sentence in self._bag_full_text:
            return True
        if self._normalized_original_md is not None and sentence in self._normalized_original_md:
            return True
        return False

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str, float]:
        actual_sentence_bag = self._extract_normalized_sentences(md_content)

        total_actual = sum(actual_sentence_bag.values())
        if total_actual == 0:
            return True, "", 1.0

        expected_hits = sum(count for sentence, count in actual_sentence_bag.items() if self._is_expected(sentence))
        score = max(0.0, min(1.0, expected_hits / total_actual))

        unexpected = [
            (sentence, actual_count)
            for sentence, actual_count in actual_sentence_bag.items()
            if not self._is_expected(sentence)
        ]
        if not unexpected:
            return True, "", score

        unexpected.sort(key=lambda item: (-item[1], item[0]))
        preview = "; ".join(f"{self._format_sentence_debug(sentence)} ({count}x)" for sentence, count in unexpected[:5])
        return (
            False,
            f"Unexpected sentence percent score={score:.3f}; unexpected: {preview}",
            score,
        )


class TooManySentenceOccurenceRule(SentenceBagRule):
    """Fail when a configured sentence appears more times than allowed.

    Uses the maximum of bag-based and substring-based counts so that
    sentence-boundary misalignment does not hide genuine duplications.
    """

    def __init__(self, rule_data: ParseTooManySentenceOccurrenceRule):
        super().__init__(rule_data, ParseRuleType.TOO_MANY_SENTENCE_OCCURENCE.value)

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        actual_sentence_bag = self._extract_normalized_sentences(md_content)
        full_text = self._normalize_full_text(md_content)

        too_many: list[tuple[str, int, int, int]] = []
        for sentence, allowed_count in self.sentence_bag.items():
            bag_count = actual_sentence_bag.get(sentence, 0)
            substr_count = self._count_sentence_in_full_text(sentence, full_text)
            actual_count = max(bag_count, substr_count)
            if actual_count > allowed_count:
                too_many.append((sentence, actual_count - allowed_count, actual_count, allowed_count))

        if not too_many:
            return True, ""

        too_many.sort(key=lambda item: (-item[1], item[0]))
        preview = "; ".join(
            f"{self._format_sentence_debug(sentence)} ({actual}>{allowed})" for sentence, _, actual, allowed in too_many
        )
        return False, f"Found too many sentence occurrence(s): {preview}"


class TooManySentenceOccurencePercentRule(SentenceBagRule):
    """Score over-limit sentence compliance in [0, 1].

    1.0 means no configured sentence exceeds its allowed count.
    0.0 means fully over-limit behavior for configured sentences.

    Uses the maximum of bag-based and substring-based counts for consistency.
    """

    def __init__(self, rule_data: ParseTooManySentenceOccurrencePercentRule):
        super().__init__(rule_data, ParseRuleType.TOO_MANY_SENTENCE_OCCURENCE_PERCENT.value)

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str, float]:
        actual_sentence_bag = self._extract_normalized_sentences(md_content)
        full_text = self._normalize_full_text(md_content)

        total_excess = 0
        total_denominator = 0
        too_many: list[tuple[str, int, int, int]] = []

        for sentence, allowed_count in self.sentence_bag.items():
            bag_count = actual_sentence_bag.get(sentence, 0)
            substr_count = self._count_sentence_in_full_text(sentence, full_text)
            actual_count = max(bag_count, substr_count)
            excess = max(0, actual_count - allowed_count)
            total_excess += excess
            total_denominator += max(actual_count, allowed_count)
            if excess > 0:
                too_many.append((sentence, excess, actual_count, allowed_count))

        if total_denominator == 0:
            score = 1.0
        else:
            score = max(0.0, min(1.0, 1.0 - (total_excess / total_denominator)))

        if not too_many:
            return True, "", score

        too_many.sort(key=lambda item: (-item[1], item[0]))
        preview = "; ".join(
            f"{self._format_sentence_debug(sentence)} ({actual}>{allowed})"
            for sentence, _, actual, allowed in too_many[:5]
        )
        return (
            False,
            f"Too-many sentence percent score={score:.3f}; over-limit: {preview}",
            score,
        )


class MissingSentenceRule(SentenceBagRule):
    """Fail when a configured sentence appears fewer times than required.

    Uses substring matching in normalized full text (consistent with
    ``MissingSentencePercentRule``) to avoid false negatives from
    sentence-boundary misalignment.  Falls back to bag-based counting
    only when substring matching finds fewer occurrences.
    """

    def __init__(self, rule_data: ParseMissingSentenceRule):
        super().__init__(rule_data, ParseRuleType.MISSING_SENTENCE.value)

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        actual_sentence_bag = self._extract_normalized_sentences(
            md_content,
            include_table_cells=True,
        )
        full_text = self._normalize_full_text(md_content)

        missing: list[tuple[str, int, int, int]] = []
        for sentence, required_count in self.sentence_bag.items():
            bag_count = actual_sentence_bag.get(sentence, 0)
            substr_count = self._count_sentence_in_full_text(sentence, full_text)
            actual_count = max(bag_count, substr_count)
            if actual_count < required_count:
                missing.append((sentence, required_count - actual_count, actual_count, required_count))

        if not missing:
            return True, ""

        missing.sort(key=lambda item: (-item[1], item[0]))
        preview = "; ".join(
            f"{self._format_sentence_debug(sentence)} ({actual}<{required})"
            for sentence, _, actual, required in missing
        )
        return False, f"Missing sentence occurrence(s): {preview}"


class MissingSentencePercentRule(SentenceBagRule):
    """Score required-sentence coverage in [0, 1].

    1.0 means all required sentence occurrences are present.
    0.0 means none of the required sentence occurrences are present.

    Checks whether each required sentence appears as a substring in the
    normalized full markdown text (rather than splitting into a bag of
    discrete sentences). This avoids false negatives caused by sentence-
    boundary misalignment.
    """

    def __init__(self, rule_data: ParseMissingSentencePercentRule):
        super().__init__(rule_data, ParseRuleType.MISSING_SENTENCE_PERCENT.value)

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str, float]:
        full_text = self._normalize_full_text(md_content)

        total_required = sum(self.sentence_bag.values())
        matched_required = 0
        missing: list[tuple[str, int, int, int]] = []

        for sentence, required_count in self.sentence_bag.items():
            actual_count = self._count_sentence_in_full_text(sentence, full_text)
            matched_required += min(actual_count, required_count)
            if actual_count < required_count:
                missing.append((sentence, required_count - actual_count, actual_count, required_count))

        if total_required == 0:
            score = 1.0
        else:
            score = max(0.0, min(1.0, matched_required / total_required))

        if not missing:
            return True, "", score

        missing.sort(key=lambda item: (-item[1], item[0]))
        preview = "; ".join(
            f"{self._format_sentence_debug(sentence)} ({actual}<{required})"
            for sentence, _, actual, required in missing[:5]
        )
        return (
            False,
            f"Missing sentence percent score={score:.3f}; missing: {preview}",
            score,
        )


class MissingSpecificSentenceRule(ParseTestRule):
    """Fail when a specific sentence is not found in the content.

    Unlike `MissingSentenceRule`, this rule targets a single sentence rather
    than a bag of sentences, making it simpler to author for one-off checks.
    Checks whether the normalized sentence appears as a substring in the
    normalized full markdown text, so sentences split by line breaks are
    still matched.
    """

    def __init__(self, rule_data: ParseMissingSpecificSentenceRule):
        super().__init__(rule_data)
        if self.type != ParseRuleType.MISSING_SPECIFIC_SENTENCE.value:
            raise ValueError(f"Invalid type for MissingSpecificSentenceRule: {self.type}")
        rule_data = cast(ParseMissingSpecificSentenceRule, self._rule_data)
        raw_sentence = rule_data.sentence
        if not isinstance(raw_sentence, str) or not raw_sentence.strip():
            raise ValueError("sentence must be a non-empty string")
        self.normalized_sentence = SentenceBagRule._normalize_sentence_fragment(raw_sentence)
        if not self.normalized_sentence:
            raise ValueError(f"sentence is too short or empty after normalization: {raw_sentence!r}")

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        full_text = SentenceBagRule._normalize_full_text(md_content)
        if self.normalized_sentence in full_text:
            return True, ""
        preview = SentenceBagRule._format_sentence_debug(self.normalized_sentence)
        return False, f"Missing specific sentence: {preview}"


class WordBagRule(ParseTestRule):
    """Shared utilities for word-bag parse rules."""

    MIN_WORD_LENGTH = 2
    _LEADING_MARKDOWN_PATTERN = re.compile(r"^(?:\s*(?:#{1,6}\s+|>\s+|[*+-]\s+|\d+[.)]\s+))+")
    _HTML_TAG_PATTERN = re.compile(r"</?[^>]+>")

    def __init__(self, rule_data: WordBagRuleData | dict, expected_type: str):
        super().__init__(rule_data)
        rule_data = cast(WordBagRuleData, self._rule_data)

        if self.type != expected_type:
            raise ValueError(f"Invalid type for {self.__class__.__name__}: {self.type}")

        bag_of_word = rule_data.bag_of_word
        if not isinstance(bag_of_word, dict) or not bag_of_word:
            raise ValueError("bag_of_word must be a non-empty dictionary")

        self.word_bag: Counter[str] = Counter()
        for word, occurrences in bag_of_word.items():
            if not isinstance(word, str) or not word.strip():
                raise ValueError("bag_of_word keys must be non-empty strings")
            if not isinstance(occurrences, int):
                raise ValueError("bag_of_word values must be integers")
            if occurrences < 0:
                raise ValueError("bag_of_word values cannot be negative")

            normalized_word = self._normalize_word_fragment(word)
            if not normalized_word:
                continue
            self.word_bag[normalized_word] += occurrences

        if not self.word_bag:
            raise ValueError(
                f"bag_of_word has no valid word after normalization "
                f"(words must be >= {WordBagRule.MIN_WORD_LENGTH} characters)"
            )

    @staticmethod
    def _normalize_word_fragment(text: str) -> str:
        """Normalize a word token for robust matching.

        Uses Unicode-aware tokenization so accented characters and CJK
        characters are handled correctly.
        """
        normalized = normalize_text(text)
        unescaped_word = _unescape_html_entities(normalized)
        if unescaped_word != normalized:
            logger.debug("Decoded HTML entities in bag_of_word entry during normalization")
        normalized = unescaped_word
        normalized = SentenceBagRule._AUTOLINK_PATTERN.sub(r"\1", normalized)
        normalized = WordBagRule._HTML_TAG_PATTERN.sub(" ", normalized)
        normalized = WordBagRule._LEADING_MARKDOWN_PATTERN.sub("", normalized).strip()
        words = _tokenize_unicode_words(normalized, min_length=WordBagRule.MIN_WORD_LENGTH)
        if not words:
            return ""
        return words[0]

    @staticmethod
    def _normalize_full_word_text(md_content: str) -> str:
        """Normalize full markdown content for word-level substring matching.

        Applies the same pipeline as ``_extract_normalized_words_static`` but
        returns the full normalized text instead of tokenizing it, so that
        word-boundary regex searches can be used as a fallback.
        """
        md_content = _strip_fenced_code_blocks(md_content)
        md_content = _strip_and_replace_latex(md_content)
        md_content = _augment_with_table_cell_text(md_content)
        normalized_content = normalize_text(md_content)
        unescaped_content = _unescape_html_entities(normalized_content)
        normalized_content = unescaped_content
        normalized_content = SentenceBagRule._AUTOLINK_PATTERN.sub(r"\1", normalized_content)
        normalized_content = WordBagRule._HTML_TAG_PATTERN.sub(" ", normalized_content)
        return normalized_content

    @staticmethod
    def _count_word_in_full_text(word: str, full_text: str) -> int:
        """Count word-boundary-delimited occurrences of *word* in *full_text*.

        This is a substring fallback for bag-based word lookup.  Uses
        Unicode-aware word boundaries so accented and CJK words are
        handled correctly.
        """
        return _word_boundary_count(word, full_text)

    def _extract_normalized_words(self, md_content: str, include_table_cells: bool = False) -> Counter[str]:
        """Tokenize normalized content and return word occurrence counts."""
        return WordBagRule._extract_normalized_words_static(
            md_content,
            include_table_cells=include_table_cells,
        )

    @staticmethod
    def _extract_normalized_words_static(md_content: str, include_table_cells: bool = False) -> Counter[str]:
        """Tokenize normalized content and return word occurrence counts.

        Fenced code blocks (mermaid, description) and LaTeX are stripped before
        tokenization.
        """
        md_content = _strip_fenced_code_blocks(md_content)
        md_content = _strip_and_replace_latex(md_content)
        if include_table_cells:
            md_content = _augment_with_table_cell_text(md_content)
        else:
            md_content = _strip_html_tables_and_content(md_content)
        normalized_content = normalize_text(md_content)
        unescaped_content = _unescape_html_entities(normalized_content)
        if unescaped_content != normalized_content:
            logger.debug("Decoded HTML entities in content before word tokenization")
        normalized_content = unescaped_content
        normalized_content = SentenceBagRule._AUTOLINK_PATTERN.sub(r"\1", normalized_content)
        normalized_content = WordBagRule._HTML_TAG_PATTERN.sub(" ", normalized_content)
        words = _tokenize_unicode_words(normalized_content, min_length=WordBagRule.MIN_WORD_LENGTH)
        return Counter(words)


class UnexpectedWordRule(WordBagRule):
    """Fail when output contains words not listed in bag_of_word."""

    def __init__(self, rule_data: ParseUnexpectedWordRule):
        super().__init__(rule_data, ParseRuleType.UNEXPECTED_WORD.value)

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        actual_word_bag = self._extract_normalized_words(md_content)

        unexpected = [
            (word, actual_count) for word, actual_count in actual_word_bag.items() if word not in self.word_bag
        ]
        if not unexpected:
            return True, ""

        unexpected.sort(key=lambda item: (-item[1], item[0]))
        preview = "; ".join(f"'{word}' ({count}x)" for word, count in unexpected)
        return False, f"Found unexpected word(s): {preview}"


class UnexpectedWordPercentRule(WordBagRule):
    """Score unexpected-word compliance in [0, 1].

    1.0 means all observed words are in `bag_of_word`.
    0.0 means no observed word is in `bag_of_word`.
    """

    def __init__(self, rule_data: ParseUnexpectedWordPercentRule):
        super().__init__(rule_data, ParseRuleType.UNEXPECTED_WORD_PERCENT.value)

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str, float]:
        actual_word_bag = self._extract_normalized_words(md_content)

        total_actual = sum(actual_word_bag.values())
        if total_actual == 0:
            return True, "", 1.0

        expected_hits = sum(count for word, count in actual_word_bag.items() if word in self.word_bag)
        score = max(0.0, min(1.0, expected_hits / total_actual))

        unexpected = [
            (word, actual_count) for word, actual_count in actual_word_bag.items() if word not in self.word_bag
        ]
        if not unexpected:
            return True, "", score

        unexpected.sort(key=lambda item: (-item[1], item[0]))
        preview = "; ".join(f"'{word}' ({count}x)" for word, count in unexpected[:5])
        return (
            False,
            f"Unexpected word percent score={score:.3f}; unexpected: {preview}",
            score,
        )


class TooManyWordOccurenceRule(WordBagRule):
    """Fail when a configured word appears more times than allowed.

    Uses the maximum of tokenized-bag and word-boundary substring counts
    for consistency with missing-word rules.
    """

    def __init__(self, rule_data: ParseTooManyWordOccurrenceRule):
        super().__init__(rule_data, ParseRuleType.TOO_MANY_WORD_OCCURENCE.value)

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        actual_word_bag = self._extract_normalized_words(md_content)
        full_text = self._normalize_full_word_text(md_content)

        too_many: list[tuple[str, int, int, int]] = []
        for word, allowed_count in self.word_bag.items():
            bag_count = actual_word_bag.get(word, 0)
            substr_count = self._count_word_in_full_text(word, full_text)
            actual_count = max(bag_count, substr_count)
            if actual_count > allowed_count:
                too_many.append((word, actual_count - allowed_count, actual_count, allowed_count))

        if not too_many:
            return True, ""

        too_many.sort(key=lambda item: (-item[1], item[0]))
        preview = "; ".join(f"'{word}' ({actual}>{allowed})" for word, _, actual, allowed in too_many)
        return False, f"Found too many word occurrence(s): {preview}"


class TooManyWordOccurencePercentRule(WordBagRule):
    """Score over-limit compliance in [0, 1].

    1.0 means no configured word exceeds its allowed count.
    0.0 means every counted token among configured words is over-limit.

    Uses the maximum of tokenized-bag and word-boundary substring counts
    for consistency with missing-word rules.
    """

    def __init__(self, rule_data: ParseTooManyWordOccurrencePercentRule):
        super().__init__(rule_data, ParseRuleType.TOO_MANY_WORD_OCCURENCE_PERCENT.value)

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str, float]:
        actual_word_bag = self._extract_normalized_words(md_content)
        full_text = self._normalize_full_word_text(md_content)

        total_excess = 0
        total_denominator = 0
        too_many: list[tuple[str, int, int, int]] = []

        for word, allowed_count in self.word_bag.items():
            bag_count = actual_word_bag.get(word, 0)
            substr_count = self._count_word_in_full_text(word, full_text)
            actual_count = max(bag_count, substr_count)
            excess = max(0, actual_count - allowed_count)
            total_excess += excess
            total_denominator += max(actual_count, allowed_count)
            if excess > 0:
                too_many.append((word, excess, actual_count, allowed_count))

        if total_denominator == 0:
            score = 1.0
        else:
            score = max(0.0, min(1.0, 1.0 - (total_excess / total_denominator)))

        if not too_many:
            return True, "", score

        too_many.sort(key=lambda item: (-item[1], item[0]))
        preview = "; ".join(f"'{word}' ({actual}>{allowed})" for word, _, actual, allowed in too_many[:5])
        return (
            False,
            f"Too-many word percent score={score:.3f}; over-limit: {preview}",
            score,
        )


class MissingWordRule(WordBagRule):
    """Fail when a configured word appears fewer times than required.

    Uses the maximum of tokenized-bag and word-boundary substring counts
    so that tokenization artifacts do not cause false negatives.
    """

    def __init__(self, rule_data: ParseMissingWordRule):
        super().__init__(rule_data, ParseRuleType.MISSING_WORD.value)

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        actual_word_bag = self._extract_normalized_words(md_content, include_table_cells=True)
        full_text = self._normalize_full_word_text(md_content)

        missing: list[tuple[str, int, int, int]] = []
        for word, required_count in self.word_bag.items():
            bag_count = actual_word_bag.get(word, 0)
            substr_count = self._count_word_in_full_text(word, full_text)
            actual_count = max(bag_count, substr_count)
            if actual_count < required_count:
                missing.append((word, required_count - actual_count, actual_count, required_count))

        if not missing:
            return True, ""

        missing.sort(key=lambda item: (-item[1], item[0]))
        preview = "; ".join(f"'{word}' ({actual}<{required})" for word, _, actual, required in missing)
        return False, f"Missing word occurrence(s): {preview}"


class MissingWordPercentRule(WordBagRule):
    """Score required-word coverage in [0, 1].

    1.0 means all required occurrences are present.
    0.0 means none of the required occurrences are present.

    Uses the maximum of tokenized-bag and word-boundary substring counts
    so that tokenization artifacts do not cause false negatives.
    """

    def __init__(self, rule_data: ParseMissingWordPercentRule):
        super().__init__(rule_data, ParseRuleType.MISSING_WORD_PERCENT.value)

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str, float]:
        actual_word_bag = self._extract_normalized_words(md_content, include_table_cells=True)
        full_text = self._normalize_full_word_text(md_content)

        total_required = sum(self.word_bag.values())
        matched_required = 0
        missing: list[tuple[str, int, int, int]] = []

        for word, required_count in self.word_bag.items():
            bag_count = actual_word_bag.get(word, 0)
            substr_count = self._count_word_in_full_text(word, full_text)
            actual_count = max(bag_count, substr_count)
            matched_required += min(actual_count, required_count)
            if actual_count < required_count:
                missing.append((word, required_count - actual_count, actual_count, required_count))

        if total_required == 0:
            score = 1.0
        else:
            score = max(0.0, min(1.0, matched_required / total_required))

        if not missing:
            return True, "", score

        missing.sort(key=lambda item: (-item[1], item[0]))
        preview = "; ".join(f"'{word}' ({actual}<{required})" for word, _, actual, required in missing[:5])
        return (
            False,
            f"Missing word percent score={score:.3f}; missing: {preview}",
            score,
        )


class MissingSpecificWordRule(ParseTestRule):
    """Fail when a specific word is not found in the content.

    Unlike `MissingWordRule`, this rule targets a single word rather than a
    bag of words, making it simpler to author for one-off checks.
    It reuses the same normalization and tokenization logic as word-bag rules.
    """

    _APOSTROPHE_PATTERN = re.compile(r"['\u2019]")

    @classmethod
    def strip_apostrophes(cls, content: str) -> str:
        """Normalize apostrophe variants used by missing_specific_word fallback matching."""
        return cls._APOSTROPHE_PATTERN.sub("", content)

    def __init__(self, rule_data: ParseMissingSpecificWordRule):
        super().__init__(rule_data)
        if self.type != ParseRuleType.MISSING_SPECIFIC_WORD.value:
            raise ValueError(f"Invalid type for MissingSpecificWordRule: {self.type}")
        rule_data = cast(ParseMissingSpecificWordRule, self._rule_data)
        raw_word = rule_data.word
        if not isinstance(raw_word, str) or not raw_word.strip():
            raise ValueError("word must be a non-empty string")
        # Use a lenient normalization: pick the longest valid token from the
        # fragments (e.g. "d'équipage" → ['d', 'equipage'] → 'equipage').
        # Annotated words should never be rejected outright.
        normalized = normalize_text(raw_word)
        unescaped = _unescape_html_entities(normalized)
        cleaned = SentenceBagRule._AUTOLINK_PATTERN.sub(r"\1", unescaped)
        cleaned = WordBagRule._HTML_TAG_PATTERN.sub(" ", cleaned)
        cleaned = WordBagRule._LEADING_MARKDOWN_PATTERN.sub("", cleaned).strip()
        fragments = _tokenize_unicode_words(cleaned, min_length=1)
        # Pick the longest fragment that meets the minimum length, falling back
        # to the longest fragment overall so annotated words are never discarded.
        valid = [f for f in fragments if len(f) >= WordBagRule.MIN_WORD_LENGTH]
        if valid:
            self.normalized_word: str = max(valid, key=len)
        elif fragments:
            # For contractions like "can't" → ['can', 't'], join all
            # fragments to form a single token ("cant", 4 chars) instead
            # of picking the longest short fragment.
            joined = "".join(fragments)
            if len(joined) >= WordBagRule.MIN_WORD_LENGTH:
                self.normalized_word = joined
            else:
                self.normalized_word = max(fragments, key=len)
        else:
            raise ValueError(f"word is empty after normalization: {raw_word!r}")
        self.actual_words: Counter[str] | None = None
        self.apostrophe_stripped_content: str | None = None

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        actual_words = self.actual_words
        if actual_words is None:
            actual_words = WordBagRule._extract_normalized_words_static(
                md_content,
                include_table_cells=True,
            )
        if self.normalized_word in actual_words:
            return True, ""
        # Fallback: search in apostrophe-stripped normalized content.
        # Handles contractions (e.g. "can't" → "cant") where the token-based
        # approach splits on the apostrophe and individual fragments are too
        # short to survive the MIN_WORD_LENGTH filter.
        content = self.apostrophe_stripped_content
        if content is None:
            # Standalone rule usage recomputes normalized content here; the
            # RuleBasedMetric cache path reuses its pre-normalized document text.
            content = normalize_text(md_content)
            content = self.strip_apostrophes(content)
        if _word_boundary_count(self.normalized_word, content) > 0:
            return True, ""
        return False, f"Missing specific word: '{self.normalized_word}'"


# --- Digit count patterns for bag_of_digit_percent ---


def _extract_digit_counts(md_content: str, include_table_cells: bool = False) -> Counter[str]:
    """Extract digit (0-9) occurrence counts from markdown content.

    Digits inside HTML tag attributes (e.g. colspan="2", rowspan="3") are excluded.
    Digits inside table cell text content ARE included.
    """
    if include_table_cells:
        md_content = _augment_with_table_cell_text(md_content)

    # Remove HTML tags (including attributes) but keep their text content
    content = _HTML_TAG_WITH_ATTRS_PATTERN.sub(" ", md_content)
    # Count each digit character
    return Counter(ch for ch in content if ch in "0123456789")


class BagOfDigitPercentRule(ParseTestRule):
    """Score digit-frequency match between expected and actual markdown in [0, 1].

    Compares the count of each digit (0-9) in the actual output against the
    expected counts in ``bag_of_digit``.  Digits inside HTML tag attributes
    (e.g. ``colspan="2"``) are excluded so only meaningful content digits
    are compared.

    Score = matched / total_expected, where matched is the sum of
    min(actual_count, expected_count) for each digit.
    """

    def __init__(self, rule_data: ParseBagOfDigitPercentRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseBagOfDigitPercentRule, self._rule_data)

        if self.type != ParseRuleType.BAG_OF_DIGIT_PERCENT.value:
            raise ValueError(f"Invalid type for BagOfDigitPercentRule: {self.type}")

        bag_of_digit = rule_data.bag_of_digit
        if not isinstance(bag_of_digit, dict) or not bag_of_digit:
            raise ValueError("bag_of_digit must be a non-empty dictionary")

        self.digit_bag: Counter[str] = Counter()
        for digit, count in bag_of_digit.items():
            if not isinstance(digit, str) or digit not in "0123456789":
                raise ValueError(f"bag_of_digit keys must be single digit characters (0-9), got: {digit!r}")
            if not isinstance(count, int):
                raise ValueError("bag_of_digit values must be integers")
            if count < 0:
                raise ValueError("bag_of_digit values cannot be negative")
            self.digit_bag[digit] += count

        if not self.digit_bag:
            raise ValueError("bag_of_digit has no valid digits")

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str, float]:
        actual_digits = _extract_digit_counts(md_content, include_table_cells=True)

        total_expected = sum(self.digit_bag.values())
        if total_expected == 0:
            return True, "", 1.0

        matched = 0
        missing: list[tuple[str, int, int, int]] = []

        for digit, expected_count in self.digit_bag.items():
            actual_count = actual_digits.get(digit, 0)
            matched += min(actual_count, expected_count)
            if actual_count < expected_count:
                missing.append((digit, expected_count - actual_count, actual_count, expected_count))

        score = max(0.0, min(1.0, matched / total_expected))

        if not missing:
            return True, "", score

        missing.sort(key=lambda item: (-item[1], item[0]))
        preview = "; ".join(f"'{d}' ({actual}<{expected})" for d, _, actual, expected in missing[:5])
        return (
            False,
            f"Bag of digit percent score={score:.3f}; missing: {preview}",
            score,
        )


class ExtraContentRule(SentenceBagRule):
    """Backward-compatible combined extra-content check.

    This remains available for older datasets and is equivalent to:
    - Unexpected sentence detection (sentence not listed in bag_of_sentence), and
    - Too many occurrence detection (sentence count exceeds allowed value).

    An actual sentence is considered expected if it matches a bag entry
    exactly or is a substring of any bag entry (handles boundary misalignment).
    """

    def __init__(self, rule_data: ParseExtraContentRule):
        super().__init__(rule_data, ParseRuleType.EXTRA_CONTENT.value)
        self._bag_full_text: str = " ".join(self.sentence_bag.keys())

    def _expected_count(self, sentence: str) -> int:
        """Return the expected count, using substring fallback for unrecognized sentences."""
        exact = self.sentence_bag.get(sentence, 0)
        if exact > 0:
            return exact
        # Substring fallback: if the sentence is a sub-piece of a bag entry, treat as expected once
        if sentence in self._bag_full_text:
            return 1
        return 0

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        actual_sentence_bag = self._extract_normalized_sentences(md_content)

        extras: list[tuple[str, int]] = []
        for sentence, actual_count in actual_sentence_bag.items():
            expected_count = self._expected_count(sentence)
            if actual_count > expected_count:
                extras.append((sentence, actual_count - expected_count))

        if not extras:
            return True, ""

        extras.sort(key=lambda item: (-item[1], item[0]))
        preview = "; ".join(
            f"{self._format_sentence_debug(sentence)} (+{extra_count})" for sentence, extra_count in extras[:5]
        )
        return False, f"Found extra content sentences: {preview}"
