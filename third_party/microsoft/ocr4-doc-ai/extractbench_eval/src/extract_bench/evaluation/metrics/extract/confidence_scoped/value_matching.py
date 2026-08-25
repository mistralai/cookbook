from __future__ import annotations

import difflib
import math
import re
import unicodedata
from typing import Any

from extract_bench.evaluation.metrics.field_grounding.value_compare import (
    compare_attributed_value,
    expected_type_for_field_path,
)

from .paths import field_name, is_array_schema

# Fraction of one side's tokens that must appear in the other side for a
# token-containment match; below this the strings are treated as different.
TOKEN_CONTAINMENT_MIN = 0.95
# Short expected values (1-2 tokens) contained in a longer prediction are
# usually a different/augmented value, not an equivalent one ("Bluemercury"
# vs "Bluemercury - 68th Street").
TOKEN_CONTAINMENT_MIN_EXPECTED_TOKENS = 3
# A predicted value counts as citation-supported when it is a substring of the
# cited text or shares at least this fraction of its tokens with it.
CITATION_SUPPORT_TOKEN_OVERLAP_MIN = 0.85
# Values shorter than this are too ambiguous to ground via citation text.
CITATION_SUPPORT_MIN_CHARS = 3
# A predicted string contained verbatim in the expected text counts as a match
# only when it is long enough to be distinctive.
LONG_SUBSTRING_MIN_CHARS = 30
LONG_SUBSTRING_MIN_TOKENS = 5
# A concise prediction fully contained in a longer expected value must still
# be substantial (and numerically identical) to count as equivalent.
CONCISE_CONTAINED_MIN_TOKENS = 4
CONCISE_CONTAINED_MIN_CHARS = 18
# Month-only GTs are encoded as instants at the end of the prior month
# (June 2016 -> 2016-05-31T22:00Z): roll forward only on end-of-month day AND
# late-evening hour — an intraday end-of-month timestamp is a genuine date.
MONTH_END_SHIFT_MIN_DAY = 28
MONTH_END_SHIFT_MIN_HOUR = 22
# Tolerances for signed-magnitude numeric equality.
NUMERIC_REL_TOL = 1e-9
NUMERIC_ABS_TOL = 1e-9
# Separator-insensitive whole-string equality is for formatted IDs and spacing
# artifacts, gated to ID/text shapes (letters present, or a long digit run) so
# short numeric ranges ("123-456") are not equated with numbers.
SEPARATOR_INSENSITIVE_MIN_DIGITS = 7
# Jaro-Winkler post-guards: whole-token divergence bounds and the near-typo
# similarity a single differing token pair must reach.
JW_TOKEN_DIVERGENCE_MIN_TOKENS = 2
JW_ENUM_TOKEN_MAX_CHARS = 3
JW_TYPO_PAIR_MIN_RATIO = 0.7
# The bench comparator's diagnostic annotation_truncated pass is credited only
# when the clipped remainder is a few characters, not appended whole tokens.
ANNOTATION_TRUNCATED_MAX_CLIP_CHARS = 3
# Score granted when a diagnostic comparator mode is credited here.
DIAGNOSTIC_EQUIVALENCE_MIN_SCORE = 0.99

# Letters NFKD cannot reduce to ASCII; fold explicitly so diacritic-preserving
# predictions ("Norðureyri") match ASCII-folded GT ("Nordureyri").
_DIACRITIC_FOLD_MAP = str.maketrans(
    {"ß": "ss", "æ": "ae", "œ": "oe", "ø": "o", "ð": "d", "þ": "th", "đ": "d", "ł": "l", "ħ": "h", "ı": "i"}
)


def fold_diacritics(text: str) -> str:
    text = text.translate(_DIACRITIC_FOLD_MAP)
    if text.isascii():
        return text
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def canonical_text(value: Any) -> str:
    if value is None:
        return ""
    text = fold_diacritics(str(value).casefold())
    text = re.sub(r"\bhttps?://", "", text)
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[*•·]", " ", text)
    text = re.sub(r"[_`#]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ;,:\n\t")


def tokens_for(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", canonical_text(value)))


def numeric_tokens_for(value: Any) -> set[str]:
    # Strip thousands separators inside numbers so "1,000" and "1000" tokenize equally.
    text = re.sub(r"(?<=\d),(?=\d)", "", canonical_text(value))
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def token_containment_score(expected: Any, predicted: Any) -> float:
    expected_tokens = tokens_for(expected)
    predicted_tokens = tokens_for(predicted)
    if not expected_tokens or not predicted_tokens:
        return 0.0
    return len(expected_tokens & predicted_tokens) / len(expected_tokens)


def value_supported_by_citation(predicted: Any, cited_text: str) -> bool:
    if predicted is None or not cited_text:
        return False
    pred = canonical_text(predicted)
    cite = canonical_text(cited_text)
    if not pred or len(pred) < CITATION_SUPPORT_MIN_CHARS:
        return False
    if pred in cite:
        return True
    pred_tokens = tokens_for(pred)
    cite_tokens = tokens_for(cite)
    return bool(pred_tokens) and len(pred_tokens & cite_tokens) / len(pred_tokens) >= CITATION_SUPPORT_TOKEN_OVERLAP_MIN


def strip_parenthetical(value: Any) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(value or "")).strip()


def year_month(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    iso_full = re.match(r"^(\d{4})-(\d{2})-(\d{2})(?:T.*)?$", text)
    if iso_full:
        year = int(iso_full.group(1))
        month_num = int(iso_full.group(2))
        day = int(iso_full.group(3))
        hour_match = re.search(r"T(\d{2}):", text)
        hour = int(hour_match.group(1)) if hour_match else 0
        if "T" in text and day >= MONTH_END_SHIFT_MIN_DAY and hour >= MONTH_END_SHIFT_MIN_HOUR:
            month_num += 1
            if month_num == 13:
                year += 1
                month_num = 1
        return f"{year:04d}-{month_num:02d}"
    iso = re.match(r"^(\d{4})-(\d{2})", text)
    if iso:
        return f"{iso.group(1)}-{iso.group(2)}"
    month = re.match(
        r"^(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{4})$",
        text.casefold(),
    )
    if not month:
        return None
    months = {
        "jan": "01",
        "january": "01",
        "feb": "02",
        "february": "02",
        "mar": "03",
        "march": "03",
        "apr": "04",
        "april": "04",
        "may": "05",
        "jun": "06",
        "june": "06",
        "jul": "07",
        "july": "07",
        "aug": "08",
        "august": "08",
        "sep": "09",
        "sept": "09",
        "september": "09",
        "oct": "10",
        "october": "10",
        "nov": "11",
        "november": "11",
        "dec": "12",
        "december": "12",
    }
    return f"{month.group(2)}-{months[month.group(1)]}"


def month_day_without_year(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    text = str(value).strip()
    match = re.match(r"^(\d{1,2})[/-](\d{1,2})$", text)
    if not match:
        return None
    month_num = int(match.group(1))
    day = int(match.group(2))
    if not (1 <= month_num <= 12 and 1 <= day <= 31):
        return None
    return month_num, day


def iso_year_month_day(value: Any) -> tuple[int, int, int] | None:
    if value is None:
        return None
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})(?:T.*)?$", str(value).strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def is_date_field_path(path: str) -> bool:
    name = field_name(path).casefold()
    return "date" in name


def canonical_url(value: Any) -> str:
    text = str(value or "").strip().casefold()
    text = re.sub(r"^https?://", "", text)
    text = re.sub(r"^www\.", "", text)
    return text.rstrip("/")


_SINGLE_NUMBER_BODY = re.compile(r"^\d[\d,]*(?:\.\d+)?$")


def parse_single_number(value: Any) -> float | None:
    """Return the signed float if ``value`` is essentially ONE number, else None.

    Accepts thousands-separator commas, a leading currency ``$``, a leading sign,
    and accounting parentheses ``(123)`` == ``-123``. Rejects anything with letters,
    ``%``, ``/``, ranges, or multiple internal dashes (e.g. ISO dates) so that dates
    and identifiers are left to the normal comparators.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if not isinstance(value, str):
        return None
    body = value.strip()
    if not body or "%" in body or "/" in body:
        return None
    negative = False
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1].strip()
        negative = True
    if body[:1] in "+-":
        negative = negative or body[0] == "-"
        body = body[1:].strip()
    body = body.lstrip("$").strip()
    if not _SINGLE_NUMBER_BODY.match(body):
        return None
    try:
        magnitude = float(body.replace(",", ""))
    except ValueError:
        return None
    return -magnitude if negative else magnitude


def numbers_equal(a: float, b: float, rel: float = NUMERIC_REL_TOL, abs_tol: float = NUMERIC_ABS_TOL) -> bool:
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


_SEPARATOR_STRIP_RE = re.compile(r"[-\s/().,]")


def separator_insensitive_key(value: Any) -> str:
    """Casefolded value with separators removed: whole-string equality for
    formatted IDs and spacing artifacts."""
    if not isinstance(value, str):
        return ""
    return _SEPARATOR_STRIP_RE.sub("", fold_diacritics(value.casefold()))


_NUMBER_WITH_UNIT_RE = re.compile(r"^([$\-(]?[\d,]+(?:\.\d+)?\)?)\s*([A-Za-z%]{1,5})\.?$")


def parse_number_with_unit_suffix(value: Any) -> float | None:
    """Parse "<number> <short unit>" strings ("22 FT", "45%", "12km")."""
    if not isinstance(value, str):
        return None
    match = _NUMBER_WITH_UNIT_RE.match(value.strip())
    if not match:
        return None
    return parse_single_number(match.group(1))


def is_effectively_null(value: Any) -> bool:
    """True for null, blank strings, and containers whose leaves are all null
    (annotators emit these interchangeably with null)."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, dict):
        return all(is_effectively_null(item) for item in value.values())
    if isinstance(value, list):
        return all(is_effectively_null(item) for item in value)
    return False


_TIME_ONLY_RE = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?$")


def looks_temporal(value: Any) -> bool:
    """True when a value parses as a date or clock time; such values may only
    match via the dedicated date/number rules (token-set comparisons are
    order- and separator-blind: 02/01 vs 02/02)."""
    if value is None or isinstance(value, (int, float, bool)):
        return False
    text = str(value).strip()
    if not text:
        return False
    if _TIME_ONLY_RE.match(text):
        return True
    return (
        iso_year_month_day(text) is not None or month_day_without_year(text) is not None or year_month(text) is not None
    )


_EMBEDDED_DATE_RES = (
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
    re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"),
    re.compile(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
        r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
        re.IGNORECASE,
    ),
)
_MONTH_NUM = {
    m: i + 1
    for i, m in enumerate(
        [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ]
    )
}
_MONTH_NUM.update({m[:3]: n for m, n in list(_MONTH_NUM.items())})
_MONTH_NUM["sept"] = 9


def extract_embedded_dates(value: Any) -> set[tuple[int, int, int]]:
    """All (y, m, d) dates found inside a prose string ("On or about 06/15/2024").

    US month/day order for slashed dates (dataset convention)."""
    if not isinstance(value, str):
        return set()
    found: set[tuple[int, int, int]] = set()
    text = value.strip()
    for pattern in _EMBEDDED_DATE_RES:
        for match in pattern.finditer(text):
            a, b, c = match.groups()
            if a.isdigit() and len(a) == 4:
                found.add((int(a), int(b), int(c)))
            elif a.isdigit():
                found.add((int(c), int(a), int(b)))
            else:
                found.add((int(c), _MONTH_NUM[a.casefold()], int(b)))
    return found


def compare_values(
    expected: Any,
    predicted: Any,
    *,
    schema: Any,
    predicted_path: str,
    expected_path: str,
) -> tuple[bool, float, str]:
    if predicted is None and expected == [] and is_array_schema(schema, expected_path):
        return True, 1.0, "null_empty_list_equivalent"

    if predicted == expected:
        return True, 1.0, "identity"

    if expected is None and isinstance(predicted, str) and predicted.strip().casefold() in {"present", "current"}:
        return True, 1.0, "present_null_current_equivalent"

    if is_effectively_null(expected) and is_effectively_null(predicted):
        return True, 1.0, "null_equivalent_empty"

    if predicted is None and expected is not None:
        # Blocks downstream type coercions (None -> False) from crediting misses.
        return False, 0.0, "null_vs_value"

    # When BOTH sides parse as a single number, signed-magnitude equality is
    # decisive in both directions.
    expected_number = parse_single_number(expected)
    predicted_number = parse_single_number(predicted)
    if expected_number is not None and predicted_number is not None:
        if numbers_equal(expected_number, predicted_number):
            if isinstance(expected, str) and isinstance(predicted, str):
                digits_expected = re.sub(r"[^0-9]", "", expected)
                digits_predicted = re.sub(r"[^0-9]", "", predicted)
                if digits_expected != digits_predicted and (
                    digits_expected.startswith("0") or digits_predicted.startswith("0")
                ):
                    # Leading zeros are meaningful in code strings ("0034" is
                    # not the code "34"), never in quantities.
                    return False, 0.0, "leading_zero_code_mismatch"
            return True, 1.0, "numeric_equal"
        return False, 0.0, "numeric_value_mismatch"

    # One side a pure number, the other that number plus a short unit suffix
    # ("22 FT"): decide numerically.
    expected_unit_number = parse_number_with_unit_suffix(expected)
    predicted_unit_number = parse_number_with_unit_suffix(predicted)
    if expected_unit_number is not None or predicted_unit_number is not None:
        left = expected_number if expected_number is not None else expected_unit_number
        right = predicted_number if predicted_number is not None else predicted_unit_number
        if left is not None and right is not None:
            if numbers_equal(left, right):
                return True, 1.0, "numeric_equal_unit_suffix"
            return False, 0.0, "numeric_value_mismatch"

    if isinstance(expected, str) and isinstance(predicted, str):
        expected_numeric_tokens = numeric_tokens_for(expected)
        predicted_numeric_tokens = numeric_tokens_for(predicted)
        if canonical_url(expected) and canonical_url(expected) == canonical_url(predicted):
            url_like = "/" in expected or expected.strip().casefold().startswith(("http", "www."))
            return True, 1.0, "url_protocol_equivalent" if url_like else "casefold_equivalent"
        # Formatted IDs ("35-1134872" vs "351134872") and spacing artifacts;
        # never when BOTH sides are true numbers (sign semantics: -5 != 5).
        if expected_number is None or predicted_number is None:
            key_expected = separator_insensitive_key(expected)
            if key_expected and key_expected == separator_insensitive_key(predicted):
                digit_count = sum(ch.isdigit() for ch in key_expected)
                has_alpha = any(ch.isalpha() for ch in key_expected)
                if has_alpha or digit_count >= SEPARATOR_INSENSITIVE_MIN_DIGITS:
                    return True, 1.0, "separator_insensitive_equivalent"
        if strip_parenthetical(expected).casefold() == predicted.strip().casefold():
            return True, 1.0, "parenthetical_detail_omitted"
        ym_expected = year_month(expected)
        ym_predicted = year_month(predicted)
        if ym_expected is not None and ym_expected == ym_predicted:
            # Only when at least one side is month-resolution: two full dates
            # in the same month are NOT equivalent.
            if iso_year_month_day(expected) is None or iso_year_month_day(predicted) is None:
                return True, 1.0, "year_month_equivalent"
        expected_iso = iso_year_month_day(expected)
        predicted_iso = iso_year_month_day(predicted)
        if expected_iso is not None and predicted_iso is not None and expected_iso != predicted_iso:
            # Decided here: the downstream comparator's date parsing can
            # equate day/month transpositions.
            return False, 0.0, "date_mismatch"
        if expected_iso is not None and predicted_iso is None:
            # GT is a pure date, the prediction may be prose ("On or about
            # 06/15/2024"): decide on a single embedded date.
            embedded = extract_embedded_dates(predicted)
            if len(embedded) == 1:
                if next(iter(embedded)) == expected_iso:
                    return True, 1.0, "date_equivalent_embedded"
                return False, 0.0, "date_mismatch"
        predicted_month_day = month_day_without_year(predicted)
        if (
            expected_iso is not None
            and predicted_month_day is not None
            and (is_date_field_path(expected_path) or is_date_field_path(predicted_path))
            and expected_iso[1:] == predicted_month_day
        ):
            return True, 1.0, "month_day_without_year_equivalent"
        values_temporal = looks_temporal(expected) or looks_temporal(predicted)
        containment = token_containment_score(expected, predicted)
        expected_tokens = tokens_for(expected)
        if (
            containment >= TOKEN_CONTAINMENT_MIN
            and not values_temporal
            and len(expected_tokens) >= TOKEN_CONTAINMENT_MIN_EXPECTED_TOKENS
            and expected_numeric_tokens == predicted_numeric_tokens
        ):
            return True, containment, "token_containment"
        canonical_expected = canonical_text(expected)
        canonical_predicted = canonical_text(predicted)
        if (
            not values_temporal
            and len(canonical_predicted) >= LONG_SUBSTRING_MIN_CHARS
            and len(tokens_for(canonical_predicted)) >= LONG_SUBSTRING_MIN_TOKENS
            and canonical_predicted in canonical_expected
        ):
            return True, 1.0, "long_exact_substring"
        reverse_containment = token_containment_score(predicted, expected)
        predicted_tokens = tokens_for(predicted)
        if (
            reverse_containment >= TOKEN_CONTAINMENT_MIN
            and not values_temporal
            and len(predicted_tokens) >= CONCISE_CONTAINED_MIN_TOKENS
            and len(canonical_text(predicted)) >= CONCISE_CONTAINED_MIN_CHARS
            and len(predicted_tokens) < len(expected_tokens)
            and predicted_numeric_tokens == expected_numeric_tokens
        ):
            return True, reverse_containment, "concise_string_contained_in_expected"
        if expected_numeric_tokens and predicted_numeric_tokens and expected_numeric_tokens != predicted_numeric_tokens:
            return False, 0.0, "numeric_mismatch"

    expected_type = expected_type_for_field_path(schema, expected_path, expected)
    comparison = compare_attributed_value(
        expected,
        predicted,
        expected_type=expected_type,
        source_kind="structured_value_no_citation_text",
    )
    if comparison.passed:
        if comparison.mode == "jaro_winkler":
            expected_tok = tokens_for(expected)
            predicted_tok = tokens_for(predicted)
            diff_expected = expected_tok - predicted_tok
            diff_predicted = predicted_tok - expected_tok
            if (
                len(diff_expected) >= JW_TOKEN_DIVERGENCE_MIN_TOKENS
                and len(diff_predicted) >= JW_TOKEN_DIVERGENCE_MIN_TOKENS
            ):
                # >=2 whole tokens differ on each side: different entities
                # sharing a template, not a typo.
                return False, float(comparison.score), "jaro_winkler_token_divergence"
            if bool(diff_expected) != bool(diff_predicted):
                # Whole token(s) on one side only: a truncation, not a typo —
                # subset matches must earn credit via the containment rules.
                return False, float(comparison.score), "jaro_winkler_token_drop"
            if diff_expected and diff_predicted:
                # Prefix-heavy JW cannot discriminate enumerated labels
                # ("Schedule A" vs "Schedule B").
                if min(len(t) for t in (diff_expected | diff_predicted)) <= JW_ENUM_TOKEN_MAX_CHARS:
                    return False, float(comparison.score), "jaro_winkler_token_divergence"
                # A single differing token pair must itself be a near-typo
                # ("Jonhson"~"Johnson"), not a different word.
                best_pair = max(
                    difflib.SequenceMatcher(None, a, b).ratio() for a in diff_expected for b in diff_predicted
                )
                if best_pair < JW_TYPO_PAIR_MIN_RATIO:
                    return False, float(comparison.score), "jaro_winkler_token_divergence"
        return True, float(comparison.score), comparison.mode
    if comparison.mode == "annotation_truncated":
        # Diagnostic-only comparator pass: credit only when the clipped
        # remainder is a few characters, not appended whole tokens.
        if len(canonical_text(predicted)) - len(canonical_text(expected)) <= ANNOTATION_TRUNCATED_MAX_CLIP_CHARS:
            return True, max(float(comparison.score), DIAGNOSTIC_EQUIVALENCE_MIN_SCORE), comparison.mode
    if comparison.mode == "ocr_noise_prefix":
        return True, max(float(comparison.score), DIAGNOSTIC_EQUIVALENCE_MIN_SCORE), comparison.mode
    return False, float(comparison.score), comparison.mode
