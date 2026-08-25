"""Conservative equivalence relation E for bbox text vs expected_value.

This module is one of three deterministic verification stages used to
partition extract-field annotations into trustworthy gold rules
(``verified=true``) and triage-tagged unverified rules.
The full pipeline is documented in ``DATASET_CARD.md``; the equivalence
ladder below is the heart of stage 2 (rule comparison).

Equivalence ladder (first match wins):

  1. null ↔ empty / dash-only (after trim) — accepts ``—`` ``–`` ``-`` as
     visually-empty cells (financial-table convention for "zero" / "n/a").

  2. identity (whitespace-collapsed + trimmed). Catches the byte-identical
     match common case.

  3. case-insensitive (string types only). Numbers, booleans, and dates
     don't case-fold.

  3b. trailing_punct_fold — string types: case-insensitive identity after
      stripping trailing ``.,;:!?*†‡§°™®©•·`` from both sides. Bbox often
      includes a trailing footnote marker the source annotation omits
      (e.g. ``inventory_listing_1`` items ending with ``*``).

  4. numeric canonicalization — fires for declared ``number`` types or
     when expected_value is a genuine Python int/float (catches schema
     typos where a numeric field is declared as "string"). The numeric
     parser strips trailing footnote markers and accepts the pymupdf
     space-separated digit groups pattern (``"2 671 43"`` ≡ 2671.43)
     for rotated PDFs whose text stream stores number components as
     separate spans.

  5. date canonicalization — for string types whose value parses as
     ISO ``YYYY-MM-DD``, accept any extracted text that parses
     unambiguously to the same date (handling US ``MM/DD/YYYY`` and
     EU ``DD/MM/YYYY`` interpretations).

  6. boolean canonicalization — for boolean types, accept ``true``/``yes``
     ``y``/``1``/``x``/``✓``/``[x]``/``checked`` as True; ``false``/``no``
     ``n``/``0``/``[ ]``/``unchecked``/`` `` as False. With checkbox-noise
     tolerance: short OCR garbage from drawn checkbox borders (``LI``,
     ``LJ``, ``oO``, ``im]``) treated as empty checkbox → False.

  7. OCR punctuation/spacing fold — OCR-only, string-typed. Accept text
     whose alphanumeric content (case-folded) matches exactly, ignoring
     punctuation and whitespace differences. Tesseract on cell-cropped
     fonts routinely confuses commas with periods and drops surrounding
     spaces (e.g. ``SMITH, JOHN`` vs ``SMITH. JOHN``).

  7b. icelandic_diacritic_fold — string types. The source
     export degrades Icelandic special characters to lossy ASCII
     fallbacks (``þ`` → ``b``, ``ð`` → ``d``, ``ó`` → ``o``, ``Þ`` → ``P``,
     ``æ`` → ``ae``, etc.) but the OCR'd text preserves them. Fold both
     sides through the lossy mapping; verified with tag noting the loss
     so the data vendor is aware of the round-trip degradation.

  8. annotation_truncated — string types. When the expected_value's
     alphanumeric content is a strict prefix of the extracted text's
     alphanumeric content (≥10 chars), the source annotation was
     truncated but the bbox grounds the correct cell. Verified=True
     with ``annotation_truncated:review`` tag.

For string and number types whose verdict is ``verified=False``, a
Jaro-Winkler similarity score is computed between expected and extracted
(after whitespace + numeric-presentation normalization) and exposed via
``Verdict.similarity_score``. Downstream tooling tags rules with score
≥ ``TYPO_LIKELY_JW_THRESHOLD`` (0.90) as ``annotation_typo_likely`` to
separate transcription typos / OCR digit confusions from wholly-wrong
cell attributions.

This file is part of a 3-stage verification pipeline:

  Stage 1 — ``source_bbox_text_extractor.py``: extract text at each
            source-export bbox (native pdfium first, OCR fallback).
  Stage 2 — THIS FILE: compare extracted text to expected_value under E.
  Stage 3 — ``add_extract_field_rules_corrected_v05.py``: apply per-bbox
            verdicts to the generated rules' verified flag, with
            additional row-level fixes (multi-bbox concat re-verification,
            value-based bbox re-attribution for permuted columns).

See ``DATASET_CARD.md`` for the full taxonomy of verification tags and
their semantics.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal

from dateutil import parser as dateutil_parser  # type: ignore[import-untyped]
from dateutil.parser import ParserError  # type: ignore[import-untyped]
from rapidfuzz.distance import JaroWinkler

ExpectedType = Literal["string", "number", "boolean", "date", "null"]
ExtractionSource = Literal["native", "ocr", "failed"]
EquivalenceUsed = Literal[
    "null_empty",
    "identity",
    "case_insensitive",
    "trailing_punct_fold",
    "unicode_compat_fold",
    "footer_text_trim",
    "numeric",
    "date",
    "boolean",
    "ocr_punctuation_fold",
    "alnum_text_fold",
    "annotation_truncated",
    "ocr_noise_prefix",
    "icelandic_diacritic_fold",
    "none",
]

# Bump when ladder semantics change.
COMPARATOR_VERSION = "1.6.0"

# Rules below this score are not interesting for "near match" tagging.
TYPO_LIKELY_JW_THRESHOLD = 0.90

# Minimum alphanumeric length for the annotation_truncated rule to fire.
# Avoids accepting trivial prefixes like "Inc" matching anything.
_ANNOTATION_TRUNCATED_MIN_ALNUM = 10


@dataclass(frozen=True)
class Verdict:
    verified: bool
    equivalence_used: EquivalenceUsed
    normalized_expected: str
    normalized_extracted: str
    reason: str  # human-readable diagnostic; empty when verified=true
    # Jaro-Winkler similarity between normalized expected and extracted, in
    # [0.0, 1.0]. Computed for string-typed verdicts when verified=False so
    # downstream tooling can distinguish near-misses (typos, OCR confusions)
    # from totally-wrong cell attributions. None for verified=True or
    # non-string types.
    similarity_score: float | None = None


_WHITESPACE_RUN = re.compile(r"\s+", re.UNICODE)

# Unicode dashes that financial tables use to denote "blank / zero / N/A":
# em-dash (U+2014), en-dash (U+2013), minus sign (U+2212), hyphen-minus.
_DASH_OR_WS_ONLY = re.compile(r"^[\s—–−\-]*$", re.UNICODE)
# Sentence terminators + footnote/trademark markers + OCR-detected box
# borders that appear at the end of cells but not in the source annotation
# (or vice versa). Witnessed:
#   inventory_listing_1.stock_list[*].item_name → trailing ``*`` footnote
#   annual_financial_statement_1 line items      → trailing ``.``
#   k_1_bw_portfolio_limited part_iii numerics   → trailing ``]`` ``|`` ``/``
#     when the K-1 form's box border bleeds into the OCR output, e.g.
#     ``951797]``, ``3759 |``, ``401 9|``
# Note: do NOT include ``)`` or ``}`` in this class — closing parens are
# load-bearing for parens-as-negative numeric parsing (e.g. ``(123)`` =
# -123 via ``_NUMERIC_TEXT_RE``). We only add the chars witnessed as
# OCR-detected box borders that don't carry semantic meaning.
_TRAILING_PUNCT = re.compile(r"[.,;:!?*†‡§°™®©•·\]\|/]+\s*$", re.UNICODE)


def normalize_whitespace(s: str) -> str:
    """Collapse all whitespace runs to a single space and trim."""
    return _WHITESPACE_RUN.sub(" ", s).strip()


def _is_visually_empty(s: str) -> bool:
    """True if ``s`` is empty, whitespace-only, or a dash-only placeholder."""
    return bool(_DASH_OR_WS_ONLY.match(s))


def _strip_trailing_punct(s: str) -> str:
    """Strip trailing ``.,;:!?*†‡§°™®©•·]|/)}`` (with optional trailing whitespace)."""
    return _TRAILING_PUNCT.sub("", s).rstrip()


# Common payroll / report column-header words that appear appended to a
# cell's content when the bbox extends slightly past the row boundary
# into the next-band column header. Case-insensitive match.
_FOOTER_SUFFIX_RE = re.compile(
    r"\s*(?:Monthly|Weekly|Annual|Quarterly|Daily|Bi-?Weekly|Semi-?Monthly|Hourly)\s*$",
    re.IGNORECASE,
)


def _strip_footer_suffix(s: str) -> str:
    """Strip a trailing pay-period word (Monthly/Weekly/Annual/etc.) that
    OCR'd out of the column header into the cell."""
    return _FOOTER_SUFFIX_RE.sub("", s).rstrip()


# Map common PDF-rendering glyph variants to their ASCII equivalents so
# the comparator can fold visual-typographic differences. NFKC is applied
# first to handle ligatures, full-width digits, and CJK compatibility
# forms; this dict handles the residual glyphs whose NFKC decomposition
# is non-ASCII (e.g. µ U+00B5 → μ U+03BC, both still non-ASCII).
_PDF_GLYPH_TO_ASCII: dict[str, str] = {
    # Micro / mu — NFKC maps U+00B5 (MICRO SIGN) to U+03BC (GREEK SMALL
    # LETTER MU), which still isn't ASCII. Map both to ``u``.
    "µ": "u",
    "μ": "u",
    "Μ": "M",  # Greek Capital Mu (rare but symmetric)
    # Multiplication sign — appears in dimensions like ``4in × 5in``
    "×": "x",
    "÷": "/",
    # Superscripts / subscripts (only the common digits — Unicode has
    # tons of compatibility forms but these are what we see in PDFs)
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁰": "0",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
    # Hyphens / dashes (the dash-empty rule already handles these for
    # null comparison; this mapping helps when they appear MID-string)
    "—": "-",
    "–": "-",
    "−": "-",
    "‐": "-",
    "‑": "-",
    "‒": "-",
    # Quotes (curly to straight)
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    # Spaces (NBSP, thin space, em-quad, en-quad — some PDFs use these)
    " ": " ",
    " ": " ",
    " ": " ",
    " ": " ",
}


def _fold_pdf_glyphs(s: str) -> str:
    """Apply NFKC + explicit glyph→ASCII mapping for typographic variants.

    Used by Rule 3c (``unicode_compat_fold``) so visual differences like
    ``µ`` ↔ ``u`` and ``×`` ↔ ``x`` don't keep otherwise-equivalent
    cells from matching.
    """
    s = unicodedata.normalize("NFKC", s)
    return "".join(_PDF_GLYPH_TO_ASCII.get(ch, ch) for ch in s)


def _jaro_winkler(a: str, b: str) -> float:
    """Case-insensitive Jaro-Winkler similarity. 1.0 = identical, 0.0 = nothing in common."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return JaroWinkler.normalized_similarity(a.casefold(), b.casefold())


def compare(
    expected_value: Any,
    expected_type: ExpectedType,
    extracted_text: str,
    *,
    extraction_source: ExtractionSource = "native",
) -> Verdict:
    """Compare extracted bbox text against expected value under E.

    ``extraction_source`` gates rule 7 (OCR punctuation fold). It defaults to
    ``"native"`` so the comparator stays strict when the source is unknown
    or comes from the deterministic native pdfium text layer.
    """
    norm_extracted = normalize_whitespace(extracted_text)

    # Rule 1: null ↔ empty / dash-only.
    # Financial tables use ``—`` / ``–`` / ``-`` to denote a blank/zero cell;
    # treat those as visually empty for null-expected comparisons.
    if expected_value is None:
        if _is_visually_empty(norm_extracted):
            return Verdict(True, "null_empty", "", norm_extracted, "")
        return Verdict(
            False,
            "none",
            "",
            norm_extracted,
            f"expected null but extracted text is non-empty: {norm_extracted[:60]!r}",
        )

    norm_expected = normalize_whitespace(str(expected_value))

    # Rule 1b: dash-only ↔ empty when expected is the dash placeholder.
    # The bbox text-layer typically renders the em-dash as nothing or whitespace.
    # This is type-independent: financial statements often encode numeric
    # empty cells as dash placeholders while the schema still says ``number``.
    if _is_visually_empty(norm_expected) and _is_visually_empty(norm_extracted):
        return Verdict(True, "null_empty", norm_expected, norm_extracted, "")

    # Rule 2: identity (after whitespace normalization)
    if norm_expected == norm_extracted:
        return Verdict(True, "identity", norm_expected, norm_extracted, "")

    # Rule 3: case-insensitive (string types only — numbers/booleans/dates don't case-fold)
    if expected_type == "string" and norm_expected.casefold() == norm_extracted.casefold():
        return Verdict(True, "case_insensitive", norm_expected, norm_extracted, "")

    # Rule 3b: trailing-punctuation fold (string types only).
    # The bbox often includes the trailing period/comma at the end of the cell
    # while the source annotation omits it (or vice versa).
    if expected_type == "string":
        e_no_punct = _strip_trailing_punct(norm_expected)
        x_no_punct = _strip_trailing_punct(norm_extracted)
        if e_no_punct and e_no_punct.casefold() == x_no_punct.casefold():
            return Verdict(True, "trailing_punct_fold", norm_expected, norm_extracted, "")

    # Rule 3c: unicode_compat_fold (string types only).
    # The PDF text layer often renders typographic variants (µ U+00B5,
    # × U+00D7, ², ™, ½, full-width digits, ligatures, etc.) that the
    # source annotation stores as their ASCII compatibility equivalents.
    # NFKC normalization handles ligatures and full-width digits
    # automatically; an explicit ``_PDF_GLYPH_TO_ASCII`` fold handles the
    # remaining glyphs that NFKC alone doesn't reduce to ASCII (e.g. µ
    # NFKC-decomposes to greek μ, not ASCII u). After folding, we also
    # strip trailing punctuation so a ``"100uL"`` expected value matches
    # an OCR-extracted ``"100µL*"`` (Greek mu PLUS a footnote-marker
    # asterisk that ``trailing_punct_fold`` alone couldn't bridge with
    # the µ-vs-u glyph delta still present).
    # Witnessed in inventory_listing_1: ``50uL`` ↔ ``50µL``,
    # ``4in × 5in`` ↔ ``4in x 5in``, ``25cm²`` ↔ ``25cm2``.
    if expected_type == "string":
        e_glyph = _strip_trailing_punct(_fold_pdf_glyphs(norm_expected))
        x_glyph = _strip_trailing_punct(_fold_pdf_glyphs(norm_extracted))
        if (
            e_glyph
            and e_glyph.casefold() == x_glyph.casefold()
            and (e_glyph != norm_expected or x_glyph != norm_extracted)
        ):
            return Verdict(True, "unicode_compat_fold", norm_expected, norm_extracted, "")

    # Rule 3d: footer_text_trim (string types only).
    # When a bbox extends slightly past the row boundary, OCR can pick up
    # the column-header / footer text (e.g. ``Monthly``, ``Weekly``,
    # ``Annual``) appended to the cell content. Witnessed in payroll_15:
    # ``CALUGAY, MARIA IRA AN`` extracted as ``CALUGAY, MARIA IRA ANMonthly``.
    # Strip from both sides; only fire when at least one side actually
    # changes (ensures we're addressing real bleed, not random text ending
    # in "Monthly").
    if expected_type == "string":
        e_no_footer = _strip_footer_suffix(norm_expected)
        x_no_footer = _strip_footer_suffix(norm_extracted)
        if (
            e_no_footer
            and (e_no_footer != norm_expected or x_no_footer != norm_extracted)
            and e_no_footer.casefold() == x_no_footer.casefold()
        ):
            return Verdict(True, "footer_text_trim", norm_expected, norm_extracted, "")

    # Rule 4: numeric canonicalization. Fires for declared "number" type or
    # when the expected value is a genuine Python int/float — catches schema-
    # type bugs where a numeric field is mistakenly declared "string". The
    # boolean check is necessary because Python's `bool` is an int subclass.
    if expected_type == "number" or (isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool)):
        verdict = _compare_numeric(expected_value, norm_extracted, norm_expected)
        return _annotate_similarity(verdict, expected_type)

    # Rule 5: date canonicalization (string-typed in the source schema; check value shape)
    if expected_type in ("string", "date") and _looks_like_iso_date(norm_expected):
        verdict = _compare_date(norm_expected, norm_extracted)
        return _annotate_similarity(verdict, expected_type)

    # Rule 6: boolean canonicalization
    if expected_type == "boolean":
        verdict = _compare_boolean(expected_value, norm_extracted, norm_expected)
        return _annotate_similarity(verdict, expected_type)

    # Rule 7: OCR punctuation/spacing fold. Tesseract routinely misreads
    # commas as periods (and vice versa) and drops spaces around them in
    # cell-cropped fonts. Accept these only when the alphanumeric content
    # is byte-identical (case-folded), so structural mis-attributions like
    # ``"JONES, MARY" vs "EWILLIAMS\nSARAH"`` still fail. Limited to
    # string-typed values so numeric/date paths above are unaffected.
    if extraction_source == "ocr" and expected_type == "string":
        expected_alnum = _ALPHANUMERIC_STRIP.sub("", norm_expected).casefold()
        extracted_alnum = _ALPHANUMERIC_STRIP.sub("", norm_extracted).casefold()
        if expected_alnum and expected_alnum == extracted_alnum:
            return Verdict(True, "ocr_punctuation_fold", norm_expected, norm_extracted, "")

    # Rule 7c: alnum_text_fold — generalized counterpart to
    # ocr_punctuation_fold targeting numeric-rich text where the visual
    # punctuation (decimals, thousand separators) is stored as vector
    # glyphs rather than text characters. Witnessed in payroll_7's
    # rotated PDF native extraction: the visual ``"160.18 FIT 74.08 SS
    # 17.33 MED"`` is stored as separate digit/word runs joined by
    # whitespace (``"160 18 FIT 74 08 SS 17 33 MED"``).
    #
    # Gates (must satisfy ALL to avoid false positives):
    #   - alnum length >= 8 chars (substantive content)
    #   - digit count >= 4 chars (numeric-rich; rules out name-with-
    #     internal-punctuation cases like ``"Apple, Inc"`` ↔ ``"Apple Inc"``
    #     where punctuation IS load-bearing)
    #   - alnum content matches case-folded
    if expected_type == "string":
        expected_alnum = _ALPHANUMERIC_STRIP.sub("", norm_expected).casefold()
        extracted_alnum = _ALPHANUMERIC_STRIP.sub("", norm_extracted).casefold()
        digit_count = sum(1 for ch in expected_alnum if ch.isdigit())
        if len(expected_alnum) >= 8 and digit_count >= 4 and expected_alnum == extracted_alnum:
            return Verdict(True, "alnum_text_fold", norm_expected, norm_extracted, "")

    # Rule 7b: icelandic_diacritic_fold. The source export
    # appears to lose Icelandic special characters (``þ`` ``ð`` ``ó`` etc.)
    # and stores their ASCII fallbacks (``b`` ``d`` ``o``). The OCR-
    # extracted text correctly preserves the proper characters when
    # rendered with ``tesseract --lang isl``. Fold both sides through
    # the lossy ASCII mapping; if they match → verified=True with a tag
    # noting the loss. Marked for review so the data vendor is aware
    # the round-trip is lossy for these characters.
    if extraction_source == "ocr" and expected_type == "string":
        e_folded = _fold_icelandic(norm_expected).casefold()
        x_folded = _fold_icelandic(norm_extracted).casefold()
        if e_folded and e_folded == x_folded:
            return Verdict(True, "icelandic_diacritic_fold", norm_expected, norm_extracted, "")

    # Rule 8: annotation_truncated. The annotation was cut off but the bbox still
    # points at the correct cell — extracted starts with expected at the
    # alphanumeric level. Verified=True, and an ``annotation_truncated`` tag is
    # added.
    if expected_type == "string":
        expected_alnum = _ALPHANUMERIC_STRIP.sub("", norm_expected).casefold()
        extracted_alnum = _ALPHANUMERIC_STRIP.sub("", norm_extracted).casefold()
        if (
            len(expected_alnum) >= _ANNOTATION_TRUNCATED_MIN_ALNUM
            and len(extracted_alnum) > len(expected_alnum)
            and extracted_alnum.startswith(expected_alnum)
        ):
            return Verdict(True, "annotation_truncated", norm_expected, norm_extracted, "")

    # Rule 8b: ocr_noise_prefix — symmetric counterpart to annotation_truncated
    # for the case where OCR has a small (≤2 alnum chars) noise PREFIX on
    # the extracted text. Witnessed in k_1_bw_portfolio_limited multi-line
    # addresses joined: ``"1462 N Rodeo Dr,\nBeverly Hills, CA 90210"``
    # extracted (line-number bleed adds ``"1"``) for expected
    # ``"462 N Rodeo Dr,\nBeverly Hills, CA 90210"``. The expected_alnum is
    # a clean SUFFIX of extracted_alnum with 1-2 chars of leading noise.
    # Conservative: cap the noise budget tight to avoid false positives.
    if expected_type == "string":
        expected_alnum = _ALPHANUMERIC_STRIP.sub("", norm_expected).casefold()
        extracted_alnum = _ALPHANUMERIC_STRIP.sub("", norm_extracted).casefold()
        noise_budget = len(extracted_alnum) - len(expected_alnum)
        if (
            len(expected_alnum) >= _ANNOTATION_TRUNCATED_MIN_ALNUM
            and 1 <= noise_budget <= 2
            and extracted_alnum.endswith(expected_alnum)
        ):
            return Verdict(True, "ocr_noise_prefix", norm_expected, norm_extracted, "")

    # Diagnostic refinement: detect "annotation_partial" — the extracted text
    # is a clean alphanumeric substring of the expected value, suggesting the
    # source bbox covers only a fragment of the cell's content (e.g. a
    # multi-line cell where the source export marked only the first line). This is
    # NOT promoted to verified=True; we only refine the reason so downstream
    # categorization can tag these distinctly (`bronze_annotation_partial`).
    if _looks_like_annotation_partial(norm_expected, norm_extracted):
        return _annotate_similarity(
            Verdict(
                False,
                "none",
                norm_expected,
                norm_extracted,
                f"annotation_partial: extracted text is a strict substring of expected (type={expected_type})",
            ),
            expected_type,
        )

    return _annotate_similarity(
        Verdict(
            False,
            "none",
            norm_expected,
            norm_extracted,
            f"no equivalence rule matched (type={expected_type})",
        ),
        expected_type,
    )


# The source export of Icelandic strings appears to lose the
# special Icelandic characters (þ, ð, æ, ó, etc.) and substitute their
# closest ASCII equivalents. The OCR-extracted text from the PDF (via
# Tesseract with --lang isl) preserves the proper characters. Map the
# proper Icelandic letters to their lossy ASCII fallbacks so that:
#
#   "Strætó" → "Straeto"  matches  "Stræto"  (annotation lost the ó)
#   "Þór"     → "Por"      matches  "Por"     (Þ → P, ó → o)
#   "þjónusta"→ "bjonusta" matches  "bjónusta"(þ → b)
#   "erfðagreining" → "erfdagreining" matches "erfdagreining" (ð → d)
#
# The fold is applied to BOTH sides before comparison — if extracted has
# the proper character and expected has the ASCII fallback (or vice
# versa), they normalize to the same lossy form. Limited to extracted
# Icelandic OCR + string types so we don't accidentally match unrelated
# Latin-script strings.
_ICELANDIC_TO_ASCII_FOLD: dict[str, str] = {
    "þ": "b",
    "Þ": "P",  # thorn (source-export lossy: thorn → b/P)
    "ð": "d",
    "Ð": "D",  # eth
    "á": "a",
    "Á": "A",
    "é": "e",
    "É": "E",
    "í": "i",
    "Í": "I",
    "ó": "o",
    "Ó": "O",
    "ú": "u",
    "Ú": "U",
    "ý": "y",
    "Ý": "Y",
    "ö": "o",
    "Ö": "O",
    "æ": "ae",
    "Æ": "AE",
    # Tesseract on Icelandic company-name OCR sometimes reads ``&``
    # (literal ampersand on the page) as ``8``. Witnessed in
    # long_table_1: ``Mekka Wines & Spirits`` extracted as
    # ``Mekka Wines 8 Spirits``.
    "&": "8",
}


def _fold_icelandic(s: str) -> str:
    """Map Icelandic special characters to their lossy ASCII equivalents
    (matching how the source export degrades them)."""
    return "".join(_ICELANDIC_TO_ASCII_FOLD.get(ch, ch) for ch in s)


_NUMERIC_PRESENTATION_STRIP = re.compile(r"[\s,$€£¥%()]+")


def _canonicalize_numeric_for_jw(s: str) -> str:
    """Strip thousand-separators / currency / whitespace for fair numeric JW.

    Without this, ``"2671.43"`` vs ``"2,671.48"`` scores ~0.88 because the
    comma counts as a difference. With it, both reduce to ``"2671.43"`` vs
    ``"2671.48"`` and score ~0.93 — a fair "one digit off" signal.
    """
    return _NUMERIC_PRESENTATION_STRIP.sub("", s)


def _annotate_similarity(verdict: Verdict, expected_type: ExpectedType) -> Verdict:
    """Attach a Jaro-Winkler similarity score to unverified verdicts.

    Computed for ``string`` and ``number`` types — both can have OCR/typo
    near-misses worth flagging. Numeric values are pre-stripped of
    thousand-separators / currency / whitespace before scoring so
    presentation differences don't penalize the similarity.
    """
    if verdict.verified:
        return verdict
    if expected_type not in ("string", "number"):
        return verdict
    if not verdict.normalized_expected:
        return verdict
    if expected_type == "number":
        a = _canonicalize_numeric_for_jw(verdict.normalized_expected)
        b = _canonicalize_numeric_for_jw(verdict.normalized_extracted)
    else:
        a = verdict.normalized_expected
        b = verdict.normalized_extracted
    score = _jaro_winkler(a, b)
    return Verdict(
        verified=verdict.verified,
        equivalence_used=verdict.equivalence_used,
        normalized_expected=verdict.normalized_expected,
        normalized_extracted=verdict.normalized_extracted,
        reason=verdict.reason,
        similarity_score=score,
    )


_ALPHANUMERIC_STRIP = re.compile(r"[^a-zA-Z0-9]+")


def _looks_like_annotation_partial(expected: str, extracted: str) -> bool:
    """Return True iff ``extracted`` is a clean partial fragment of ``expected``.

    Detection criteria (kept conservative to avoid false positives):
      - Both sides have non-empty alphanumeric content after stripping
        punctuation/whitespace and case-folding.
      - ``extracted_alnum`` appears as a substring of ``expected_alnum``.
      - ``extracted_alnum`` is meaningfully shorter (≤ 75 % of expected length).
      - ``extracted_alnum`` length ≥ 2.
    """
    expected_alnum = _ALPHANUMERIC_STRIP.sub("", expected).casefold()
    extracted_alnum = _ALPHANUMERIC_STRIP.sub("", extracted).casefold()
    if len(extracted_alnum) < 2 or not expected_alnum:
        return False
    if extracted_alnum not in expected_alnum:
        return False
    return len(extracted_alnum) <= int(len(expected_alnum) * 0.75)


def _compare_numeric(expected: Any, norm_extracted: str, norm_expected: str) -> Verdict:
    expected_float = _coerce_to_float(expected)
    if expected_float is None:
        return Verdict(
            False,
            "none",
            norm_expected,
            norm_extracted,
            f"expected_value not coercible to number: {expected!r}",
        )
    extracted_float = _parse_numeric_text(norm_extracted)
    if extracted_float is None:
        return Verdict(
            False,
            "none",
            norm_expected,
            norm_extracted,
            "extracted text does not parse as a number",
        )
    if math.isclose(expected_float, extracted_float, rel_tol=1e-9, abs_tol=1e-6):
        return Verdict(True, "numeric", norm_expected, norm_extracted, "")
    return Verdict(
        False,
        "none",
        norm_expected,
        norm_extracted,
        f"numeric mismatch: expected={expected_float!r}, extracted={extracted_float!r}",
    )


def _coerce_to_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        return _parse_numeric_text(v)
    return None


_NUMERIC_TEXT_RE = re.compile(
    r"""^[\s$€£¥]*       # leading currency / whitespace
        (?P<sign>[-+]?)
        [\s$€£¥]*        # also allow currency BETWEEN sign and digits
                         # (e.g. ``-$21,948,982.12`` — bank statement
                         #  total_debits prefixes the sign before $)
        (?P<paren>\(?)
        \s*
        (?P<digits>\d{1,3}(?:[, ]\d{3})*(?:\.\d+)?|\.\d+|\d+(?:\.\d+)?)
        \s*
        (?P<paren_close>\)?)
        \s*[$€£¥%]*\s*$""",
    re.VERBOSE,
)

_DIGIT_GROUPS_WITH_2DIGIT_CENTS = re.compile(r"^\d+(?: \d+)+$")


def _parse_digit_groups_with_cents(text: str) -> float | None:
    """Parse ``"2 671 43"`` → 2671.43, also ``"55,450 45"`` → 55450.45.

    Witnessed in:
      - pymupdf native extraction of rotated PDFs where the visual
        separators (``,``, ``.``) are vector glyphs not in the text stream.
        Pymupdf returns each digit run as a separate "word" joined by
        spaces (e.g. ``"2 671 43"``).
      - Canadian/EU number formatting where commas separate thousands
        and a SPACE separates thousands from cents (e.g. ``"55,450 45"``,
        canadian_tax_form_t4_1.box_14_employment_income).

    Pre-strips thousand-separator commas so both shapes go through the
    same digit-groups regex. When the LAST group is exactly 2 digits,
    it's overwhelmingly the cents portion of a money value; concatenate
    all preceding groups as the integer part and divide.

    Returns None when:
      - text doesn't match the digit-groups-with-spaces pattern (after
        stripping commas)
      - last group is not exactly 2 digits
    """
    cleaned = text.strip().replace(",", "")
    if not _DIGIT_GROUPS_WITH_2DIGIT_CENTS.match(cleaned):
        return None
    parts = cleaned.split()
    if len(parts[-1]) != 2:
        return None
    integer_part = "".join(parts[:-1])
    cents_part = parts[-1]
    try:
        return float(f"{integer_part}.{cents_part}")
    except ValueError:
        return None


def _parse_numeric_text(text: str) -> float | None:
    """Parse ``'$1,739.12'``, ``'(123)'``, ``'1.5%'``, ``"2 671 43"``, ``"20098 *"`` → float.

    Strips trailing footnote / sentence-terminator markers (``.,;:!?*†‡§°™®©•·``)
    before parsing — bank-statement check numbers in particular show up as
    ``"20098 *"`` where the asterisk is a return-item indicator. Falls back
    to ``_parse_digit_groups_with_cents`` for the pymupdf-native pattern
    when the standard regex doesn't match. Returns None on failure.
    """
    if not text:
        return None
    cleaned = _strip_trailing_punct(text)
    match = _NUMERIC_TEXT_RE.match(cleaned)
    if not match:
        return _parse_digit_groups_with_cents(cleaned.strip())
    digits = match.group("digits").replace(",", "").replace(" ", "")
    try:
        value = float(digits)
    except ValueError:
        return None
    if match.group("sign") == "-" or (match.group("paren") and match.group("paren_close")):
        value = -value
    return value


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _looks_like_iso_date(s: str) -> bool:
    return bool(_ISO_DATE_RE.match(s))


def _compare_date(norm_expected: str, norm_extracted: str) -> Verdict:
    """Verify when expected is ``YYYY-MM-DD`` and extracted parses unambiguously to the same date."""
    try:
        expected_date = datetime.strptime(norm_expected, "%Y-%m-%d").date()
    except ValueError:
        return Verdict(
            False,
            "none",
            norm_expected,
            norm_extracted,
            "expected_value declared date-shaped but does not parse",
        )

    if not norm_extracted:
        return Verdict(False, "none", norm_expected, "", "extracted text is empty")

    parsed_us = _safe_dateutil_parse(norm_extracted, dayfirst=False)
    parsed_eu = _safe_dateutil_parse(norm_extracted, dayfirst=True)

    if parsed_us is None and parsed_eu is None:
        return Verdict(
            False,
            "none",
            norm_expected,
            norm_extracted,
            "extracted text does not parse as a date",
        )

    candidates = {d for d in (parsed_us, parsed_eu) if d is not None}
    # If the expected_date matches ONE of the parse candidates, the user-
    # provided expected value disambiguates between the US and EU
    # interpretations — accept. Without this, ``"12/06/1979"`` would be
    # rejected as ambiguous (US: 1979-12-06, EU: 1979-06-12) even when
    # expected is exactly ``"1979-12-06"``.
    if expected_date in candidates:
        return Verdict(True, "date", norm_expected, norm_extracted, "")
    if len(candidates) > 1:
        return Verdict(
            False,
            "none",
            norm_expected,
            norm_extracted,
            f"ambiguous date format (US={parsed_us}, EU={parsed_eu})",
        )

    extracted_date = next(iter(candidates))
    if extracted_date == expected_date:
        return Verdict(True, "date", norm_expected, norm_extracted, "")

    return Verdict(
        False,
        "none",
        norm_expected,
        norm_extracted,
        f"date mismatch: expected={expected_date.isoformat()}, extracted={extracted_date.isoformat()}",
    )


def _safe_dateutil_parse(text: str, *, dayfirst: bool) -> date | None:
    try:
        parsed = dateutil_parser.parse(text, dayfirst=dayfirst)
    except (ParserError, ValueError, OverflowError):
        return None
    if not isinstance(parsed, datetime):
        return None
    return parsed.date()


_BOOLEAN_TRUE = {"true", "yes", "y", "1", "x", "✓", "✗", "✘", "☒", "[x]", "checked"}
_BOOLEAN_FALSE = {"false", "no", "n", "0", "", "[ ]", "☐", "unchecked"}

# Maximum length of a short OCR token treated as "checkbox noise" → False.
# K-1 / tax-form checkboxes are drawn squares. Tesseract sometimes reads the
# square's outline as 1-4 letters/punctuation: 'LI', 'LJ', 'oO', 'im]', '|]'.
# When the expected_type is boolean and the extracted is a short token that
# isn't a known True/False marker, treat it as an empty checkbox.
_CHECKBOX_NOISE_MAX_LEN = 4


def _compare_boolean(expected: Any, norm_extracted: str, norm_expected: str) -> Verdict:
    expected_bool: bool
    if isinstance(expected, bool):
        expected_bool = expected
    elif isinstance(expected, str) and expected.lower() in _BOOLEAN_TRUE | _BOOLEAN_FALSE:
        expected_bool = expected.lower() in _BOOLEAN_TRUE
    else:
        return Verdict(
            False,
            "none",
            norm_expected,
            norm_extracted,
            f"expected_value not coercible to boolean: {expected!r}",
        )

    # Strip non-alphanumerics for token-set lookup so "yes." or "[X]" still
    # match the canonical "yes" / "x" markers.
    extracted_clean = re.sub(r"[^\w]", "", norm_extracted).casefold()

    if extracted_clean in _BOOLEAN_TRUE or norm_extracted in _BOOLEAN_TRUE:
        extracted_bool = True
    elif extracted_clean in _BOOLEAN_FALSE or norm_extracted in _BOOLEAN_FALSE:
        extracted_bool = False
    elif len(norm_extracted) <= _CHECKBOX_NOISE_MAX_LEN:
        # Short token, not a recognized marker — Tesseract noise from a
        # drawn checkbox border. Treat as empty checkbox → False. The
        # field that bbox-points to a checkbox glyph is binary by
        # construction; if we can't see an X-like mark, it's unchecked.
        extracted_bool = False
    else:
        return Verdict(
            False,
            "none",
            norm_expected,
            norm_extracted,
            f"extracted text not coercible to boolean: {norm_extracted!r}",
        )
    if extracted_bool == expected_bool:
        return Verdict(True, "boolean", norm_expected, norm_extracted, "")
    return Verdict(
        False,
        "none",
        norm_expected,
        norm_extracted,
        f"boolean mismatch: expected={expected_bool}, extracted={extracted_bool}",
    )


__all__ = [
    "COMPARATOR_VERSION",
    "EquivalenceUsed",
    "ExpectedType",
    "ExtractionSource",
    "TYPO_LIKELY_JW_THRESHOLD",
    "Verdict",
    "compare",
    "normalize_whitespace",
]
