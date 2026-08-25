"""Utility functions for parse evaluation."""

import re
import unicodedata

_SINGLE_QUOTE_CHARS = (
    "‘’‚‛"  # curly / low-9 / high-reversed-9
    "`´"  # grave accent, acute accent
    "ʼʹ＇"  # modifier apostrophe, modifier prime, fullwidth apostrophe
    "′‵"  # prime (U+2032), reversed prime (U+2035)
    "ʻˊˋ"  # turned comma (U+02BB), modifier acute (U+02CA), modifier grave (U+02CB)
)
_DOUBLE_QUOTE_CHARS = (
    "“”"  # left / right double quotation marks
    "„‟"  # double low-9 / high-reversed-9
    "〝〞"  # reversed double prime / double prime quotation
    "＂"  # fullwidth quotation mark
    "″‶"  # double prime (U+2033), reversed double prime (U+2036)
    "ˮ"  # modifier letter double apostrophe (U+02EE)
)
# Fullwidth punctuation forms (U+FF01..U+FF5E) → ASCII (U+0021..U+007E).
# Only the most common punctuation is listed explicitly; extend as needed.
_FULLWIDTH_PUNCT_CHARS = {
    "\uff0c": ",",  # fullwidth comma
    "\uff0e": ".",  # fullwidth full stop
    "\uff1a": ":",  # fullwidth colon
    "\uff1b": ";",  # fullwidth semicolon
    "\uff01": "!",  # fullwidth exclamation mark
    "\uff1f": "?",  # fullwidth question mark
    "\uff08": "(",  # fullwidth left parenthesis
    "\uff09": ")",  # fullwidth right parenthesis
    "\u3001": ",",  # ideographic comma (、)
    "\u3002": ".",  # ideographic full stop (。)
}

_QUOTE_TRANSLATION_TABLE = str.maketrans(
    {
        **dict.fromkeys(_SINGLE_QUOTE_CHARS, "'"),
        **dict.fromkeys(_DOUBLE_QUOTE_CHARS, '"'),
        **_FULLWIDTH_PUNCT_CHARS,
    }
)

# Ranges of CJK base characters whose combining marks (dakuten, handakuten)
# must be preserved during NFD accent stripping.
_CJK_BASE_RANGES = (
    ("\u3040", "\u309f"),  # Hiragana
    ("\u30a0", "\u30ff"),  # Katakana
    ("\u4e00", "\u9fff"),  # CJK Unified Ideographs
    ("\u3400", "\u4dbf"),  # CJK Extension A
    ("\uf900", "\ufaff"),  # CJK Compatibility Ideographs
    ("\uac00", "\ud7af"),  # Hangul Syllables
    ("\u1100", "\u11ff"),  # Hangul Jamo
    ("\u0b80", "\u0bff"),  # Tamil
    ("\u0c80", "\u0cff"),  # Kannada
)


def _is_cjk_base_char(ch: str) -> bool:
    """Return True if *ch* is a CJK/Kana/Hangul/Indic base character."""
    return any(lo <= ch <= hi for lo, hi in _CJK_BASE_RANGES)


# ---------------------------------------------------------------------------
# Unicode symbol equivalence classes
#
# Each entry maps a set of visually-similar Unicode characters to a single
# canonical character.  Used by normalize_cell_text() (and transitively by
# normalize_text / normalize_text_light) so that TEDS, GriTS, and other
# cell-level comparisons treat these variants as identical.
# ---------------------------------------------------------------------------

_UNICODE_SYMBOL_CLASSES: list[tuple[str, str]] = [
    # Bullet-like dots → standard bullet (U+2022)
    (
        "●"  # U+25CF BLACK CIRCLE
        "○"  # U+25CB WHITE CIRCLE
        "◦"  # U+25E6 WHITE BULLET
        "∙"  # U+2219 BULLET OPERATOR
        "⦁"  # U+2981 Z NOTATION SPOT
        "·",  # U+00B7 MIDDLE DOT
        "•",  # U+2022 BULLET (canonical)
    ),
    # Circled x / cross marks → ⊗ (U+2297 CIRCLED TIMES)
    (
        "⮾"  # U+2BBE CIRCLED X
        "ⓧ"  # U+24E7 CIRCLED LATIN SMALL LETTER X
        "⨂",  # U+2A02 N-ARY CIRCLED TIMES OPERATOR
        "⊗",  # U+2297 CIRCLED TIMES (canonical)
    ),
]

_UNICODE_SYMBOL_TABLE = str.maketrans(
    {char: canonical for chars, canonical in _UNICODE_SYMBOL_CLASSES for char in chars}
)


def _normalize_unicode_symbols(text: str) -> str:
    """Collapse Unicode symbol variants to their canonical forms."""
    return text.translate(_UNICODE_SYMBOL_TABLE)


def _normalize_quotes(text: str) -> str:
    """Map common Unicode quote/punctuation variants to ASCII equivalents."""
    return text.translate(_QUOTE_TRANSLATION_TABLE)


# ---------------------------------------------------------------------------
# Formatting / markup patterns stripped by normalize_cell_text()
# ---------------------------------------------------------------------------

# HTML formatting tags to strip (same set as header_accuracy_metric._FORMATTING_RE)
_HTML_FORMATTING_RE = re.compile(
    r"</?(?:b|i|u|s|em|strong|del|strike|mark|ins)>",
    re.IGNORECASE,
)

# <span> tags (with optional attributes like style, color) — strip tag, keep content
_HTML_SPAN_RE = re.compile(r"</?span\b[^>]*>", re.IGNORECASE)

# Markdown bold: **text** or __text__
_MD_BOLD_RE = re.compile(r"\*\*(.*?)\*\*|__(.*?)__")
# Markdown italic: *text* or _text_  (must not match ** or __)
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)|(?<!_)_(?!_)(.*?)(?<!_)_(?!_)")
# Markdown strikethrough: ~~text~~
_MD_STRIKETHROUGH_RE = re.compile(r"~~(.*?)~~")

# ---------------------------------------------------------------------------
# Sup/sub conversion (shared between GriTS normalize_cell_text and TRM
# normalize_table). Converts <sup>x</sup> / <sub>x</sub> tags to plain text
# and translates Unicode super/subscript characters to ASCII equivalents.
# ---------------------------------------------------------------------------

# Unicode superscript → ASCII mappings (digits + common letters/symbols)
_SUPERSCRIPT_TO_ASCII = str.maketrans(
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ",
    "0123456789+-=()ni",
)

# Unicode subscript → ASCII mappings
_SUBSCRIPT_TO_ASCII = str.maketrans(
    "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ",
    "0123456789+-=()aehijklmnoprstuvx",
)


def _normalize_sub_sup_for_table(text: str) -> str:
    """Convert sub/sup tags and Unicode chars to plain text for table comparison.

    Unlike ``normalize_text()`` which strips sup/sub entirely (correct for
    footnote markers in prose), tables use sup/sub for meaningful content
    like chemical formulas (H₂O) and exponents (x²).
    """
    text = re.sub(r"<sup[^>]*>(.*?)</sup>", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"<sub[^>]*>(.*?)</sub>", r"\1", text, flags=re.IGNORECASE)
    text = text.translate(_SUPERSCRIPT_TO_ASCII)
    text = text.translate(_SUBSCRIPT_TO_ASCII)
    return text


# Dash-like characters to normalize to ASCII hyphen
_DASH_CHARS = str.maketrans(
    {
        "–": "-",  # en-dash U+2013
        "—": "-",  # em-dash U+2014
        "‑": "-",  # non-breaking hyphen U+2011
        "‒": "-",  # figure dash U+2012
        "−": "-",  # minus sign U+2212
    }
)

# A cell consisting entirely of dash-like characters and whitespace
_DASH_ONLY_RE = re.compile(r"^[\s\-–—‑‒−]+$")


def normalize_cell_text(text: str) -> str:
    """Normalize a table cell's text content for metric comparison.

    Applies transformations suitable for cell-level comparison
    in TEDS, GriTS, and header-accuracy metrics:
    - Sup/sub tag conversion (<sup>x</sup> → x, ¹ → 1, etc.)
    - HTML formatting tag removal (<b>, <i>, <mark>, <em>, <strong>, etc.)
    - Markdown bold/italic/strikethrough removal
    - Unicode symbol equivalence (bullets, circled-x, etc.)
    - Quote / fullwidth punctuation canonicalization
    - Dash character normalization (en-dash, em-dash, etc. → ASCII hyphen)
    - Dash-only cells collapsed to a single "-"
    - Dot-leader stripping (trailing runs of 2+ dots)
    - Whitespace collapsing and stripping

    Formatting is intentionally stripped (not preserved as a signal): in
    practice, models routinely emphasize totals/headers with bold while
    ground-truth tables don't, and treating that mismatch as a content
    error penalizes parsing quality for an unrelated convention. This
    intentionally does NOT lowercase or strip accents.
    """
    # Sup/sub: tag conversion + Unicode → ASCII (shared with TRM normalization)
    text = _normalize_sub_sup_for_table(text)
    # Strip HTML formatting tags
    text = _HTML_FORMATTING_RE.sub("", text)
    # Strip <span> tags (keep content) — e.g. <span color="red">text</span> → text
    text = _HTML_SPAN_RE.sub("", text)
    # Strip markdown bold, then italic, then strikethrough
    text = _MD_BOLD_RE.sub(r"\1\2", text)
    text = _MD_ITALIC_RE.sub(r"\1\2", text)
    text = _MD_STRIKETHROUGH_RE.sub(r"\1", text)
    # Unicode symbol equivalence
    text = _normalize_unicode_symbols(text)
    # Quote / fullwidth punctuation canonicalization
    text = _normalize_quotes(text)
    # Normalize dash characters to ASCII hyphen
    text = text.translate(_DASH_CHARS)
    # Strip trailing dot-leaders (2+ consecutive dots at end)
    text = re.sub(r"\.{2,}\s*$", "", text)
    # Collapse whitespace and strip
    text = re.sub(r"\s+", " ", text).strip()
    # If cell is entirely dashes (after normalization), collapse to single dash
    if _DASH_ONLY_RE.match(text):
        return "-"
    return text


def normalize_text(md_content: str | None) -> str:
    """
    Normalize markdown text for comparison.

    This function:
    - Normalizes whitespace
    - Removes markdown formatting (bold, italics)
    - Normalizes unicode characters
    - Replaces fancy quotes and dashes with ASCII equivalents

    :param md_content: Markdown content to normalize
    :return: Normalized text
    """
    if md_content is None:
        return ""

    # Strip autolink angle brackets: <http://foo.bar> → http://foo.bar
    # Also handles mailto: and bare email autolinks (<user@host.tld>)
    md_content = re.sub(
        r"<((?:https?://|mailto:)[^>\s]+|[^>@\s]+@[^>@\s]+\.[^>@\s]+)>",
        r"\1",
        md_content,
        flags=re.IGNORECASE,
    )

    # Normalize <br>, <br/>, and <br /> to spaces
    md_content = re.sub(r"<br\s*/?>", " ", md_content)

    # Canonicalize Unicode quote variants early for robust text matching.
    md_content = _normalize_quotes(md_content)

    # Normalize whitespace in the md_content
    md_content = re.sub(r"\s+", " ", md_content)

    # Remove markdown bold formatting (** or __ for bold)
    md_content = re.sub(r"\*\*(.*?)\*\*", r"\1", md_content)
    md_content = re.sub(r"__(.*?)__", r"\1", md_content)
    md_content = re.sub(r"</?b>", "", md_content)  # Remove <b> tags if they exist
    md_content = re.sub(r"</?i>", "", md_content)  # Remove <i> tags if they exist

    # Remove markdown italics formatting (* or _ for italics)
    md_content = re.sub(r"\*(.*?)\*", r"\1", md_content)
    md_content = re.sub(r"_(.*?)_", r"\1", md_content)

    # Replace remaining underscores with spaces so filenames like
    # "099_20090718白山祭り088" split into separate tokens.  Paired italic
    # markers (_..._) are already stripped above; any leftover _ is a literal
    # underscore (e.g. in image filenames embedded in OCR output).
    # Stays aligned with JS annotation tool which does text.replace(/[*_~]+/g, " ").
    md_content = md_content.replace("_", " ")

    # Convert accented letters to ASCII equivalents (e.g., é -> e)
    # NFD decomposing separates base characters from combining marks
    md_content = unicodedata.normalize("NFD", md_content)
    # Remove combining characters (accents, diacritics) but KEEP combining marks
    # that follow CJK base characters (e.g. Japanese dakuten ゙ and handakuten ゚
    # which distinguish が from か, ぱ from は, etc.)
    result_chars: list[str] = []
    for char in md_content:
        if unicodedata.category(char) != "Mn":
            result_chars.append(char)
        else:
            # Keep combining mark if it follows a CJK base character
            # (Hiragana \u3040-\u309f, Katakana \u30a0-\u30ff, CJK ideographs, Hangul)
            if result_chars and _is_cjk_base_char(result_chars[-1]):
                result_chars.append(char)
            # else: strip (Latin/Cyrillic accents → ASCII)
    md_content = "".join(result_chars)
    # Convert back to NFC form for consistency
    md_content = unicodedata.normalize("NFC", md_content)

    # Dictionary of characters to replace: keys are fancy characters, values are ASCII equivalents
    replacements = {
        "＿": "_",
        "–": "-",
        "—": "-",
        "‑": "-",
        "‒": "-",
        "−": "-",
        "…": "...",
        "<ins>": "",
        "</ins>": "",
        "<u>": "",
        "</u>": "",
        "~~": "",
        "<mark>": "",
        "</mark>": "",
        "<br/>": " ",
        "<br />": " ",
        "\n": " ",
        "$$": "",  # Remove $$ signs as Latex delimiters are that way
        "\u00b5": "\u03bc",  # micro sign to greek mu
    }

    # Apply all replacements from the dictionary
    for fancy_char, ascii_char in replacements.items():
        md_content = md_content.replace(fancy_char, ascii_char)

    # Normalize Unicode symbol variants (bullets, circled-x, etc.)
    md_content = _normalize_unicode_symbols(md_content)

    # Strip <s>, <del>, <strike> tags (keep content) — equivalent to ~~ stripping above
    md_content = re.sub(r"</?(?:s|del|strike)>", "", md_content, flags=re.IGNORECASE)

    # Strip <span> tags with any attributes (keep content)
    # e.g. <span color="red">text</span> → text
    md_content = _HTML_SPAN_RE.sub("", md_content)

    # Remove <sup>...</sup> and <sub>...</sub> tags AND their content
    # (e.g., footnote markers like "84.1<sup>(2)</sup>" → "84.1")
    md_content = re.sub(r"<sup[^>]*>.*?</sup>", "", md_content, flags=re.IGNORECASE)
    md_content = re.sub(r"<sub[^>]*>.*?</sub>", "", md_content, flags=re.IGNORECASE)

    # Strip Unicode superscript digits (footnote markers like "84.1¹" → "84.1").
    # These are standalone codepoints that NFD decomposition does not decompose.
    # We strip rather than convert to regular digits to avoid changing values
    # (e.g., "84.1¹" → "84.11" would be wrong). Consistent with <sup> removal above.
    md_content = re.sub(r"[\u00b9\u00b2\u00b3\u2070\u2074-\u2079]+", "", md_content)

    # Strip Unicode subscript digits (e.g. "H₂O" → "HO"), consistent with
    # <sub> removal above and superscript digit stripping.
    md_content = re.sub(r"[\u2080-\u2089]+", "", md_content)

    # Normalize multiple consecutive dashes to single dash
    # This handles cases like "--" or "---" becoming "-"
    md_content = re.sub(r"-{2,}", "-", md_content)

    # Strip trailing dot-leaders (e.g., "Operating income.........." → "Operating income")
    # These are formatting dots used in tables to connect labels to values
    # Only strip 2+ consecutive dots at the end to preserve grammatical periods ("Inc.")
    md_content = re.sub(r"\.{2,}\s*$", "", md_content)

    # lowerCase the content for case-insensitive comparison
    md_content = md_content.lower()
    return md_content


def normalize_text_light(md_content: str | None) -> str:
    """
    Light normalization that preserves text formatting/styling.

    Unlike normalize_text(), this function:
    - KEEPS markdown formatting (bold **, italics *)
    - KEEPS HTML styling tags (<i>, <b>, <u>, <sup>, <sub>, <mark>, <ins>)
    - KEEPS dots/periods
    - KEEPS original case
    - Still normalizes whitespace and unicode quotes/dashes for reliable matching

    Use this when testing that formatting is correctly preserved in the output.

    :param md_content: Markdown content to normalize
    :return: Lightly normalized text with formatting preserved
    """
    if md_content is None:
        return ""

    # Strip autolink angle brackets: <http://foo.bar> → http://foo.bar
    md_content = re.sub(
        r"<((?:https?://|mailto:)[^>\s]+|[^>@\s]+@[^>@\s]+\.[^>@\s]+)>",
        r"\1",
        md_content,
        flags=re.IGNORECASE,
    )

    # Normalize <br> and <br/> to spaces (these are layout, not styling)
    md_content = re.sub(r"<br/?>", " ", md_content)

    # Strip <span> tags (keep content) — these are layout wrappers, not styling
    md_content = _HTML_SPAN_RE.sub("", md_content)

    # Canonicalize Unicode quote variants for robust matching while preserving styling.
    md_content = _normalize_quotes(md_content)

    # Normalize whitespace (collapse multiple spaces/newlines to single space)
    md_content = re.sub(r"\s+", " ", md_content)

    # Convert accented letters to ASCII equivalents (e.g., é -> e)
    # This helps with matching even when accents differ
    md_content = unicodedata.normalize("NFD", md_content)
    md_content = "".join(
        char
        for char in md_content
        if unicodedata.category(char) != "Mn"  # Mn = Nonspacing_Mark (accents)
    )
    md_content = unicodedata.normalize("NFC", md_content)

    # Only normalize dashes/symbols to ASCII equivalents
    # Keep dots, keep case, keep formatting tags
    replacements = {
        "＿": "_",
        "–": "-",
        "—": "-",
        "‑": "-",
        "‒": "-",
        "−": "-",
        "…": "...",
        "\u00b5": "\u03bc",  # micro sign to greek mu
    }

    for fancy_char, ascii_char in replacements.items():
        md_content = md_content.replace(fancy_char, ascii_char)

    # Normalize Unicode symbol variants (bullets, circled-x, etc.)
    md_content = _normalize_unicode_symbols(md_content)

    # Normalize multiple consecutive dashes to single dash
    md_content = re.sub(r"-{2,}", "-", md_content)

    return md_content.strip()
