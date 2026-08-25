"""Formatting, LaTeX, code block, title, and page section test rules."""

import re
import unicodedata
from typing import Any, cast

from extract_bench.evaluation.metrics.parse.rules_base import (
    ParseTestRule,
    _unescape_html_entities,
)
from extract_bench.evaluation.metrics.parse.test_types import ParseRuleType
from extract_bench.evaluation.metrics.parse.utils import normalize_text
from extract_bench.test_cases.parse_rule_schemas import (
    ParseCodeBlockRule,
    ParseFormattingRule,
    ParseLatexRule,
    ParseMarkColorRule,
    ParsePageSectionRule,
    ParseTitleHierarchyPercentRule,
    ParseTitleRule,
)

# ---------------------------------------------------------------------------
# Unicode superscript / subscript character tables
# ---------------------------------------------------------------------------
# Inline markup tolerance – allows regex patterns built from stripped rule text
# to still match raw markdown that contains nested formatting markers.
# E.g. rule text "hello world" matches raw "hello ~~world~~".
# ---------------------------------------------------------------------------

# Matches optional inline formatting tokens that may appear between words
# in raw markdown: strikethrough (~~), bold (**), italic (* or _), and
# any HTML open/close/self-closing tags (<b>, </b>, <mark>, <sup>, etc.).
_INLINE_MARKUP_OPT = r"(?:\*{1,2}|~~|__?|</?\w+(?:\s[^>]*)?>)*"

# Markdown allows backslash-escaping of ASCII punctuation characters.
# Model output often contains these (e.g. ``\~``, ``\*``) which prevent
# rule text from matching.  This pattern strips such escapes.
_MD_BACKSLASH_ESCAPE_RE = re.compile(r"\\([!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])")


def _make_markup_tolerant(escaped_text: str) -> str:
    """Make an escaped regex pattern tolerant of inline formatting markup.

    Between each word boundary (space) and at the start/end of the text,
    allow optional inline markup tokens (``~~``, HTML tags) so that
    clean rule text can match raw content with nested formatting.
    """
    # Split on escaped spaces (re.escape turns " " into "\\ ")
    # and rejoin with markup-tolerant whitespace
    parts = escaped_text.split(r"\ ")
    joiner = _INLINE_MARKUP_OPT + r"\s+" + _INLINE_MARKUP_OPT
    tolerant = joiner.join(parts)
    # Allow leading and trailing markup adjacent to the text
    return _INLINE_MARKUP_OPT + tolerant + _INLINE_MARKUP_OPT


# Regex patterns for stripping inline formatting by kind.
# Used by _strip_other_formatting() to remove all formatting EXCEPT the type
# being tested, so the outer markers remain for pattern matching.
_STRIP_PATTERNS: dict[str, list[tuple[re.Pattern, str]]] = {
    "bold": [
        (re.compile(r"\*\*\*(.+?)\*\*\*", re.DOTALL), r"\1"),
        (re.compile(r"\*\*(.+?)\*\*", re.DOTALL), r"\1"),
        (re.compile(r"</?b>", re.IGNORECASE), ""),
    ],
    "italic": [
        (re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)"), r"\1"),
        (re.compile(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)"), r"\1"),
        (re.compile(r"</?(?:i|em)>", re.IGNORECASE), ""),
    ],
    "underline": [
        (re.compile(r"</?(?:u|ins)>", re.IGNORECASE), ""),
    ],
    "strikeout": [
        (re.compile(r"~~"), ""),
        (re.compile(r"</?(?:s|del|strike)>", re.IGNORECASE), ""),
    ],
    "mark": [
        (re.compile(r"</?mark\b[^>]*>", re.IGNORECASE), ""),
    ],
    "sup": [
        (re.compile(r"</?sup>", re.IGNORECASE), ""),
    ],
    "sub": [
        (re.compile(r"</?sub>", re.IGNORECASE), ""),
    ],
}


def _strip_other_formatting(text: str, keep_kind: str) -> str:
    """Strip all inline formatting markers EXCEPT those of *keep_kind*.

    This lets us re-try pattern matching after removing nested markup that
    would otherwise prevent the outer-marker regex from matching the clean
    rule text.
    """
    result = text
    for kind, replacements in _STRIP_PATTERNS.items():
        if kind == keep_kind:
            continue
        for pattern, repl in replacements:
            result = pattern.sub(repl, result)
    # Collapse whitespace (preserving line structure — the heading arm of
    # the bold patterns is line-anchored) and fix stray spaces before
    # punctuation
    result = re.sub(r"[^\S\n]+", " ", result)
    result = re.sub(r" ?\n ?", "\n", result)
    result = re.sub(r" ([,;:!?.\)\]\}])", r"\1", result)
    return result


# Used by FormattingRule to detect Unicode-encoded sup/sub alongside HTML tags
# ---------------------------------------------------------------------------
_UNICODE_SUPERSCRIPT_CHARS = set("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻᴬᴮᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᴿᵀᵁⱽᵂ")

_UNICODE_SUBSCRIPT_CHARS = set("₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")


class FormattingRule(ParseTestRule):
    """Test rule to verify that specific text has (or lacks) a formatting style.

    Each formatting type defines regex patterns to detect formatting markers
    (markdown syntax or HTML tags) wrapping the target text. The rule searches
    the *raw* markdown content (before normalize_text strips markers) so that
    formatting information is still available.

    Supported formatting kinds and their detection patterns:
    - bold:      **text** or <b>text</b>
    - italic:    *text* or _text_ or <i>text</i>
    - underline: <u>text</u> or <ins>text</ins>
    - strikeout: ~~text~~
    - mark:      <mark>text</mark>
    - sup:       <sup>text</sup> or Unicode superscript characters
    - sub:       <sub>text</sub> or Unicode subscript characters
    """

    # Map formatting kind -> list of regex-builder functions.
    # Each function receives the escaped query text and returns a compiled
    # pattern that should match the formatted occurrence in raw content.
    _FORMATTING_PATTERNS: dict[str, list] = {}  # populated below after class body

    def __init__(self, rule_data: ParseFormattingRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseFormattingRule, self._rule_data)

        # Derive formatting kind and polarity from the test type value.
        # E.g. "is_bold" -> kind="bold", expect_present=True
        #      "is_not_bold" -> kind="bold", expect_present=False
        type_value = self.type
        if type_value.startswith("is_not_"):
            self.formatting_kind = type_value[len("is_not_") :]
            self.expect_present = False
        elif type_value.startswith("is_"):
            self.formatting_kind = type_value[len("is_") :]
            self.expect_present = True
        else:
            raise ValueError(f"Invalid type for FormattingRule: {type_value}")

        if self.formatting_kind not in self._FORMATTING_PATTERNS:
            raise ValueError(f"Unsupported formatting kind: {self.formatting_kind}")

        raw_text = rule_data.text
        if not raw_text.strip():
            raise ValueError("Text field cannot be empty")
        self.text = raw_text.strip()

    # ------------------------------------------------------------------
    # Pattern builders – called with re.escape(query) to build detectors
    # ------------------------------------------------------------------

    @staticmethod
    def _build_bold_patterns(escaped_query: str) -> list[re.Pattern]:
        """Detect **text**, <b>text</b>, or markdown heading lines (# text).

        The query may be a substring of the bold span (e.g. query
        ``Population`` matches ``**Population:**``).

        The ``**`` delimiters follow CommonMark flanking (an opener is not
        followed by whitespace, a closer is not preceded by it) and the
        ``<b>`` filler cannot cross a closing tag — otherwise the closer of
        one span pairs with the opener of the next and the unformatted text
        between two bold spans (``**Name:** John **Age:**``) tests as bold.
        The heading arm's fillers stay on the heading's own line for the
        same reason. (The whitespace joiners inside a multi-word query can
        still cross these boundaries; only the fillers are constrained.)
        """
        # Tempered greedy token: match any char that doesn't start a new **
        _not_bold_close = r"(?:(?!\*\*).)*?"
        # Same idea for the HTML arm: don't cross a closing </b>
        _not_b_close_tag = r"(?:(?!</b>).)*?"
        return [
            re.compile(
                r"\*\*(?!\s)" + _not_bold_close + escaped_query + _not_bold_close + r"(?<!\s)\*\*",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"<b>" + _not_b_close_tag + escaped_query + _not_b_close_tag + r"</b>",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"^[ \t]*#{1,6}[ \t]+[^\n]*?" + escaped_query + r"[^\n]*?[ \t]*(?:#+[ \t]*)?$",
                re.IGNORECASE | re.MULTILINE,
            ),
        ]

    @staticmethod
    def _build_italic_patterns(escaped_query: str) -> list[re.Pattern]:
        """Detect *text* (not **), _text_ (not __), or <i>text</i>.

        The query may be a substring of the italic span (e.g. query
        ``Grazing Line`` matches ``*Grazing Line, Macquarie Marshes, NSW*``).
        """
        # Tempered greedy token: any char that isn't a lone * (i.e. not * unless followed by *)
        _not_italic_close_star = r"(?:(?!\*(?!\*)).)*?"
        # Same idea for _
        _not_italic_close_under = r"(?:(?!_(?!_)).)*?"
        # And for the HTML arms: the fillers must not cross a closing tag,
        # or adjacent spans (<i>A</i> x <i>B</i>) merge into one match.
        _not_i_close_tag = r"(?:(?!</i>).)*?"
        _not_em_close_tag = r"(?:(?!</em>).)*?"
        return [
            # *text* but NOT **text** – use negative lookbehind/lookahead
            re.compile(
                r"(?<!\*)\*(?!\*)" + _not_italic_close_star + escaped_query + _not_italic_close_star + r"\*(?!\*)",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"(?<!_)_(?!_)" + _not_italic_close_under + escaped_query + _not_italic_close_under + r"_(?!_)",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"<i>" + _not_i_close_tag + escaped_query + _not_i_close_tag + r"</i>",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"<em>" + _not_em_close_tag + escaped_query + _not_em_close_tag + r"</em>",
                re.IGNORECASE | re.DOTALL,
            ),
        ]

    @staticmethod
    def _build_underline_patterns(escaped_query: str) -> list[re.Pattern]:
        """Detect <u>text</u> or <ins>text</ins>."""
        return [
            re.compile(r"<u>" + escaped_query + r"</u>", re.IGNORECASE),
            re.compile(r"<ins>" + escaped_query + r"</ins>", re.IGNORECASE),
        ]

    @staticmethod
    def _build_strikeout_patterns(escaped_query: str) -> list[re.Pattern]:
        """Detect ~~text~~ or <s>text</s> / <del>text</del> / <strike>text</strike>."""
        return [
            re.compile(r"~~" + escaped_query + r"~~", re.IGNORECASE),
            re.compile(
                r"<(?:s|del|strike)>" + escaped_query + r"</(?:s|del|strike)>",
                re.IGNORECASE,
            ),
        ]

    @staticmethod
    def _build_mark_patterns(escaped_query: str) -> list[re.Pattern]:
        """Detect <mark>text</mark>."""
        return [
            re.compile(r"<mark>" + escaped_query + r"</mark>", re.IGNORECASE),
        ]

    @staticmethod
    def _build_sup_patterns(escaped_query: str) -> list[re.Pattern]:
        """Detect <sup>text</sup>. Unicode superscripts handled separately."""
        return [
            re.compile(r"<sup>" + escaped_query + r"</sup>", re.IGNORECASE),
        ]

    @staticmethod
    def _build_sub_patterns(escaped_query: str) -> list[re.Pattern]:
        """Detect <sub>text</sub>. Unicode subscripts handled separately."""
        return [
            re.compile(r"<sub>" + escaped_query + r"</sub>", re.IGNORECASE),
        ]

    def _has_unicode_superscript(self, content: str) -> bool:
        """Check if content contains consecutive Unicode superscript chars
        that correspond to self.text (case-insensitive)."""
        query_lower = self.text.lower()
        # Build a string of superscript chars found consecutively
        for i in range(len(content)):
            if content[i] in _UNICODE_SUPERSCRIPT_CHARS:
                # Collect the full run of superscript characters
                run = []
                j = i
                while j < len(content) and content[j] in _UNICODE_SUPERSCRIPT_CHARS:
                    run.append(content[j])
                    j += 1
                # Transliterate to plain ASCII/Latin and compare
                transliterated = unicodedata.normalize("NFKD", "".join(run)).lower()
                if query_lower in transliterated:
                    return True
        return False

    def _has_unicode_subscript(self, content: str) -> bool:
        """Check if content contains consecutive Unicode subscript chars
        that correspond to self.text (case-insensitive)."""
        query_lower = self.text.lower()
        for i in range(len(content)):
            if content[i] in _UNICODE_SUBSCRIPT_CHARS:
                run = []
                j = i
                while j < len(content) and content[j] in _UNICODE_SUBSCRIPT_CHARS:
                    run.append(content[j])
                    j += 1
                transliterated = unicodedata.normalize("NFKD", "".join(run)).lower()
                if query_lower in transliterated:
                    return True
        return False

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        """Check if the target text has the expected formatting in raw markdown.

        We search the *raw* md_content (not the normalized version) because
        normalize_text strips all formatting markers.

        The query is made markup-tolerant so that clean rule text (e.g.
        ``"hello world"``) still matches raw content where the words are
        wrapped in nested formatting (e.g. ``"hello ~~world~~"``).

        As a fallback, other formatting markers are stripped from the content
        while keeping the markers for the kind being tested, so the regex
        can match even when nested markup is adjacent to words (no space).
        """
        # Un-escape markdown backslash sequences (e.g. \~ → ~) so that
        # clean rule text can match content produced by parsers that emit
        # backslash-escaped punctuation.
        md_clean = _MD_BACKSLASH_ESCAPE_RE.sub(r"\1", md_content)

        escaped_query = re.escape(self.text)
        # Allow flexible whitespace *and* optional inline markup between words
        flexible_query = _make_markup_tolerant(escaped_query)

        patterns = self._FORMATTING_PATTERNS[self.formatting_kind](flexible_query)  # type: ignore[operator]
        found = any(p.search(md_clean) for p in patterns)

        # Fallback: strip OTHER formatting from content, keeping only the kind
        # being tested, then retry with a simple flexible-whitespace pattern.
        if not found:
            stripped = _strip_other_formatting(md_clean, self.formatting_kind)
            simple_query = re.sub(r"\\ ", r"\\s+", escaped_query)
            simple_patterns = self._FORMATTING_PATTERNS[self.formatting_kind](simple_query)  # type: ignore[operator]
            found = any(p.search(stripped) for p in simple_patterns)

        # For sup/sub, also check Unicode superscript/subscript characters
        if not found and self.formatting_kind == "sup":
            found = self._has_unicode_superscript(md_content)
        if not found and self.formatting_kind == "sub":
            found = self._has_unicode_subscript(md_content)

        if self.expect_present:
            if found:
                return True, ""
            return (
                False,
                f"Expected '{self.text[:40]}' to be formatted as {self.formatting_kind}, "
                f"but no {self.formatting_kind} formatting found",
            )
        else:
            if not found:
                return True, ""
            return (
                False,
                f"Expected '{self.text[:40]}' NOT to be formatted as {self.formatting_kind}, "
                f"but unexpectedly had {self.formatting_kind} formatting",
            )


# Wire up pattern builders after class body is defined
FormattingRule._FORMATTING_PATTERNS = {
    "bold": FormattingRule._build_bold_patterns,  # type: ignore[dict-item]
    "italic": FormattingRule._build_italic_patterns,  # type: ignore[dict-item]
    "underline": FormattingRule._build_underline_patterns,  # type: ignore[dict-item]
    "strikeout": FormattingRule._build_strikeout_patterns,  # type: ignore[dict-item]
    "mark": FormattingRule._build_mark_patterns,  # type: ignore[dict-item]
    "sup": FormattingRule._build_sup_patterns,  # type: ignore[dict-item]
    "sub": FormattingRule._build_sub_patterns,  # type: ignore[dict-item]
}

# Collect all formatting TestType values handled by FormattingRule
_FORMATTING_TEST_TYPES = {
    ParseRuleType.IS_UNDERLINE.value,
    ParseRuleType.IS_NOT_UNDERLINE.value,
    ParseRuleType.IS_BOLD.value,
    ParseRuleType.IS_NOT_BOLD.value,
    ParseRuleType.IS_STRIKEOUT.value,
    ParseRuleType.IS_NOT_STRIKEOUT.value,
    ParseRuleType.IS_ITALIC.value,
    ParseRuleType.IS_NOT_ITALIC.value,
    ParseRuleType.IS_MARK.value,
    ParseRuleType.IS_NOT_MARK.value,
    ParseRuleType.IS_SUP.value,
    ParseRuleType.IS_NOT_SUP.value,
    ParseRuleType.IS_SUB.value,
    ParseRuleType.IS_NOT_SUB.value,
    ParseRuleType.MARK_COLOR.value,
}


# Regex to match <mark ...>text</mark> capturing the opening tag attributes and inner text.
_MARK_TAG_PATTERN = re.compile(
    r"<mark\b([^>]*)>([\s\S]+?)</mark>",
    re.IGNORECASE,
)


class MarkColorRule(ParseTestRule):
    """Test rule to verify that text is wrapped in a ``<mark>`` tag with a specific color.

    Passes when:
    1. The text is found inside a ``<mark>`` tag, AND
    2. The ``<mark>`` tag contains the expected color string in any of its attributes
       (e.g. ``style="background-color: yellow"``, ``background="yellow"``,
       ``backgroundColor="yellow"``).
    """

    def __init__(self, rule_data: ParseMarkColorRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseMarkColorRule, self._rule_data)

        if self.type != ParseRuleType.MARK_COLOR.value:
            raise ValueError(f"Invalid type for MarkColorRule: {self.type}")

        raw_text = rule_data.text
        if not raw_text.strip():
            raise ValueError("Text field cannot be empty")
        self.text = raw_text.strip()

        raw_color = rule_data.color
        if not raw_color.strip():
            raise ValueError("Color field cannot be empty")
        self.color = raw_color.strip().lower()

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        """Check if text is inside a <mark> tag that contains the expected color."""
        escaped_query = re.escape(self.text)
        flexible_query = _make_markup_tolerant(escaped_query)
        text_pattern = re.compile(flexible_query, re.IGNORECASE)

        for match in _MARK_TAG_PATTERN.finditer(md_content):
            attrs_str = match.group(1)
            inner_text = match.group(2)

            # Check if the target text is inside this <mark> tag
            if not text_pattern.search(inner_text):
                continue

            # Check if the color string appears in the tag attributes
            if self.color in attrs_str.lower():
                return True, ""

        # Fallback: strip other formatting and retry
        stripped = _strip_other_formatting(md_content, "mark")
        for match in _MARK_TAG_PATTERN.finditer(stripped):
            attrs_str = match.group(1)
            inner_text = match.group(2)

            if not text_pattern.search(inner_text):
                continue

            if self.color in attrs_str.lower():
                return True, ""

        return (
            False,
            f"Expected '{self.text[:40]}' to be inside a <mark> tag with color '{self.color}', "
            f"but no matching <mark> tag found",
        )


def _strip_latex_delimiters(formula: str) -> str:
    stripped = formula.strip()
    if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) >= 4:
        return stripped[2:-2].strip()
    if stripped.startswith("$") and stripped.endswith("$") and len(stripped) >= 2:
        return stripped[1:-1].strip()
    if stripped.startswith(r"\(") and stripped.endswith(r"\)") and len(stripped) >= 4:
        return stripped[2:-2].strip()
    if stripped.startswith(r"\[") and stripped.endswith(r"\]") and len(stripped) >= 4:
        return stripped[2:-2].strip()
    return stripped


def _normalize_latex_formula(formula: str) -> str:
    body = _strip_latex_delimiters(formula)
    body = _unescape_html_entities(body)
    body = re.sub(r"\s+", "", body)
    return body


def _extract_latex_formulas(md_content: str) -> set[str]:
    formulas: set[str] = set()
    block_dollar = re.compile(r"(?<!\\)\$\$(.+?)(?<!\\)\$\$", re.DOTALL)
    inline_dollar = re.compile(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$(?!\$)", re.DOTALL)
    inline_paren = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
    block_bracket = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)

    for candidate in _title_content_candidates(md_content):
        for pattern in (block_dollar, inline_dollar, inline_paren, block_bracket):
            for match in pattern.finditer(candidate):
                normalized = _normalize_latex_formula(match.group(1))
                if normalized:
                    formulas.add(normalized)
    return formulas


class LatexRule(ParseTestRule):
    """Check that a specific inline or block LaTeX formula is present."""

    def __init__(self, rule_data: ParseLatexRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseLatexRule, self._rule_data)

        if self.type != ParseRuleType.IS_LATEX.value:
            raise ValueError(f"Invalid type for LatexRule: {self.type}")

        raw_formula = rule_data.formula
        if not isinstance(raw_formula, str) or not raw_formula.strip():
            raise ValueError("formula must be a non-empty string")

        self.formula = raw_formula.strip()
        self.normalized_formula = _normalize_latex_formula(self.formula)
        if not self.normalized_formula:
            raise ValueError("formula is empty after normalization")

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        found_formulas = _extract_latex_formulas(md_content)
        if self.normalized_formula in found_formulas:
            return True, ""

        preview = ", ".join(sorted(found_formulas)[:3])
        placeholder_hint = ""
        if "LATEX" in md_content and not found_formulas:
            placeholder_hint = (
                " Content appears to contain 'LATEX' placeholder tokens "
                "and no raw formula delimiters,"
                " suggesting upstream preprocessing replaced formulas before this rule."
            )
        return (
            False,
            (
                f"Expected LaTeX formula '{self.formula[:80]}' not found. "
                f"Detected formulas (normalized preview): {preview}"
                if preview
                else f"Expected LaTeX formula '{self.formula[:80]}' not found.{placeholder_hint}"
            ),
        )


def _extract_fenced_code_blocks(md_content: str) -> list[tuple[str, str]]:
    """Extract markdown fenced code blocks as (language, code)."""
    blocks: list[tuple[str, str]] = []
    # ```lang\n...\n``` (language optional)
    pattern = re.compile(r"(?ms)^[ \t]*```(?P<lang>[^\n`]*)\n(?P<body>.*?)[ \t]*\n[ \t]*```[ \t]*$")

    for candidate in _title_content_candidates(md_content):
        for match in pattern.finditer(candidate):
            lang = match.group("lang").strip().lower()
            body = match.group("body").strip()
            blocks.append((lang, body))
    return blocks


class CodeBlockRule(ParseTestRule):
    """Check that a fenced code block with a given language contains target code.

    Matching is permissive: whitespace is collapsed before comparison so that
    minor indentation or line-break differences do not cause failures.
    """

    _WS_COLLAPSE = re.compile(r"\s+")

    def __init__(self, rule_data: ParseCodeBlockRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseCodeBlockRule, self._rule_data)

        if self.type != ParseRuleType.IS_CODE_BLOCK.value:
            raise ValueError(f"Invalid type for CodeBlockRule: {self.type}")

        raw_language = rule_data.language
        raw_code = rule_data.code

        if not isinstance(raw_language, str) or not raw_language.strip():
            raise ValueError("language must be a non-empty string")
        if not isinstance(raw_code, str) or not raw_code.strip():
            raise ValueError("code must be a non-empty string")

        self.language = raw_language.strip().lower()
        self.code = raw_code.strip()
        self._code_normalized = self._WS_COLLAPSE.sub(" ", self.code)

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        blocks = _extract_fenced_code_blocks(md_content)

        matching_lang_blocks = [body for lang, body in blocks if lang == self.language]
        if not matching_lang_blocks:
            available = ", ".join(sorted({lang for lang, _ in blocks if lang}))
            return (
                False,
                (
                    f"No fenced code block found with language '{self.language}'."
                    + (f" Available languages: {available}" if available else "")
                ),
            )

        for body in matching_lang_blocks:
            # Exact substring first, then whitespace-normalized fallback
            if self.code in body:
                return True, ""
            if self._code_normalized in self._WS_COLLAPSE.sub(" ", body):
                return True, ""

        return (
            False,
            f"Found '{self.language}' code block(s), but none contained snippet '{self.code[:80]}'",
        )


def _title_content_candidates(md_content: str) -> list[str]:
    """Return raw + decoded/de-escaped content variants for title matching."""
    candidates: list[str] = []

    def _append_unique(value: str) -> None:
        if value not in candidates:
            candidates.append(value)

    _append_unique(md_content)
    unescaped_content = _unescape_html_entities(md_content)
    _append_unique(unescaped_content)

    markdown_unescaped = re.sub(r"(?m)^\\(#{1,6}\s+)", r"\1", unescaped_content)
    markdown_unescaped = markdown_unescaped.replace(r"\*\*", "**")
    _append_unique(markdown_unescaped)
    return candidates


def _normalize_title_label(text: str) -> str:
    """Normalize title text for case/spacing-insensitive comparisons."""
    return normalize_text(text).strip()


def _extract_title_events(md_content: str) -> list[tuple[int, int, str]]:
    """Extract title events as (line_index, level, normalized_text).

    Levels: 1-6 for heading levels, 7 for bold-title lines (lowest title level).
    A bold title line must start at line beginning and contain only the bold text.
    """
    events: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int, str]] = set()

    html_heading_regex = re.compile(r"<h([1-6])[^>]*>\s*(.*?)\s*</h\1>", re.IGNORECASE)
    html_bold_line_regex = re.compile(r"^\s*<b[^>]*>\s*(.*?)\s*</b>\s*$", re.IGNORECASE)
    md_heading_regex = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$")
    md_bold_line_regex = re.compile(r"^\s*\*\*\s*(.+?)\s*\*\*\s*$")

    for candidate in _title_content_candidates(md_content):
        for line_idx, line in enumerate(candidate.splitlines()):
            md_heading_match = md_heading_regex.match(line)
            if md_heading_match:
                level = len(md_heading_match.group(1))
                title_text = re.sub(r"\s+#+\s*$", "", md_heading_match.group(2)).strip()
                normalized = _normalize_title_label(title_text)
                if normalized:
                    key = (line_idx, level, normalized)
                    if key not in seen:
                        seen.add(key)
                        events.append(key)

            for match in html_heading_regex.finditer(line):
                level = int(match.group(1))
                title_text = re.sub(r"<[^>]+>", " ", match.group(2)).strip()
                normalized = _normalize_title_label(title_text)
                if normalized:
                    key = (line_idx, level, normalized)
                    if key not in seen:
                        seen.add(key)
                        events.append(key)

            md_bold_match = md_bold_line_regex.match(line)
            if md_bold_match:
                normalized = _normalize_title_label(md_bold_match.group(1))
                if normalized:
                    key = (line_idx, 7, normalized)
                    if key not in seen:
                        seen.add(key)
                        events.append(key)

            html_bold_match = html_bold_line_regex.match(line)
            if html_bold_match:
                bold_text = re.sub(r"<[^>]+>", " ", html_bold_match.group(1)).strip()
                normalized = _normalize_title_label(bold_text)
                if normalized:
                    key = (line_idx, 7, normalized)
                    if key not in seen:
                        seen.add(key)
                        events.append(key)

    events.sort(key=lambda item: (item[0], item[1], item[2]))
    return events


class TitleLevelRule(ParseTestRule):
    """Test rule to verify that text appears as a title.

    A title is satisfied if the text appears either:
    - as a markdown/HTML heading (`#`, `##`, ..., `<h1>`, ..., `<h6>`), or
    - as bold text (`**text**` or `<b>text</b>`).

    The `level` field is currently ignored for matching; any heading level
    (1-6) or standalone bold title line can satisfy the rule.
    """

    def __init__(self, rule_data: ParseTitleRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseTitleRule, self._rule_data)

        if self.type != ParseRuleType.IS_TITLE.value:
            raise ValueError(f"Invalid type for TitleLevelRule: {self.type}")

        raw_text = rule_data.text
        if not raw_text.strip():
            raise ValueError("Text field cannot be empty")
        self.text = raw_text.strip()

        self.level = rule_data.level

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        """Check if text appears as heading or bold title in raw markdown.

        Also tolerates escaped markup payloads (e.g. `&lt;h1&gt;...&lt;/h1&gt;`,
        `\\# Title`, `\\*\\*Title\\*\\*`) by checking decoded/de-escaped variants.
        """
        escaped = re.escape(self.text)
        # Allow flexible whitespace and optional inline markup between words
        flexible = _make_markup_tolerant(escaped)

        md_heading_pattern = re.compile(
            r"^#{1,6}\s+" + flexible,
            re.MULTILINE | re.IGNORECASE,
        )
        html_heading_pattern = re.compile(
            r"<h[1-6][^>]*>\s*" + flexible + r"\s*</h[1-6]>",
            re.IGNORECASE,
        )

        # Bold title forms (standalone line, bold at line beginning)
        md_bold_pattern = re.compile(
            r"^\s*\*\*\s*" + flexible + r"\s*\*\*\s*$",
            re.MULTILINE | re.IGNORECASE,
        )
        html_bold_pattern = re.compile(
            r"^\s*<b[^>]*>\s*" + flexible + r"\s*</b>\s*$",
            re.MULTILINE | re.IGNORECASE,
        )

        for candidate in _title_content_candidates(md_content):
            if (
                md_heading_pattern.search(candidate)
                or html_heading_pattern.search(candidate)
                or md_bold_pattern.search(candidate)
                or html_bold_pattern.search(candidate)
            ):
                return True, ""

        # Fallback: normalized comparison using title event extraction.
        # This catches cases where inner markup prevents literal regex matching.
        normalized_self = _normalize_title_label(self.text)
        if normalized_self:
            events = _extract_title_events(md_content)
            for _, _, normalized_title in events:
                if normalized_title == normalized_self:
                    return True, ""

        return (
            False,
            (f"Expected '{self.text[:40]}' to be a title, but no matching heading or bold formatting found"),
        )


class TitleHierarchyPercentRule(ParseTestRule):
    """Score title hierarchy compliance using expected nested title map.

    Expected hierarchy is provided via `title_hierarchy` as nested dict/list.
    Bold title lines are treated as the lowest heading level (level 7).
    """

    def __init__(self, rule_data: ParseTitleHierarchyPercentRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParseTitleHierarchyPercentRule, self._rule_data)
        if self.type != ParseRuleType.TITLE_HIERARCHY_PERCENT.value:
            raise ValueError(f"Invalid type for TitleHierarchyPercentRule: {self.type}")

        if not isinstance(rule_data.title_hierarchy, dict) or not rule_data.title_hierarchy:
            raise ValueError("title_hierarchy must be a non-empty dictionary")
        self.title_hierarchy = rule_data.title_hierarchy

    @staticmethod
    def _collect_constraints(
        hierarchy: dict[str, Any],
    ) -> tuple[set[str], list[tuple[str, str, bool]]]:
        titles: set[str] = set()
        # (parent, child, require_deeper_level)
        edges: list[tuple[str, str, bool]] = []

        def normalize_title(value: str) -> str:
            return _normalize_title_label(value)

        def walk_children(parent_title: str, children: Any) -> None:
            if children is None:
                return

            child_titles_in_order: list[str] = []

            if isinstance(children, str):
                normalized_child = normalize_title(children)
                if normalized_child:
                    titles.add(normalized_child)
                    edges.append((parent_title, normalized_child, True))
                    child_titles_in_order.append(normalized_child)
            elif isinstance(children, dict):
                for child_raw, grand_children in children.items():
                    if not isinstance(child_raw, str):
                        continue
                    normalized_child = normalize_title(child_raw)
                    if not normalized_child:
                        continue
                    titles.add(normalized_child)
                    edges.append((parent_title, normalized_child, True))
                    child_titles_in_order.append(normalized_child)
                    walk_children(normalized_child, grand_children)
            elif isinstance(children, list):
                for child in children:
                    if isinstance(child, str):
                        normalized_child = normalize_title(child)
                        if not normalized_child:
                            continue
                        titles.add(normalized_child)
                        edges.append((parent_title, normalized_child, True))
                        child_titles_in_order.append(normalized_child)
                    elif isinstance(child, dict):
                        for child_raw, grand_children in child.items():
                            if not isinstance(child_raw, str):
                                continue
                            normalized_child = normalize_title(child_raw)
                            if not normalized_child:
                                continue
                            titles.add(normalized_child)
                            edges.append((parent_title, normalized_child, True))
                            child_titles_in_order.append(normalized_child)
                            walk_children(normalized_child, grand_children)

            # Preserve sibling ordering when children are explicitly ordered.
            for i in range(len(child_titles_in_order) - 1):
                edges.append((child_titles_in_order[i], child_titles_in_order[i + 1], False))

        for root_raw, root_children in hierarchy.items():
            if not isinstance(root_raw, str):
                continue
            normalized_root = normalize_title(root_raw)
            if not normalized_root:
                continue
            titles.add(normalized_root)
            walk_children(normalized_root, root_children)

        return titles, edges

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str, float]:
        events = _extract_title_events(md_content)
        first_pos: dict[str, int] = {}
        first_level: dict[str, int] = {}
        for idx, level, title in events:
            if title not in first_pos:
                first_pos[title] = idx
                first_level[title] = level

        expected_titles, edges = self._collect_constraints(self.title_hierarchy)
        if not expected_titles:
            return False, "title_hierarchy has no valid titles after normalization", 0.0

        total_constraints = len(expected_titles) + len(edges)
        if total_constraints == 0:
            return False, "title_hierarchy has no evaluable constraints", 0.0

        satisfied = 0
        failures: list[str] = []

        for title in sorted(expected_titles):
            if title in first_pos:
                satisfied += 1
            else:
                failures.append(f"missing title '{title}'")

        for parent, child, require_deeper_level in edges:
            if parent not in first_pos or child not in first_pos:
                failures.append(f"missing edge '{parent}' -> '{child}'")
                continue
            parent_pos = first_pos[parent]
            child_pos = first_pos[child]
            parent_level = first_level[parent]
            child_level = first_level[child]

            order_ok = parent_pos < child_pos
            depth_ok = parent_level < child_level if require_deeper_level else True
            if order_ok and depth_ok:
                satisfied += 1
            else:
                violation_kind = "order/level" if require_deeper_level else "order"
                failures.append(
                    f"{violation_kind} violation '{parent}'(line={parent_pos},lvl={parent_level}) "
                    f"-> '{child}'(line={child_pos},lvl={child_level})"
                )

        score = max(0.0, min(1.0, satisfied / total_constraints))
        passed = score >= 0.999
        if passed:
            return True, "", score

        preview = "; ".join(failures[:5])
        return False, f"Title hierarchy score={score:.3f}; {preview}", score


_PAGE_SECTION_DASH_CHARS = "-\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
_PAGE_SECTION_DASH_PATTERN = f"[{re.escape(_PAGE_SECTION_DASH_CHARS)}]"


def _build_page_section_query_pattern(text: str) -> str:
    parts: list[str] = []
    for char in text:
        if char in _PAGE_SECTION_DASH_CHARS:
            parts.append(_PAGE_SECTION_DASH_PATTERN)
        elif char == " ":
            parts.append(r"\s+")
        else:
            parts.append(re.escape(char))
    return "".join(parts)


class PageSectionRule(ParseTestRule):
    """Test rule to verify that text appears in page header/footer sections.

    Structured page metadata (parse_output.layout_pages) is the primary source.
    Raw markdown tag scanning is retained as a backward-compatible fallback.
    """

    # Map type value -> tag name
    _TAG_MAP = {
        ParseRuleType.IS_HEADER.value: "page_header",
        ParseRuleType.IS_FOOTER.value: "page_footer",
    }

    def __init__(self, rule_data: ParsePageSectionRule | dict):
        super().__init__(rule_data)
        rule_data = cast(ParsePageSectionRule, self._rule_data)

        if self.type not in self._TAG_MAP:
            raise ValueError(f"Invalid type for PageSectionRule: {self.type}")

        raw_text = rule_data.text
        if not raw_text.strip():
            raise ValueError("Text field cannot be empty")
        self.text = raw_text.strip()
        self.tag = self._TAG_MAP[self.type]

    def run(self, md_content: str, normalized_content: str | None = None) -> tuple[bool, str]:
        """Check if the target text appears inside the expected page section."""
        flexible = _build_page_section_query_pattern(self.text)

        # Primary path: evaluate against structured per-page sections.
        structured_sections = self._get_structured_page_sections()
        if structured_sections is not None:
            section_values = structured_sections.get(self.tag, [])
            section_pattern = re.compile(
                r"[^<]*?" + flexible + r"[^<]*?",
                re.IGNORECASE | re.DOTALL,
            )

            for section_value in section_values:
                if section_pattern.search(section_value):
                    return True, ""

            section_label = "header" if self.tag == "page_header" else "footer"
            preview = "; ".join(repr(value[:80]) for value in section_values[:3])
            return (
                False,
                (
                    f"Expected '{self.text[:40]}' to appear in structured page {section_label} "
                    f"content, but it was not found" + (f" (sample: {preview})" if preview else "")
                ),
            )

        # Backward-compatible path for artifacts/rules that only provide markdown.
        pattern = re.compile(
            r"<" + self.tag + r">" + r"[^<]*?" + flexible + r"[^<]*?" + r"</" + self.tag + r">",
            re.IGNORECASE | re.DOTALL,
        )

        if pattern.search(md_content):
            return True, ""

        section_label = "header" if self.tag == "page_header" else "footer"
        return (
            False,
            f"Expected '{self.text[:40]}' to appear inside a page {section_label} "
            f"(<{self.tag}>...</{self.tag}>), but it was not found",
        )
