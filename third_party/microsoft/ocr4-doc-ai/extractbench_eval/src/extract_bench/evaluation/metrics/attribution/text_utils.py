"""Text normalization and tokenization for layout attribution evaluation.

The normalization pipeline is designed to be tolerant of OCR differences
while preserving semantic content for attribution comparison.
"""

import re
import unicodedata
from html.parser import HTMLParser


class _HTMLTextExtractor(HTMLParser):
    """Extract plain text from HTML, stripping all tags."""

    def __init__(self):  # type: ignore[no-untyped-def]
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str):  # type: ignore[no-untyped-def]
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def extract_text_from_html(html: str, *, ignore_thead: bool = False) -> str:
    """Extract plain text content from HTML (e.g., table HTML).

    :param html: HTML string
    :param ignore_thead: When true, drop <thead> content before extracting text
    :return: Plain text with tags stripped
    """
    if ignore_thead:
        html = re.sub(r"<thead\b[^>]*>.*?</thead>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    extractor = _HTMLTextExtractor()
    extractor.feed(html)
    return extractor.get_text()


# Ligature map for normalization
_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
}

# Characters to normalize
_CHAR_REPLACEMENTS = {
    "\u2018": "'",  # left single quote
    "\u2019": "'",  # right single quote
    "\u201a": "'",  # single low-9 quote
    "\u201c": '"',  # left double quote
    "\u201d": '"',  # right double quote
    "\u201e": '"',  # double low-9 quote
    "\uff3f": "_",  # fullwidth underscore
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2011": "-",  # non-breaking hyphen
    "\u2012": "-",  # figure dash
    "\u2212": "-",  # minus sign
    "\u2026": "...",  # ellipsis
    "\u00b5": "\u03bc",  # micro sign to greek mu
    "\u203a": ">",  # single right-pointing angle quote
    "\u2039": "<",  # single left-pointing angle quote
}


def _replace_markdown_image_with_alt(text: str) -> str:
    """Replace markdown image syntax with its alt text."""
    return re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)


def strip_leading_markdown_image(text: str) -> str:
    """Strip a leading inline image reference from a non-image text item.

    When a prediction segment for a Text/Section item starts with an image
    reference (e.g. ``![icon](image.jpg) Administrator of Shareholders'``)
    the image markdown adds extra tokens (alt-text words + URL path tokens)
    that reduce token F1 against GT text that contains only the caption.

    This function strips the leading ``![...](...)`` only when text content
    follows it, leaving the rest of the string intact.  If the text consists
    solely of an image reference, it is returned unchanged so that image-type
    attribution still works via the normal strip_image_markup path.

    Witness: annotated_v0.5/ar2024e_p49 (icon-preceded section headers),
    annotated_v0.5/938c3dc8-b424-40fc-836b-101415d323cd_p19 (icon+caption)
    """
    stripped = text.lstrip()
    match = re.match(r"!\[([^\]]*)\]\([^)]*\)\s*", stripped)
    if match and match.end() < len(stripped):
        return stripped[match.end() :]
    return text


def _replace_html_img_with_alt(text: str) -> str:
    """Replace HTML image tags with their alt text.

    If the tag does not expose an alt attribute, strip the tag entirely.
    """

    def _replacement(match: re.Match[str]) -> str:
        tag = match.group(0)
        alt_match = re.search(r'\balt\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s>]+))', tag, flags=re.IGNORECASE)
        if not alt_match:
            return " "
        alt_text = next((group for group in alt_match.groups() if group is not None), "")
        return alt_text

    return re.sub(r"<img\b[^>]*>", _replacement, text, flags=re.IGNORECASE)


def normalize_attribution_text(text: str | None, *, strip_image_markup: bool = False) -> str:
    """Normalize text for attribution comparison.

    Pipeline:
    1. Unicode NFKC normalization
    2. Strip LaTeX/math content
    2. Ligature expansion
    3. Strip HTML/markdown formatting tags
    4. Character replacements (fancy quotes, dashes)
    5. Dehyphenate line-break hyphens
    6. Normalize whitespace
    7. Lowercase
    8. Strip leading/trailing whitespace

    :param text: Raw text to normalize
    :return: Normalized text
    """
    if not text:
        return ""

    # Unicode NFKC normalization (canonical decomposition + compatibility composition)
    text = unicodedata.normalize("NFKC", text)

    # Strip diacritics for robust matching across accent variations (French/German, etc.)
    # Example: "bâtiment" -> "batiment", "évacué" -> "evacue"
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    # German sharp s -> ss
    text = text.replace("ß", "ss")

    # Remove LaTeX/math content entirely (applies to GT and pred equally)
    # Strip display math blocks ($$...$$) unconditionally
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)

    # Strip inline math ($...$) only when the content looks like LaTeX
    # (contains ^, _, \, {, }). Plain dollar amounts ($100, $52.6bn) are
    # NOT LaTeX and should be preserved as numeric tokens.
    def _strip_latex_inline(m: "re.Match[str]") -> str:
        content = m.group(0)[1:-1]  # strip surrounding $
        if re.search(r"[\\^_{}]", content):
            return " "  # LaTeX content — strip
        return str(m.group(0))  # currency — keep the $..$ as-is

    text = re.sub(r"\$[^$]+\$", _strip_latex_inline, text)
    text = re.sub(r"\\\\\\[.*?\\\\\\]", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\\\\\(.*?\\\\\\)", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\\\begin\\{.*?\\}.*?\\\\end\\{.*?\\}", " ", text, flags=re.DOTALL)

    # Strip remaining LaTeX commands (with optional brace arguments)
    text = re.sub(r"\\\\[a-zA-Z]+(\\s*\\{[^}]*\\})*", " ", text)
    # Strip leftover superscripts/subscripts
    text = re.sub(r"[_^]\\{[^}]*\\}", " ", text)
    text = re.sub(r"[_^][a-zA-Z0-9]+", " ", text)

    # Expand ligatures
    for lig, replacement in _LIGATURES.items():
        text = text.replace(lig, replacement)

    if strip_image_markup:
        # Reduce image syntax to visible alt text before stripping generic markup.
        text = _replace_markdown_image_with_alt(text)
        text = _replace_html_img_with_alt(text)

    # Strip HTML tags (sup, sub, b, i, span, div, etc.)
    text = re.sub(r"</?(?:sup|sub|b|i|u|ins|mark|span|div|br/?)\b[^>]*>", " ", text, flags=re.IGNORECASE)

    # Strip residual HTML-like attribute patterns from partially stripped tags
    # (e.g., 'style="color: green; font-size: 1.5em;"' left over when angle
    # brackets were stripped but tag content survived)
    text = re.sub(r'\bstyle\s*=\s*"[^"]*"', " ", text)

    # Strip markdown formatting
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # bold
    text = re.sub(r"__(.*?)__", r"\1", text)  # bold
    text = re.sub(r"\*(.*?)\*", r"\1", text)  # italic
    text = re.sub(r"_(.*?)_", r"\1", text)  # italic
    text = re.sub(r"#{1,6}\s*", "", text)  # headings
    text = re.sub(r"\$\$?", "", text)  # LaTeX delimiters
    # Strip markdown links [text](url) → text. Without this, URLs inside
    # link syntax appear as extra tokens (the display text and URL are both
    # kept, duplicating domain/path tokens).
    text = re.sub(r"(?<!!)\[([^\]]*)\]\([^)]*\)", r"\1", text)

    # Dehyphenate BEFORE character replacements (› is used as hyphen in GT)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"(\w)\u203a\s*\n\s*(\w)", r"\1\2", text)

    # Character replacements
    for old, new in _CHAR_REPLACEMENTS.items():
        text = text.replace(old, new)

    # Lowercase
    text = text.lower()

    # Normalize whitespace (newlines, tabs, multiple spaces -> single space)
    text = re.sub(r"\s+", " ", text)

    # Remove remaining non-ASCII symbols (math glyphs, greek letters, etc.)
    # Keep basic punctuation so text remains readable in reports.
    text = re.sub(r"[^a-z0-9\s\.,;:!\?\-\(\)\[\]\{\}/'\"%]", " ", text)

    return text.strip()


# Token pattern: sequences of alphanumeric characters (letters + digits)
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize normalized text into word/digit tokens.

    Extracts sequences of lowercase alphanumeric characters.
    Punctuation is ignored (not a token).

    :param text: Normalized text (should be output of normalize_attribution_text)
    :return: List of tokens
    """
    return _TOKEN_PATTERN.findall(text)
