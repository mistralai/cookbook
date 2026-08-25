"""Tests for FormattingRule span boundaries.

Three ways unformatted text used to test as formatted:

* markdown ``**`` spans: the closer of one span paired with the opener of
  the next, so the plain text between two bold spans matched
  (``**Name:** John **Age:**`` -> ``** John **``),
* HTML spans: the greedy fillers in ``<b>.*?query.*?</b>`` crossed closing
  tags, merging adjacent elements, and
* the heading arm: its DOTALL fillers crossed newlines, so any text
  anywhere after any heading in the document tested as bold.
"""

from __future__ import annotations

from extract_bench.evaluation.metrics.parse.rules_base import create_test_rule

KV_MD = "**Name:** John **Age:** 42"
KV_HTML = "<b>Name:</b> John <b>Age:</b> 42"


def _run(rule_type: str, text: str, md: str) -> bool:
    passed, _ = create_test_rule({"type": rule_type, "text": text}).run(md)
    return passed


class TestAdjacentMarkdownSpans:
    def test_text_between_bold_spans_is_not_bold(self):
        assert not _run("is_bold", "John", KV_MD)
        assert _run("is_not_bold", "John", KV_MD)

    def test_bold_span_content_still_detected(self):
        assert _run("is_bold", "Name:", KV_MD)
        assert _run("is_bold", "Age:", KV_MD)

    def test_substring_of_bold_span_still_detected(self):
        assert _run("is_bold", "Population", "**Population:** 5,000")

    def test_multiline_bold_span_still_detected(self):
        assert _run("is_bold", "multi line", "**multi\nline**")

    def test_bold_italic_still_detected(self):
        assert _run("is_bold", "tri", "some ***tri*** here")


class TestAdjacentHtmlSpans:
    def test_text_between_b_tags_is_not_bold(self):
        assert not _run("is_bold", "John", KV_HTML)
        assert _run("is_not_bold", "John", KV_HTML)

    def test_b_tag_content_still_detected(self):
        assert _run("is_bold", "Name:", KV_HTML)

    def test_text_between_i_tags_is_not_italic(self):
        assert not _run("is_italic", "John", "<i>Name:</i> John <i>Age:</i> 42")

    def test_text_between_em_tags_is_not_italic(self):
        assert not _run("is_italic", "John", "<em>Name:</em> John <em>Age:</em> 42")

    def test_i_tag_content_still_detected(self):
        assert _run("is_italic", "Name:", "<i>Name:</i> John")

    def test_nested_markup_inside_element_still_detected(self):
        assert _run("is_bold", "A B", "<b>A <mark>B</mark></b>")


class TestHeadingArmStaysOnItsLine:
    def test_body_text_after_heading_is_not_bold(self):
        md = "# Some Heading\n\nparagraph with later text here"
        assert not _run("is_bold", "later text here", md)
        assert _run("is_not_bold", "later text here", md)

    def test_heading_text_still_detected(self):
        assert _run("is_bold", "Actual Heading", "# Actual Heading\n\nbody")

    def test_heading_substring_still_detected(self):
        assert _run("is_bold", "Heading", "## Longer Actual Heading Line\nbody")

    def test_closed_atx_heading_still_detected(self):
        assert _run("is_bold", "Title", "## Title ##\nbody")

    def test_indented_heading_still_detected(self):
        assert _run("is_bold", "Indented", "   # Indented\nbody")
