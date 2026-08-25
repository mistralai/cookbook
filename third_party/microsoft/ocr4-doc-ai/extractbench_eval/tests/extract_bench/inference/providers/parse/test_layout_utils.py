"""Unit tests for ``_layout_utils.items_to_markdown``.

The layout prompt asks the model for clean markdown inside each div, so
Title/Section-header items usually already carry a leading ``#`` marker and
Formula items their own ``$$`` delimiters. These tests pin down that the
assembler does not double those markers, which would break downstream
heading/formula scoring (``TitleLevelRule`` matches ``^#{1,6}\\s+<text>``, so
``# # Heading`` never matches), while never mutating item text in ambiguous
cases.
"""

from __future__ import annotations

import unittest

from extract_bench.inference.providers.parse._layout_utils import items_to_markdown


class TestHeadingAssembly(unittest.TestCase):
    def test_unmarked_title_gets_h1(self) -> None:
        out = items_to_markdown([{"label": "title", "text": "CHRISM MASS"}])
        self.assertEqual(out, "# CHRISM MASS")

    def test_unmarked_section_header_gets_h2(self) -> None:
        out = items_to_markdown([{"label": "section-header", "text": "Schedule"}])
        self.assertEqual(out, "## Schedule")

    def test_premarked_title_not_doubled(self) -> None:
        out = items_to_markdown([{"label": "title", "text": "# CHRISM MASS"}])
        self.assertEqual(out, "# CHRISM MASS")

    def test_premarked_section_header_not_doubled(self) -> None:
        out = items_to_markdown([{"label": "section-header", "text": "## Schedule"}])
        self.assertEqual(out, "## Schedule")

    def test_label_level_wins_over_model_level(self) -> None:
        # Title always renders H1 and Section-header H2, regardless of the
        # marker level the model emitted — same contract as before.
        out = items_to_markdown([{"label": "section-header", "text": "### Subsection"}])
        self.assertEqual(out, "## Subsection")
        out = items_to_markdown([{"label": "title", "text": "## Doc Title"}])
        self.assertEqual(out, "# Doc Title")

    def test_indented_marker_stripped(self) -> None:
        out = items_to_markdown([{"label": "section-header", "text": "  ## Schedule"}])
        self.assertEqual(out, "## Schedule")

    def test_literal_hash_content_preserved(self) -> None:
        # "#1" is content, not a heading marker (no whitespace after the
        # hashes) — it must not be consumed.
        out = items_to_markdown([{"label": "title", "text": "#1 Best Seller"}])
        self.assertEqual(out, "# #1 Best Seller")

    def test_marker_only_text_kept_verbatim(self) -> None:
        # Degenerate marker-only text falls back to the old verbatim behavior
        # instead of producing an empty heading.
        out = items_to_markdown([{"label": "title", "text": "# "}])
        self.assertEqual(out, "# # ")

    def test_underscore_label_variant(self) -> None:
        out = items_to_markdown([{"label": "section_header", "text": "## Schedule"}])
        self.assertEqual(out, "## Schedule")


class TestFormulaAssembly(unittest.TestCase):
    def test_bare_formula_wrapped(self) -> None:
        out = items_to_markdown([{"label": "formula", "text": "E = mc^2"}])
        self.assertEqual(out, "$$\nE = mc^2\n$$")

    def test_predelimited_formula_not_nested(self) -> None:
        out = items_to_markdown([{"label": "formula", "text": "$$E = mc^2$$"}])
        self.assertEqual(out, "$$\nE = mc^2\n$$")

    def test_inline_delimited_formula_not_nested(self) -> None:
        out = items_to_markdown([{"label": "formula", "text": "$E = mc^2$"}])
        self.assertEqual(out, "$$\nE = mc^2\n$$")

    def test_multiline_predelimited_formula(self) -> None:
        out = items_to_markdown([{"label": "formula", "text": "$$\na + b\n$$"}])
        self.assertEqual(out, "$$\na + b\n$$")

    def test_multiple_formulas_left_intact(self) -> None:
        # First/last '$' of two separate formulas are not an outer pair;
        # stripping them would corrupt the math.
        out = items_to_markdown([{"label": "formula", "text": "$x = 1$, $y = 2$"}])
        self.assertEqual(out, "$$\n$x = 1$, $y = 2$\n$$")

    def test_interior_dollar_left_intact(self) -> None:
        # An unbalanced interior '$' is not an outer delimiter.
        out = items_to_markdown([{"label": "formula", "text": "x = $5.00"}])
        self.assertEqual(out, "$$\nx = $5.00\n$$")

    def test_delimiter_only_text_kept_verbatim(self) -> None:
        out = items_to_markdown([{"label": "formula", "text": "$$"}])
        self.assertEqual(out, "$$\n$$\n$$")


class TestPassthrough(unittest.TestCase):
    def test_plain_text_unchanged(self) -> None:
        out = items_to_markdown([{"label": "text", "text": "# not a heading item"}])
        self.assertEqual(out, "# not a heading item")

    def test_empty_items_skipped(self) -> None:
        out = items_to_markdown(
            [
                {"label": "title", "text": ""},
                {"label": "text", "text": "body"},
            ]
        )
        self.assertEqual(out, "body")

    def test_items_joined_with_blank_line(self) -> None:
        out = items_to_markdown(
            [
                {"label": "title", "text": "# Title"},
                {"label": "text", "text": "body"},
            ]
        )
        self.assertEqual(out, "# Title\n\nbody")


if __name__ == "__main__":
    unittest.main()
