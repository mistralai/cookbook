"""Unit tests for InfinityParser2 table-header heuristics.

These tests pin down the rule-driven behavior of the post-processing helpers
in ``infinity_parser2.py`` so future model/format changes don't silently
regress them.
"""

from __future__ import annotations

import unittest

import pytest
from bs4 import BeautifulSoup

pytest.importorskip("pdf2image", reason="dev and runners extras required; run: uv sync --extra dev --extra runners")

from extract_bench.inference.providers.parse.infinity_parser2 import (
    _convert_nonstandard_table,
    _convert_table_header,
    _determine_header_row_count,
    _find_column_number,
    _is_gender_cell,
    _is_nonstandard_table,
    _is_pure_number_cell,
    _is_pure_text_cell,
    _is_year_cell,
)


class TestCellClassifiers(unittest.TestCase):
    """Cell-level predicates used by the header-row heuristics."""

    def test_year_cell(self) -> None:
        self.assertTrue(_is_year_cell("2024"))
        self.assertTrue(_is_year_cell("202401"))
        self.assertTrue(_is_year_cell("2024-01-15"))
        self.assertFalse(_is_year_cell("Revenue"))

    def test_gender_cell(self) -> None:
        self.assertTrue(_is_gender_cell("Male"))
        self.assertTrue(_is_gender_cell("female"))
        self.assertFalse(_is_gender_cell("Total"))

    def test_pure_text_vs_pure_number(self) -> None:
        # pure text: has alpha, no all-numeric requirement
        self.assertTrue(_is_pure_text_cell("Revenue"))
        self.assertFalse(_is_pure_text_cell("123"))
        self.assertFalse(_is_pure_text_cell(""))

        # pure number: digits + permitted symbols only
        self.assertTrue(_is_pure_number_cell("1,234.56"))
        self.assertTrue(_is_pure_number_cell("$(45.00)"))
        self.assertTrue(_is_pure_number_cell("-12%"))
        self.assertFalse(_is_pure_number_cell("12 apples"))
        self.assertFalse(_is_pure_number_cell(""))


class TestNonstandardTable(unittest.TestCase):
    """Detection and conversion of '&'-separated tables emitted by the model."""

    def test_is_nonstandard_table(self) -> None:
        # Has '&' and does not start with '|' → nonstandard
        self.assertTrue(_is_nonstandard_table("a | b | c & 1 | 2 | 3"))
        # Already a proper markdown table → not nonstandard
        self.assertFalse(_is_nonstandard_table("| a | b |\n| - | - |"))
        # No '&' → not nonstandard
        self.assertFalse(_is_nonstandard_table("plain text"))
        self.assertFalse(_is_nonstandard_table(""))

    def test_find_column_number(self) -> None:
        # 3 columns → header has 2 pipes between cells
        self.assertEqual(_find_column_number("a | b | c & 1 | 2 | 3"), 3)
        self.assertEqual(_find_column_number("no ampersand here"), 0)

    def test_convert_nonstandard_table_roundtrip(self) -> None:
        raw = "Year | Revenue | Profit & 2023 | 100 | 20 & 2024 | 150 | 35"
        out = _convert_nonstandard_table(raw)
        lines = out.splitlines()
        # Header + separator + 2 data rows
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[0].startswith("|") and lines[0].endswith("|"))
        self.assertEqual(lines[1], "| --- | --- | --- |")
        self.assertIn("2023", lines[2])
        self.assertIn("2024", lines[3])

    def test_convert_nonstandard_table_passthrough(self) -> None:
        # Already-valid markdown tables must be returned unchanged.
        already_md = "| a | b |\n| - | - |\n| 1 | 2 |"
        self.assertEqual(_convert_nonstandard_table(already_md), already_md)


class TestDetermineHeaderRowCount(unittest.TestCase):
    """Header-row count is determined by year/gender/value rules, then rowspan."""

    @staticmethod
    def _rows(html: str) -> list:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        return table.find_all("tr", recursive=False)

    def test_year_rule_single_header_row(self) -> None:
        html = """
        <table>
          <tr><td>2022</td><td>2023</td><td>2024</td></tr>
          <tr><td>10</td><td>20</td><td>30</td></tr>
          <tr><td>11</td><td>21</td><td>31</td></tr>
        </table>
        """
        self.assertEqual(_determine_header_row_count(self._rows(html)), 1)

    def test_value_rule_text_then_numbers(self) -> None:
        # First row pure text, rest pure numbers → 1 header row by value rule.
        html = """
        <table>
          <tr><td>Region</td><td>Revenue</td><td>Profit</td></tr>
          <tr><td>100</td><td>200</td><td>30</td></tr>
          <tr><td>110</td><td>210</td><td>35</td></tr>
        </table>
        """
        self.assertEqual(_determine_header_row_count(self._rows(html)), 1)

    def test_rowspan_fallback(self) -> None:
        # No year/gender/value signal → fallback to rowspan of first row.
        html = """
        <table>
          <tr><td rowspan="2">A</td><td rowspan="2">B</td></tr>
          <tr></tr>
          <tr><td>x</td><td>y</td></tr>
        </table>
        """
        self.assertEqual(_determine_header_row_count(self._rows(html)), 2)


class TestConvertTableHeader(unittest.TestCase):
    """End-to-end: <td> in detected header rows is rewritten to <th>."""

    def test_td_to_th_in_header_row(self) -> None:
        html = "<table><tr><td>2022</td><td>2023</td></tr><tr><td>10</td><td>20</td></tr></table>"
        out = _convert_table_header(html)
        soup = BeautifulSoup(out, "html.parser")
        rows = soup.find_all("tr")
        # Row 0: both cells become <th>
        self.assertEqual(len(rows[0].find_all("th")), 2)
        self.assertEqual(len(rows[0].find_all("td")), 0)
        # Row 1: data cells remain <td>
        self.assertEqual(len(rows[1].find_all("td")), 2)
        self.assertEqual(len(rows[1].find_all("th")), 0)

    def test_non_table_html_unchanged(self) -> None:
        self.assertEqual(_convert_table_header(""), "")
        self.assertEqual(_convert_table_header("plain text"), "plain text")


if __name__ == "__main__":
    unittest.main()
