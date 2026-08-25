"""Tests for ParseFormFieldRule / FormFieldRule (parse-side form KV evaluation)."""

from __future__ import annotations

import pytest

from extract_bench.evaluation.metrics.parse.rules_base import create_test_rule
from extract_bench.evaluation.metrics.parse.test_rules import FormFieldRule
from extract_bench.evaluation.metrics.parse.test_types import ParseRuleType

# ---------------------------------------------------------------------------
# Factory dispatch + schema
# ---------------------------------------------------------------------------


def test_factory_dispatches_form_field_to_form_field_rule():
    rule = create_test_rule({"type": "form_field", "label": "Last Name", "value": "Collins"})
    assert isinstance(rule, FormFieldRule)
    assert rule.type == ParseRuleType.FORM_FIELD.value


def test_empty_label_raises():
    with pytest.raises(ValueError, match="label"):
        FormFieldRule({"type": "form_field", "label": "", "value": "X"})


# ---------------------------------------------------------------------------
# value_type = "text"
# ---------------------------------------------------------------------------


class TestTextValueMatching:
    def _rule(self, label: str, value: str) -> FormFieldRule:
        return FormFieldRule({"type": "form_field", "label": label, "value": value, "value_type": "text"})

    def test_bold_colon_match(self):
        md = "**Last Name:** Collins\n"
        passed, _, score = self._rule("Last Name", "Collins").run(md)
        assert passed
        assert score == 1.0

    def test_bold_no_colon_inside_then_colon_outside(self):
        md = "**Last Name**: Collins\n"
        passed, _, _ = self._rule("Last Name", "Collins").run(md)
        assert passed

    def test_plain_colon_match(self):
        md = "Last Name: Collins\n"
        passed, _, _ = self._rule("Last Name", "Collins").run(md)
        assert passed

    def test_2col_markdown_table_match(self):
        md = "| Field | Value |\n|---|---|\n| Last Name | Collins |\n| First Name | Maya |\n"
        passed, _, _ = self._rule("Last Name", "Collins").run(md)
        assert passed

    def test_label_not_found_returns_failure(self):
        md = "**Other Field:** Something\n"
        passed, expl, score = self._rule("Last Name", "Collins").run(md)
        assert not passed
        assert score == 0.0
        assert "label not found" in expl

    def test_value_mismatch_returns_failure(self):
        md = "**Last Name:** Smith\n"
        passed, expl, score = self._rule("Last Name", "Collins").run(md)
        assert not passed
        assert score == 0.0
        assert "expected" in expl and "got" in expl

    def test_numeric_value_with_thousands_separator(self):
        # Numeric strict equality via normalize_number_string: "1,234" == "1234"
        # is the same numeric value, just written differently.
        md = "**Salary:** 1234\n"
        passed, _, _ = self._rule("Salary", "1,234").run(md)
        assert passed

    def test_label_fuzzy_match(self):
        # Label matching tolerates minor parser rendering noise (single-character
        # typo). Value matching is strict; label matching uses fuzz at 0.8 so
        # that "Last Nam" / "Last Name" still resolves the right field.
        md = "**Last Nam:** Collins\n"
        passed, _, _ = self._rule("Last Name", "Collins").run(md)
        assert passed


class TestHtmlCellNeighborFallback:
    """Wide-form HTML table cell-neighbor fallback (well-log style headers).

    Targets the ``_iter_html_cell_neighbor_pairs`` last-resort source that
    recovers KV pairs when labels and values are interleaved inside a single
    wide ``<table>`` rather than being separated into a header row + data
    rows. Only fires when the strong-source patterns (bold-colon, plain-colon,
    2-col table, multi-col header×data) have all missed.
    """

    def _rule(self, label: str, value) -> FormFieldRule:
        return FormFieldRule({"type": "form_field", "label": label, "value": value, "value_type": "text"})

    def test_right_neighbor_value_wins(self):
        # Three-cell row: label cell, label-shaped middle cell, value-shaped
        # right cell. The right-neighbor matcher picks the third cell as the
        # value of the *middle* label and the *first* label.
        md = "<table><tr><th>FILE NO:</th><th>COMPANY</th><th>KEBO OIL &amp; GAS, INC.</th></tr></table>"
        passed, _, _ = self._rule("COMPANY", "KEBO OIL & GAS, INC.").run(md)
        assert passed

    def test_below_neighbor_when_right_is_label_like(self):
        # API NO: in row 0 → right neighbor "WELL" is itself label-shaped, so
        # the matcher must fall through to the below neighbor (row 1 col 0).
        md = (
            "<table>"
            "<tr><th>API NO:</th><th>WELL</th><th>LEHMAN #1</th></tr>"
            "<tr><th>42-157-33282</th><th>FIELD</th><th>NEEDVILLE</th></tr>"
            "</table>"
        )
        passed, _, _ = self._rule("API NO:", "42-157-33282").run(md)
        assert passed

    def test_value_with_punctuation_not_confused_for_label(self):
        # LEHMAN #1 looks ALL_CAPS but has '#' / digit → must be classified
        # as a value, not a label. So WELL → LEHMAN #1 must pass.
        md = "<table><tr><th>API NO:</th><th>WELL</th><th>LEHMAN #1</th></tr></table>"
        passed, _, _ = self._rule("WELL", "LEHMAN #1").run(md)
        assert passed

    def test_colspan_duplicates_skipped(self):
        # ``parse_html_tables`` expands colspan by duplicating cell text across
        # the covered columns. The right-scan must skip those duplicates and
        # land on the next distinct non-empty cell.
        md = (
            "<table><tr>"
            '<th colspan="2">FILE NO:</th>'
            '<th colspan="2">COMPANY</th>'
            '<th colspan="4">KEBO OIL &amp; GAS, INC.</th>'
            "</tr></table>"
        )
        passed, _, _ = self._rule("COMPANY", "KEBO OIL & GAS, INC.").run(md)
        assert passed

    def test_two_col_html_table_still_uses_strong_source(self):
        # Sanity: ordinary 2-col HTML tables continue to match via the
        # existing pair iterator — the new fallback must never override the
        # strong-source path. With value=Smith the 2-col matcher returns
        # Smith; with value=Collins it returns failure (not whatever the
        # neighbor matcher might dredge up).
        md = "<table><tr><th>Last Name</th><td>Smith</td></tr></table>"
        passed_match, _, _ = self._rule("Last Name", "Smith").run(md)
        assert passed_match
        passed_miss, expl, _ = self._rule("Last Name", "Collins").run(md)
        assert not passed_miss
        assert "expected" in expl and "got" in expl

    def test_no_html_table_means_no_change(self):
        # Bold-colon already handles this and the fallback never fires; this
        # is a guard against accidental new false positives when there is
        # no ``<table>`` in the content.
        md = "**Last Name:** Smith\n"
        passed, _, _ = self._rule("Last Name", "Smith").run(md)
        assert passed

    def test_empty_expected_passes_on_label_presence(self):
        # FILE NO: has no value-shaped right or below neighbor in this
        # snippet; the matcher returns label_seen=True with value="" so an
        # empty-expected rule still passes (matching the contract of the
        # other pair sources).
        md = "<table><tr><th>FILE NO:</th><th>COMPANY</th><th>KEBO</th></tr></table>"
        passed, _, _ = self._rule("FILE NO:", "").run(md)
        assert passed

    def test_single_th_cell_inline_kv_still_parses_via_plain_colon(self):
        # ``<th>Label: Value</th>`` on its own line carries exactly one inline
        # KV pair — ``_iter_plain_colon_pairs`` must still match this so we
        # don't regress the wide-form well-log case where one tier emits
        # ``<th colspan="2">Company: CIMARRON ENGINEERING, LLC</th>``. Only
        # *multi-cell* rows (multiple ``<th>`` / ``<td>`` openings on the
        # same line) are skipped by the plain-colon iterator.
        md = '<table><tr><th colspan="2">Company: CIMARRON ENGINEERING, LLC</th></tr></table>'
        passed, _, _ = self._rule("Company", "CIMARRON ENGINEERING, LLC").run(md)
        assert passed

    def test_iterates_past_label_position_with_only_label_like_neighbor(self):
        # The same label text ``API`` appears twice in this table:
        #   row 0 col 0 → right neighbor ``WELL`` (label-shaped, ALL-CAPS
        #                 short) and below neighbor ``API`` (rowspan-style
        #                 dup, skipped). No value-shaped neighbor at this
        #                 position, so the matcher must NOT bail out with
        #                 ``label_seen=True, value=""``.
        #   row 1 col 0 → right neighbor ``42-001`` (digits → value-shaped) —
        #                 the matcher must keep iterating past the first
        #                 match position and surface this neighbor.
        # Locks in the "keep iterating" behavior for repeated label tokens.
        md = (
            "<table>"
            "<tr><th>API</th><th>WELL</th><th>STATE</th></tr>"
            "<tr><th>API</th><th>42-001</th><th>TEXAS</th></tr>"
            "</table>"
        )
        passed, _, _ = self._rule("API", "42-001").run(md)
        assert passed

    def test_repeated_short_text_treated_as_label(self):
        # ``KB`` appears three times in this elevation block. The matcher's
        # text-count map must classify the right neighbor of ELEVATIONS as
        # label-shaped (it repeats ≥2 times) and fall through to a value-
        # shaped neighbor — here the matcher returns the elevation reading
        # for the ELEVATIONS → KB row when looking up "KB" the label.
        md = (
            "<table>"
            "<tr><th>ELEVATIONS:</th><th>KB</th><th>96.8 FT</th></tr>"
            "<tr><th></th><th>DF</th><th>95.8 FT</th></tr>"
            "<tr><th></th><th>GL</th><th>80.0 FT</th></tr>"
            "</table>"
        )
        passed, _, _ = self._rule("KB", "96.8 FT").run(md)
        assert passed


class TestMultiOccurrenceLabelPairing:
    """A single label text can legitimately appear at multiple page positions
    at the *same* best score (``KB`` exact-matching in both an elevation
    block label-row and a ``LOG MEASURED FROM`` value-row; ``Address`` in
    a W-15 form for Cementer vs Operator). The matcher must let the rule's
    expected value disambiguate among those tied candidates without
    weakening the cross-label boundary (Country vs County, etc.).
    """

    def _rule(self, label: str, value) -> FormFieldRule:
        return FormFieldRule({"type": "form_field", "label": label, "value": value, "value_type": "text"})

    def test_second_occurrence_with_matching_value_wins(self):
        # Two bold-colon ``Address`` rows; both score exactly 1.0 against
        # the rule label ``Address``. The first carries the operator's
        # address, the second carries the cementer's — which is what GT
        # expects. Without GT-oracle disambiguation, the source-priority
        # + doc-order tie-break would lock onto the first row and fail.
        md = "**Address:** 100 Operator Way\n\n**Address:** 200 Cementer Blvd\n"
        passed, _, _ = self._rule("Address", "200 Cementer Blvd").run(md)
        assert passed

    def test_first_occurrence_still_wins_when_it_matches(self):
        # Symmetric: with GT matching the first row, the matcher returns it.
        md = "**Address:** 200 Cementer Blvd\n\n**Address:** 100 Operator Way\n"
        passed, _, _ = self._rule("Address", "200 Cementer Blvd").run(md)
        assert passed

    def test_no_occurrence_matches_returns_top_for_diagnostics(self):
        # When no tied-top candidate matches expected, the legacy
        # source-priority + doc-order tie-break still surfaces a value so
        # the error message reads "expected X, got Y".
        md = "**Address:** 100 Operator Way\n\n**Address:** 200 Cementer Blvd\n"
        passed, expl, _ = self._rule("Address", "999 Nowhere St").run(md)
        assert not passed
        assert "expected '999 Nowhere St'" in expl
        assert "got '100 Operator Way'" in expl

    def test_gt_oracle_does_not_leak_across_adjacent_labels(self):
        # Cross-label boundary: ``Country`` is an exact match (score 1.0)
        # for the rule label, ``County`` is a partial match (score ~0.92
        # after the partial-ratio penalty). They live at *different*
        # score levels, so even if ``County``'s value coincidentally
        # equals what GT expects for ``Country``, ``County`` must not be
        # eligible. The rule must read the *correct* row and fail (or
        # pass) based on what's actually there — never grab the wrong
        # row just because its value matches GT.
        md = "**County**: U.S.A.\n**Country**: Mexico\n"
        # The correct value for ``Country`` is ``Mexico`` (extraction
        # error in the document), so the rule must FAIL when GT says
        # ``U.S.A.``. If the GT oracle leaked across labels, the matcher
        # would grab ``U.S.A.`` from the County row and wrongly pass.
        passed, expl, _ = self._rule("Country", "U.S.A.").run(md)
        assert not passed
        assert "expected 'U.S.A.'" in expl
        assert "got 'Mexico'" in expl

    def test_gt_oracle_does_not_promote_lower_score_partial_hit(self):
        # Symmetric to the previous test: when the rule label is the
        # shorter ``County``, ``County`` exact-matches (score 1.0) and
        # ``Country`` partial-matches (score ~0.95 with penalty). Only
        # the County row is eligible regardless of which value GT names.
        md = "**County**: U.S.A.\n**Country**: Mexico\n"
        passed, _, _ = self._rule("County", "U.S.A.").run(md)
        assert passed
        passed_neg, expl, _ = self._rule("County", "Mexico").run(md)
        assert not passed_neg
        assert "got 'U.S.A.'" in expl

    def test_cross_source_pairing_picks_matching_value(self):
        # Same label surfaces in bold-colon (wrong value) AND an HTML
        # 2-col table (right value). Both score 1.0 on the label. The
        # bold-colon entry has higher source priority so the legacy
        # tie-break would return ``99-999-99999``; the GT oracle must
        # promote the HTML row instead because its value matches.
        md = "**API NO:** 99-999-99999\n<table><tr><th>API NO:</th><td>42-157-33282</td></tr></table>"
        passed, _, _ = self._rule("API NO:", "42-157-33282").run(md)
        assert passed

    def test_cell_neighbor_multi_position_disambiguates(self):
        # ``KB`` appears as a value (col 1 in "LOG MEASURED FROM" row)
        # AND as a label (col 1 in "ELEVATIONS:" row with elevation
        # reading to the right). The cell-neighbor matcher yields both
        # candidates at score 1.0. The expected value "96.8 FT" must
        # steer the matcher to the ELEVATIONS position, not the LOG
        # MEASURED FROM one which would give "16.0 FT".
        md = (
            "<table>"
            "<tr><th>LOG MEASURED FROM</th><th>KB</th><th>16.0 FT</th></tr>"
            "<tr><th>ELEVATIONS:</th><th>KB</th><th>96.8 FT</th></tr>"
            "</table>"
        )
        passed, _, _ = self._rule("KB", "96.8 FT").run(md)
        assert passed

    def test_cell_neighbor_first_position_still_picked_when_it_matches(self):
        # Symmetric: if GT expects the value at the first KB position,
        # the matcher must return it.
        md = (
            "<table>"
            "<tr><th>LOG MEASURED FROM</th><th>KB</th><th>16.0 FT</th></tr>"
            "<tr><th>ELEVATIONS:</th><th>KB</th><th>96.8 FT</th></tr>"
            "</table>"
        )
        passed, _, _ = self._rule("KB", "16.0 FT").run(md)
        assert passed

    def test_legacy_call_without_expected_returns_top_candidate(self):
        # Internal contract: callers that don't have an expected value
        # (signature rule, label-presence-only paths) keep the legacy
        # source-priority + doc-order pick — same as #978's behaviour.
        from extract_bench.evaluation.metrics.parse.rules_form import (
            _find_text_value_for_label,
        )

        md = "**Address:** 100 Operator Way\n\n**Address:** 200 Cementer Blvd\n"
        seen, value = _find_text_value_for_label(md, "Address")
        assert seen
        assert value == "100 Operator Way"

    def test_empty_expected_passes_via_label_seen(self):
        # Empty-expected rule: when GT value is "", a label_seen=True /
        # no-value outcome already satisfies the rule via
        # _values_match_text("", ""). The GT oracle must not promote an
        # adjacent label-like neighbor as a spurious value.
        md = "Last Name: \n"
        passed, _, _ = self._rule("Last Name", "").run(md)
        assert passed


class TestTextValueAlternatives:
    """List-of-strings `value` declares acceptable alternatives for ambiguous fields."""

    def _rule(self, label: str, value) -> FormFieldRule:
        return FormFieldRule({"type": "form_field", "label": label, "value": value, "value_type": "text"})

    def test_first_alternative_matches(self):
        md = "**Lease/ID No.:** N/A DRY\n"
        passed, _, _ = self._rule("Lease/ID No.", ["N/A DRY", "NIA DRY"]).run(md)
        assert passed

    def test_second_alternative_matches(self):
        md = "**Lease/ID No.:** NIA DRY\n"
        passed, _, _ = self._rule("Lease/ID No.", ["N/A DRY", "NIA DRY"]).run(md)
        assert passed

    def test_no_alternative_matches_fails(self):
        md = "**Lease/ID No.:** Something Else\n"
        passed, expl, _ = self._rule("Lease/ID No.", ["N/A DRY", "NIA DRY"]).run(md)
        assert not passed
        assert "any of" in expl

    def test_single_element_list_behaves_like_string(self):
        md = "**Last Name:** Collins\n"
        passed, _, _ = self._rule("Last Name", ["Collins"]).run(md)
        assert passed

    def test_empty_string_alternative_passes_when_field_blank(self):
        md = "| Field | Value |\n|---|---|\n| Last Name |  |\n"
        passed, _, _ = self._rule("Last Name", ["", "N/A"]).run(md)
        assert passed

    def test_string_value_still_works_after_list_support(self):
        # Backward compat: pre-existing string GTs must keep passing exactly as before.
        md = "**Last Name:** Collins\n"
        passed, _, _ = self._rule("Last Name", "Collins").run(md)
        assert passed


# ---------------------------------------------------------------------------
# value_type = "checkbox"
# ---------------------------------------------------------------------------


class TestCheckboxValueMatching:
    def _rule(self, label: str, value) -> FormFieldRule:
        return FormFieldRule({"type": "form_field", "label": label, "value": value, "value_type": "checkbox"})

    def test_glyph_checked_match(self):
        md = "☑ Married\n☐ Single\n"
        passed, _, _ = self._rule("Married", True).run(md)
        assert passed

    def test_list_value_rejected_for_checkbox(self):
        # Checkboxes are bool-or-bust: list-of-alternatives is meaningless for a
        # binary state, so _coerce_bool returns None and the existing
        # "must be coercible to bool" error fires with the value in the message.
        md = "☑ Married\n"
        passed, expl, _ = self._rule("Married", ["Yes", "Y"]).run(md)
        assert not passed
        assert "must be coercible to bool" in expl
        assert "['Yes', 'Y']" in expl

    def test_glyph_unchecked_match(self):
        md = "☑ Married\n☐ Single\n"
        passed, _, _ = self._rule("Single", False).run(md)
        assert passed

    def test_glyph_state_mismatch(self):
        md = "☐ Married\n"
        passed, expl, _ = self._rule("Married", True).run(md)
        assert not passed
        assert "expected True" in expl

    def test_md_task_list_checked(self):
        md = "- [x] Routine service\n- [ ] Expedited service\n"
        passed, _, _ = self._rule("Routine service", True).run(md)
        assert passed

    def test_md_task_list_unchecked(self):
        md = "- [x] Routine service\n- [ ] Expedited service\n"
        passed, _, _ = self._rule("Expedited service", False).run(md)
        assert passed

    def test_text_yes_coerces_to_true(self):
        md = "**Married:** Yes\n"
        passed, _, _ = self._rule("Married", True).run(md)
        assert passed

    def test_text_no_coerces_to_false(self):
        md = "**Married:** No\n"
        passed, _, _ = self._rule("Married", False).run(md)
        assert passed

    def test_label_not_found(self):
        md = "**Other:** Yes\n"
        passed, expl, _ = self._rule("Married", True).run(md)
        assert not passed
        assert "label not found" in expl


# ---------------------------------------------------------------------------
# value_type = "signature"
# ---------------------------------------------------------------------------


class TestSignatureValueMatching:
    def _rule(self, label: str, value, value_type: str = "signature") -> FormFieldRule:
        return FormFieldRule({"type": "form_field", "label": label, "value": value, "value_type": value_type})

    def test_signed_when_text_after_label(self):
        md = "**Applicant Signature:** Maya Collins\n"
        passed, _, _ = self._rule("Applicant Signature", True).run(md)
        assert passed

    def test_signed_mismatch_when_label_missing(self):
        md = "**Other Field:** something\n"
        passed, _, _ = self._rule("Applicant Signature", True).run(md)
        assert not passed

    def test_unsigned_table_blank_value(self):
        # Ground truth says "should be unsigned" — and the parser surfaces
        # the field with an empty value cell. The label IS present (in column 0)
        # so _run_signature must distinguish this from "label absent" and pass.
        md = "| Field | Value |\n|---|---|\n| Applicant Signature |  |\n"
        passed, _, _ = self._rule("Applicant Signature", False).run(md)
        assert passed

    def test_unsigned_does_not_pass_when_label_absent(self):
        # Regression: a signature=false rule must FAIL when the parser omitted
        # the signature label entirely. Otherwise pipelines that drop pages of
        # signatures get full credit on every "unsigned" GT tuple.
        md = "**Other Field:** something\n"
        passed, expl, _ = self._rule("Applicant Signature", False).run(md)
        assert not passed
        assert "label not found" in expl

    # ---- Relaxed semantics: non-empty string value == True ----

    def test_string_value_passes_when_any_text_signed(self):
        # Rule stores the actual signed name as documentation; parser surfaces
        # *some* non-empty value under the label — that counts as "signed".
        md = "**Applicant Signature:** Maya Collins\n"
        passed, _, _ = self._rule("Applicant Signature", "Robert Hal Thompson").run(md)
        assert passed

    def test_string_value_passes_even_when_text_differs(self):
        # Signature matching is relaxed: handwriting need not match the rule's
        # text, only that *something* is signed there.
        md = "**Applicant Signature:** Maya Collins\n"
        passed, _, _ = self._rule("Applicant Signature", "CHRIS BUSH").run(md)
        assert passed

    def test_string_value_fails_when_field_empty(self):
        # Rule has non-empty string (expecting signed); parser shows empty cell.
        md = "| Field | Value |\n|---|---|\n| Applicant Signature |  |\n"
        passed, expl, _ = self._rule("Applicant Signature", "Robert Hal Thompson").run(md)
        assert not passed
        assert "signed=True" in expl and "signed=False" in expl

    def test_empty_string_value_treated_as_unsigned(self):
        # Empty string == expected unsigned, same as False.
        md = "| Field | Value |\n|---|---|\n| Applicant Signature |  |\n"
        passed, _, _ = self._rule("Applicant Signature", "").run(md)
        assert passed

    def test_empty_string_value_fails_when_signed(self):
        md = "**Applicant Signature:** Maya Collins\n"
        passed, _, _ = self._rule("Applicant Signature", "").run(md)
        assert not passed

    # ---- List value: any non-empty alternative means "expected signed" ----

    def test_list_value_with_any_non_empty_passes_when_signed(self):
        # A list with at least one non-empty alternative collapses to "expected
        # signed" — alternatives describe handwriting variants but the matcher
        # only checks presence.
        md = "**Applicant Signature:** Maya Collins\n"
        passed, _, _ = self._rule("Applicant Signature", ["CHRIS BUSH", "Chris Bush"]).run(md)
        assert passed

    def test_list_value_all_empty_treated_as_unsigned(self):
        # An all-empty list collapses to "expected unsigned", same as False.
        md = "| Field | Value |\n|---|---|\n| Applicant Signature |  |\n"
        passed, _, _ = self._rule("Applicant Signature", ["", ""]).run(md)
        assert passed

    def test_list_value_non_empty_fails_when_field_blank(self):
        md = "| Field | Value |\n|---|---|\n| Applicant Signature |  |\n"
        passed, expl, _ = self._rule("Applicant Signature", ["CHRIS BUSH"]).run(md)
        assert not passed
        assert "signed=True" in expl and "signed=False" in expl


# ---------------------------------------------------------------------------
# Inline checkbox groups (regression for ☐ Single ☑ Married ☐ Head)
# ---------------------------------------------------------------------------


class TestInlineCheckboxGroups:
    def _rule(self, label: str, value: bool) -> FormFieldRule:
        return FormFieldRule({"type": "form_field", "label": label, "value": value, "value_type": "checkbox"})

    def test_glyph_first_inline_group_finds_middle_label(self):
        # Regression: the previous regex consumed the next glyph as a delimiter,
        # so the middle label was either skipped or paired with the wrong glyph.
        md = "☐ Single ☑ Married ☐ Head of Household\n"
        passed, _, _ = self._rule("Married", True).run(md)
        assert passed

    def test_glyph_first_inline_group_finds_last_label(self):
        md = "☐ Single ☑ Married ☐ Head of Household\n"
        passed, _, _ = self._rule("Head of Household", False).run(md)
        assert passed

    def test_glyph_first_inline_group_finds_first_label(self):
        md = "☐ Single ☑ Married ☐ Head of Household\n"
        passed, _, _ = self._rule("Single", False).run(md)
        assert passed

    def test_label_first_inline_group(self):
        # Older form layout: label-first ordering (label precedes its glyph).
        md = "Single ☐  Married ☑  Head of Household ☐\n"
        passed, _, _ = self._rule("Married", True).run(md)
        assert passed

    def test_label_first_inline_group_unchecked(self):
        md = "Single ☐  Married ☑  Head of Household ☐\n"
        passed, _, _ = self._rule("Single", False).run(md)
        assert passed


# ---------------------------------------------------------------------------
# Inline bold-colon multi-pair  (**First:** Maya **Last:** Collins)
# ---------------------------------------------------------------------------


class TestInlineBoldColonMultiPair:
    def _rule(self, label: str, value: str) -> FormFieldRule:
        return FormFieldRule({"type": "form_field", "label": label, "value": value, "value_type": "text"})

    def test_first_field_in_inline_pair(self):
        # Regression: the old $-anchored regex only matched the LAST pair on a line.
        md = "**First Name:** Maya **Last Name:** Collins\n"
        passed, _, _ = self._rule("First Name", "Maya").run(md)
        assert passed

    def test_last_field_in_inline_pair(self):
        md = "**First Name:** Maya **Last Name:** Collins\n"
        passed, _, _ = self._rule("Last Name", "Collins").run(md)
        assert passed

    def test_three_pairs_inline(self):
        md = "**A:** 1 **B:** 2 **C:** 3\n"
        for lbl, val in [("A", "1"), ("B", "2"), ("C", "3")]:
            passed, _, _ = self._rule(lbl, val).run(md)
            assert passed, f"{lbl}={val} should match"


# ---------------------------------------------------------------------------
# HTML tables
# ---------------------------------------------------------------------------


class TestHtmlTableMatching:
    def _rule(self, label: str, value: str) -> FormFieldRule:
        return FormFieldRule({"type": "form_field", "label": label, "value": value, "value_type": "text"})

    def test_2col_html_table_match(self):
        html = (
            "<table>\n"
            "<tr><th>Field</th><th>Value</th></tr>\n"
            "<tr><td>Last Name</td><td>Collins</td></tr>\n"
            "<tr><td>First Name</td><td>Maya</td></tr>\n"
            "</table>\n"
        )
        passed, _, _ = self._rule("Last Name", "Collins").run(html)
        assert passed

    def test_html_table_value_mismatch(self):
        html = "<table><tr><td>Last Name</td><td>Smith</td></tr></table>"
        passed, _, _ = self._rule("Last Name", "Collins").run(html)
        assert not passed


# ---------------------------------------------------------------------------
# Strict value matching — no fuzzy, no relative tolerance
#
# Form values are extracted, not estimated, so any wrong digit / letter / token
# is a real mismatch. The only normalization applied is case-folding +
# whitespace (via `normalize_text`) and numeric equivalence ("1,234" == "1234").
# ---------------------------------------------------------------------------


class TestStrictValueMatching:
    def _rule(self, label: str, value: str) -> FormFieldRule:
        return FormFieldRule({"type": "form_field", "label": label, "value": value, "value_type": "text"})

    def test_phone_off_by_one_digit_fails(self):
        md = "**Phone:** 555-0139\n"
        passed, _, _ = self._rule("Phone", "555-0138").run(md)
        assert not passed

    def test_phone_exact_match_passes(self):
        md = "**Phone:** 555-0138\n"
        passed, _, _ = self._rule("Phone", "555-0138").run(md)
        assert passed

    def test_ssn_off_by_one_digit_fails(self):
        md = "**SSN:** 900-12-3457\n"
        passed, _, _ = self._rule("SSN", "900-12-3456").run(md)
        assert not passed

    def test_zip_off_by_one_digit_fails(self):
        md = "**ZIP:** 53704\n"
        passed, _, _ = self._rule("ZIP", "53703").run(md)
        assert not passed

    def test_date_off_by_one_day_fails(self):
        md = "**Date of Birth:** 1991-08-15\n"
        passed, _, _ = self._rule("Date of Birth", "1991-08-14").run(md)
        assert not passed

    def test_text_case_insensitive(self):
        # `normalize_text` case-folds, so "madison" matches "Madison" exactly
        # — no fuzz needed for that.
        md = "**City:** madison\n"
        passed, _, _ = self._rule("City", "Madison").run(md)
        assert passed

    def test_text_one_letter_off_fails(self):
        # Single-letter typo in a name — not a fuzzy match, real mismatch.
        md = "**First Name:** Mara\n"
        passed, _, _ = self._rule("First Name", "Maya").run(md)
        assert not passed

    def test_currency_normalization(self):
        # Same value written with currency + thousands separator should match.
        md = "**Salary:** $1,234.00\n"
        passed, _, _ = self._rule("Salary", "1234").run(md)
        assert passed


# ---------------------------------------------------------------------------
# Page scoping via injected parse_output
# ---------------------------------------------------------------------------


class _FakePageIR:
    def __init__(self, page_index: int, markdown: str) -> None:
        self.page_index = page_index
        self.markdown = markdown


class _FakeParseOutput:
    def __init__(self, pages: list[_FakePageIR]) -> None:
        self.pages = pages
        self.layout_pages: list[object] = []


class _LayoutItem:
    def __init__(self, md: str = "", html: str = "", value: str = "") -> None:
        self.md = md
        self.html = html
        self.value = value


class _LayoutPage:
    def __init__(self, page_number: int, items: list[_LayoutItem], md: str = "") -> None:
        self.page_number = page_number
        self.items = items
        self.md = md


class _LayoutOnlyOutput:
    def __init__(self, layout_pages: list[_LayoutPage]) -> None:
        self.pages: list[object] = []
        self.layout_pages = layout_pages


def _layout_only_output(specs: list[tuple]) -> _LayoutOnlyOutput:
    # Each spec is (page_number, items[, md]). Compact factory keeps tests terse.
    pages = [_LayoutPage(s[0], s[1], s[2] if len(s) > 2 else "") for s in specs]
    return _LayoutOnlyOutput(pages)


class TestRelaxedLabelMatching:
    def _rule(self, label: str, value: str) -> FormFieldRule:
        return FormFieldRule({"type": "form_field", "label": label, "value": value, "value_type": "text"})

    def test_label_with_numeric_prefix_and_hint_suffix(self):
        # Parser preserves the field's core label but wraps it in form-specific
        # noise: a "(3)" numeric prefix and a long parenthetical hint. The test
        # writer's compact label should still resolve via partial-ratio.
        md = (
            "**(3) Account number (maximum 15 digits, include any leading zeros, "
            "do not include check number)**: TEST-CHK-782194\n"
        )
        passed, _, _ = self._rule("Account number", "TEST-CHK-782194").run(md)
        assert passed

    def test_relaxed_match_does_not_fire_for_short_fragments(self):
        # 3-char fragments must not partial-match into longer labels —
        # otherwise "Tax" would match "Taxonomy of biological classifiers".
        md = "**Income Tax Withheld:** 100\n"
        passed, _, _ = self._rule("Tax", "100").run(md)
        assert not passed

    def test_relaxed_match_blocks_dropped_short_disambiguator(self):
        # Borderline-on-purpose: when the parser drops a disambiguator and the
        # remaining bare label is shorter than the length gate (6 chars), we
        # do NOT auto-match. A form could have both "Date (Supervisor)" and
        # "Date (Employee)"; matching either to bare "Date" would be wrong.
        md = "**Date:** 04/27/2026\n"
        passed, _, _ = self._rule("Date (Supervisor)", "04/27/2026").run(md)
        assert not passed


class TestEscapedAndLabelFirstCheckbox:
    def _rule(self, label: str, value: bool) -> FormFieldRule:
        return FormFieldRule({"type": "form_field", "label": label, "value": value, "value_type": "checkbox"})

    def test_escaped_brackets_in_md_tasklist(self):
        md = "* \\[x] Routine service\n* \\[ ] Expedited service\n"
        passed, _, _ = self._rule("Routine service", True).run(md)
        assert passed

    def test_label_first_bullet_checked(self):
        # Common form layout: bullet text is the option label, brackets follow.
        md = "* Checking \\[x]\n* Savings \\[ ]\n"
        passed, _, _ = self._rule("Checking", True).run(md)
        assert passed

    def test_label_first_bullet_unchecked(self):
        md = "* Checking \\[x]\n* Savings \\[ ]\n"
        passed, _, _ = self._rule("Savings", False).run(md)
        assert passed

    def test_escaped_dash_label_first_bullet(self):
        # Some parsers emit ``\-`` so a literal dash survives markdown rendering
        # of nested bullets. The matcher should treat it like a regular dash.
        md = (
            "* **Purpose of Training (mark all that apply)**:\n"
            "  \\- Improve current job skills \\[x]\n"
            "  \\- Learn new job skills \\[x]\n"
            "  \\- Personal development \\[ ]\n"
        )
        passed, _, _ = self._rule("Improve current job skills", True).run(md)
        assert passed
        passed, _, _ = self._rule("Personal development", False).run(md)
        assert passed

    def test_parent_qualified_label_resolves_to_child_bullet(self):
        # GT label is "Type of Account: Checking" but the parser only emits
        # the child bullet "Checking [x]". Relaxed label match bridges that.
        md = "* Checking \\[x]\n* Savings \\[ ]\n"
        passed, _, _ = self._rule("Type of Account: Checking", True).run(md)
        assert passed


class TestMultiColumnTableRowLabel:
    def _rule(self, label: str, value: str) -> FormFieldRule:
        return FormFieldRule({"type": "form_field", "label": label, "value": value, "value_type": "text"})

    def test_html_table_row_lookup(self):
        html = (
            "<table>\n"
            "<thead><tr>"
            "<th>Course Title</th><th>Start Date</th><th>End Date</th>"
            "</tr></thead>\n"
            "<tbody>"
            "<tr><td>Federal Procurement Basics</td><td>05/04/2026</td><td>06/01/2026</td></tr>\n"
            "<tr><td>Records Management 101</td><td>05/11/2026</td><td>05/29/2026</td></tr>\n"
            "</tbody></table>"
        )
        passed, _, _ = self._rule("Course Title (row 1)", "Federal Procurement Basics").run(html)
        assert passed
        passed, _, _ = self._rule("Start Date (row 2)", "05/11/2026").run(html)
        assert passed

    def test_html_table_row_value_mismatch(self):
        html = (
            "<table><thead><tr><th>Course Title</th><th>Start Date</th></tr></thead>\n"
            "<tbody><tr><td>Federal Procurement Basics</td><td>05/04/2026</td></tr></tbody></table>"
        )
        passed, _, _ = self._rule("Start Date (row 1)", "06/01/2026").run(html)
        assert not passed

    def test_html_table_empty_cell_with_empty_expected_passes(self):
        # Out-of-range row with empty expected value: column was found, the
        # cell is genuinely empty, so the rule should pass.
        html = (
            "<table><thead><tr><th>Course Title</th></tr></thead>\n"
            "<tbody><tr><td>Federal Procurement Basics</td></tr></tbody></table>"
        )
        passed, _, _ = self._rule("Course Title (row 5)", "").run(html)
        assert passed

    def test_html_table_colspan_header_concatenated(self):
        # A colspan parent header ("Hours") above a sub-header ("During duty")
        # should match a GT label that combines them.
        html = (
            "<table>\n"
            "<thead>\n"
            "<tr><th rowspan='2'>Course Title</th><th colspan='2'>Hours</th></tr>\n"
            "<tr><th>During duty</th><th>Non duty</th></tr>\n"
            "</thead>\n"
            "<tbody><tr><td>Federal Procurement Basics</td><td>8</td><td>2</td></tr></tbody>\n"
            "</table>"
        )
        passed, _, _ = self._rule("Hours During duty (row 1)", "8").run(html)
        assert passed

    def test_markdown_table_row_lookup(self):
        md = (
            "| Course Title | Start Date | End Date |\n"
            "|---|---|---|\n"
            "| Federal Procurement Basics | 05/04/2026 | 06/01/2026 |\n"
            "| Records Management 101 | 05/11/2026 | 05/29/2026 |\n"
        )
        passed, _, _ = self._rule("Course Title (row 2)", "Records Management 101").run(md)
        assert passed


class TestEmptyExpectedValue:
    def _rule(self, label: str, value: str) -> FormFieldRule:
        return FormFieldRule({"type": "form_field", "label": label, "value": value, "value_type": "text"})

    def test_label_present_value_blank_with_empty_expected_passes(self):
        # Parser surfaces the field but the cell is empty; GT also empty.
        md = "**Account number (see instructions)**:\n"
        passed, _, _ = self._rule("Account number (see instructions)", "").run(md)
        assert passed

    def test_label_absent_with_empty_expected_still_fails(self):
        # An empty expected value does NOT excuse a parser that dropped the
        # field entirely — that would let pipelines silently lose pages.
        md = "**Other Field:** something\n"
        passed, expl, _ = self._rule("Account number", "").run(md)
        assert not passed
        assert "label not found" in expl


class TestBoldColonDoesNotCrossBlankLines:
    def _rule(self, label: str, value: str) -> FormFieldRule:
        return FormFieldRule({"type": "form_field", "label": label, "value": value, "value_type": "text"})

    def test_empty_bold_colon_does_not_capture_next_block_value(self):
        # Regression: previously the ``\s*`` segments in _BOLD_COLON_RE allowed
        # the regex to consume blank lines and headings, so an empty
        # ``**SUFFIX**:\n\nFiling Office Copy ...`` would attribute the heading
        # text to SUFFIX. After the fix, the value is empty and an
        # empty-expected rule passes.
        md = "**SUFFIX**:\n\nFILING OFFICE COPY — INFORMATION STATEMENT\n"
        passed, _, _ = self._rule("SUFFIX", "").run(md)
        assert passed

    def test_empty_bold_colon_with_blank_line_followed_by_other_field(self):
        # Two distinct fields separated by a blank line and a heading. The
        # empty Agency Case No. should not absorb the URLA title.
        md = "**Agency Case No.**:\n\n# Uniform Residential Loan Application\n\n**Other Field**: x\n"
        passed, _, _ = self._rule("Agency Case No.", "").run(md)
        assert passed

    def test_trailing_backslash_value_treated_as_empty(self):
        # ``**Label**: \\`` is a markdown line-continuation; the value should
        # normalize to empty so an empty-expected rule passes.
        md = "**Suffix**: \\\n"
        passed, _, _ = self._rule("Suffix", "").run(md)
        assert passed


class TestMultiColTableHeaderValueRow:
    def _rule(self, label: str, value: str) -> FormFieldRule:
        return FormFieldRule({"type": "form_field", "label": label, "value": value, "value_type": "text"})

    def test_html_3col_header_then_data_row(self):
        # 3-column HTML table with header row + single data row. Used by
        # ours_cost_effective and gemini for vehicle/personal info blocks.
        html = (
            "<table>"
            "<thead><tr>"
            "<th>Vehicle Identification Number</th><th>Year</th><th>Make</th>"
            "</tr></thead>"
            "<tbody><tr>"
            "<td><strong>TESTVIN0001</strong></td><td><strong>2020</strong></td><td><strong>Toyota</strong></td>"
            "</tr></tbody>"
            "</table>"
        )
        passed, _, _ = self._rule("Vehicle Identification Number", "TESTVIN0001").run(html)
        assert passed
        passed, _, _ = self._rule("Year", "2020").run(html)
        assert passed

    def test_html_in_cell_br_label_value_split(self):
        # ``<td>Label<br/><strong>Value</strong></td>`` — label and value
        # stacked inside one cell via <br/>.
        html = (
            "<table><tr>"
            "<td>Last Name (Family Name)<br/><strong>Nguyen</strong></td>"
            "<td>First Name (Given Name)<br/><strong>Erin</strong></td>"
            "</tr></table>"
        )
        passed, _, _ = self._rule("Last Name (Family Name)", "Nguyen").run(html)
        assert passed
        passed, _, _ = self._rule("First Name (Given Name)", "Erin").run(html)
        assert passed


class TestPlainColonInListItems:
    def _rule(self, label: str, value: str) -> FormFieldRule:
        return FormFieldRule({"type": "form_field", "label": label, "value": value, "value_type": "text"})

    def test_dash_bullet_with_label_colon_value(self):
        # OpenAI emits ``- Defendant: Devon Reed`` style list items that
        # used to be silently dropped by the plain-colon scanner.
        md = "Form fields (filled):\n- Defendant: Devon Marcus Reed\n- Plaintiff: Anthony Cole Jackson\n"
        passed, _, _ = self._rule("Defendant", "Devon Marcus Reed").run(md)
        assert passed
        passed, _, _ = self._rule("Plaintiff", "Anthony Cole Jackson").run(md)
        assert passed

    def test_tasklist_bullets_are_still_skipped(self):
        # ``- [ ] Foo: bar`` should remain a checkbox bullet, not a colon pair.
        md = "- [x] Single\n- [ ] Married\n"
        passed, _, _ = self._rule("Single", "Married").run(md)
        assert not passed


class TestExtendedCheckboxGlyphs:
    def _rule(self, label: str, value: bool) -> FormFieldRule:
        return FormFieldRule({"type": "form_field", "label": label, "value": value, "value_type": "checkbox"})

    def test_filled_circle_is_checked(self):
        md = "● Checking account\n○ Savings account\n"
        passed, _, _ = self._rule("Checking account", True).run(md)
        assert passed
        passed, _, _ = self._rule("Savings account", False).run(md)
        assert passed

    def test_fisheye_is_checked(self):
        md = "Married ◉  Single ○\n"
        passed, _, _ = self._rule("Married", True).run(md)
        assert passed


class TestNumberedTaskListAndBareBracketAndInlineAscii:
    def _rule(self, label: str, value: bool) -> FormFieldRule:
        return FormFieldRule({"type": "form_field", "label": label, "value": value, "value_type": "checkbox"})

    def test_numbered_tasklist(self):
        # USCIS citizenship attestation: ``1. [x] Citizen`` numbered list with brackets.
        md = "1. [x] A citizen of the United States\n2. [ ] A noncitizen national\n"
        passed, _, _ = self._rule("A citizen of the United States", True).run(md)
        assert passed
        passed, _, _ = self._rule("A noncitizen national", False).run(md)
        assert passed

    def test_bare_bracket_no_bullet(self):
        # IRS W-9 / UCC5 line: ``\[x] Individual/sole proprietor`` with no
        # leading bullet marker.
        md = "\\[x] Individual/sole proprietor\n\\[ ] C corporation\n"
        passed, _, _ = self._rule("Individual/sole proprietor", True).run(md)
        assert passed
        passed, _, _ = self._rule("C corporation", False).run(md)
        assert passed

    def test_inline_ascii_bracket_group(self):
        # W-9 inline classification group: ``\[ ] A \[x] B \[ ] C`` on one line.
        md = "\\[ ] Individual/sole proprietor \\[x] C corporation \\[ ] S corporation\n"
        passed, _, _ = self._rule("C corporation", True).run(md)
        assert passed
        passed, _, _ = self._rule("Individual/sole proprietor", False).run(md)
        assert passed

    def test_mid_line_bracket_after_bold_label(self):
        # UCC5 inline: ``2a. RECORD IS INACCURATE \[x] enter explanation...``
        # The label-first inline tokenizer treats the bold label as the label
        # for the trailing bracket marker.
        md = "**Inaccuracy in financing statement** \\[x]\n"
        passed, _, _ = self._rule("Inaccuracy in financing statement", True).run(md)
        assert passed


class TestUnderscoreBlankField:
    def _rule(self, label: str, value: str) -> FormFieldRule:
        return FormFieldRule({"type": "form_field", "label": label, "value": value, "value_type": "text"})

    def test_label_followed_by_underscore_blank_passes_empty(self):
        # HUD-2993 layout: ``Processor's Name _________________``.
        md = "Processor's Name _________________\n"
        passed, _, _ = self._rule("Processor's Name", "").run(md)
        assert passed


class TestAdjacentLineFallback:
    def _rule(self, label: str, value: str, value_type: str = "text") -> FormFieldRule:
        return FormFieldRule({"type": "form_field", "label": label, "value": value, "value_type": value_type})

    def test_italic_caption_below_value_returns_value_above(self):
        # AO398 court caption: ``Anthony Cole Jackson )\n*Plaintiff* )``.
        md = "Anthony Cole Jackson )\n*Plaintiff* )\n"
        passed, _, _ = self._rule("Plaintiff", "Anthony Cole Jackson").run(md)
        assert passed

    def test_italic_caption_with_blank_line_above(self):
        # E-mail line followed by blank then ``*E-mail address*`` italic.
        md = "anthony.jackson@example.com\n\n*E-mail address*\n"
        passed, _, _ = self._rule("E-mail address", "anthony.jackson@example.com").run(md)
        assert passed

    def test_numbered_label_then_value_below(self):
        # UCC5 sub-section: ``1a. INITIAL FINANCING STATEMENT FILE NUMBER\nOR-UCC-2025-...``.
        md = "1a. INITIAL FINANCING STATEMENT FILE NUMBER\nOR-UCC-2025-00532600\n"
        passed, _, _ = self._rule("1a. Initial Financing Statement File Number", "OR-UCC-2025-00532600").run(md)
        assert passed

    def test_short_label_in_paragraph_does_not_fire(self):
        # Safety: a paragraph that *contains* the label as a substring must
        # NOT be treated as a label line (strict ratio match required).
        md = "This document is a Statement of Address for the user.\n123 Main Street\n"
        passed, _, _ = self._rule("Address", "123 Main Street").run(md)
        assert not passed

    def test_signature_uses_adjacent_line_fallback(self):
        # AO398: ``True\n*Signature of the attorney or unrepresented party*``.
        md = "True\n*Signature of the attorney or unrepresented party*\n"
        passed, _, _ = self._rule("Signature of the attorney or unrepresented party", True, value_type="signature").run(
            md
        )
        assert passed


class TestDialectAgnosticExtraction:
    """Regression tests for cross-provider output dialects.

    Form-field GT was authored against a canonical ``**Label**: value``
    style. Other providers emit valid but stylistically different markdown
    (HTML wrapping, ``<u>`` fill-in, ``[FORM FIELD]`` tagged lines, pipe-
    concatenated single-line layouts, multi-line values below the header).
    These tests lock in the matcher's tolerance of those dialects so the
    benchmark measures parser quality rather than format compatibility.
    """

    def _rule(self, label: str, value: str) -> FormFieldRule:
        return FormFieldRule({"type": "form_field", "label": label, "value": value, "value_type": "text"})

    def test_html_wrapped_value_in_table_cell_is_stripped(self):
        # haiku-style output: cell contents wrapped in <strong>/<br/>, no
        # markdown KV pair. The plain-colon path captures the line; HTML
        # tags must be stripped from the value before comparison.
        md = "<table><tr><td><strong>Account Balance:</strong><br/>$4,820.00</td></tr></table>"
        passed, _, _ = self._rule("Account Balance", "$4,820.00").run(md)
        assert passed

    def test_underline_template_fill_inline_in_prose(self):
        # gemini-style TREC contract: filled values wrapped in <u>...</u>
        # inline in the form's printed prose.
        md = "**2. PROPERTY:** Lot <u>12</u>, Block <u>C</u>, City of <u>Austin</u>."
        assert self._rule("Block", "C").run(md)[0]
        assert self._rule("Lot", "12").run(md)[0]
        assert self._rule("City of", "Austin").run(md)[0]

    def test_form_field_tag_prefix_stripped_from_label(self):
        # gpt5_mini emits ``[FORM FIELD] <visible-label>: value``. The
        # bracketed prefix is parser noise; the visible label after it must
        # match the rule's label.
        md = "[FORM FIELD] Property/Development Name: Maple Yard Houses\n"
        passed, _, _ = self._rule("Property/Development Name", "Maple Yard Houses").run(md)
        assert passed

    def test_pipe_concatenated_single_line_yields_each_pair(self):
        # cost_effective collapses a multi-field row into a single line with
        # ``|`` separators. The matcher must yield one pair per labelled
        # segment and not bleed the run-on tail into the first value.
        md = "Date: April 27, 2026 | Borrower's Name: Maya Lynn Hernandez | Other: foo\n"
        assert self._rule("Date", "April 27, 2026").run(md)[0]
        assert self._rule("Borrower's Name", "Maya Lynn Hernandez").run(md)[0]
        assert self._rule("Other", "foo").run(md)[0]

    def test_multiline_address_below_bold_colon_header_is_aggregated(self):
        # Audit pattern A3: bold-colon header with empty inline value, the
        # actual value laid out on subsequent bullet lines.
        md = "**C. SEND ACKNOWLEDGMENT TO**:\n- 123 Main St\n- Suite 4\n- Springfield, IL 62701\n"
        passed, _, _ = self._rule("C. SEND ACKNOWLEDGMENT TO", "123 Main St, Suite 4, Springfield, IL 62701").run(md)
        assert passed

    def test_legitimate_pipe_in_value_is_preserved(self):
        # Safety: a value with an embedded pipe but no following ``Label:``
        # shape after it should NOT be split. Form data rarely contains
        # raw pipes, but we shouldn't split on every one.
        md = "Reference: ABC | XYZ | DEF\n"
        passed, _, _ = self._rule("Reference", "ABC | XYZ | DEF").run(md)
        assert passed

    def test_html_wrapped_label_still_resolves(self):
        # Symmetric to the value case: a label can also pick up surrounding
        # HTML tag noise from the parser. Stripping must apply both sides.
        md = "<p><strong>Customer name</strong>: Alex Rivers</p>\n"
        passed, _, _ = self._rule("Customer name", "Alex Rivers").run(md)
        assert passed

    def test_email_autolink_value_is_not_html_stripped(self):
        # Regression: ``<email@host>`` is a markdown autolink, not an HTML
        # tag. The HTML stripper must leave it intact so the email value
        # survives extraction. This used to fail when a permissive
        # ``<[^>]+>`` stripper consumed the entire autolink.
        md = "* **E-mail**: <wei.lin.p019@example.com>\n"
        passed, _, _ = self._rule("E-mail", "wei.lin.p019@example.com").run(md)
        assert passed

    def test_url_autolink_value_is_not_html_stripped(self):
        md = "* **Website**: <https://example.com/profile>\n"
        passed, _, _ = self._rule("Website", "https://example.com/profile").run(md)
        assert passed


class TestPageScoping:
    def _rule(self, label: str, value: str, page: int) -> FormFieldRule:
        return FormFieldRule(
            {
                "type": "form_field",
                "label": label,
                "value": value,
                "value_type": "text",
                "page": page,
            }
        )

    def test_duplicate_label_across_pages_resolved_by_page_field(self):
        # Same label on both pages with different values. Without page scoping
        # the first match wins — wrong page may pass. With injected
        # parse_output + rule.page=2 we must scope to page 2 and pick "Smith".
        page1_md = "**Last Name:** Collins\n"
        page2_md = "**Last Name:** Smith\n"
        full_md = page1_md + page2_md

        rule = self._rule("Last Name", "Smith", page=2)
        rule.parse_output = _FakeParseOutput([_FakePageIR(0, page1_md), _FakePageIR(1, page2_md)])

        passed, _, _ = rule.run(full_md)
        assert passed

    def test_falls_back_to_full_content_when_no_parse_output(self):
        # Without parse_output injection the rule scans full content. Page
        # field is metadata-only in that case.
        page1_md = "**Last Name:** Collins\n"
        rule = self._rule("Last Name", "Collins", page=1)
        # No parse_output set.
        passed, _, _ = rule.run(page1_md)
        assert passed

    def test_fail_closed_when_pages_populated_but_page_missing(self):
        # Provider produced per-page IR but the requested page (3) isn't in
        # the list. Old behavior leaked full-doc content; new behavior
        # returns "" so the rule fails closed.
        page1_md = "**Buyer:** Acme\n"
        page2_md = "**Buyer:** Globex\n"
        full_md = page1_md + page2_md
        rule = self._rule("Buyer", "Acme", page=3)
        rule.parse_output = _FakeParseOutput([_FakePageIR(0, page1_md), _FakePageIR(1, page2_md)])
        passed, _, _ = rule.run(full_md)
        assert not passed

    def test_layout_pages_items_synthesize_when_md_empty(self):
        # Providers like datalab/pulse populate ``layout_pages[*].items`` but
        # leave ``md`` empty. Synthesizing from items should let per-page
        # scoping work without needing each provider to change.
        full_md = "**Last Name:** Collins\n**Last Name:** Smith\n"
        rule = self._rule("Last Name", "Smith", page=2)
        rule.parse_output = _layout_only_output(
            [
                (1, [_LayoutItem(value="**Last Name:** Collins")]),
                (2, [_LayoutItem(value="**Last Name:** Smith")]),
            ]
        )
        passed, _, _ = rule.run(full_md)
        assert passed

    def test_fail_closed_when_matched_page_markdown_is_empty(self):
        # ``pages`` has the requested page but its markdown is empty.
        # Old behavior leaked full-doc content via ``... or content``;
        # new behavior returns "" so the rule fails closed rather than
        # silently scoring against page-1 text.
        full_md = "**Buyer:** Acme\n**Buyer:** Globex\n"
        rule = self._rule("Buyer", "Acme", page=2)
        rule.parse_output = _FakeParseOutput([_FakePageIR(0, "x"), _FakePageIR(1, "")])
        passed, _, _ = rule.run(full_md)
        assert not passed

    def test_layout_pages_md_used_directly_when_populated(self):
        # When ``lp.md`` is non-empty the synthesis path is skipped and
        # ``md`` is returned verbatim — items must not be re-joined on top.
        full_md = "ignored full doc"
        rule = self._rule("Buyer", "Acme", page=2)
        rule.parse_output = _layout_only_output(
            [
                (1, [_LayoutItem(value="Buyer: Globex")], "noise"),
                (2, [_LayoutItem(value="Buyer: Globex")], "**Buyer:** Acme"),
            ]
        )
        passed, _, _ = rule.run(full_md)
        assert passed

    def test_layout_pages_items_priority_md_beats_html_beats_value(self):
        # Items synthesis priority is md > html > value. When ``md`` is
        # populated, ``html`` and ``value`` are ignored.
        full_md = "ignored"
        rule = self._rule("Buyer", "Acme", page=1)
        rule.parse_output = _layout_only_output(
            [
                (
                    1,
                    [
                        _LayoutItem(
                            md="**Buyer:** Acme",
                            html="<p>Buyer: Wrong</p>",
                            value="Buyer: Wrong",
                        )
                    ],
                ),
            ]
        )
        passed, _, _ = rule.run(full_md)
        assert passed


# ---------------------------------------------------------------------------
# Adjacent-label collision resolution (best-score wins)
# ---------------------------------------------------------------------------


class TestAdjacentLabelCollision:
    """When two visually similar labels coexist in the markdown, the rule must
    pick the exact match — not whichever fuzzy hit happened to come first.

    These regressions guard the matcher against the classic
    ``Country`` / ``County`` adjacent-label collision triaged in the
    well_log run, plus a few other near-miss label pairs we have seen in
    the wild (``Operator`` / ``Operator Name``, ``Bill To`` / ``Ship To``,
    etc.).
    """

    def _rule(self, label: str, value: str) -> FormFieldRule:
        return FormFieldRule({"type": "form_field", "label": label, "value": value, "value_type": "text"})

    def test_country_vs_county_county_first(self):
        # County appears BEFORE Country in the markdown. First-match-wins
        # would latch onto County's value (Washington) when looking for
        # Country (U.S.A.). Best-score-wins picks the exact Country match.
        md = "**County**: Washington\n**Country**: U.S.A.\n"
        passed, expl, _ = self._rule("Country", "U.S.A.").run(md)
        assert passed, f"expected pass, got {expl!r}"

    def test_country_vs_county_country_first(self):
        # Reverse order — should still pick the right field.
        md = "**Country**: U.S.A.\n**County**: Washington\n"
        passed, expl, _ = self._rule("Country", "U.S.A.").run(md)
        assert passed, f"expected pass, got {expl!r}"

    def test_county_value_still_resolvable_when_country_also_present(self):
        # Symmetric: the rule for County must still find Washington when
        # Country is also present.
        md = "**Country**: U.S.A.\n**County**: Washington\n"
        passed, expl, _ = self._rule("County", "Washington").run(md)
        assert passed, f"expected pass, got {expl!r}"

    def test_html_table_adjacent_labels(self):
        # Same collision in an HTML 2-col table layout.
        md = "<table><tr><td>County</td><td>Washington</td></tr><tr><td>Country</td><td>U.S.A.</td></tr></table>"
        passed, expl, _ = self._rule("Country", "U.S.A.").run(md)
        assert passed, f"expected pass, got {expl!r}"

    def test_bill_to_vs_ship_to(self):
        # Common invoice layout where Bill To and Ship To share three words
        # and the fuzzy threshold can confuse them.
        md = "**Ship To**: Warehouse 42, Reno NV\n**Bill To**: 100 Main St, Austin TX\n"
        passed, expl, _ = self._rule("Bill To", "100 Main St, Austin TX").run(md)
        assert passed, f"expected pass, got {expl!r}"

    def test_operator_vs_operator_name(self):
        # Exact label `Operator` should not be hijacked by the longer
        # `Operator Name` that happens to appear first in the markdown.
        md = "**Operator Name**: Jane Doe\n**Operator**: ACME Drilling LLC\n"
        passed, expl, _ = self._rule("Operator", "ACME Drilling LLC").run(md)
        assert passed, f"expected pass, got {expl!r}"

    def test_exact_match_in_lower_priority_source_beats_fuzzy_higher_priority(self):
        # A bold-colon hit on a fuzzy near-miss should NOT outrank an exact
        # HTML-table hit on the right label. Best-score-across-all-sources
        # is the whole point.
        md = "**Customer Name**: Wrong Co\n<table><tr><td>Customer</td><td>Right Co</td></tr></table>"
        passed, expl, score = self._rule("Customer", "Right Co").run(md)
        assert passed, f"expected pass, got {expl!r}"
        assert score == 1.0

    def test_two_exact_matches_tie_break_by_source_priority(self):
        # When two sources both surface an EXACT label match, the higher-
        # priority source (bold-colon) wins, preserving legacy behavior on
        # docs where multiple parses agree on the same field.
        md = "**Customer**: Bold Value\n<table><tr><td>Customer</td><td>Table Value</td></tr></table>"
        passed, expl, _ = self._rule("Customer", "Bold Value").run(md)
        assert passed, f"expected pass, got {expl!r}"

    def test_partial_match_loses_to_exact_match(self):
        # A fuzzy/partial hit on a near-miss label must not outrank an exact
        # hit elsewhere in the doc. Without the partial-ratio penalty an
        # equally-strong partial could tie a strict match and the legacy
        # ordering would slip back in.
        md = (
            "**API NO. (if available)**: 4239133299\n"  # partial-ratio match for "API NO."
            "**API NO.**: 1111111111\n"  # exact match for "API NO."
        )
        passed, expl, _ = self._rule("API NO.", "1111111111").run(md)
        assert passed, f"expected pass, got {expl!r}"

    def test_partial_match_still_used_when_no_exact(self):
        # Sanity check: when the only candidate is a partial-ratio hit, the
        # rule still finds it. The penalty only matters for tie-breaking
        # against strict matches; partials remain valid fallbacks.
        md = "**API NO. (if available)**: 4239133299\n"
        passed, expl, _ = self._rule("API NO.", "4239133299").run(md)
        assert passed, f"expected pass, got {expl!r}"


# ---------------------------------------------------------------------------
# Bold connector splicing — "**Depth Drilled**: 105 **to** 15437"
# ---------------------------------------------------------------------------


class TestBoldConnectorValueSplice:
    def _rule(self, label: str, value: str) -> FormFieldRule:
        return FormFieldRule({"type": "form_field", "label": label, "value": value, "value_type": "text"})

    def test_to_connector_in_range(self):
        md = "**Depth Drilled**: 105 **to** 15437\n"
        passed, _, _ = self._rule("Depth Drilled", "105 to 15437").run(md)
        assert passed

    def test_to_with_colon_after_connector(self):
        # Parser sometimes renders `**to**:`
        md = "**Range**: 0.0 **to**: 22888.0\n"
        passed, _, _ = self._rule("Range", "0.0 to 22888.0").run(md)
        assert passed

    def test_and_connector(self):
        md = "**Operators**: Acme **and** Beta\n"
        passed, _, _ = self._rule("Operators", "Acme and Beta").run(md)
        assert passed

    def test_real_label_after_value_still_terminates(self):
        # The connector list should NOT include arbitrary bold spans.
        # ``**Phone**`` after the value is a real label boundary.
        md = "**Name**: Maya **Phone**: 555-0138\n"
        # Name's value must remain "Maya", not "Maya Phone 555-0138".
        passed, _, _ = self._rule("Name", "Maya").run(md)
        assert passed
        passed2, _, _ = self._rule("Phone", "555-0138").run(md)
        assert passed2


# ---------------------------------------------------------------------------
# Dot-strip in label match: K.B. ≡ KB, Tel. ≡ Tel, API NO: ≡ API NO
# ---------------------------------------------------------------------------


class TestDotStripLabelMatch:
    def _rule(self, label: str, value: str) -> FormFieldRule:
        return FormFieldRule({"type": "form_field", "label": label, "value": value, "value_type": "text"})

    def test_gt_dotted_pred_undotted(self):
        md = "**KB**: 60.0 FT\n"
        passed, _, _ = self._rule("K.B.", "60.0 FT").run(md)
        assert passed

    def test_gt_undotted_pred_dotted(self):
        md = "**D.F.**: 59.0 FT\n"
        passed, _, _ = self._rule("DF", "59.0 FT").run(md)
        assert passed

    def test_trailing_colon_in_gt(self):
        md = "**API NO**: 4239133299\n"
        passed, _, _ = self._rule("API NO:", "4239133299").run(md)
        assert passed

    def test_unrelated_labels_still_distinguished(self):
        # Dot-strip must not collapse semantically distinct labels onto
        # each other. "K.B." and "KBC" share 2 letters of overlap; we
        # don't want them treated as equivalent.
        md = "**KBC**: 99\n"
        passed, _, _ = self._rule("K.B.", "60.0 FT").run(md)
        assert not passed

    def test_dot_strip_in_score_path_beats_fuzzy_near_miss(self):
        # Regression for the #976 ⨯ #978 interaction. With dot-strip moved
        # into ``_label_match_score`` (rather than a parallel boolean path
        # outside the scorer), the score path must rank an exact dot-strip
        # equivalent ABOVE a fuzzy near-miss when both appear in the doc.
        #
        # GT label: ``K.B.``. Both candidates fire on the score axis:
        #   - ``KBC`` scores 0.80 via strict fuzz.ratio (the dot-strip
        #     leak we fixed: ``fuzz.ratio("kbc", "kb") == 80.0``).
        #   - ``KB``  scores 1.0  via the dot-strip exact-equality path.
        # Best-score-wins means KB's value ("60.0 FT") must win, not KBC's.
        # If dot-strip lived outside the score path again, the first label
        # match in the document would dictate the value — exactly the bug
        # #978 was added to prevent.
        md = "**KBC**: 99\n**KB**: 60.0 FT\n"
        passed, expl, _ = self._rule("K.B.", "60.0 FT").run(md)
        assert passed, f"expected pass, got {expl!r}"


# ---------------------------------------------------------------------------
# Strikethrough span stripping in predicted values
# ---------------------------------------------------------------------------


class TestStrikethroughStrip:
    def _rule(self, label: str, value: str) -> FormFieldRule:
        return FormFieldRule({"type": "form_field", "label": label, "value": value, "value_type": "text"})

    def test_strikethrough_value_collapses_to_kept(self):
        md = "**Operator**: ~~Old Corp~~ New Corp\n"
        passed, _, _ = self._rule("Operator", "New Corp").run(md)
        assert passed

    def test_no_strikethrough_unaffected(self):
        md = "**Operator**: New Corp\n"
        passed, _, _ = self._rule("Operator", "New Corp").run(md)
        assert passed

    def test_only_strikethrough_present_then_empty(self):
        # If everything was crossed out and the GT records empty, it should
        # still pass (value cleared).
        md = "**Operator**: ~~Old Corp~~\n"
        passed, _, _ = self._rule("Operator", "").run(md)
        assert passed
