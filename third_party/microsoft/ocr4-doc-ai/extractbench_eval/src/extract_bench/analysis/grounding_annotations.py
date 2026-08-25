"""Shared loaders for extract grounding annotations used by HTML reports.

Both the detailed report and comparison report need camelCase citation / GT
evidence payloads for PDF overlays. Callers choose payload richness via flags:
comparison keeps HTML small (bbox-only); detailed keeps text metadata.
"""

from __future__ import annotations

from typing import Any


def load_pred_grounding_citations(
    result_data: dict[str, Any] | None,
    *,
    require_bbox: bool = False,
    include_reference_text: bool = True,
) -> list[dict[str, Any]]:
    """Load predicted field citations from an inference result payload.

    Args:
        result_data: Parsed ``*.result.json`` (or equivalent) dict.
        require_bbox: When True, drop page-only citations that lack a bbox.
        include_reference_text: When True, keep non-empty ``referenceText``.
    """
    if not result_data:
        return []
    output = result_data.get("output") or {}
    citations = output.get("field_citations") if isinstance(output, dict) else None
    if not isinstance(citations, list):
        return []

    annotations: list[dict[str, Any]] = []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        field_path = citation.get("field_path")
        page = citation.get("page")
        bbox = citation.get("bbox")
        if not field_path or page is None:
            continue
        has_bbox = isinstance(bbox, list) and len(bbox) >= 4
        if require_bbox and not has_bbox:
            continue
        entry: dict[str, Any] = {
            "fieldPath": str(field_path),
            "page": int(page),
        }
        if has_bbox:
            entry["bbox"] = [float(v) for v in bbox[:4]]
        if include_reference_text:
            reference_text = citation.get("reference_text")
            if isinstance(reference_text, str) and reference_text:
                entry["referenceText"] = reference_text
        annotations.append(entry)
    return annotations


def load_gt_grounding_annotations(
    test_data: dict[str, Any] | None,
    *,
    include_verified: bool = True,
    include_quote: bool = True,
) -> list[dict[str, Any]]:
    """Load GT evidence boxes from ``_field_rules`` and legacy ``test_rules``.

    Only evidence entries with both ``page`` and a 4+ float ``bbox`` are kept.
    """
    if not test_data:
        return []

    annotations: list[dict[str, Any]] = []

    def _append_from_rule(field_path: str, rule: dict[str, Any]) -> None:
        verified = bool(rule.get("verified", False))
        # Prefer the v0.2 ``evidence`` list. Rules still on the v0.1 shape carry
        # the same page/normalized-COCO pairs under ``bboxes`` instead; without
        # the fallback those test cases render no GT overlay at all.
        entries = rule.get("evidence") or rule.get("bboxes")
        for evidence in entries or []:
            if not isinstance(evidence, dict):
                continue
            page = evidence.get("page")
            bbox = evidence.get("bbox")
            if page is None or not isinstance(bbox, list) or len(bbox) < 4:
                continue
            entry: dict[str, Any] = {
                "fieldPath": str(field_path),
                "page": int(page),
                "bbox": [float(v) for v in bbox[:4]],
            }
            if include_verified:
                entry["verified"] = verified
            if include_quote:
                entry["quote"] = evidence.get("quote")
            annotations.append(entry)

    field_rules = test_data.get("_field_rules") or {}
    if isinstance(field_rules, dict):
        for field_path, rule in field_rules.items():
            if isinstance(rule, dict):
                _append_from_rule(str(field_path), rule)

    test_rules = test_data.get("test_rules") or []
    if isinstance(test_rules, list):
        for rule in test_rules:
            if not isinstance(rule, dict):
                continue
            rule_type = rule.get("type")
            if rule_type not in (None, "extract_field"):
                continue
            field_path = rule.get("field_path") or rule.get("id")
            if not field_path:
                continue
            _append_from_rule(str(field_path), rule)

    return annotations
