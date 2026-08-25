"""Builders for grounded-confidence document payloads.

Precomputes the payloads that reports and viewers read back.

All entry points take explicit paths; this module holds no configuration
state of its own.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, TypedDict, cast
from urllib.parse import quote

from extract_bench.evaluation.metrics.extract.confidence_scoped.paths import iter_leaf_values
from extract_bench.evaluation.metrics.extract.confidence_scoped.scoring import build_confidence_field_rows
from extract_bench.evaluation.metrics.extract.confidence_scoped.summary import (
    DEFAULT_CONFIDENCE_THRESHOLDS,
    compute_confidence_threshold_rows,
    summarize_confidence_rows,
)
from extract_bench.evaluation.metrics.extract.confidence_scoped.types import ConfidenceFieldRow
from extract_bench.schemas.extract_output import FieldCitation
from extract_bench.test_cases.loader import load_test_case, load_test_cases
from extract_bench.test_cases.schema import ExtractFieldTestRule

logger = logging.getLogger(__name__)

# Directory written next to a pipeline's results, holding the grounded
# confidence artifacts that reports and viewers read back. The name is part of
# the on-disk output layout, so changing it orphans artifacts from earlier runs.
GROUNDED_CONFIDENCE_ARTIFACT_DIRNAME = "_grounded_confidence"

# Keys present only on a full document detail payload; stripping them yields
# the lightweight per-document summary used in listings.
DOC_DETAIL_KEYS = frozenset(
    {
        "parsed_markdown",
        "test_json",
        "ground_truth",
        "data_schema",
        "extracted_data",
        "field_citations",
        "field_rows",
        "pdf_url",
    }
)


class DocumentPaths(TypedDict):
    dataset_dir: str
    test_json: str
    pdf: str | None
    pdf_url: str | None
    markdown: str | None
    result_json: str


class GroundedDocumentRecord(TypedDict):
    id: str
    description: Any
    domain: Any
    source: Any
    notes: Any
    phi_present: bool
    schema_leaves: Any
    gt_leaves: Any
    pages: Any
    paths: DocumentPaths
    metrics_from_report: dict[str, float]


class SkippedDocument(TypedDict, total=False):
    id: str
    reason: str
    path: str


class SerializedConfidenceRow(TypedDict):
    id: str
    document_id: str
    field_path: str
    field_pattern: str
    predicted_path: str | None
    expected_value: Any
    predicted_value: Any
    confidence: float | None
    correct: bool
    comparison_score: float
    comparison_mode: str
    verified: bool
    matched_gt_path: str
    alignment_mode: str
    alignment_score: float
    cited_text: str
    schema_valid: bool
    citation: dict[str, Any] | None


class GroundedDocumentPayload(TypedDict, total=False):
    id: str
    description: Any
    domain: Any
    source: Any
    notes: Any
    phi_present: bool
    schema_leaves: Any
    gt_leaves: Any
    pages: Any
    paths: DocumentPaths
    metrics_from_report: dict[str, float]
    metrics: dict[str, Any]
    thresholds: list[dict[str, Any]]
    field_count: int
    correct_field_count: int
    fields_with_confidence: int
    parsed_markdown: str | None
    test_json: Any
    ground_truth: Any
    data_schema: Any
    extracted_data: Any
    field_citations: list[dict[str, Any]]
    field_rows: list[SerializedConfidenceRow]
    pdf_url: str | None


def load_json_file(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def safe_read_text(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def first_file(directory: Path, pattern: str) -> Path | None:
    return next(iter(sorted(directory.glob(pattern))), None)


def _result_json_for_doc(run_dir: Path, doc_id: str) -> Path | None:
    """Resolve the canonical result path, then the nested v0.1 layout."""
    result_json = run_dir / f"{doc_id}.result.json"
    if result_json.exists():
        return result_json
    return first_file(run_dir / doc_id, "*.result.json")


def _relative_posix(base_dir: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return str(path)


def _test_sidecar_for_case(test_case: Any) -> Path:
    """Resolve the sidecar convention used by the canonical test-case loader."""
    source_path = Path(test_case.file_path).resolve()
    source_sidecar = source_path.parent / f"{source_path.stem}.test.json"
    if source_sidecar.exists():
        return source_sidecar

    # Multi-file Extract groups use ``<group>/<group>.test.json`` while the
    # loaded case's primary ``file_path`` points at the base input file.
    if getattr(test_case, "input_files", None):
        return source_path.parent / f"{source_path.parent.name}.test.json"
    return source_sidecar


def _doc_ids_from_example(value: str | None) -> list[str]:
    if not value:
        return []
    text = str(value).strip()
    if not text:
        return []
    candidates = [text]
    if "/" in text:
        candidates.extend([text.split("/", 1)[0], text.rsplit("/", 1)[-1]])
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def _metric_values_by_name(result: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in result.get("metrics", []):
        if not isinstance(metric, dict):
            continue
        name = metric.get("metric_name")
        value = metric.get("value")
        if isinstance(name, str) and isinstance(value, (int, float)) and not isinstance(value, bool):
            out[name] = float(value)
    return out


def load_report_metrics(report_dir: Path) -> dict[str, dict[str, float]]:
    """Per-doc metric values from an ``_evaluation_report.json``, if present."""
    report_path = report_dir / "_evaluation_report.json"
    if not report_path.exists():
        return {}
    report = load_json_file(report_path)
    out: dict[str, dict[str, float]] = {}
    for result in report.get("per_example_results", []):
        if not isinstance(result, dict):
            continue
        doc_ids = _doc_ids_from_example(result.get("example_id")) or _doc_ids_from_example(result.get("test_id"))
        metrics = _metric_values_by_name(result)
        for doc_id in doc_ids:
            out.setdefault(doc_id, metrics)
    return out


def synthesize_manifest_payload(dataset_dir: Path) -> dict[str, Any]:
    """Build a manifest-shaped payload for datasets without a manifest.json."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Grounded confidence dataset dir not found: {dataset_dir}")

    docs: list[dict[str, Any]] = []
    test_cases = load_test_cases(dataset_dir, require_test_json=True, product_type="EXTRACT")
    for test_case in test_cases:
        source_path = Path(test_case.file_path).resolve()
        test_json = _test_sidecar_for_case(test_case)
        markdown_path = source_path.with_suffix(".md")
        doc: dict[str, Any] = {
            "id": test_case.test_id,
            "description": "",
            "domain": "",
            "source": "",
            "phi_present": False,
            "test_json": _relative_posix(dataset_dir, test_json),
        }
        pdf_rel = _relative_posix(dataset_dir, source_path) if source_path.exists() else None
        markdown_rel = _relative_posix(dataset_dir, markdown_path) if markdown_path.exists() else None
        if pdf_rel:
            doc["pdf"] = pdf_rel
        if markdown_rel:
            doc["source_md"] = markdown_rel
        docs.append(doc)

    return {"name": dataset_dir.name, "version": "synthetic", "synthetic": True, "docs": docs}


def load_manifest_payload(manifest_path: Path, *, dataset_dir: Path | None = None) -> dict[str, Any]:
    if not manifest_path.exists():
        if dataset_dir is not None and manifest_path == (dataset_dir / "manifest.json").resolve():
            return synthesize_manifest_payload(dataset_dir)
        raise FileNotFoundError(f"Grounded confidence manifest not found: {manifest_path}")
    manifest = load_json_file(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Grounded confidence manifest must be a JSON object: {manifest_path}")
    if not isinstance(manifest.get("docs"), list):
        raise ValueError(f"Grounded confidence manifest must include a docs array: {manifest_path}")
    return manifest


def _manifest_string(doc: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _dataset_relative_path(dataset_dir: Path, relative_path: str | None) -> Path | None:
    if not relative_path:
        return None
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return (dataset_dir / path).resolve()


def _dataset_url(relative_path: str | None, manifest: dict[str, Any], *, dataset_url_base: str) -> str | None:
    if not relative_path or Path(relative_path).is_absolute():
        return None
    if dataset_url_base:
        return f"{dataset_url_base.rstrip('/')}/{quote(relative_path, safe='/')}"
    # Optional: a manifest may name the host serving the source documents.
    data_root = manifest.get("dataset_url_base")
    if isinstance(data_root, str) and data_root.strip():
        return f"{data_root.rstrip('/')}/{quote(relative_path, safe='/')}"
    return None


def _record_from_manifest_doc(
    *,
    dataset_dir: Path,
    run_dir: Path,
    manifest: dict[str, Any],
    manifest_doc: dict[str, Any],
    report_metrics: dict[str, dict[str, float]],
    dataset_url_base: str,
) -> tuple[GroundedDocumentRecord | None, SkippedDocument | None]:
    doc_id = manifest_doc.get("id")
    if not isinstance(doc_id, str) or not doc_id.strip():
        return None, {"id": str(doc_id or ""), "reason": "missing_doc_id"}
    doc_id = doc_id.strip()

    test_rel = _manifest_string(manifest_doc, "test_json")
    pdf_rel = _manifest_string(manifest_doc, "pdf", "source_pdf")
    markdown_rel = _manifest_string(manifest_doc, "source_md", "parsed_markdown", "markdown")
    doc_dir = (dataset_dir / doc_id).resolve()

    test_json = _dataset_relative_path(dataset_dir, test_rel) if test_rel else None
    if test_json is None and doc_dir.exists():
        test_json = first_file(doc_dir, "*.test.json") or (doc_dir / "test.json")
    if test_json is None or not test_json.exists():
        path = str(test_json) if test_json else ""
        logger.warning("Skipping confidence artifact for %s: missing test JSON at %s", doc_id, path)
        return None, {"id": doc_id, "reason": "missing_test_json", "path": path}

    result_json = _result_json_for_doc(run_dir, doc_id)
    if result_json is None:
        return None, {"id": doc_id, "reason": "missing_result_json", "path": str(run_dir / doc_id)}

    pdf_path = _dataset_relative_path(dataset_dir, pdf_rel)
    if pdf_path is None and doc_dir.exists():
        pdf_path = first_file(doc_dir, "*.pdf")
    if pdf_path is not None and not pdf_path.exists():
        pdf_path = None

    markdown_path = _dataset_relative_path(dataset_dir, markdown_rel)
    if markdown_path is None and doc_dir.exists():
        markdown_path = first_file(doc_dir, "*.md")
    if markdown_path is not None and not markdown_path.exists():
        markdown_path = None

    paths: DocumentPaths = {
        "dataset_dir": str(doc_dir if doc_dir.exists() else test_json.parent),
        "test_json": str(test_json),
        "pdf": str(pdf_path) if pdf_path else None,
        "pdf_url": _dataset_url(pdf_rel, manifest, dataset_url_base=dataset_url_base),
        "markdown": str(markdown_path) if markdown_path else None,
        "result_json": str(result_json),
    }
    record: GroundedDocumentRecord = {
        "id": doc_id,
        "description": manifest_doc.get("description", ""),
        "domain": manifest_doc.get("domain", ""),
        "source": manifest_doc.get("source", ""),
        "notes": manifest_doc.get("notes", ""),
        "phi_present": bool(manifest_doc.get("phi_present", False)),
        "schema_leaves": manifest_doc.get("schema_leaves"),
        "gt_leaves": manifest_doc.get("gt_leaves"),
        "pages": manifest_doc.get("pages"),
        "paths": paths,
        "metrics_from_report": report_metrics.get(doc_id, {}),
    }
    return record, None


def doc_records_with_skips(
    *,
    dataset_dir: Path,
    run_dir: Path,
    manifest_path: Path,
    report_dir: Path | None = None,
    dataset_url_base: str = "",
) -> tuple[list[GroundedDocumentRecord], list[SkippedDocument]]:
    """Resolve manifest docs against a run dir into (records, skipped)."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Grounded confidence dataset dir not found: {dataset_dir}")
    if not run_dir.exists():
        raise FileNotFoundError(f"Grounded confidence run dir not found: {run_dir}")

    manifest = load_manifest_payload(manifest_path, dataset_dir=dataset_dir)
    report_metrics = load_report_metrics(report_dir) if report_dir is not None else {}
    records: list[GroundedDocumentRecord] = []
    skipped: list[SkippedDocument] = []
    for manifest_doc in manifest.get("docs", []):
        if not isinstance(manifest_doc, dict):
            skipped.append({"id": "", "reason": "invalid_manifest_doc"})
            continue
        record, skip = _record_from_manifest_doc(
            dataset_dir=dataset_dir,
            run_dir=run_dir,
            manifest=manifest,
            manifest_doc=manifest_doc,
            report_metrics=report_metrics,
            dataset_url_base=dataset_url_base,
        )
        if record is not None:
            records.append(record)
        elif skip is not None:
            skipped.append(skip)
    return records, skipped


def confidence_field_rules(test_case: Any) -> list[Any]:
    """Return explicit extract_field rules, or derive leaf rules from expected_output.

    Older extract benchmarks often carry only ``expected_output`` sidecars. The
    confidence sweep needs per-leaf rows, so we derive temporary value-only
    rules when curated ``extract_field`` rules are absent.

    Note: derived rules have no ``comparator`` / ``structural`` / ``evidence``,
    so the typed comparator path collapses to ``case_insensitive`` for every
    leaf — dates compared as strings, numbers as strings, no ``match_by:`` for
    arrays. The bench loader converts v0.2 ``_field_rules`` dicts to
    ``test_rules``, so this branch only fires for legacy datasets that ship
    only ``expected_output``. We warn so the loss of comparator nuance shows
    up in run logs.
    """
    rules = list(test_case.get_extract_field_rules())
    if rules or getattr(test_case, "expected_output", None) is None:
        return rules
    logger.warning(
        "Confidence sweep deriving rules from expected_output for %s — "
        "typed comparators (date/number/enum/match_by) are not applied. "
        "Add explicit test_rules or _field_rules to recover them.",
        getattr(test_case, "test_id", "<unknown test case>"),
    )
    return [
        ExtractFieldTestRule(
            field_path=field_path,
            expected_value=expected_value,
            verified=False,
            evidence_required=False,
        )
        for field_path, expected_value in iter_leaf_values(test_case.expected_output)
    ]


def serialize_citation(citation: Any) -> dict[str, Any]:
    if hasattr(citation, "model_dump"):
        data = citation.model_dump(mode="json")
        return data if isinstance(data, dict) else {}
    if isinstance(citation, dict):
        return citation
    return {}


def field_citation_lookup(citations: list[Any]) -> dict[str, dict[str, Any]]:
    """Highest-confidence citation per field path.

    Row confidence is the max citation confidence for the path, so the
    citation shown next to a row must be the one that produced that number.
    """
    out: dict[str, dict[str, Any]] = {}
    for citation in citations:
        data = serialize_citation(citation)
        field_path = data.get("field_path")
        if not isinstance(field_path, str):
            continue
        current = out.get(field_path)
        confidence = data.get("confidence")
        current_confidence = current.get("confidence") if current else None
        if current is None or (
            isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and (not isinstance(current_confidence, (int, float)) or confidence > current_confidence)
        ):
            out[field_path] = data
    return out


def serialize_confidence_row(
    doc_id: str, index: int, row: ConfidenceFieldRow, citations_by_path: dict[str, dict[str, Any]]
) -> SerializedConfidenceRow:
    citation = citations_by_path.get(row.predicted_path or "") or citations_by_path.get(row.field_path or "")
    return {
        "id": f"{doc_id}:{index}",
        "document_id": doc_id,
        "field_path": row.field_path,
        "field_pattern": row.field_pattern,
        "predicted_path": row.predicted_path,
        "expected_value": row.expected_value,
        "predicted_value": row.predicted_value,
        "confidence": row.confidence,
        "correct": row.correct,
        "comparison_score": row.comparison_score,
        "comparison_mode": row.comparison_mode,
        "verified": row.verified,
        "matched_gt_path": row.matched_gt_path,
        "alignment_mode": row.alignment_mode,
        "alignment_score": row.alignment_score,
        "cited_text": row.cited_text,
        "schema_valid": row.schema_valid,
        "citation": citation,
    }


def build_doc_payload(
    record: GroundedDocumentRecord,
    *,
    include_details: bool,
    pdf_api_url: str | None = None,
) -> GroundedDocumentPayload:
    """Score one document record and build its dashboard payload.

    ``pdf_api_url`` is the caller's serving route for the document PDF, used
    only when the dataset does not publish a direct ``pdf_url``.
    """
    record_paths = record["paths"]
    test_json_path = Path(record_paths["test_json"])
    file_path = Path(record_paths["pdf"] or test_json_path)
    result_json_path = Path(record_paths["result_json"])
    result = load_json_file(result_json_path)
    output = result.get("output", {})
    field_citations = [
        FieldCitation.model_validate(citation)
        for citation in output.get("field_citations", [])
        if isinstance(citation, dict)
    ]
    test_case = load_test_case(file_path, test_json_path)
    if test_case is None or not hasattr(test_case, "get_extract_field_rules"):
        raise ValueError(f"Could not load extract test case from {test_json_path}")
    field_rules = confidence_field_rules(test_case)

    # `eval_data_schema`, not `data_schema`: this lane shares the alignment code
    # with ExtractEvaluator, and a schema-declared identity_key is authoritative
    # while the `match_by:` rule path is gated on global uniqueness. Reading the
    # plain schema here would silently score migrated GT with gated-only
    # alignment and diverge from the main evaluator on duplicate-identity rows.
    eval_schema = getattr(test_case, "eval_data_schema", None) or getattr(test_case, "data_schema", None)
    rows = build_confidence_field_rows(
        extracted_data=output.get("extracted_data", {}),
        field_rules=field_rules,
        field_citations=field_citations,
        data_schema=eval_schema,
        expected_output=getattr(test_case, "expected_output", None),
    )
    citations_by_path = field_citation_lookup(field_citations)
    serialized_rows: list[SerializedConfidenceRow] = [
        serialize_confidence_row(record["id"], index, row, citations_by_path) for index, row in enumerate(rows)
    ]
    summary = summarize_confidence_rows(
        rows,
        thresholds=DEFAULT_CONFIDENCE_THRESHOLDS,
        expected_output=getattr(test_case, "expected_output", None),
        data_schema=eval_schema,
    )
    payload: GroundedDocumentPayload = {
        "id": record["id"],
        "description": record["description"],
        "domain": record["domain"],
        "source": record["source"],
        "notes": record["notes"],
        "phi_present": record["phi_present"],
        "schema_leaves": record["schema_leaves"],
        "gt_leaves": record["gt_leaves"],
        "pages": record["pages"],
        "metrics_from_report": record["metrics_from_report"],
        "paths": record_paths,
        # Scoped metrics intentionally win over same-named report metrics
        # (e.g. "accuracy"): this payload is the confidence-scoped view, and
        # the report copy is only carried along for context.
        "metrics": {**record["metrics_from_report"], **summary},
        "thresholds": compute_confidence_threshold_rows(rows, DEFAULT_CONFIDENCE_THRESHOLDS),
        "field_count": len(rows),
        "correct_field_count": sum(1 for row in rows if row.correct),
        "fields_with_confidence": sum(1 for row in rows if row.confidence is not None),
    }
    if include_details:
        # Every key added here must also be listed in DOC_DETAIL_KEYS so that
        # doc_summary_from_detail() strips it from the lightweight summary.
        test_json = load_json_file(test_json_path)
        markdown = record_paths["markdown"]
        payload["parsed_markdown"] = safe_read_text(Path(markdown)) if markdown else None
        payload["test_json"] = test_json
        payload["ground_truth"] = test_json.get("expected_output")
        payload["data_schema"] = test_json.get("data_schema")
        payload["extracted_data"] = output.get("extracted_data", {})
        payload["field_citations"] = [serialize_citation(citation) for citation in field_citations]
        payload["field_rows"] = serialized_rows
        payload["pdf_url"] = record_paths["pdf_url"] or (pdf_api_url if record_paths["pdf"] else None)
    return payload


def doc_summary_from_detail(detail: GroundedDocumentPayload) -> GroundedDocumentPayload:
    # mypy cannot express "TypedDict minus a key set"; the comprehension only
    # ever drops keys from a total=False TypedDict, so the cast is faithful.
    return cast(
        "GroundedDocumentPayload",
        {key: value for key, value in detail.items() if key not in DOC_DETAIL_KEYS},
    )


def aggregate_rows_summary(
    rows: list[SerializedConfidenceRow],
    documents: list[GroundedDocumentPayload],
) -> dict[str, Any]:
    """Aggregate serialized field rows across documents into one summary.

    GT-denominator totals are summed from the per-document metrics because the
    aggregated rows only cover emitted fields.
    """
    full_gt_totals = {
        "expected_leaf_total": sum(
            int(doc.get("metrics", {}).get("confidence_full_gt_expected_leaf_count") or 0) for doc in documents
        ),
        "unique_expected_leaf_covered": sum(
            int(doc.get("metrics", {}).get("confidence_full_gt_unique_leaf_covered") or 0) for doc in documents
        ),
        "invalid_schema_fields": sum(
            int(doc.get("metrics", {}).get("confidence_full_gt_invalid_schema_fields") or 0) for doc in documents
        ),
    }
    return summarize_confidence_rows(
        rows,
        thresholds=DEFAULT_CONFIDENCE_THRESHOLDS,
        full_gt_totals=full_gt_totals,
    )
