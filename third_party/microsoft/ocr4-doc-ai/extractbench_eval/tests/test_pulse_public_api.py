from datetime import datetime
from pathlib import Path

from extract_bench.inference.pipelines import get_pipeline
from extract_bench.inference.providers.parse.pulse import (
    PulseProvider,
    _build_layout_pages,
    _iter_bbox_elements,
    _merge_tables_endpoint_into_markdown,
    _normalize_coords,
    _strip_html_table_blocks,
)
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult
from extract_bench.schemas.product import ProductType


class _FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def _provider(config: dict | None = None) -> PulseProvider:
    return PulseProvider("pulse", {"api_key": "test-key", **(config or {})})


def test_should_run_tables_endpoint_matches_substring_case_insensitively() -> None:
    provider = _provider(
        {
            "use_tables_endpoint": True,
            "tables_endpoint_categories": ["chart", "table"],
        }
    )

    assert provider._should_run_tables_endpoint(Path("/tmp/charts_core/doc.pdf"))
    assert provider._should_run_tables_endpoint(Path("/tmp/Tables_Extended/doc.pdf"))
    assert provider._should_run_tables_endpoint(Path("/tmp/staged/chart-v2/doc.pdf"))
    assert provider._should_run_tables_endpoint(Path("/tmp/docs/table/doc.pdf"))
    assert not provider._should_run_tables_endpoint(Path("/tmp/docs/layout/doc.pdf"))


def test_should_run_tables_endpoint_respects_disabled_and_empty_categories() -> None:
    disabled = _provider({"use_tables_endpoint": False, "tables_endpoint_categories": ["chart"]})
    all_categories = _provider({"use_tables_endpoint": True})

    assert not disabled._should_run_tables_endpoint(Path("/tmp/chart/doc.pdf"))
    assert all_categories._should_run_tables_endpoint(Path("/tmp/layout/doc.pdf"))


def test_registered_pulse_pipeline_runs_tables_endpoint_for_all_paths() -> None:
    pipeline = get_pipeline("pulse")
    provider = _provider(pipeline.config)

    assert provider._use_tables_endpoint is True
    assert provider._tables_endpoint_categories == set()
    assert provider._should_run_tables_endpoint(Path("/tmp/customer/upload/invoice.pdf"))
    assert provider._should_run_tables_endpoint(Path("/tmp/customer/upload/report.pdf"))


def test_normalize_coords_handles_pixels_polygons_and_invalid_boxes() -> None:
    assert _normalize_coords([100, 200, 300, 500]) == [
        0.1,
        0.2,
        0.3,
        0.2,
        0.3,
        0.5,
        0.1,
        0.5,
    ]
    assert _normalize_coords([0.1, 0.2, 0.3, 0.2, 0.3, 0.5, 0.1, 0.5]) == [
        0.1,
        0.2,
        0.3,
        0.2,
        0.3,
        0.5,
        0.1,
        0.5,
    ]
    assert _normalize_coords([0, 0, 0, 10]) is None
    assert _normalize_coords(["bad", 0, 1, 1]) is None


def test_strip_html_table_blocks_removes_tables_without_matching_notable_text() -> None:
    content = "before\n<table><tr><td>A</td></tr></table>\nafter\n<notable>keep</notable>"

    assert _strip_html_table_blocks(content) == "before\nafter\n<notable>keep</notable>"
    assert _strip_html_table_blocks("x<table><tr><td><table></table></td></tr></table>y") == "xy"


def test_merge_tables_endpoint_into_markdown_replaces_or_appends_tables() -> None:
    markdown = "Intro\n<table><tr><td>old</td></tr></table>\nOutro"
    tables_result = {
        "tables_output": {
            "tables": [
                {
                    "table_content": "<table><tr><td>new</td></tr></table>",
                    "from_chart": True,
                    "citations": ["chart-1"],
                }
            ]
        }
    }

    replaced = _merge_tables_endpoint_into_markdown(markdown, tables_result, replace_existing_tables=True)
    assert "old" not in replaced
    assert "Outro" in replaced
    assert "from_chart=true; citations=chart-1" in replaced
    assert "<td>new</td>" in replaced

    appended = _merge_tables_endpoint_into_markdown(markdown, tables_result, replace_existing_tables=False)
    assert "old" in appended
    assert "<td>new</td>" in appended


def _raw_result(example_id: str, source_file_path: str, raw_output: dict) -> RawInferenceResult:
    now = datetime.now()
    return RawInferenceResult(
        request=InferenceRequest(
            example_id=example_id,
            source_file_path=source_file_path,
            product_type=ProductType.PARSE,
        ),
        pipeline=PipelineSpec(
            pipeline_name="pulse",
            provider_name="pulse",
            product_type=ProductType.PARSE,
            config={},
        ),
        pipeline_name="pulse",
        product_type=ProductType.PARSE,
        raw_output=raw_output,
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )


def test_normalize_uses_native_markdown_when_configured() -> None:
    provider = _provider({"markdown_source": "markdown"})
    raw_result = _raw_result(
        example_id="table/doc-1",
        source_file_path="/tmp/parsebench/docs/table/doc.pdf",
        raw_output={
            "markdown": "# **Native Title**\n\n__native bold__",
            "extensions": {
                "altOutputs": {
                    "html": "<html><body><h1><strong>HTML Title</strong></h1></body></html>",
                }
            },
            "bounding_boxes": {},
        },
    )

    normalized = provider.normalize(raw_result)

    assert normalized.output.markdown == "# **Native Title**\n\n__native bold__"


def test_normalize_uses_html_when_configured() -> None:
    provider = _provider({"markdown_source": "html"})
    raw_result = _raw_result(
        example_id="text/text_simple__doc",
        source_file_path="/tmp/parsebench/docs/text/doc.pdf",
        raw_output={
            "markdown": "# **Native Title**\n\n__native bold__",
            "extensions": {
                "altOutputs": {
                    "html": "<html><body><h1><strong>HTML Title</strong></h1></body></html>",
                }
            },
            "bounding_boxes": {},
        },
    )

    normalized = provider.normalize(raw_result)

    assert normalized.output.markdown == "<html><body><h1><strong>HTML Title</strong></h1></body></html>"


def test_normalize_falls_back_to_native_markdown_when_html_is_missing() -> None:
    provider = _provider()
    raw_result = _raw_result(
        example_id="chart/doc-1",
        source_file_path="/tmp/parsebench/docs/chart/doc.pdf",
        raw_output={
            "markdown": "# Native Title",
            "bounding_boxes": {},
        },
    )

    normalized = provider.normalize(raw_result)

    assert normalized.output.markdown == "# Native Title"


def test_run_inference_records_cost_from_configured_credits(tmp_path, monkeypatch) -> None:
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    provider = _provider({"credits_per_page": 10})
    monkeypatch.setattr(
        provider,
        "_extract",
        lambda _: {"markdown": "ok", "bounding_boxes": {}, "plan_info": {"pages_used": 2}},
    )
    pipeline = PipelineSpec(
        pipeline_name="pulse_ultra_2",
        provider_name="pulse",
        product_type=ProductType.PARSE,
        config={},
    )
    request = InferenceRequest(
        example_id="doc",
        source_file_path=str(source),
        product_type=ProductType.PARSE,
    )

    result = provider.run_inference(pipeline, request)

    assert result.raw_output["cost_per_page_usd"] == 0.15
    assert result.raw_output["cost_usd"] == 0.30
    assert result.raw_output["num_pages"] == 2


def test_extract_passes_request_timeout(tmp_path, monkeypatch) -> None:
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    provider = _provider({"request_timeout": 12})
    seen: dict[str, float] = {}

    def fake_post(*_, timeout: float, **__) -> _FakeResponse:
        seen["timeout"] = timeout
        return _FakeResponse({"markdown": "ok", "bounding_boxes": {}})

    monkeypatch.setattr("extract_bench.inference.providers.parse.pulse.requests.post", fake_post)

    assert provider._extract(str(source))["markdown"] == "ok"
    assert seen["timeout"] == 12


def test_iter_bbox_elements_skips_ordered_metadata_when_grouped_boxes_exist() -> None:
    bounding_boxes = {
        "markdown_with_ids": "# title",
        "ordered_elements": [{"source_category": "Text", "bounding_box": [0, 0, 1, 1], "text": "duplicate"}],
        "Text": [{"bounding_box": [0, 0, 100, 100], "text": "body"}],
    }

    assert list(_iter_bbox_elements(bounding_boxes)) == [("Text", {"bounding_box": [0, 0, 100, 100], "text": "body"})]


def test_build_layout_pages_normalizes_flat_and_grouped_bbox_outputs() -> None:
    pages = _build_layout_pages(
        {
            "ordered_elements": [
                {
                    "source_category": "Header",
                    "bbox": [100, 200, 900, 250],
                    "page": 1,
                    "text": "section heading",
                    "confidence": 0.8,
                }
            ]
        }
    )
    assert len(pages) == 1
    assert pages[0].items[0].bbox.label == "Section-header"
    assert pages[0].items[0].value == "section heading"

    list_pages = _build_layout_pages(
        {
            "ordered_elements": [
                {
                    "source_category": "List Items",
                    "bbox": [100, 200, 900, 250],
                    "page": 1,
                    "text": "bullet",
                }
            ]
        }
    )
    assert list_pages[0].items[0].bbox.label == "List-item"

    formula_pages = _build_layout_pages(
        {
            "ordered_elements": [
                {
                    "source_category": "Formulas",
                    "bbox": [100, 200, 900, 250],
                    "page": 1,
                    "text": "x = y",
                }
            ]
        }
    )
    assert formula_pages[0].items[0].bbox.label == "Formula"

    grouped_pages = _build_layout_pages(
        {
            "Tables": [
                {
                    "table_info": {"location": {"coordinates": [50, 60, 500, 300], "page": 2}, "confidence": 0.9},
                    "cell_data": [{"text": "0t-Name"}, {"text": "Value"}],
                }
            ]
        }
    )
    assert grouped_pages[0].page_number == 2
    assert grouped_pages[0].items[0].type == "table"
    assert grouped_pages[0].items[0].bbox.label == "Table"
    assert grouped_pages[0].items[0].value == "Name Value"
