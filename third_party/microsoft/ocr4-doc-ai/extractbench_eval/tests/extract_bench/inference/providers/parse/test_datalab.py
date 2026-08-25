"""Unit tests for the Datalab parse provider's mode-specific pricing and
image-input handling."""

import asyncio
from dataclasses import dataclass

import pytest

pytest.importorskip("datalab_sdk", reason="dev and runners extras required; run: uv sync --extra dev --extra runners")

from extract_bench.inference.providers.parse import datalab as datalab_provider
from extract_bench.inference.providers.parse.datalab import DatalabProvider


def test_datalab_parse_pricing_is_mode_specific() -> None:
    assert DatalabProvider.COST_PER_PAGE_USD_BY_MODE == {
        "fast": 0.004,
        "balanced": 0.004,
        "accurate": 0.01,
    }


def test_datalab_image_inputs_use_single_page_fallback(tmp_path) -> None:
    image_path = tmp_path / "page.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")

    assert datalab_provider._get_source_page_count(image_path) == 1


def test_datalab_non_pdf_bytes_do_not_raise_during_page_count(tmp_path) -> None:
    mislabelled_path = tmp_path / "page.pdf"
    mislabelled_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")

    assert datalab_provider._get_source_page_count(mislabelled_path) is None


def test_datalab_parse_image_input_does_not_require_pdf_reader(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "page.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
    seen_paths: list[str] = []

    @dataclass
    class FakeConvertResult:
        html: str = "<p>ok</p>"
        page_count: int | None = None

    class FakeDatalabClient:
        def __init__(self, api_key: str) -> None:
            assert api_key == "test-key"

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def convert(self, source_path: str, options):
            seen_paths.append(source_path)
            return FakeConvertResult()

    monkeypatch.setattr(datalab_provider, "AsyncDatalabClient", FakeDatalabClient)
    provider = DatalabProvider("datalab", {"api_key": "test-key", "mode": "fast"})

    raw_output = asyncio.run(provider._parse_file_async(str(image_path)))

    assert seen_paths == [str(image_path)]
    assert raw_output["_config"]["total_pages"] == 1
    assert raw_output["cost_usd"] == 0.004
    assert raw_output["cost_per_page_usd"] == 0.004
