"""Tests for layout adapter selection order and lazy registry loading."""

from __future__ import annotations

import datetime as dt
import importlib.util
import subprocess
import sys
from types import SimpleNamespace

import pytest

from extract_bench.evaluation.layout_adapters import registry
from extract_bench.evaluation.layout_adapters.base import LayoutAdapter
from extract_bench.evaluation.layout_adapters.registry import (
    _LayoutAdapterRegistration,
    create_layout_adapter_for_result,
)
from extract_bench.schemas.parse_output import ParseLayoutPageIR, ParseOutput
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from extract_bench.schemas.product import ProductType

# Loading the real adapter bindings imports provider modules that require the
# runners extra; fake-registry tests below run without it.
requires_adapters = pytest.mark.skipif(
    importlib.util.find_spec("llama_cloud") is None,
    reason="dev and runners extras required; run: uv sync --extra dev --extra runners",
)


class _NonMatchingAdapter(LayoutAdapter):
    def to_layout_output(self, inference_result, *, page_filter=None):
        raise NotImplementedError


class _MatchingAdapter(LayoutAdapter):
    @classmethod
    def matches(cls, inference_result):
        return True

    def to_layout_output(self, inference_result, *, page_filter=None):
        raise NotImplementedError


class _MatchingAdapterHighPriority(_MatchingAdapter):
    pass


class _DefaultAdapter(_NonMatchingAdapter):
    pass


def _registration(cls, *keys, priority=0):
    return _LayoutAdapterRegistration(keys=tuple(keys), priority=priority, adapter_cls=cls)


def _make_result(pipeline_name="test_pipeline", layout_pages=False, raw_output=None):
    now = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    pages = [ParseLayoutPageIR(page_number=1)] if layout_pages else []
    return InferenceResult(
        request=InferenceRequest(
            example_id="example",
            source_file_path="example.pdf",
            product_type=ProductType.PARSE,
        ),
        pipeline_name=pipeline_name,
        product_type=ProductType.PARSE,
        raw_output=raw_output or {},
        output=ParseOutput(
            example_id="example",
            pipeline_name=pipeline_name,
            markdown="content",
            layout_pages=pages,
        ),
        started_at=now,
        completed_at=now,
        latency_in_ms=1,
    )


@pytest.fixture
def fake_registry(monkeypatch):
    """Replace the registry with a controlled set of fake registrations."""
    monkeypatch.setattr(registry, "_adapters_loaded", True)
    monkeypatch.setattr(
        registry,
        "_LAYOUT_ADAPTER_REGISTRY",
        [
            _registration(_NonMatchingAdapter, "prov_keyed"),
            _registration(_MatchingAdapter, "prov_matcher", priority=10),
            _registration(_MatchingAdapterHighPriority, "prov_matcher_high", priority=90),
            _registration(_DefaultAdapter, "__default__", priority=-100),
        ],
    )


def test_provider_key_wins_over_higher_priority_matcher(fake_registry, monkeypatch):
    monkeypatch.setattr(registry, "get_pipeline", lambda name: SimpleNamespace(provider_name="prov_keyed"))
    adapter = create_layout_adapter_for_result(_make_result())
    assert isinstance(adapter, _NonMatchingAdapter)


def test_unregistered_provider_key_falls_back_to_default_not_matchers(fake_registry, monkeypatch):
    # A resolvable provider without a registered adapter must land on the
    # default adapter. Shape matchers cannot distinguish providers (many emit
    # the same shared layout IR), so running them here would silently score one
    # provider's results with another provider's adapter.
    monkeypatch.setattr(registry, "get_pipeline", lambda name: SimpleNamespace(provider_name="prov_unregistered"))
    adapter = create_layout_adapter_for_result(_make_result())
    assert isinstance(adapter, _DefaultAdapter)


def test_unresolvable_pipeline_falls_back_to_matchers(fake_registry, monkeypatch):
    # Results whose pipeline is unknown (renamed or external) are routed via
    # shape matchers; field-grounding granular units depend on this path.
    monkeypatch.setattr(registry, "get_pipeline", lambda name: (_ for _ in ()).throw(KeyError(name)))
    adapter = create_layout_adapter_for_result(_make_result())
    assert isinstance(adapter, _MatchingAdapterHighPriority)


def test_highest_priority_wins_among_matchers(fake_registry, monkeypatch):
    monkeypatch.setattr(registry, "get_pipeline", lambda name: (_ for _ in ()).throw(KeyError(name)))
    adapter = create_layout_adapter_for_result(_make_result())
    assert isinstance(adapter, _MatchingAdapterHighPriority)


def test_no_key_and_no_matcher_returns_default(fake_registry, monkeypatch):
    monkeypatch.setattr(
        registry,
        "_LAYOUT_ADAPTER_REGISTRY",
        [
            _registration(_NonMatchingAdapter, "prov_keyed"),
            _registration(_DefaultAdapter, "__default__", priority=-100),
        ],
    )
    monkeypatch.setattr(registry, "get_pipeline", lambda name: (_ for _ in ()).throw(KeyError(name)))
    adapter = create_layout_adapter_for_result(_make_result())
    assert isinstance(adapter, _DefaultAdapter)


@requires_adapters
def test_unregistered_provider_with_shared_ir_gets_default_adapter(monkeypatch):
    # Real-registry regression test: a resolvable parse provider without a
    # provider key must land on the default adapter even though its ParseOutput
    # carries the shared layout IR that LlamaParse's matcher would claim.
    from extract_bench.evaluation.layout_adapters.adapters import NormalizedLayoutOutputAdapter

    monkeypatch.setattr(registry, "get_pipeline", lambda name: SimpleNamespace(provider_name="prov_unregistered"))
    adapter = create_layout_adapter_for_result(_make_result(pipeline_name="paddleocr_default", layout_pages=True))
    assert isinstance(adapter, NormalizedLayoutOutputAdapter)


@requires_adapters
def test_unresolvable_pipeline_with_shared_ir_matches_llamaparse(monkeypatch):
    # Real-registry contract test: unknown pipelines with the shared layout IR
    # are consumed by the LlamaParse adapter via the matcher fallback.
    from extract_bench.evaluation.layout_adapters.adapters import LlamaParseLayoutAdapter

    monkeypatch.setattr(registry, "get_pipeline", lambda name: (_ for _ in ()).throw(KeyError(name)))
    adapter = create_layout_adapter_for_result(_make_result(pipeline_name="unknown_pipeline", layout_pages=True))
    assert isinstance(adapter, LlamaParseLayoutAdapter)


@requires_adapters
def test_list_layout_adapters_loads_lazily():
    # Reading the registry in a fresh interpreter must populate it without any
    # prior import of the adapters module.
    code = (
        "import sys\n"
        "from extract_bench.evaluation.layout_adapters.registry import list_layout_adapters\n"
        "assert 'extract_bench.evaluation.layout_adapters.adapters' not in sys.modules\n"
        "keys = list_layout_adapters()\n"
        "assert 'llamaparse' in keys\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@requires_adapters
def test_duplicate_key_raises_before_first_registry_read():
    # The duplicate-key guard must hold even when an external registration
    # happens before anything has read the registry.
    code = (
        "from extract_bench.evaluation.layout_adapters.base import LayoutAdapter\n"
        "from extract_bench.evaluation.layout_adapters.registry import register_layout_adapter\n"
        "try:\n"
        "    @register_layout_adapter('llamaparse')\n"
        "    class Dup(LayoutAdapter):\n"
        "        def to_layout_output(self, inference_result, *, page_filter=None):\n"
        "            raise NotImplementedError\n"
        "except ValueError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('duplicate registration did not raise')\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
