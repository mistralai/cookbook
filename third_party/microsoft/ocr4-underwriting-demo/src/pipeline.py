#!/usr/bin/env python3
"""End-to-end document processing pipeline.

    file/bytes ──▶ Mistral OCR 4 (deterministic) ──▶ markdown
                ──▶ Orchestrator agent (Medium 3.5) ──▶ classify + extract + summarize
                ──▶ consolidated DocumentAnalysis (JSON)

Usage:
    python pipeline.py mydoc.pdf                 # prints JSON, writes mydoc.result.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from datetime import datetime, timezone

from agents import DocumentAnalysis
from observability import (
    extract_trace_context,
    get_tracer,
    record_since_upload,
    setup_observability,
)
from workflow import run_workflow_doc


def _since_upload_ms(correlation: dict) -> float | None:
    """Milliseconds from upload/blob-created to now, if a timestamp is available."""
    stamp = correlation.get("uploaded_at") or correlation.get("event_time")
    if not stamp:
        return None
    try:
        ts = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - ts).total_seconds() * 1000.0
    except (ValueError, AttributeError):
        return None


async def run_pipeline_bytes(data: bytes, name: str, *,
                             correlation: dict | None = None) -> DocumentAnalysis:
    """OCR 4 (+ annotation) → Extractor agent → DocumentAnalysis.

    `correlation` carries traceability from upload: doc_id, traceparent (to
    continue the upload's trace), uploaded_at / event_time, etag.
    """
    setup_observability()
    correlation = correlation or {}
    parent_ctx = extract_trace_context({"traceparent": correlation.get("traceparent")})

    with get_tracer().start_as_current_span("pipeline.run", context=parent_ctx) as span:
        span.set_attribute("document.name", name)
        span.set_attribute("document.bytes", len(data))
        for key in ("doc_id", "etag", "event_time", "uploaded_at"):
            if correlation.get(key):
                span.set_attribute(f"document.{key}", str(correlation[key]))

        since_ms = _since_upload_ms(correlation)
        if since_ms is not None:
            span.set_attribute("document.since_upload_ms", round(since_ms))

        analysis = await run_workflow_doc(data, name)
        span.set_attribute("document.type", analysis.doc_type)
        if since_ms is not None:
            record_since_upload(since_ms, analysis.doc_type)
        return analysis


async def run_pipeline(path: str | Path) -> DocumentAnalysis:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")
    return await run_pipeline_bytes(p.read_bytes(), p.name)


async def _main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end document processing pipeline (OCR 4 + Medium 3.5).")
    parser.add_argument("file", help="PDF or image to process")
    args = parser.parse_args()

    analysis = await run_pipeline(args.file)
    payload = analysis.model_dump()
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    out = Path(args.file).with_suffix(".result.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    asyncio.run(_main())
