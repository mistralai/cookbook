#!/usr/bin/env python3
"""Traced uploader — drop a document into the `inbox` container with trace context.

Stamps the blob's metadata with a document id, a W3C `traceparent`, and the
upload time, all under an `upload.document` span. The Function reads this metadata
and continues the SAME trace, so you get one connected trace from upload all the
way to the written result.

Usage:
    uv run src/upload.py mydoc.pdf
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from azure.storage.blob import BlobServiceClient

from config import load_settings
from observability import get_tracer, inject_trace_context, setup_observability


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python upload.py <file.pdf|image>")
    src = Path(sys.argv[1])
    if not src.exists():
        sys.exit(f"No such file: {src}")

    setup_observability()
    settings = load_settings()
    doc_id = uuid.uuid4().hex
    uploaded_at = datetime.now(timezone.utc).isoformat()

    with get_tracer().start_as_current_span("upload.document") as span:
        span.set_attribute("document.id", doc_id)
        span.set_attribute("document.name", src.name)
        # Capture the current trace context so the pipeline can continue it.
        carrier = inject_trace_context()
        metadata = {"doc_id": doc_id, "uploaded_at": uploaded_at, **carrier}

        client = BlobServiceClient.from_connection_string(settings.storage_connection_string)
        client.get_blob_client(settings.inbox_container, src.name).upload_blob(
            src.read_bytes(), overwrite=True, metadata=metadata
        )

    print(f"Uploaded {src.name} to {settings.inbox_container}/")
    print(f"  doc_id      = {doc_id}")
    print(f"  traceparent = {carrier.get('traceparent')}")


if __name__ == "__main__":
    main()
