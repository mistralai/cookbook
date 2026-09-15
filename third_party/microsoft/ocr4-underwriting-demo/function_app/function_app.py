"""Azure Function (Python v2) — queue-triggered document pipeline.

Flow: upload to `inbox` -> Event Grid (BlobCreated) -> `ingest` Storage Queue
(the SQS-equivalent buffer) -> this function -> OCR 4 (+ annotation) + Extractor
agent -> `<name>.result.json` in the `results` container.

The queue message is an Event Grid event (EventGridSchema), which Event Grid
writes base64-encoded — we decode defensively.

Local run:
    cp local.settings.json.example local.settings.json   # fill in values
    func start
"""
import asyncio
import base64
import json
import logging
import sys
from pathlib import Path

import azure.functions as func

# Reuse the pipeline package in ../src. For cloud deployment, vendor `src`
# into this folder (or package it) so it ships with the function.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import load_settings  # noqa: E402
from observability import flush, setup_observability  # noqa: E402
from pipeline import run_pipeline_bytes  # noqa: E402

setup_observability()

app = func.FunctionApp()


def _parse_event(raw: bytes) -> dict:
    """Decode a queue message that carries an Event Grid event (maybe base64)."""
    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
    try:
        event = json.loads(text)
    except json.JSONDecodeError:
        event = json.loads(base64.b64decode(text))
    if isinstance(event, list):  # Event Grid may batch events
        event = event[0]
    return event


def _blob_name(event: dict) -> str:
    # subject: /blobServices/default/containers/inbox/blobs/<name>
    subject = event.get("subject", "")
    if "/blobs/" in subject:
        return subject.split("/blobs/", 1)[1]
    return event.get("data", {}).get("url", "").split("/")[-1]


@app.queue_trigger(arg_name="msg", queue_name="ingest", connection="AzureWebJobsStorage")
def process_document(msg: func.QueueMessage) -> None:
    event = _parse_event(msg.get_body())
    name = _blob_name(event)
    if not name:
        logging.warning("Could not resolve blob name from event: %s", event)
        return

    logging.info("Processing blob: %s", name)
    try:
        settings = load_settings()

        from azure.storage.blob import BlobServiceClient

        client = BlobServiceClient.from_connection_string(settings.storage_connection_string)
        blob = client.get_blob_client(settings.inbox_container, name)

        # Pull trace context + doc id stamped at upload (if any) from blob metadata,
        # and the blob-created time from the Event Grid event, for end-to-end traceability.
        metadata = blob.get_blob_properties().metadata or {}
        correlation = {
            "doc_id": metadata.get("doc_id") or name,
            "traceparent": metadata.get("traceparent"),
            "uploaded_at": metadata.get("uploaded_at"),
            "event_time": event.get("eventTime"),
            "etag": (event.get("data") or {}).get("eTag"),
        }

        data = blob.download_blob().readall()
        analysis = asyncio.run(run_pipeline_bytes(data, name, correlation=correlation))

        out_name = f"{Path(name).stem}.result.json"
        body = json.dumps(analysis.model_dump(), indent=2, ensure_ascii=False)
        client.get_blob_client(settings.results_container, out_name).upload_blob(
            body,
            overwrite=True,
            metadata={"doc_id": correlation["doc_id"], "source_blob": name},
        )
        logging.info("Wrote result to %s/%s (doc_id=%s)", settings.results_container, out_name, correlation["doc_id"])
    finally:
        # Export buffered OpenTelemetry spans before the Flex Consumption instance can freeze.
        # A short-lived invocation otherwise loses spans the batch processor has not flushed yet,
        # which is why the OCR span never reached App Insights from the function.
        flush()
