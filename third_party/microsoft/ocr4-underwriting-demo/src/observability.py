"""OpenTelemetry / Azure Monitor wiring for the pipeline.

Agent Framework auto-instruments agents and workflow executors (spans + GenAI
token metrics); we only configure an exporter and add pipeline-level spans and a
few custom metrics (documents, OCR pages, OCR tokens).

Exporter selection (via env):
- APPLICATIONINSIGHTS_CONNECTION_STRING set -> Azure Monitor exporter.
- else ENABLE_CONSOLE_EXPORTERS=true      -> console exporter (local dev).
Observability must never break the pipeline, so setup failures are swallowed.
"""
from __future__ import annotations

import atexit
import logging
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

_INITIALIZED = False


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def setup_observability() -> None:
    """Idempotent: configure the OTel exporter once per process."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    cs = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
    # In telemetryMode=OpenTelemetry the Azure Functions host installs and owns the OTel tracer
    # provider and exports worker spans under each invocation. Calling configure_azure_monitor in
    # that process installs a SECOND provider the host never drains, so the pipeline's manual
    # spans (ocr.process / docchat.ocr) and the OCR-4 POST attach to the orphan provider and are
    # dropped — while the host-owned request/log telemetry still exports. When the host signals
    # this mode, defer to its provider and only add HTTP-client instrumentation, so the OCR span
    # lands under the invocation. Locally this env is unset, so the connection-string path runs.
    host_owns_telemetry = _truthy("PYTHON_APPLICATIONINSIGHTS_ENABLE_TELEMETRY")
    try:
        if host_owns_telemetry:
            if _truthy("ENABLE_SENSITIVE_TELEMETRY"):
                from agent_framework.observability import enable_sensitive_telemetry

                enable_sensitive_telemetry()
        elif cs:
            from agent_framework.observability import create_resource
            from azure.monitor.opentelemetry import configure_azure_monitor

            configure_azure_monitor(connection_string=cs, resource=create_resource())
            if _truthy("ENABLE_SENSITIVE_TELEMETRY"):
                from agent_framework.observability import enable_sensitive_telemetry

                enable_sensitive_telemetry()
        elif _truthy("ENABLE_CONSOLE_EXPORTERS"):
            from agent_framework.observability import configure_otel_providers

            configure_otel_providers(enable_console_exporters=True)
        # Instrument the outbound HTTP client so the OCR call (a requests.post to the OCR
        # route) and any other HTTP dependency emit a client span in the same trace. Without
        # this the OCR read leaves no span, only the manual ocr.process/docchat.ocr wrappers.
        if host_owns_telemetry or cs or _truthy("ENABLE_CONSOLE_EXPORTERS"):
            try:
                from opentelemetry.instrumentation.requests import RequestsInstrumentor

                if not RequestsInstrumentor().is_instrumented_by_opentelemetry:
                    RequestsInstrumentor().instrument()
            except Exception as exc:  # instrumentation is best-effort
                logging.debug("requests instrumentation skipped: %s", exc)
    except Exception as exc:  # never let telemetry setup break the pipeline
        logging.warning("Observability setup skipped: %s", exc)
    # Flush on interpreter exit so a short-lived run (or Ctrl+C) does not drop its spans;
    # the long-running app also flushes on the batch processor's interval.
    atexit.register(flush)


def flush() -> None:
    """Force-export buffered spans and metrics to the configured exporter."""
    try:
        from opentelemetry import metrics, trace

        for provider in (trace.get_tracer_provider(), metrics.get_meter_provider()):
            force = getattr(provider, "force_flush", None)
            if callable(force):
                force()
    except Exception:  # flushing must never raise
        pass


# ---- Tracer + lazily-created custom metrics ----
def get_tracer():
    from opentelemetry import trace

    return trace.get_tracer("doc-pipeline")


# ---- W3C trace-context propagation (carried in blob metadata) ----
def inject_trace_context() -> dict:
    """Serialize the current span context into a carrier dict (traceparent)."""
    from opentelemetry.propagate import inject

    carrier: dict = {}
    inject(carrier)
    return carrier


def extract_trace_context(carrier: dict | None):
    """Rebuild a parent Context from a carrier dict, or None if absent/empty."""
    if not carrier or not carrier.get("traceparent"):
        return None
    from opentelemetry.propagate import extract

    return extract(carrier)


_INSTRUMENTS: dict = {}


def _instruments():
    if not _INSTRUMENTS:
        from opentelemetry import metrics

        meter = metrics.get_meter("doc-pipeline")
        _INSTRUMENTS["docs"] = meter.create_counter(
            "pipeline.documents", description="Documents processed"
        )
        _INSTRUMENTS["pages"] = meter.create_histogram(
            "pipeline.ocr.pages", description="OCR pages per document"
        )
        _INSTRUMENTS["tokens"] = meter.create_counter(
            "pipeline.ocr.tokens", unit="token", description="OCR tokens used"
        )
        _INSTRUMENTS["since_upload"] = meter.create_histogram(
            "pipeline.since_upload_ms", unit="ms",
            description="Latency from blob upload to pipeline start",
        )
    return _INSTRUMENTS


def record_ocr(pages: int, usage: dict, doc_type: str = "unknown") -> None:
    """Emit document / page / token metrics for one OCR call."""
    try:
        inst = _instruments()
        inst["docs"].add(1, {"doc_type": doc_type})
        if pages:
            inst["pages"].record(pages, {"doc_type": doc_type})
        tokens = 0
        if isinstance(usage, dict):
            tokens = usage.get("total_tokens") or usage.get("prompt_tokens") or 0
        if tokens:
            inst["tokens"].add(int(tokens), {"doc_type": doc_type})
    except Exception as exc:
        logging.debug("record_ocr skipped: %s", exc)


def record_since_upload(ms: float, doc_type: str = "unknown") -> None:
    """Emit the upload -> pipeline-start latency."""
    try:
        if ms and ms >= 0:
            _instruments()["since_upload"].record(ms, {"doc_type": doc_type})
    except Exception as exc:
        logging.debug("record_since_upload skipped: %s", exc)
