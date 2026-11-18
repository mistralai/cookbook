"""OTLP export of GenAI evaluation results.

An evaluation is an OTLP log record with the standard ``gen_ai.evaluation.result`` event name and
``gen_ai.evaluation.*`` attributes. Configuration is passed explicitly so this shared utility does
not depend on application settings.
"""

from collections.abc import Mapping

from opentelemetry._logs import get_logger, set_logger_provider
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, set_span_in_context

EVALUATION_EVENT_NAME = "gen_ai.evaluation.result"

_INSTRUMENTATION_SCOPE = "app.telemetry"
_MILLISECONDS = 1000

_provider: LoggerProvider | None = None


def configure_telemetry(
    *,
    endpoint: str,
    api_key: str | None,
    service_name: str,
    export_timeout_seconds: float = 5.0,
) -> bool:
    """Install the exporter, returning whether telemetry is now live.

    Answers ``False`` rather than raising when there is no API key: a deployment without one
    is un-instrumented, not broken, and every emitter below is inert in that state.
    """
    global _provider
    if _provider is not None:
        return True
    if not api_key:
        return False

    _provider = LoggerProvider(resource=Resource.create({"service.name": service_name}))
    _provider.add_log_record_processor(
        BatchLogRecordProcessor(
            OTLPLogExporter(endpoint=endpoint, headers={"Authorization": f"Bearer {api_key}"}),
            export_timeout_millis=int(export_timeout_seconds * _MILLISECONDS),
        )
    )
    set_logger_provider(_provider)
    return True


def shutdown_telemetry() -> None:
    """Flush what is queued, bounded by the configured export timeout."""
    global _provider
    if _provider is None:
        return
    _provider.shutdown()
    _provider = None


def _span_context(trace_id: str | None, span_id: str | None) -> Context | None:
    """The evaluated span, as a context the log record can be attached to.

    Both ids or neither: a trace id without a span id addresses nothing.
    """
    if trace_id is None or span_id is None:
        return None
    span = NonRecordingSpan(
        SpanContext(
            trace_id=int(trace_id, 16),
            span_id=int(span_id, 16),
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
    )
    return set_span_in_context(span)


def record_evaluation_result(
    *,
    name: str,
    score_value: float,
    score_label: str,
    attributes: Mapping[str, str] | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
) -> None:
    """Emit one evaluation result.

    Safe to call unconfigured: the OTel API returns a no-op logger until a provider is
    installed. A caller can emit unconditionally instead of branching on whether this
    deployment exports.
    """
    get_logger(_INSTRUMENTATION_SCOPE).emit(
        event_name=EVALUATION_EVENT_NAME,
        context=_span_context(trace_id, span_id),
        attributes={
            "gen_ai.evaluation.name": name,
            "gen_ai.evaluation.score.value": score_value,
            "gen_ai.evaluation.score.label": score_label,
            **(attributes or {}),
        },
    )
