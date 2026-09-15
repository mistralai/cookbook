"""Document pipeline via the agent-framework Workflow engine (Option B).

    {data, name} ──▶ OcrExecutor ──▶ ExtractorExecutor ──▶ DocumentAnalysis
                     (OCR 4 +          (Medium 3.5:
                      annotation:       fields + entities)
                      classify+summary)

OCR is a first-class workflow node (OcrExecutor), so the agent-framework graph
owns the pipeline end-to-end; the Azure Function just feeds it the blob.
"""
from __future__ import annotations

from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler
from typing_extensions import Never

from agents import DocumentAnalysis, build_chat_client, build_extractor, parse_json_object
from foundry_agents import EXTRACTOR_AGENT, agents_enabled, build_agent
from observability import get_tracer, record_ocr
from ocr_client import ocr_with_annotation


class OcrExecutor(Executor):
    """Start node: OCR 4 + document annotation (classification + summary)."""

    @handler
    async def handle(self, doc: dict, ctx: WorkflowContext[dict]) -> None:
        with get_tracer().start_as_current_span("ocr.process") as span:
            span.set_attribute("document.name", doc.get("name", ""))
            result = ocr_with_annotation(doc["data"], doc["name"])
            annotation = result.get("annotation") or {}
            doc_type = annotation.get("document_type", "unknown")
            span.set_attribute("document.type", doc_type)
            span.set_attribute("ocr.pages", result.get("pages_count", 0))
            record_ocr(result.get("pages_count", 0), result.get("usage", {}), doc_type)
            await ctx.send_message(
                {
                    "markdown": result.get("markdown", ""),
                    "doc_type": doc_type,
                    "language": annotation.get("language", "unknown"),
                    "summary": annotation.get("summary", ""),
                }
            )


class ExtractorExecutor(Executor):
    """Reasoning node: Medium 3.5 pulls deep fields + entities, then merges."""

    def __init__(self, agent, id: str) -> None:
        super().__init__(id=id)
        self._agent = agent

    @handler
    async def handle(self, ocr: dict, ctx: WorkflowContext[Never, dict]) -> None:
        with get_tracer().start_as_current_span("extract.run"):
            response = await self._agent.run(ocr["markdown"])
        extracted = parse_json_object(response.text)
        analysis = DocumentAnalysis.model_validate(
            {
                "doc_type": ocr.get("doc_type", "unknown"),
                "language": ocr.get("language", "unknown"),
                "summary": ocr.get("summary", ""),
                "fields": extracted.get("fields", {}),
                "entities": extracted.get("entities", []),
            }
        )
        await ctx.yield_output(analysis.model_dump())


def build_workflow():
    ocr = OcrExecutor(id="ocr")
    # Hosted by default: when USE_FOUNDRY_AGENTS is on and a project endpoint is set, the
    # extractor runs server-side through Foundry Agent Service (per-agent portal traces).
    # Otherwise it runs inline via /chat/completions. Both objects expose .run(text) -> .text,
    # so ExtractorExecutor consumes either unchanged. The hosted agent already falls back to
    # /chat/completions internally if a server-side run errors.
    if agents_enabled():
        agent = build_agent(EXTRACTOR_AGENT)
    else:
        agent = build_extractor(build_chat_client())
    extractor = ExtractorExecutor(agent, id="extractor")
    return (
        WorkflowBuilder(start_executor=ocr)
        .add_edge(ocr, extractor)
        .build()
    )


async def run_workflow_doc(data: bytes, name: str) -> DocumentAnalysis:
    workflow = build_workflow()
    output: dict | None = None
    async for event in workflow.run({"data": data, "name": name}, stream=True):
        if getattr(event, "type", None) == "output":
            output = event.data
    if output is None:
        raise RuntimeError("Workflow produced no output.")
    return DocumentAnalysis.model_validate(output)
