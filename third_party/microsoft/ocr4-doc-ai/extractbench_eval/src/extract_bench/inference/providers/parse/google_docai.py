"""Provider for Google Document AI PARSE."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai
from pypdf import PdfReader

from extract_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from extract_bench.inference.providers.parse.google_docai_layout_normalization import normalize_layout_document
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.parse_output import LayoutItemIR, LayoutSegmentIR, PageIR, ParseLayoutPageIR, ParseOutput
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from extract_bench.schemas.product import ProductType

try:
    from google.cloud import documentai_v1beta3 as documentai_v1beta3
except ImportError:  # pragma: no cover - dependency guarded by runtime validation
    documentai_v1beta3 = None  # type: ignore[assignment]


_REQUIRED_LAYOUT_CONFIG_FIELDS = {
    "return_bounding_boxes",
    "return_images",
    "enable_image_annotation",
    "enable_table_annotation",
}

_VIRTUAL_PAGE_DIM = 1000.0


@register_provider("google_docai")
class GoogleDocAIProvider(Provider):
    """
    Provider for Google Document AI PARSE.

    OCR mode uses `documentai_v1`.
    Layout Parser mode uses the first SDK surface that exposes the full layout
    config contract, preferring `documentai_v1beta3` on current installs.
    """

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        self._project_id = self.base_config.get("project_id") or os.getenv("GOOGLE_DOCAI_PROJECT_ID")
        if not self._project_id:
            raise ProviderConfigError(
                "Google Cloud project ID is required. "
                "Set GOOGLE_DOCAI_PROJECT_ID environment variable or pass project_id in base_config."
            )

        self._location = self.base_config.get("location") or os.getenv("GOOGLE_DOCAI_LOCATION", "us")

        self._processor_id = self.base_config.get("processor_id") or os.getenv("GOOGLE_DOCAI_PROCESSOR_ID")
        if not self._processor_id:
            raise ProviderConfigError(
                "Google Document AI processor ID is required. "
                "Set GOOGLE_DOCAI_PROCESSOR_ID environment variable or pass processor_id in base_config."
            )

        self._processor_version = self.base_config.get("processor_version") or os.getenv(
            "GOOGLE_DOCAI_PROCESSOR_VERSION"
        )

        self._enable_native_pdf_parsing = self.base_config.get("enable_native_pdf_parsing", True)
        self._enable_symbol_detection = self.base_config.get("enable_symbol_detection", False)

        self._use_layout_parser = self.base_config.get("use_layout_parser", False)
        self._layout_processor_id = (
            self.base_config.get("layout_processor_id")
            or os.getenv("GOOGLE_DOCAI_LAYOUT_PROCESSOR_ID")
            or self._processor_id
        )
        self._chunking_config = self.base_config.get("chunking_config")

        self._layout_api_surface_label: str | None = None
        self._layout_documentai: Any | None = None
        self._layout_config_fields: set[str] = set()
        if self._use_layout_parser:
            self._layout_api_surface_label, self._layout_documentai = self._resolve_layout_api_surface()
            self._layout_config_fields = set(
                self._layout_documentai.ProcessOptions.LayoutConfig()._pb.DESCRIPTOR.fields_by_name
            )

    def _resolve_layout_api_surface(self) -> tuple[str, Any]:
        candidates: list[tuple[str, Any]] = []
        if documentai_v1beta3 is not None:
            candidates.append(("v1beta3", documentai_v1beta3))
        candidates.append(("v1", documentai))

        for surface_label, module in candidates:
            layout_fields = set(module.ProcessOptions.LayoutConfig()._pb.DESCRIPTOR.fields_by_name)
            if _REQUIRED_LAYOUT_CONFIG_FIELDS.issubset(layout_fields):
                return surface_label, module

        raise ProviderConfigError(
            "Google DocAI layout mode requires a Document AI SDK surface exposing "
            f"{sorted(_REQUIRED_LAYOUT_CONFIG_FIELDS)}. "
            "Current install does not provide a compatible layout API surface."
        )

    def _is_pdf_file(self, file_path: str) -> bool:
        try:
            with open(file_path, "rb") as file_handle:
                return file_handle.read(4) == b"%PDF"
        except Exception:
            return False

    def _get_page_count(self, file_path: str) -> int:
        if self._is_pdf_file(file_path):
            try:
                reader = PdfReader(file_path)
                return len(reader.pages)
            except Exception:
                return 1
        return 1

    def _get_mime_type(self, file_path: str) -> str:
        suffix = Path(file_path).suffix.lower()
        return {
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
        }.get(suffix, "application/pdf")

    def _is_image_file(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".tiff", ".tif", ".bmp", ".webp"}

    def _convert_image_to_pdf(self, file_path: str) -> bytes:
        try:
            import io

            from PIL import Image
        except ImportError as exc:
            raise ProviderConfigError("Pillow library not installed. Run: pip install Pillow") from exc

        try:
            with Image.open(file_path) as image:
                if image.mode in ("RGBA", "LA", "P"):
                    background = Image.new("RGB", image.size, (255, 255, 255))
                    if image.mode == "P":
                        image = image.convert("RGBA")
                    background.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
                    image = background
                elif image.mode != "RGB":
                    image = image.convert("RGB")

                pdf_buffer = io.BytesIO()
                image.save(pdf_buffer, format="PDF", resolution=100.0)
                pdf_buffer.seek(0)
                return pdf_buffer.read()
        except Exception as exc:  # pragma: no cover - filesystem/PIL failure
            raise ProviderPermanentError(f"Failed to convert image to PDF: {exc}") from exc

    def _build_layout_config(self, layout_module: Any) -> Any:
        chunking_config = None
        if self._chunking_config:
            chunking_kwargs: dict[str, Any] = {}
            if "chunk_size" in self._chunking_config:
                chunking_kwargs["chunk_size"] = self._chunking_config["chunk_size"]
            if "include_ancestor_headings" in self._chunking_config:
                chunking_kwargs["include_ancestor_headings"] = self._chunking_config["include_ancestor_headings"]
            if chunking_kwargs:
                chunking_config = layout_module.ProcessOptions.LayoutConfig.ChunkingConfig(**chunking_kwargs)

        # Visual grounding in bench depends on native layout bounding boxes, so this
        # provider is intentionally optimized for the stable Layout Parser surfaces
        # that still expose bbox geometry. Newer parser versions can improve table
        # understanding, but some do not expose layout bboxes and therefore cannot
        # support the visual-grounding column honestly.
        #
        # Keep LLM image annotations enabled because they materially improve picture
        # detection. Keep LLM table annotations disabled because the native
        # `tableBlock` structure is already present on the stable bbox-capable path,
        # and the extra table annotations did not improve merged-cell fidelity in our
        # verification runs.
        kwargs: dict[str, Any] = {
            "chunking_config": chunking_config,
            "return_bounding_boxes": True,
            "return_images": True,
            "enable_image_annotation": True,
            "enable_table_annotation": False,
        }
        if "enable_image_extraction" in self._layout_config_fields:
            kwargs["enable_image_extraction"] = True

        return layout_module.ProcessOptions.LayoutConfig(**kwargs)

    def _build_ocr_response(self, document_obj: Any) -> dict[str, Any]:
        raw_response = {
            "text": document_obj.text,
            "mime_type": document_obj.mime_type,
            "pages": [],
            "entities": [],
            "tables": [],
            "mode": "ocr",
        }

        for page in document_obj.pages:
            page_data = {
                "page_number": page.page_number,
                "width": page.dimension.width if page.dimension else None,
                "height": page.dimension.height if page.dimension else None,
                "blocks": [],
                "paragraphs": [],
                "lines": [],
                "tokens": [],
                "tables": [],
            }

            for block in page.blocks:
                block_text = self._get_text_from_layout(block.layout, document_obj.text)
                page_data["blocks"].append(
                    {
                        "text": block_text,
                        "confidence": block.layout.confidence if block.layout else None,
                    }
                )

            for para in page.paragraphs:
                para_text = self._get_text_from_layout(para.layout, document_obj.text)
                para_entry: dict[str, Any] = {
                    "text": para_text,
                    "confidence": para.layout.confidence if para.layout else None,
                }
                if para.layout and para.layout.bounding_poly and para.layout.bounding_poly.vertices:
                    vertices = para.layout.bounding_poly.vertices
                    para_entry["y_position"] = min(v.y for v in vertices if v.y is not None) if vertices else None
                if para.layout and para.layout.bounding_poly:
                    normalized_vertices = para.layout.bounding_poly.normalized_vertices
                    if normalized_vertices and len(normalized_vertices) >= 4:
                        para_entry["normalized_bbox"] = {
                            "x1": normalized_vertices[0].x or 0.0,
                            "y1": normalized_vertices[0].y or 0.0,
                            "x2": normalized_vertices[2].x or 0.0,
                            "y2": normalized_vertices[2].y or 0.0,
                        }
                page_data["paragraphs"].append(para_entry)

            for line in page.lines:
                line_text = self._get_text_from_layout(line.layout, document_obj.text)
                page_data["lines"].append(
                    {
                        "text": line_text,
                        "confidence": line.layout.confidence if line.layout else None,
                    }
                )

            for table in page.tables:
                table_data = self._extract_table(table, document_obj.text)
                if table.layout and table.layout.bounding_poly and table.layout.bounding_poly.vertices:
                    vertices = table.layout.bounding_poly.vertices
                    table_data["y_position"] = min(v.y for v in vertices if v.y is not None) if vertices else None
                if table.layout and table.layout.bounding_poly:
                    normalized_vertices = table.layout.bounding_poly.normalized_vertices
                    if normalized_vertices and len(normalized_vertices) >= 4:
                        table_data["normalized_bbox"] = {
                            "x1": normalized_vertices[0].x or 0.0,
                            "y1": normalized_vertices[0].y or 0.0,
                            "x2": normalized_vertices[2].x or 0.0,
                            "y2": normalized_vertices[2].y or 0.0,
                        }
                page_data["tables"].append(table_data)

            raw_response["pages"].append(page_data)

        for entity in document_obj.entities:
            raw_response["entities"].append(
                {
                    "type": entity.type_,
                    "mention_text": entity.mention_text,
                    "confidence": entity.confidence,
                }
            )

        raw_response["_config"] = {
            "project_id": self._project_id,
            "location": self._location,
            "processor_id": self._processor_id,
            "processor_version": self._processor_version,
            "enable_native_pdf_parsing": self._enable_native_pdf_parsing,
            "enable_symbol_detection": self._enable_symbol_detection,
            "total_pages": len(document_obj.pages),
        }
        return raw_response

    def _serialize_api_document(self, document_obj: Any) -> dict[str, Any]:
        try:
            from google.protobuf.json_format import MessageToDict  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - protobuf always available with SDK
            raise ProviderConfigError("google.protobuf is required to serialize Document AI payloads.") from exc
        return cast(dict[str, Any], MessageToDict(document_obj._pb))

    def _materialize_internal_raw_output(
        self,
        raw_payload: dict[str, Any],
        *,
        use_layout_parser: bool,
    ) -> dict[str, Any]:
        if "mode" in raw_payload and ("pages" in raw_payload or "blocks" in raw_payload):
            return raw_payload

        try:
            from google.protobuf.json_format import ParseDict
        except ImportError as exc:  # pragma: no cover
            raise ProviderConfigError("Google protobuf support is required to parse Document AI payloads.") from exc

        if use_layout_parser:
            raise ProviderPermanentError(
                "Legacy Google DocAI layout raw outputs are no longer normalized through provider-shaped blocks. "
                "Re-run inference to regenerate raw outputs from the untouched DocAI payload."
            )

        document_pb = ParseDict(raw_payload, documentai.Document()._pb)
        document_obj = documentai.Document(document_pb)
        return self._build_ocr_response(document_obj)

    def _materialize_layout_document(self, raw_payload: dict[str, Any]) -> Any:
        if self._layout_documentai is None:
            raise ProviderConfigError("Layout Parser requested without an initialized layout API surface.")
        try:
            from google.protobuf.json_format import ParseDict
        except ImportError as exc:  # pragma: no cover
            raise ProviderConfigError("Google protobuf support is required to parse Document AI payloads.") from exc

        document_pb = ParseDict(raw_payload, self._layout_documentai.Document()._pb)
        return self._layout_documentai.Document(document_pb)

    def _parse_document(self, file_path: str) -> dict[str, Any]:
        try:
            docai_module = self._layout_documentai if self._use_layout_parser else documentai
            if docai_module is None:
                raise ProviderConfigError("Layout Parser requested without a compatible Document AI SDK surface.")

            opts = ClientOptions(api_endpoint=f"{self._location}-documentai.googleapis.com")
            client = docai_module.DocumentProcessorServiceClient(client_options=opts)

            processor_id = str(self._layout_processor_id if self._use_layout_parser else self._processor_id)
            processor_name = self._build_processor_name(processor_id)

            with open(file_path, "rb") as file_handle:
                file_content = file_handle.read()

            mime_type = self._get_mime_type(file_path)
            if self._use_layout_parser and self._is_image_file(file_path):
                file_content = self._convert_image_to_pdf(file_path)
                mime_type = "application/pdf"

            raw_document = docai_module.RawDocument(content=file_content, mime_type=mime_type)

            if self._use_layout_parser:
                process_options = docai_module.ProcessOptions(layout_config=self._build_layout_config(docai_module))
            else:
                process_options = docai_module.ProcessOptions(
                    ocr_config=docai_module.OcrConfig(
                        enable_native_pdf_parsing=self._enable_native_pdf_parsing,
                        enable_symbol=self._enable_symbol_detection,
                    )
                )

            result = client.process_document(
                request=docai_module.ProcessRequest(
                    name=processor_name,
                    raw_document=raw_document,
                    process_options=process_options,
                )
            )
            return self._serialize_api_document(result.document)
        except Exception as exc:
            error_str = str(exc).lower()
            transient_keywords = ["timeout", "deadline", "unavailable", "503", "502", "504", "connection", "network"]
            if any(keyword in error_str for keyword in transient_keywords):
                raise ProviderTransientError(f"Transient error during Document AI processing: {exc}") from exc
            raise ProviderPermanentError(f"Error during Document AI processing: {exc}") from exc

    def _build_processor_name(self, processor_id: str) -> str:
        if self._processor_version:
            return (
                f"projects/{self._project_id}/locations/{self._location}/"
                f"processors/{processor_id}/processorVersions/{self._processor_version}"
            )
        return f"projects/{self._project_id}/locations/{self._location}/processors/{processor_id}"

    def _get_text_from_layout(self, layout: Any, full_text: str) -> str:
        if not layout or not layout.text_anchor or not layout.text_anchor.text_segments:
            return ""

        text_parts: list[str] = []
        for segment in layout.text_anchor.text_segments:
            start_index = int(segment.start_index) if segment.start_index else 0
            end_index = int(segment.end_index) if segment.end_index else 0
            text_parts.append(full_text[start_index:end_index])
        return "".join(text_parts)

    def _extract_table(self, table: Any, full_text: str) -> dict[str, Any]:
        table_data: dict[str, Any] = {
            "header_rows": [],
            "body_rows": [],
        }

        for row in table.header_rows:
            row_data = []
            for cell in row.cells:
                row_data.append(
                    {
                        "text": self._get_text_from_layout(cell.layout, full_text).strip(),
                        "row_span": cell.row_span,
                        "col_span": cell.col_span,
                    }
                )
            table_data["header_rows"].append(row_data)

        for row in table.body_rows:
            row_data = []
            for cell in row.cells:
                row_data.append(
                    {
                        "text": self._get_text_from_layout(cell.layout, full_text).strip(),
                        "row_span": cell.row_span,
                        "col_span": cell.col_span,
                    }
                )
            table_data["body_rows"].append(row_data)

        return table_data

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"GoogleDocAIProvider only supports PARSE product type, got {request.product_type}"
            )

        started_at = datetime.now()
        file_path = Path(request.source_file_path)
        if not file_path.exists():
            raise ProviderPermanentError(f"File not found: {file_path}")

        try:
            raw_output = self._parse_document(str(file_path))
            completed_at = datetime.now()
            latency_ms = int((completed_at - started_at).total_seconds() * 1000)
            return RawInferenceResult(
                request=request,
                pipeline=pipeline,
                pipeline_name=pipeline.pipeline_name,
                product_type=request.product_type,
                raw_output=raw_output,
                started_at=started_at,
                completed_at=completed_at,
                latency_in_ms=latency_ms,
            )
        except (ProviderPermanentError, ProviderTransientError):
            raise
        except Exception as exc:  # pragma: no cover
            raise ProviderPermanentError(f"Unexpected error during inference: {exc}") from exc

    def _table_to_html(self, table: dict[str, Any]) -> str:
        html_parts = ["<table>"]

        if table.get("header_rows"):
            html_parts.append("<thead>")
            for row in table["header_rows"]:
                html_parts.append("<tr>")
                for cell in row:
                    colspan = f' colspan="{cell["col_span"]}"' if cell.get("col_span", 1) > 1 else ""
                    rowspan = f' rowspan="{cell["row_span"]}"' if cell.get("row_span", 1) > 1 else ""
                    html_parts.append(f"<th{colspan}{rowspan}>{cell['text']}</th>")
                html_parts.append("</tr>")
            html_parts.append("</thead>")

        if table.get("body_rows"):
            html_parts.append("<tbody>")
            for row in table["body_rows"]:
                html_parts.append("<tr>")
                for cell in row:
                    colspan = f' colspan="{cell["col_span"]}"' if cell.get("col_span", 1) > 1 else ""
                    rowspan = f' rowspan="{cell["row_span"]}"' if cell.get("row_span", 1) > 1 else ""
                    html_parts.append(f"<td{colspan}{rowspan}>{cell['text']}</td>")
                html_parts.append("</tr>")
            html_parts.append("</tbody>")

        html_parts.append("</table>")
        return "\n".join(html_parts)

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"GoogleDocAIProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        try:
            pipeline_layout = raw_result.pipeline.config.get("use_layout_parser")
            use_layout_parser = pipeline_layout if pipeline_layout is not None else self._use_layout_parser
            if use_layout_parser:
                if isinstance(raw_result.raw_output, dict) and raw_result.raw_output.get("mode") == "layout_parser":
                    output = self._normalize_legacy_layout_output(raw_result.raw_output, raw_result)
                else:
                    layout_document = self._materialize_layout_document(raw_result.raw_output)
                    output = normalize_layout_document(document=layout_document, raw_result=raw_result)
            else:
                raw_output = self._materialize_internal_raw_output(raw_result.raw_output, use_layout_parser=False)
                output = self._normalize_ocr_output(raw_output, raw_result)

            return InferenceResult(
                request=raw_result.request,
                pipeline_name=raw_result.pipeline_name,
                product_type=raw_result.product_type,
                raw_output=raw_result.raw_output,
                output=output,
                started_at=raw_result.started_at,
                completed_at=raw_result.completed_at,
                latency_in_ms=raw_result.latency_in_ms,
            )
        except Exception as exc:
            raise ProviderPermanentError(f"Normalization failed: {exc}") from exc

    def _normalize_ocr_output(self, raw_output: dict[str, Any], raw_result: RawInferenceResult) -> ParseOutput:
        pages: list[PageIR] = []
        markdown_parts: list[str] = []

        for page_idx, page_data in enumerate(raw_output.get("pages", [])):
            elements: list[tuple[float, str]] = []

            for para in page_data.get("paragraphs", []):
                text = para.get("text", "").strip()
                if text:
                    elements.append((para.get("y_position", 0.0) or 0.0, text))

            for table in page_data.get("tables", []):
                elements.append((table.get("y_position", 0.0) or 0.0, self._table_to_html(table)))

            elements.sort(key=lambda element: element[0])
            page_markdown_parts = [element[1] for element in elements]
            page_markdown = "\n\n".join(page_markdown_parts)
            pages.append(PageIR(page_index=page_idx, markdown=page_markdown))
            markdown_parts.append(page_markdown)

        return ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=_build_layout_pages(raw_output),
            markdown="\n\n---\n\n".join(markdown_parts),
            job_id=None,
        )

    def _normalize_legacy_layout_output(
        self,
        raw_output: dict[str, Any],
        raw_result: RawInferenceResult,
    ) -> ParseOutput:
        blocks = raw_output.get("blocks", [])
        if not blocks:
            full_text = raw_output.get("text", "")
            return ParseOutput(
                task_type="parse",
                example_id=raw_result.request.example_id,
                pipeline_name=raw_result.pipeline_name,
                pages=[PageIR(page_index=0, markdown=full_text)],
                markdown=full_text,
                job_id=None,
            )

        page_content: dict[int, list[str]] = {}
        all_content: list[str] = []
        for block in blocks:
            markdown = _legacy_block_to_markdown(block)
            if not markdown:
                continue
            all_content.append(markdown)
            page_span = block.get("page_span")
            if page_span:
                page_start = page_span.get("page_start", 1) - 1
                page_end = page_span.get("page_end", page_start + 1) - 1
                for page_idx in range(page_start, page_end + 1):
                    page_content.setdefault(page_idx, []).append(markdown)
            else:
                page_content.setdefault(0, []).append(markdown)

        pages = [
            PageIR(page_index=page_idx, markdown="\n\n".join(page_content[page_idx]))
            for page_idx in sorted(page_content)
        ]
        layout_pages_payload = raw_output.get("layout_pages")
        if not layout_pages_payload:
            raise ProviderPermanentError(
                "Legacy layout raw output is missing layout_pages. Re-run inference with the native layout rewrite."
            )

        return ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=[ParseLayoutPageIR.model_validate(page_data) for page_data in layout_pages_payload],
            markdown="\n\n".join(all_content),
            job_id=None,
        )


def _build_layout_pages(raw_output: dict[str, Any]) -> list[ParseLayoutPageIR]:
    layout_pages: list[ParseLayoutPageIR] = []

    for page_idx, page_data in enumerate(raw_output.get("pages", [])):
        items: list[LayoutItemIR] = []

        for para in page_data.get("paragraphs", []):
            bbox_data = para.get("normalized_bbox")
            if not bbox_data:
                continue

            text = para.get("text", "").strip()
            if not text:
                continue

            x1 = float(bbox_data.get("x1", 0.0))
            y1 = float(bbox_data.get("y1", 0.0))
            x2 = float(bbox_data.get("x2", 0.0))
            y2 = float(bbox_data.get("y2", 0.0))
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue

            conf_raw = para.get("confidence")
            try:
                confidence = float(conf_raw) if conf_raw is not None else 1.0
            except (TypeError, ValueError):
                confidence = 1.0

            seg = LayoutSegmentIR(x=x1, y=y1, w=w, h=h, confidence=confidence, label="Text")
            items.append(LayoutItemIR(type="text", value=text, bbox=seg, layout_segments=[seg]))

        for table in page_data.get("tables", []):
            bbox_data = table.get("normalized_bbox")
            if not bbox_data:
                continue

            x1 = float(bbox_data.get("x1", 0.0))
            y1 = float(bbox_data.get("y1", 0.0))
            x2 = float(bbox_data.get("x2", 0.0))
            y2 = float(bbox_data.get("y2", 0.0))
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue

            seg = LayoutSegmentIR(x=x1, y=y1, w=w, h=h, confidence=1.0, label="Table")
            table_html = _table_dict_to_html(table)
            items.append(LayoutItemIR(type="table", value=table_html, bbox=seg, layout_segments=[seg]))

        if items:
            layout_pages.append(
                ParseLayoutPageIR(
                    page_number=page_idx + 1,
                    width=_VIRTUAL_PAGE_DIM,
                    height=_VIRTUAL_PAGE_DIM,
                    items=items,
                )
            )

    return layout_pages


def _table_dict_to_html(table: dict[str, Any]) -> str:
    parts = ["<table>"]
    for section, tag in [("header_rows", "th"), ("body_rows", "td")]:
        rows = table.get(section, [])
        if not rows:
            continue
        wrapper = "thead" if tag == "th" else "tbody"
        parts.append(f"<{wrapper}>")
        for row in rows:
            parts.append("<tr>")
            for cell in row:
                colspan = f' colspan="{cell["col_span"]}"' if cell.get("col_span", 1) > 1 else ""
                rowspan = f' rowspan="{cell["row_span"]}"' if cell.get("row_span", 1) > 1 else ""
                parts.append(f"<{tag}{colspan}{rowspan}>{cell.get('text', '')}</{tag}>")
            parts.append("</tr>")
        parts.append(f"</{wrapper}>")
    parts.append("</table>")
    return "\n".join(parts)


def _legacy_block_to_markdown(block: dict[str, Any]) -> str:
    block_type = block.get("type")
    parts: list[str] = []

    if block_type == "text":
        text = block.get("text", "").strip()
        text_type = block.get("text_type", "")
        if text:
            if text_type == "heading-1":
                parts.append(f"# {text}")
            elif text_type == "heading-2":
                parts.append(f"## {text}")
            elif text_type == "heading-3":
                parts.append(f"### {text}")
            elif text_type and text_type.startswith("heading"):
                parts.append(f"#### {text}")
            else:
                parts.append(text)
        for child in block.get("children", []):
            child_md = _legacy_block_to_markdown(child)
            if child_md:
                parts.append(child_md)

    elif block_type == "table":
        parts.append(_legacy_layout_table_to_html(block))

    elif block_type == "list":
        for entry in block.get("entries", []):
            entry_md = _legacy_block_to_markdown(entry)
            if entry_md:
                parts.append(f"- {entry_md}")

    return "\n\n".join(part for part in parts if part)


def _legacy_layout_table_to_html(table_block: dict[str, Any]) -> str:
    html_parts = ["<table>"]

    header_rows = table_block.get("header_rows", [])
    if header_rows:
        html_parts.append("<thead>")
        for row in header_rows:
            html_parts.append("<tr>")
            for cell in row:
                colspan = f' colspan="{cell["col_span"]}"' if cell.get("col_span", 1) > 1 else ""
                rowspan = f' rowspan="{cell["row_span"]}"' if cell.get("row_span", 1) > 1 else ""
                html_parts.append(f"<th{colspan}{rowspan}>{_legacy_extract_cell_text(cell)}</th>")
            html_parts.append("</tr>")
        html_parts.append("</thead>")

    body_rows = table_block.get("body_rows", [])
    if body_rows:
        html_parts.append("<tbody>")
        for row in body_rows:
            html_parts.append("<tr>")
            for cell in row:
                colspan = f' colspan="{cell["col_span"]}"' if cell.get("col_span", 1) > 1 else ""
                rowspan = f' rowspan="{cell["row_span"]}"' if cell.get("row_span", 1) > 1 else ""
                html_parts.append(f"<td{colspan}{rowspan}>{_legacy_extract_cell_text(cell)}</td>")
            html_parts.append("</tr>")
        html_parts.append("</tbody>")

    html_parts.append("</table>")
    return "\n".join(html_parts)


def _legacy_extract_cell_text(cell: dict[str, Any]) -> str:
    texts: list[str] = []
    for block in cell.get("blocks", []):
        if block.get("type") == "text":
            text = block.get("text", "").strip()
            if text:
                texts.append(text)
    return " ".join(texts)
