"""Base abstractions for layout evaluation adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from extract_bench.evaluation.metrics.attribution.core import PredBlock
from extract_bench.evaluation.metrics.attribution.text_utils import (
    extract_text_from_html,
    normalize_attribution_text,
    tokenize,
)
from extract_bench.schemas.layout_detection_output import LayoutOutput
from extract_bench.schemas.pipeline_io import InferenceResult
from extract_bench.test_cases.schema import TestCase


class LayoutAdapter(ABC):
    """Adapter contract for normalizing provider outputs to `LayoutOutput`."""

    @classmethod
    def get_provider_keys(cls) -> tuple[str, ...]:
        """Provider keys this adapter supports."""
        return ()

    @classmethod
    def matches(cls, inference_result: InferenceResult) -> bool:
        """Optional shape-based fallback matcher."""
        del inference_result
        return False

    @abstractmethod
    def to_layout_output(
        self,
        inference_result: InferenceResult,
        *,
        page_filter: int | None = None,
    ) -> LayoutOutput:
        """Convert provider output into unified `LayoutOutput`."""

    def to_attribution_blocks(
        self,
        layout_output: LayoutOutput,
        *,
        page_number: int,
        test_case: TestCase | None = None,
    ) -> list[PredBlock]:
        """Build attribution blocks from normalized prediction content.

        This generic path assumes ``layout_output.predictions`` are already
        leaf-like attribution claims. Providers with container trees or richer
        segment payloads must override this method.
        """
        del test_case
        if layout_output.image_width <= 0 or layout_output.image_height <= 0:
            return []

        blocks: list[PredBlock] = []
        for idx, prediction in enumerate(layout_output.predictions):
            if prediction.page != page_number:
                continue
            if prediction.content is None:
                continue

            if prediction.content.type == "table":
                raw_text = extract_text_from_html(prediction.content.html)
                block_type = "table"
            else:
                raw_text = prediction.content.text
                block_type = "text"

            normalized_text = normalize_attribution_text(raw_text)
            tokens = tokenize(normalized_text)
            bbox_xyxy = normalize_bbox_xyxy(
                prediction.bbox,
                width=layout_output.image_width,
                height=layout_output.image_height,
            )
            order_index = prediction.provider_metadata.get("order_index")
            if not isinstance(order_index, int):
                order_index = idx

            blocks.append(
                PredBlock(
                    bbox_xyxy=bbox_xyxy,
                    block_type=block_type,
                    label=prediction.label,
                    text=raw_text,
                    normalized_text=normalized_text,
                    tokens=tokens,
                    order_index=order_index,
                )
            )

        return blocks


def normalize_bbox_xyxy(bbox: list[float], *, width: int, height: int) -> list[float]:
    """Normalize pixel XYXY bbox coordinates into [0, 1] space."""
    return [
        bbox[0] / width,
        bbox[1] / height,
        bbox[2] / width,
        bbox[3] / height,
    ]
