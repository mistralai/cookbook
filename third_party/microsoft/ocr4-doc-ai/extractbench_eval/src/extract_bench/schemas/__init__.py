"""Schema definitions for the evaluation system."""

from extract_bench.schemas.evaluation import (
    EvaluationResult,
    EvaluationSummary,
    MetricValue,
    RunStat,
)
from extract_bench.schemas.extract_output import ExtractOutput, FieldCitation
from extract_bench.schemas.parse_output import PageIR, ParseOutput
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from extract_bench.schemas.product import ProductType

__all__ = [
    "EvaluationResult",
    "EvaluationSummary",
    "ExtractOutput",
    "FieldCitation",
    "InferenceRequest",
    "InferenceResult",
    "MetricValue",
    "RunStat",
    "PageIR",
    "ParseOutput",
    "PipelineSpec",
    "ProductType",
    "RawInferenceResult",
]
