"""Schemas for evaluation metrics and confusion matrix data."""

from typing import Literal

from pydantic import BaseModel, Field


class ConfusionMatrixCell(BaseModel):
    """Single cell in confusion matrix.

    Represents a specific (GT class, Pred class) pair with count,
    percentage, and the test case IDs that contributed to this cell.
    """

    gt_class: str = Field(..., description="Ground truth class name")
    pred_class: str = Field(..., description="Predicted class name")
    count: int = Field(..., description="Number of instances in this cell")
    percentage: float = Field(
        ...,
        description="Percentage of total GT instances for this gt_class (count / gt_total * 100)",
    )
    example_ids: list[str] = Field(
        default_factory=list,
        description="Test case IDs (test_id) that contributed to this cell",
    )


class ConfusionMatrixMetrics(BaseModel):
    """Confusion matrix computed during evaluation.

    Contains full confusion matrix data with metadata to enable
    interactive filtering in the HTML report.
    """

    iou_threshold: float = Field(default=0.5, description="IoU threshold used for matching predictions to GT")
    evaluation_view: Literal["core", "canonical"] = Field(
        default="core",
        description="Evaluation view: 'core' for Core11 labels, 'canonical' for Canonical17",
    )

    cells: list[ConfusionMatrixCell] = Field(
        default_factory=list,
        description="All confusion matrix cells (including diagonal for correct predictions)",
    )

    false_negatives: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Unmatched GT boxes by class. Maps class name → list of test_ids",
    )

    false_positives: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Unmatched predictions by class. Maps class name → list of test_ids",
    )

    gt_totals: dict[str, int] = Field(
        default_factory=dict,
        description="Total GT instances per class. Maps class name → count",
    )

    pred_totals: dict[str, int] = Field(
        default_factory=dict,
        description="Total predictions per class. Maps class name → count",
    )

    all_classes: list[str] = Field(
        default_factory=list,
        description="Sorted list of all unique class names in GT and predictions",
    )
