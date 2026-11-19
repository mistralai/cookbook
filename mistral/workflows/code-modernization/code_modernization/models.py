"""Pydantic models for the Code Modernization workflow.

All data flowing between the parent workflow, sub-workflows, and activities is
typed through these models. Using Pydantic ensures Temporal can serialize/deserialize
every payload and that LLM responses are validated before use.
"""

from pydantic import BaseModel, Field


class ModernizationParams(BaseModel):
    """Top-level input for the parent workflow."""

    repo_path: str = Field(description="Absolute path to the local repo (or GitHub URL in production).")
    target: str = Field(
        default="Python 2.7 → 3.12",
        description="Modernization target description passed verbatim to the LLM prompt.",
    )


class FileToModernize(BaseModel):
    """Represents a single source file that needs modernization."""

    path: str = Field(description="Absolute path on disk.")
    relative_path: str = Field(description="Path relative to repo_path, used in PR descriptions.")
    source: str = Field(description="Raw source content of the file.")


class ModernizeFileParams(BaseModel):
    """Input to each ModernizeFileWorkflow child."""

    file: FileToModernize
    target: str


class ModernizedFile(BaseModel):
    """Structured output from the LLM modernization call.

    The LLM is asked to return exactly this shape so we can validate and diff.
    """

    original: str
    modernized: str
    changes_summary: list[str] = Field(description="Short bullet points describing each change made.")
    confidence: float = Field(ge=0.0, le=1.0, description="LLM self-assessed confidence in the modernization.")


class FileModernizationResult(BaseModel):
    """Full result for a single file after sub-workflow completion."""

    relative_path: str
    original: str
    modernized: str
    changes_summary: list[str]
    confidence: float
    syntax_valid: bool


class ChangeSet(BaseModel):
    """Aggregated results from all per-file sub-workflows."""

    repo_path: str
    target: str
    files: list[FileModernizationResult]

    @property
    def success_count(self) -> int:
        return sum(1 for f in self.files if f.syntax_valid)


class WorkflowResult(BaseModel):
    """Final output of the parent workflow."""

    status: str = Field(description="'approved' or 'declined'.")
    change_set: ChangeSet
    pr_proposal_path: str | None = Field(
        default=None,
        description="Path to the written PR_PROPOSAL.md, set only on approval.",
    )
