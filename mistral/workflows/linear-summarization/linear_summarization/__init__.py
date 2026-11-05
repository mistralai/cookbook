"""Linear Weekly Summary Workflow

Demonstrates an end-to-end workflow using the Linear connector to fetch recently
updated issues and generate a summary. Highlights:
    - InteractiveWorkflow with wait_for_input for HITL in Studio
    - Connector-based validation of user input (team + project)
    - Structured activity outputs for rich context in the final summary
    - On-behalf-of execution to use the executor's stored Linear credentials"""

from .workflow import LinearWeeklySummaryWorkflow, LinearWorkflowParams

__all__ = ["LinearWeeklySummaryWorkflow", "LinearWorkflowParams"]
