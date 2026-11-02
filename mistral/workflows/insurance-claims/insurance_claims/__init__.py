"""Insurance Claims Triage with Vision.

Demonstrates: parallel activities, retry policies on vision calls,
deterministic severity branching, structured LLM outputs, and full
timeline observability in AI Studio.
"""

from .workflow import InsuranceClaimsTriageWorkflow

__all__ = ["InsuranceClaimsTriageWorkflow"]
