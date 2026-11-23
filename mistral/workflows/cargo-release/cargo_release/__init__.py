"""Cargo Release & Dangerous Goods Compliance workflow package.

Demonstrates: wait_for_input() HITL, child sub-workflows, activity retry policies,
deterministic branching, and structured LLM output via Pydantic, all in a
realistic maritime logistics use case.

The worker discovers workflows module by module, so the workflow classes are
not re-exported here: importing them would register each one twice.
"""
