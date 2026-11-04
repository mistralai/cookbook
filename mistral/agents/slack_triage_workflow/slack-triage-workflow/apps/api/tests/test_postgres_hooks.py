import pytest
from support.postgres import postgres_hooks


@pytest.mark.asyncio
async def test_postgres_hooks_expose_report_and_readiness_results() -> None:
    hooks = postgres_hooks(report=False, ready=True)

    assert await hooks.report() == {"database": False}
    assert await hooks.ready() == {"database": True}
