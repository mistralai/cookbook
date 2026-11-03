from mistralai_capabilities.fastapi.hooks import FastAPIHooks


def postgres_hooks(*, report: bool = True, ready: bool = True) -> FastAPIHooks:
    async def report_check() -> bool:
        return report

    async def readiness_check() -> bool:
        return ready

    return FastAPIHooks(
        report_checks=(("database", report_check),),
        readiness_checks=(("database", readiness_check),),
    )
