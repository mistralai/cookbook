from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from env.api import env as http_env
from env.logging import env as logging_env
from env.mistral import env as mistral_env
from env.telemetry import env as telemetry_env
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mistralai_capabilities.fastapi.hooks import FastAPIHooks
from mistralai_capabilities.fastapi.openapi import operation_id
from mistralai_capabilities.fastapi.routing import create_api_router
from utils.logging import configure_logging
from utils.telemetry import configure_telemetry, shutdown_telemetry


def create_app(*, hooks: FastAPIHooks | None = None) -> FastAPI:
    configure_logging(log_level=logging_env.log_level, log_format=logging_env.log_format)
    if telemetry_env.telemetry_enabled:
        configure_telemetry(
            endpoint=telemetry_env.telemetry_endpoint,
            api_key=mistral_env.mistral_api_key,
            service_name=telemetry_env.telemetry_service_name,
            export_timeout_seconds=telemetry_env.telemetry_export_timeout_seconds,
        )

    manager = hooks if hooks is not None else FastAPIHooks.discover()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        try:
            await manager.startup(app)
            yield
        finally:
            # Hooks first, so their shutdown logs still export; telemetry last, even if one raises.
            try:
                await manager.shutdown(app)
            finally:
                shutdown_telemetry()

    # FastAPI 0.142+ logs every unhandled exception, stack trace included, to the global OTel logger
    # provider. That provider exports to TELEMETRY_ENDPOINT, which takes evaluation results only, so
    # keep the API's own error logs out of it.
    # Studio HTTP connectors import this spec and name each MCP tool after its operationId.
    app = FastAPI(lifespan=lifespan, telemetry={"logs": False}, generate_unique_id_function=operation_id)
    app.state.fastapi_hooks = manager
    manager.configure(app)
    app.include_router(create_api_router("api.routers"))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[http_env.cors_origin],
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    return app


app = create_app()
