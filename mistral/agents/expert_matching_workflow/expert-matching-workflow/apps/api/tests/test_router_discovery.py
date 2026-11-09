"""What the shipped tree serves, and what its packages declare.

The loader's own behaviour is pinned in the fastapi toolkit's `tests/test_routing.py`. What stays
here is the part that is about the tree itself: that it reproduces the committed surface, and that
its groups declare the rules the loader then applies.
"""

import importlib
import pkgutil

from api.routers.api import internal
from fastapi import APIRouter, FastAPI
from mistralai_capabilities.fastapi.routing import anonymous_paths

_INFRA = {"/health", "/health/live", "/health/ready", "/health/startup"}


def _app(router: APIRouter) -> FastAPI:
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    app.include_router(router)
    return app


def test_the_anonymous_surface_is_what_the_groups_declared() -> None:
    """`internal` declares `()`, so its paths — and only its paths — carry no rule."""
    assert anonymous_paths("api.routers.api.internal") == _INFRA


def test_every_api_owned_internal_module_serves() -> None:
    """The API-owned anonymous group holds route modules and nothing else.

    The loader skips a module that exposes no `router`, so anything else put here would be
    invisible rather than rejected. Feature capabilities own their own route modules, so this
    assertion stays on the base package's invariant instead of walking the composed tree.
    """
    for info in pkgutil.iter_modules(internal.__path__, f"{internal.__name__}."):
        module = importlib.import_module(info.name)
        assert not info.ispkg, f"{info.name} is a package and must declare its own ownership rule"
        assert isinstance(getattr(module, "router", None), APIRouter), (
            f"{info.name} serves nothing — it belongs outside the URL tree"
        )
