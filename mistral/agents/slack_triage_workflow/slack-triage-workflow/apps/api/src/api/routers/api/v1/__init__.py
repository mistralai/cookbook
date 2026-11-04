"""The tracker API. Every route needs a caller signed in through the Apps gateway."""

from fastapi import HTTPException, Request, params, status
from mistralai_capabilities.apps.tokens import token_from


def require_caller(request: Request) -> None:
    # The gateway signs the caller in and forwards a token; a request without one bypassed it.
    if not token_from(request.headers):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Call this API through the app's gateway"
        )


dependencies: tuple[params.Depends, ...] = (params.Depends(require_caller),)
