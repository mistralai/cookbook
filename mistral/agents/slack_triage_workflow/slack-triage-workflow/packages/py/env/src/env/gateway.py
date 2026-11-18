"""Where the Mistral API is reachable through the Apps gateway.

Only the gateway sets it, so an app with no value here has no gateway in front of it. The two names
carry the same URL and the gateway sends whichever the deployment is on. The leading double
underscore marks a variable the platform injects at runtime, so it is owned by no capability and
never appears in the generated `.env`.
"""

from pydantic import AliasChoices, Field

from env._base import BaseEnv


class Env(BaseEnv):
    apps_proxy_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("__APPS_PROXY_URL", "__SPACES_PROXY_URL"),
    )


env = Env()
