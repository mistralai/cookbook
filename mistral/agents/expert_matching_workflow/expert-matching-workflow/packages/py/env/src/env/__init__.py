"""Typed application settings.

Import a settings instance from its concern module, for example ``from env.mistral import env``.
Each module defines ``class Env(BaseEnv)`` and ``env = Env()``. The shared
:class:`env._base.BaseEnv` loads the repository-root ``.env`` without overriding process values.
Annotate a setting with :class:`WorkspaceSecret` to let the Mistral workspace secrets fill it.
"""

from env._base import BaseEnv
from env._mistral_secrets import WorkspaceSecret, WorkspaceSecretsForbiddenError

__all__ = ["BaseEnv", "WorkspaceSecret", "WorkspaceSecretsForbiddenError"]
