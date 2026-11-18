"""Settings for the Slack triage workflow.

The connector names, the Intercom IDs and the Slack token have no defaults: set them in `.env` locally,
or as workspace secrets for the deployed worker.
"""

from typing import Annotated

from env._base import BaseEnv
from env._mistral_secrets import WorkspaceSecret


class Env(BaseEnv):
    triage_model: str = "mistral-medium-latest"
    slack_bot_token: Annotated[str | None, WorkspaceSecret()] = None
    intercom_connector_name: Annotated[str | None, WorkspaceSecret()] = None
    intercom_ticket_type_id: Annotated[str | None, WorkspaceSecret()] = None
    intercom_workspace_id: Annotated[str | None, WorkspaceSecret()] = None
    tracker_connector_name: Annotated[str | None, WorkspaceSecret()] = None
    # Where ticket links in Slack replies point when not https://app.intercom.com, such as
    # https://app.eu.intercom.com for an EU workspace.
    intercom_app_url: str | None = None

    def require(self, name: str) -> str:
        """The value of setting ``name``, or an error naming the variable to set."""
        value = getattr(self, name)
        if not value:
            raise RuntimeError(f"{name.upper()} is not set")
        return value


env = Env()
