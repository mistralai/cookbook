"""Bind the workflow's workspace secrets to the managed worker deployment.

`mistral apps deploy` binds only the database settings, and a deployed worker has no `.env`. Each
variable below is bound to the workspace secret of the same name, so create those secrets first.
"""

from typing import Annotated, Any

import httpx
import typer
from env.mistral import env as mistral_env
from env.triage import env
from utils.mistral import mistral_api_url

app = typer.Typer()

SECRETS = (
    "MISTRAL_API_KEY",
    "SLACK_BOT_TOKEN",
    "INTERCOM_CONNECTOR_NAME",
    "INTERCOM_TICKET_TYPE_ID",
    "INTERCOM_WORKSPACE_ID",
    "TRACKER_CONNECTOR_NAME",
)


@app.command(name="bind-secrets", help=__doc__)
def main(
    deployment: Annotated[
        str, typer.Argument(help="Worker deployment name, for example my-app-worker")
    ],
) -> None:
    url = f"{mistral_api_url()}/v1/workflows/deployments/{deployment}"
    headers = {"Authorization": f"Bearer {mistral_env.mistral_api_key}"}
    current = httpx.get(url, headers=headers, timeout=30).raise_for_status().json()
    backend: dict[str, Any] = dict(
        (current.get("spec") or {}).get("backend_spec") or {}
    )
    # The PATCH replaces the whole list, so keep the bindings deploy made, such as POSTGRES_*.
    bindings = {b["env_var_name"]: b for b in backend.get("secrets") or []}
    # INTERCOM_APP_URL is optional, so bind it only when it's set here and its secret should exist.
    names = (*SECRETS, "INTERCOM_APP_URL") if env.intercom_app_url else SECRETS
    bindings |= {
        name: {"env_var_name": name, "reference": f"secret:workspace:{name}"}
        for name in names
    }
    backend["secrets"] = list(bindings.values())
    httpx.patch(
        url, headers=headers, json={"spec": {"backend_spec": backend}}, timeout=30
    ).raise_for_status()
    typer.echo(f"Bound {', '.join(names)} on {deployment}")
