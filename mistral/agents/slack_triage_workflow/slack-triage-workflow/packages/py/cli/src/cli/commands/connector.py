"""Create a Studio HTTP connector whose credential is shared with the workspace."""

from typing import Annotated

import typer
from mistralai.client.models import (
    BearerAuthMethod,
    ConnectorAuthenticationHeader,
    CreateHTTPConnectorRequest,
    NoneAuthMethod,
)
from utils.mistral import mistral_client

app = typer.Typer()


@app.command(name="connector", help=__doc__)
def main(
    name: Annotated[
        str, typer.Argument(help="Connector name, for example intercom or tracker")
    ],
    url: Annotated[
        str, typer.Argument(help="API base URL, for example https://api.intercom.io")
    ],
    token: Annotated[
        str,
        typer.Option(prompt=True, hide_input=True, help="Token the connector sends"),
    ],
    header: Annotated[
        str | None,
        typer.Option(
            help="Send the token in this header, such as X-Api-Key, instead of as a bearer token"
        ),
    ] = None,
) -> None:
    if header:
        auth = NoneAuthMethod(headers=[ConnectorAuthenticationHeader(name=header)])
        credentials = {"headers": {header: token.strip()}}
    else:
        auth = BearerAuthMethod()
        credentials = {"bearer_token": token.strip()}
    client = mistral_client()
    client.beta.connectors.create(
        request=CreateHTTPConnectorRequest(
            name=name,
            title=name,
            description=f"{name} API",
            server=url,
            visibility="shared_workspace",
            auth_methods=[auth],
        )
    )
    # Workspace scope: a credential scoped to your user is invisible to triggered runs.
    client.beta.connectors.create_credentials(
        connector_id_or_name=name,
        consumer_scope="workspace",
        name="default",
        credentials=credentials,
    )
    typer.echo(f"Created connector {name} -> {url} with a workspace credential")
