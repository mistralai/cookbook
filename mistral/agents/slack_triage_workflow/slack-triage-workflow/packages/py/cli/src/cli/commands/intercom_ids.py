"""Print the Intercom workspace ID and the ticket type IDs to put in `.env`."""

import asyncio

import typer
from env.triage import env

app = typer.Typer()


@app.command(name="intercom-ids", help=__doc__)
def main() -> None:
    # The worker's connector client, so this also checks that the connector and its credential work.
    from worker.activities import INTERCOM_HEADERS, call_connector

    async def fetch() -> tuple[dict, dict]:
        name = env.require("intercom_connector_name")
        me = await call_connector(name, "GET", "/me", headers=INTERCOM_HEADERS)
        types = await call_connector(
            name, "GET", "/ticket_types", headers=INTERCOM_HEADERS
        )
        return me, types

    me, types = asyncio.run(fetch())
    typer.echo(f"INTERCOM_WORKSPACE_ID={me['app']['id_code']}")
    for ticket_type in types["data"]:
        if not ticket_type["archived"]:
            typer.echo(
                f"INTERCOM_TICKET_TYPE_ID={ticket_type['id']}  # {ticket_type['name']} ({ticket_type['category']})"
            )
