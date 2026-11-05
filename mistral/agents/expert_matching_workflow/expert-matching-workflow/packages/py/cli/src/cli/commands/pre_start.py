"""Run the setup steps the API needs before it starts on Mistral Apps, such as database migrations."""

import subprocess
import sys
from importlib.util import find_spec

import typer

app = typer.Typer()

# Each step is a `python -m cli` command a selected capability may have vendored, run in this order.
# Mistral Apps runs these on every start of every instance, so each must be idempotent. Init steps
# that other deployments run, and that are not listed here, do not run on Mistral Apps.
STEPS = ("migrations",)


def installed_steps() -> list[str]:
    """The steps whose command module a selected capability vendored."""
    return [step for step in STEPS if find_spec(f"cli.commands.{step}") is not None]


@app.command(name="pre-start", help=__doc__)
def main() -> None:
    for step in installed_steps():
        subprocess.run([sys.executable, "-m", "cli", step.replace("_", "-")], check=True)
