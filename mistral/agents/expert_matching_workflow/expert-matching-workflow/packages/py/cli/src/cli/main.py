"""The generated app's command-line entry point.

`cli` is a Typer application whose command modules each define a `typer.Typer()` app. `cli.main`
discovers those modules and mounts each app at the top level with `add_typer`, so a command is invoked
as `python -m cli <command>`. Logging is configured once in the application callback.
"""

import importlib
import pkgutil

import typer
from env.logging import env as logging_env
from utils.logging import configure_logging

from cli import commands as _commands

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Operational commands for this app, contributed by its installed capabilities.",
)


@app.callback()
def _configure() -> None:
    # Runs before every command (Click skips it for `--help`), so each command inherits one logging
    # setup instead of repeating it.
    configure_logging(log_level=logging_env.log_level, log_format=logging_env.log_format)


def _mount_commands() -> None:
    # Mount every command module present in the generated application. Discovery keeps the shared
    # entry point independent of the selected composition.
    for info in pkgutil.iter_modules(_commands.__path__):
        module = importlib.import_module(f"{_commands.__name__}.{info.name}")
        app.add_typer(module.app)


_mount_commands()
