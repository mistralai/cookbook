"""The CLI must expose exactly one command per vendored command module.

The expected command set is derived from modules on disk, including the valid empty set. Each module
ships one Typer app (`app`) that `cli.main` mounts with `add_typer`; a module the CLI failed to expose
would be a command that silently vanishes. These checks pin the discovery contract.
"""

import pkgutil

from cli import commands as commands_pkg
from cli.main import app
from typer.main import get_command

_MODULES = sorted(info.name for info in pkgutil.iter_modules(commands_pkg.__path__))


def test_cli_exposes_one_command_per_module() -> None:
    # Command names use module stems with underscores rendered as hyphens.
    exposed = sorted(getattr(get_command(app), "commands", {}))
    assert exposed == sorted(name.replace("_", "-") for name in _MODULES)
