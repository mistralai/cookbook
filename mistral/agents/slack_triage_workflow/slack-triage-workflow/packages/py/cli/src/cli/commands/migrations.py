"""Bring the database schema to head."""

import asyncio
import importlib.util
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import structlog
import typer
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from db.engine import readiness_ping
from env.init import env as init_env

logger = structlog.get_logger("init.migrations")
app = typer.Typer()
INIT_STEP = True  # run by each deployment's init or pre-start steps

_Ping = Callable[[float], Awaitable[bool]]
_Sleep = Callable[[float], Awaitable[None]]


class MigrationsProjectNotFoundError(RuntimeError):
    """Raised when the db project holding alembic.ini cannot be located."""


class DatabaseReadinessTimeoutError(RuntimeError):
    """Raised when Postgres does not stay reachable long enough to run migrations."""


async def _wait_for_database_readiness(
    *,
    ping: _Ping = readiness_ping,
    sleep: _Sleep = asyncio.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    timeout_s = init_env.init_migrations_db_ready_timeout_seconds
    required_successes = init_env.init_migrations_db_ready_successes
    interval_s = init_env.init_migrations_db_ready_interval_seconds
    ping_timeout_s = init_env.init_migrations_db_ready_ping_timeout_seconds

    started_at = monotonic()
    successes = 0
    attempts = 0
    logger.info(
        "waiting_for_database_readiness",
        timeout_s=timeout_s,
        required_successes=required_successes,
        interval_s=interval_s,
        ping_timeout_s=ping_timeout_s,
    )

    while True:
        attempts += 1
        if await ping(ping_timeout_s):
            successes += 1
            if successes >= required_successes:
                logger.info(
                    "database_ready", attempts=attempts, consecutive_successes=successes
                )
                return
        else:
            successes = 0

        remaining_s = timeout_s - (monotonic() - started_at)
        if remaining_s <= 0:
            raise DatabaseReadinessTimeoutError(
                f"database did not reach {required_successes} consecutive successful readiness checks "
                f"within {timeout_s:g}s"
            )
        await sleep(min(interval_s, remaining_s))


def _db_project_dir() -> Path:
    spec = importlib.util.find_spec("db")
    if spec is None or spec.origin is None:
        raise MigrationsProjectNotFoundError("the 'db' package is not importable")
    # <project>/src/db/__init__.py -> <project>
    project = Path(spec.origin).resolve().parents[2]
    if not (project / "alembic.ini").is_file():
        raise MigrationsProjectNotFoundError(f"alembic.ini not found under {project}")
    return project


def _has_no_revisions(config: Config) -> bool:
    """Whether the migration set is empty. Informational only, so it never blocks the upgrade."""
    try:
        script = ScriptDirectory.from_config(config)
        return next(script.walk_revisions(), None) is None
    except Exception:
        return False


@app.command(name="migrations", help=__doc__)
def main() -> None:
    asyncio.run(_wait_for_database_readiness())
    project = _db_project_dir()
    config = Config(str(project / "alembic.ini"))
    # script_location and prepend_sys_path are relative to the db project in alembic.ini,
    # but Alembic resolves them against the process cwd, which is not that project here.
    config.set_main_option(
        "script_location", str(project / "src" / "db" / "migrations")
    )
    config.set_main_option("prepend_sys_path", str(project / "src"))
    if _has_no_revisions(config):
        # Legitimate when no selected capability owns a table: each owner ships its own baseline.
        logger.info(
            "no_migrations_present",
            hint="no selected capability owns a table; add a model and an Alembic revision for it",
        )
    # "heads", not "head": every table-owning capability ships its baseline as an independent branch,
    # so a composition with more than one of them has more than one head.
    command.upgrade(config, "heads")
