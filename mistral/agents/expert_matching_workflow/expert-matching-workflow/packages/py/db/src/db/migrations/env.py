"""Alembic environment configured for async SQLAlchemy (asyncpg) + SQLModel.

The connection URL is taken from db settings (``DATABASE_URL``, normalized
to the asyncpg driver) rather than alembic.ini, and ``target_metadata`` is
``SQLModel.metadata`` populated by importing every model module. Autogenerate only diffs the
tables those models declare (``db.schema_filter``), so tables a library or extension owns are left
alone.

The migration set is one independent branch per capability that owns a table (each rooted at
``down_revision = None`` and labelled with the capability id), so it may have several heads:
upgrade to ``heads``, never ``head``.
"""

import asyncio
from logging.config import fileConfig

import db.models  # noqa: F401  (import populates SQLModel.metadata)
from alembic import context
from db import metadata as target_metadata
from db.schema_filter import include_object
from env.db import env
from sqlalchemy import pool, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Several processes can upgrade at once, such as API instances that each migrate as they start. This
# Postgres advisory lock makes them take turns, so the later ones find the database already upgraded.
MIGRATION_LOCK_KEY = int.from_bytes(b"migrate")

config = context.config
config.set_main_option("sqlalchemy.url", env.async_database_url)

# Not `fileConfig`'s default, which disables every logger that exists already: `cli migrations` and
# the migration tests run this in a process whose own loggers would then go silent.
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)


def run_migrations_offline() -> None:
    context.configure(
        url=env.async_database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        connection.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": MIGRATION_LOCK_KEY})
        context.run_migrations()


async def run_async_migrations() -> None:
    section = config.get_section(config.config_ini_section, {})
    connectable = async_engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
