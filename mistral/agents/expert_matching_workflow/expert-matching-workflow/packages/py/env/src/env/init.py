"""Settings for deployment init steps."""

from pydantic import Field

from env._base import BaseEnv


class Env(BaseEnv):
    # The migrations step waits for a stable window before running Alembic because a
    # development Postgres can briefly accept connections and then restart itself.
    init_migrations_db_ready_timeout_seconds: float = Field(default=90.0, gt=0)
    init_migrations_db_ready_successes: int = Field(default=3, ge=1)
    init_migrations_db_ready_interval_seconds: float = Field(default=2.0, gt=0)
    init_migrations_db_ready_ping_timeout_seconds: float = Field(default=2.0, gt=0)


env = Env()
