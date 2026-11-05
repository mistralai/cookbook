from typing import Self
from urllib.parse import quote

from pydantic import model_validator

from env._base import BaseEnv


def _to_async_url(url: str) -> str:
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + url[len("postgresql://") :]
    if url.startswith("postgres://"):
        return "postgresql+asyncpg://" + url[len("postgres://") :]
    return url


class Env(BaseEnv):
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"
    postgres_host: str = ""
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str = ""
    postgres_db: str = "postgres"
    postgres_ssl_mode: str = ""
    db_echo: bool = False
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_pre_ping: bool = True

    @model_validator(mode="after")
    def _url_from_postgres_parts(self) -> Self:
        # Managed Postgres on Mistral Apps injects POSTGRES_* rather than DATABASE_URL.
        # An explicit DATABASE_URL still wins.
        if not self.postgres_host or "database_url" in self.model_fields_set:
            return self
        credentials = f"{quote(self.postgres_user, safe='')}:{quote(self.postgres_password, safe='')}"
        host = f"[{self.postgres_host}]" if ":" in self.postgres_host else self.postgres_host
        url = f"postgresql+asyncpg://{credentials}@{host}:{self.postgres_port}/{quote(self.postgres_db, safe='')}"
        # asyncpg takes the libpq sslmode values under the name `ssl`.
        self.database_url = f"{url}?ssl={self.postgres_ssl_mode}" if self.postgres_ssl_mode else url
        return self

    @property
    def async_database_url(self) -> str:
        return _to_async_url(self.database_url)


env = Env()
