"""The engine module's contract, without a database.

Importing ``db.engine`` must never connect, both probes must answer ``False`` rather than raise when
Postgres is unreachable, and disposal must leave the module ready to build fresh engines. The
unreachable server is a closed local port, so every probe fails fast and offline.
"""

from collections.abc import Iterator

import pytest
from db import engine
from env.db import Env, _to_async_url

# Port 1 on loopback: nothing listens there, so a connection is refused immediately.
_UNREACHABLE = "postgresql://postgres:postgres@127.0.0.1:1/postgres"
_DB_ENV_KEYS = (
    "DATABASE_URL",
    *(f"POSTGRES_{part}" for part in ("HOST", "PORT", "USER", "PASSWORD", "DB", "SSL_MODE")),
)


@pytest.fixture
def unreachable(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(engine, "env", Env(database_url=_UNREACHABLE, _env_file=None))
    monkeypatch.setattr(engine, "_engine", None)
    monkeypatch.setattr(engine, "_session_maker", None)
    monkeypatch.setattr(engine, "_readiness_engine", None)
    yield


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        pytest.param("postgresql+asyncpg://u@h/d", "postgresql+asyncpg://u@h/d", id="already_async"),
        pytest.param("postgresql://u@h/d", "postgresql+asyncpg://u@h/d", id="postgresql"),
        pytest.param("postgres://u@h/d", "postgresql+asyncpg://u@h/d", id="postgres_alias"),
        pytest.param("sqlite:///x.db", "sqlite:///x.db", id="other_driver_untouched"),
    ],
)
def test_a_database_url_is_normalised_to_the_async_driver(url: str, expected: str) -> None:
    assert _to_async_url(url) == expected
    assert Env(database_url=url, _env_file=None).async_database_url == expected


@pytest.fixture
def no_db_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _DB_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.mark.usefixtures("no_db_env")
@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        pytest.param(
            {"postgres_host": "db.example", "postgres_user": "app", "postgres_password": "pw", "postgres_db": "main"},
            "postgresql+asyncpg://app:pw@db.example:5432/main",
            id="parts",
        ),
        pytest.param(
            {"postgres_host": "db", "postgres_port": 6543, "postgres_ssl_mode": "require"},
            "postgresql+asyncpg://postgres:@db:6543/postgres?ssl=require",
            id="port_and_ssl",
        ),
        pytest.param(
            {"postgres_host": "db", "postgres_user": "a@b", "postgres_password": "p/w:@?"},
            "postgresql+asyncpg://a%40b:p%2Fw%3A%40%3F@db:5432/postgres",
            id="credentials_escaped",
        ),
        pytest.param(
            {"postgres_host": "2001:db8::1", "postgres_db": "tenant?archive"},
            "postgresql+asyncpg://postgres:@[2001:db8::1]:5432/tenant%3Farchive",
            id="ipv6_host_and_database_escaped",
        ),
    ],
)
def test_the_url_is_built_from_postgres_parts(settings: dict[str, object], expected: str) -> None:
    assert Env(_env_file=None, **settings).async_database_url == expected


@pytest.mark.usefixtures("no_db_env")
@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        pytest.param({}, "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres", id="no_host"),
        pytest.param(
            {"postgres_host": "", "postgres_password": "pw"},
            "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres",
            id="empty_host",
        ),
        pytest.param(
            {"database_url": "postgres://u@explicit/d", "postgres_host": "db"},
            "postgresql+asyncpg://u@explicit/d",
            id="explicit_url_wins",
        ),
    ],
)
def test_the_url_is_not_built_from_postgres_parts(settings: dict[str, object], expected: str) -> None:
    assert Env(_env_file=None, **settings).async_database_url == expected


@pytest.mark.usefixtures("no_db_env")
def test_postgres_parts_are_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "db.example")
    monkeypatch.setenv("POSTGRES_PASSWORD", "pw")
    monkeypatch.setenv("POSTGRES_SSL_MODE", "require")

    expected = "postgresql+asyncpg://postgres:pw@db.example:5432/postgres?ssl=require"
    assert Env(_env_file=None).async_database_url == expected


@pytest.mark.usefixtures("unreachable")
async def test_probes_answer_false_when_postgres_is_unreachable() -> None:
    assert await engine.ping(timeout_s=5.0) is False
    assert await engine.readiness_ping(timeout_s=5.0) is False
    await engine.dispose_engine()
