"""Async SQLAlchemy engine and session factory for db.

The engine is created lazily so importing this module never opens a connection
(safe for processes that may run without a database configured).
"""

import asyncio

from env.db import env
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

_engine: AsyncEngine | None = None
_session_maker: async_sessionmaker[AsyncSession] | None = None
_readiness_engine: AsyncEngine | None = None

# Keep below the API readiness probe's timeout (3s), or the probe gives up first and a reachable
# database reads as unready.
_PING_TIMEOUT_S = 2.0


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            env.async_database_url,
            echo=env.db_echo,
            pool_size=env.db_pool_size,
            max_overflow=env.db_max_overflow,
            pool_pre_ping=env.db_pool_pre_ping,
        )
    return _engine


def get_session_maker() -> async_sessionmaker[AsyncSession]:
    global _session_maker
    if _session_maker is None:
        _session_maker = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_maker


def _get_readiness_engine() -> AsyncEngine:
    """``NullPool`` so this engine connects per call and can never queue behind the app pool."""
    global _readiness_engine
    if _readiness_engine is None:
        _readiness_engine = create_async_engine(
            env.async_database_url,
            echo=env.db_echo,
            poolclass=NullPool,
        )
    return _readiness_engine


async def ping(timeout_s: float = _PING_TIMEOUT_S) -> bool:
    """Whether the database answers through the application pool. Never raises.

    Shares the request path's pool, so ``False`` means unreachable OR saturated. That
    ambiguity suits ``/api/health``, which only reports; probes need ``readiness_ping``.
    """
    try:
        async with asyncio.timeout(timeout_s):
            async with get_session_maker()() as session:
                await session.execute(text("SELECT 1"))
    except Exception:
        return False
    return True


async def readiness_ping(timeout_s: float = _PING_TIMEOUT_S) -> bool:
    """Whether Postgres is reachable from this pod. Never raises. Not a duplicate of ``ping``.

    Readiness gates traffic, so it must not answer "this pod is busy". On the shared pool it
    would: a saturated replica marks itself unready, sheds load onto its peers, saturates
    them, and takes the service down without Postgres ever failing.
    """
    try:
        async with asyncio.timeout(timeout_s):
            async with _get_readiness_engine().connect() as conn:
                await conn.execute(text("SELECT 1"))
    except Exception:
        return False
    return True


async def dispose_engine() -> None:
    global _engine, _session_maker, _readiness_engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_maker = None
    if _readiness_engine is not None:
        await _readiness_engine.dispose()
        _readiness_engine = None
