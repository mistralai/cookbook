"""The app-local persistence package: the async engine/session factory and the SQLModel metadata.

Concrete models and accessors are contributed by the feature capabilities that own each table,
dropped into ``db/models/`` and ``db/accessors/``; this base names none of them. Importing ``db``
imports ``db.models``, whose loader discovers every contributed model module, so ``metadata`` is
complete for Alembic autogenerate.
"""

from .engine import dispose_engine, get_engine, get_session_maker, ping, readiness_ping
from .models import SQLModel

metadata = SQLModel.metadata

__all__ = [
    "SQLModel",
    "dispose_engine",
    "get_engine",
    "get_session_maker",
    "metadata",
    "ping",
    "readiness_ping",
]
