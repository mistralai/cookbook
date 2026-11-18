"""Discover every contributed model module so ``SQLModel.metadata`` is complete for Alembic.

The base package ships no models: each feature capability that owns a table drops its model module
here (``db/models/<table>.py``), and this loader imports every sibling on import, registering it on
the shared ``SQLModel.metadata``. It mirrors the file discovery the API routers and FastAPI hooks
use, so adding a table is dropping a file, never editing a list here. Import order is irrelevant --
the models carry no cross-table foreign keys.
"""

import importlib
import pkgutil

from sqlmodel import SQLModel

for _module in sorted(pkgutil.iter_modules(__path__, f"{__name__}."), key=lambda entry: entry.name):
    # One public module per table; skip private helpers (a model that needs one imports it directly).
    if _module.name.rpartition(".")[2].startswith("_"):
        continue
    importlib.import_module(_module.name)

__all__ = ["SQLModel"]
