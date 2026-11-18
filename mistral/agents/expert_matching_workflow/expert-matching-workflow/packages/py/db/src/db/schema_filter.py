"""Scope Alembic autogenerate to the tables this app's models declare.

The database holds tables no model here describes, and each has another owner: libraries provision
their own at runtime (a search plugin's per-collection ``<collection>_chunks`` tables, for example)
and extensions ship theirs (PostGIS's ``spatial_ref_sys``). Autogenerate
compares the whole schema against ``SQLModel.metadata``, so without this filter every revision would
open with ``op.drop_table`` for each of them.

The rule is ownership, not a name list: a table that exists only in the database is never this
app's to drop. Removing one of the app's own tables is therefore a hand-written ``op.drop_table``,
which is the deliberate step it should be anyway.
"""

from typing import Any


def include_object(
    _object: Any,
    _name: str | None,
    type_: str,
    reflected: bool,
    compare_to: Any,
) -> bool:
    """Alembic ``include_object`` hook: skip a reflected table that has no model counterpart."""
    return not (type_ == "table" and reflected and compare_to is None)
