"""The migration set a composition ends up with, checked without a database.

Each capability that owns a table ships its model and its baseline revision together, as the root of
its own branch. These checks hold for whatever subset of those capabilities an app selected, which
is why they iterate over what is present rather than naming a table: a model with no revision leaves
a freshly generated app without that table, and it fails here first.
"""

import io
from pathlib import Path

import db.models  # noqa: F401  (import populates SQLModel.metadata)
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from db import metadata

MIGRATIONS = Path(__file__).resolve().parents[1] / "src" / "db" / "migrations"


def _script() -> ScriptDirectory:
    return ScriptDirectory(str(MIGRATIONS))


def _upgrade_sql() -> str:
    """Every revision's upgrade rendered to SQL, the offline (``--sql``) way."""
    buffer = io.StringIO()
    context = MigrationContext.configure(dialect_name="postgresql", opts={"as_sql": True, "output_buffer": buffer})
    with Operations.context(context):
        for revision in _script().walk_revisions():
            revision.module.upgrade()
    return buffer.getvalue()


def test_every_model_table_is_created_by_a_shipped_revision() -> None:
    sql = _upgrade_sql()
    missing = [table for table in metadata.tables if f"CREATE TABLE {table} (" not in sql]
    assert not missing, f"models with no migration creating their table: {missing}"


def test_every_root_is_a_labelled_capability_branch() -> None:
    roots = [revision for revision in _script().walk_revisions() if revision.down_revision is None]
    unlabelled = [revision.revision for revision in roots if not revision.branch_labels]
    assert not unlabelled, f"a baseline must carry its capability's branch label: {unlabelled}"


def test_every_revision_id_fits_the_alembic_version_column() -> None:
    # `alembic_version.version_num` is varchar(32); a longer id fails at stamp time, after the
    # migration body has already run.
    too_long = [revision.revision for revision in _script().walk_revisions() if len(revision.revision) > 32]
    assert not too_long, f"ids longer than alembic_version.version_num's 32 characters: {too_long}"
