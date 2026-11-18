"""Create the feature_request table.

Revision ID: 0001
Revises:
Create Date: 2026-10-27

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = ("feature_request",)
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "feature_request",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("summary", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("customer", sa.String(), nullable=False),
        sa.Column("source_channel", sa.String(), nullable=False),
        sa.Column("slack_thread_url", sa.String(), nullable=False),
        sa.Column("urgency", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("feature_request")
