"""Create the role and expert tables, and add three sample roles.

Revision ID: 0001
Revises:
Create Date: 2026-11-05

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = ("expert_matching",)
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    role = op.create_table(
        "role",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("project", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
    )
    op.create_table(
        "expert",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("file_name", sa.String(), nullable=False),
        sa.Column("file_id", sa.String(), nullable=False),
        sa.Column("execution_id", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("headline", sa.String(), nullable=True),
        sa.Column("years_of_experience", sa.Integer(), nullable=True),
        sa.Column("skills", sa.JSON(), nullable=False),
        sa.Column("summary", sa.String(), nullable=True),
        sa.Column("role_id", sa.Integer(), nullable=True),
        sa.Column("reason", sa.String(), nullable=True),
        sa.Column("review", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.bulk_insert(
        role,
        [
            {
                "title": "Data platform engineer",
                "project": "Retail demand forecasting",
                "description": (
                    "Build the ingestion pipelines and feature store behind a weekly demand forecast. "
                    "Python, SQL, Spark or similar, and experience running pipelines in production."
                ),
            },
            {
                "title": "Cloud security architect",
                "project": "Bank cloud migration",
                "description": (
                    "Design identity, network and key management for a bank moving workloads to the cloud. "
                    "Experience with regulated environments and security reviews."
                ),
            },
            {
                "title": "Product designer",
                "project": "Patient booking app",
                "description": (
                    "Lead user research and design the booking flow of a mobile app for clinics. "
                    "Strong portfolio in mobile UX and accessibility."
                ),
            },
        ],
    )


def downgrade() -> None:
    op.drop_table("expert")
    op.drop_table("role")
