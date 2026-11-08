"""initial migration2

Revision ID: 9fc6f192c12c
Revises: 9785fa02deda
Create Date: 2026-11-06 23:03:47.558181

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9fc6f192c12c'
down_revision: Union[str, Sequence[str], None] = '9785fa02deda'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
