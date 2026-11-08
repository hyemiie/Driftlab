from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from driftwatch.db.models import ColumnDtype, ColumnRole, ProjectStatus, TaskType


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    task_type: TaskType

    model_config = ConfigDict(str_strip_whitespace=True)


class ProjectRead(BaseModel):
    id: UUID
    name: str
    task_type: TaskType
    status: ProjectStatus
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

# ---- auth ----
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserRead(BaseModel):
    id: UUID
    email: EmailStr
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ---- schema columns ----
class SchemaColumnRead(BaseModel):
    name: str
    dtype: ColumnDtype
    role: ColumnRole
    position: int

    model_config = ConfigDict(from_attributes=True)


class SchemaColumnProposal(SchemaColumnRead):
    reason: str  # why inference chose this dtype; computed, not stored


class SchemaColumnUpdate(BaseModel):
    name: str
    dtype: ColumnDtype
    role: ColumnRole


class SchemaUpdate(BaseModel):
    columns: list[SchemaColumnUpdate] = Field(min_length=1)


# ---- reference profile ----
class ColumnStatRead(BaseModel):
    column_name: str
    stats: dict


class ReferenceProfileRead(BaseModel):
    id: UUID
    n_rows: int
    created_at: datetime
    columns: list[ColumnStatRead]