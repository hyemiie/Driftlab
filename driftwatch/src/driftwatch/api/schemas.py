from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator

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


MAX_BATCH = 1000


class PredictionIn(BaseModel):
    row_id: str = Field(max_length=255, description="Your ID for this prediction; also the idempotency key.")
    timestamp: datetime | None = Field(None, description="When the prediction was made. Defaults to receipt time.")
    model_version: str | None = Field(None, max_length=100)
    inputs: dict[str, Any] = Field(default_factory=dict, description="Feature name -> scalar value.")
    prediction: float | str | bool
    prediction_score: float | None = Field(None, description="Predicted probability/confidence, if any.")


class PredictLogRequest(BaseModel):
    items: list[PredictionIn] = Field(min_length=1, max_length=MAX_BATCH)

    @model_validator(mode="before")
    @classmethod
    def _accept_single_item(cls, data: Any) -> Any:
        if isinstance(data, dict) and "items" not in data and "row_id" in data:
            return {"items": [data]}
        return data


class ItemResult(BaseModel):
    row_id: str
    status: Literal["accepted", "duplicate", "rejected"]
    reason: str | None = None


class FindingOut(BaseModel):
    code: str
    message: str
    severity: Literal["info", "warning"]
    count: int = 0


class PredictLogResponse(BaseModel):
    accepted: int
    duplicates: int
    rejected: int
    results: list[ItemResult]
    findings: list[FindingOut]