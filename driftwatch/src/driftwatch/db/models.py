import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


# ---------- Enums ----------


class TaskType(str, enum.Enum):
    regression = "regression"
    classification = "classification"


class ProjectStatus(str, enum.Enum):
    created = "created"
    schema_pending = "schema_pending"
    ready = "ready"


class ColumnDtype(str, enum.Enum):
    numeric = "numeric"
    categorical = "categorical"
    datetime = "datetime"
    id = "id"
    text = "text"
    ignore = "ignore"


class ColumnRole(str, enum.Enum):
    feature = "feature"
    target = "target"
    prediction = "prediction"
    id = "id"
    time = "time"


# ---------- Models ----------


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    task_type: Mapped[TaskType] = mapped_column(
        Enum(TaskType, name="task_type"), nullable=False
    )
    status: Mapped[ProjectStatus] = mapped_column(
        Enum(ProjectStatus, name="project_status"),
        nullable=False,
        default=ProjectStatus.created,
    )
    window_size: Mapped[str | None] = mapped_column(String(10), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class SchemaColumn(Base):
    __tablename__ = "schema_columns"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    dtype: Mapped[ColumnDtype] = mapped_column(
        Enum(ColumnDtype, name="column_dtype"), nullable=False
    )
    role: Mapped[ColumnRole] = mapped_column(
        Enum(ColumnRole, name="column_role"),
        nullable=False,
        default=ColumnRole.feature,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (UniqueConstraint("project_id", "name"),)


class ReferenceProfile(Base):
    __tablename__ = "reference_profiles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    n_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    sample_path: Mapped[str] = mapped_column(String(500), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ColumnStat(Base):
    __tablename__ = "column_stats"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    profile_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("reference_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    column_name: Mapped[str] = mapped_column(String(200), nullable=False)
    stats: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (UniqueConstraint("profile_id", "column_name"),)


class Batch(Base):
    __tablename__ = "batches"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False)  # sha256 hex
    n_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    warnings: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (UniqueConstraint("project_id", "file_hash"),)


class Window(Base):
    __tablename__ = "windows"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    window_size: Mapped[str | None] = mapped_column(String(10), nullable=False)

    window_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    window_end: Mapped[datetime] = mapped_column(  # exclusive
        DateTime(timezone=True), nullable=False
    )
    n_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sample_path: Mapped[str] = mapped_column(String(500), nullable=False)
    reference_profile_id: Mapped[uuid.UUID | None] = mapped_column(
    ForeignKey(
        "reference_profiles.id",
        ondelete="NO ACTION",
        deferrable=True,
        initially="DEFERRED",
    ),
    nullable=True,
)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


    __table_args__ = (UniqueConstraint("project_id", "window_size", "window_start",
                                       name="uq_window_project_size_start"),)


class WindowColumnSummary(Base):
    __tablename__ = "window_column_summaries"
    window_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("windows.id", ondelete="CASCADE"), primary_key=True)
    column_name: Mapped[str] = mapped_column(String(255), primary_key=True)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    summary: Mapped[dict] = mapped_column(JSONB, nullable=False)



class WindowStat(Base):
    __tablename__ = "window_stats"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    window_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("windows.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    column_name: Mapped[str] = mapped_column(String(200), nullable=False)
    stats: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (UniqueConstraint("window_id", "column_name"),)


class PredictionRow(Base):
    __tablename__ = "prediction_rows"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    window_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("windows.id", ondelete="CASCADE"), nullable=False, index=True)
    row_id: Mapped[str] = mapped_column(String(255), nullable=False)

    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now())
    model_version: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Regression: use the _num pair. Classification: use the _cat pair.
    prediction_num: Mapped[float | None] = mapped_column(Float, nullable=True)
    label_num: Mapped[float | None] = mapped_column(Float, nullable=True)
    prediction_cat: Mapped[str | None] = mapped_column(String(200), nullable=True)
    label_cat: Mapped[str | None] = mapped_column(String(200), nullable=True)
    prediction_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    label_received_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("project_id", "row_id", name="uq_prediction_project_row"),
        CheckConstraint("(prediction_num IS NOT NULL) <> (prediction_cat IS NOT NULL)",
                        name="ck_one_prediction_kind"),
        CheckConstraint("NOT (label_num IS NOT NULL AND label_cat IS NOT NULL)",
                        name="ck_one_label_kind"),
    )


class ApiKey(Base):
    __tablename__ = "api_keys"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    prefix: Mapped[str] = mapped_column(String(12), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now())
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)