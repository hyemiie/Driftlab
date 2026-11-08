from collections import Counter
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from driftwatch.api.deps import get_current_user, get_owned_project
from driftwatch.api.schemas import (
    ColumnStatRead, ReferenceProfileRead, SchemaColumnProposal,
    SchemaColumnRead, SchemaColumnUpdate, SchemaUpdate,
)
from driftwatch.config import settings
from driftwatch.db.models import (
    ColumnDtype, ColumnRole, ColumnStat, ProjectStatus, ReferenceProfile,
    SchemaColumn, TaskType, User,
)
from driftwatch.db.session import get_db
from driftwatch.services.ingestion import read_csv
from driftwatch.services.profiling import compute_reference_profile
from driftwatch.services.schema_infer import infer_schema
from driftwatch.storage.files import load_reference, save_reference, save_sample

router = APIRouter(prefix="/projects/{project_id}", tags=["reference"])


def _columns(db: Session, project_id: UUID) -> list[SchemaColumn]:
    return db.scalars(
        select(SchemaColumn)
        .where(SchemaColumn.project_id == project_id)
        .order_by(SchemaColumn.position)
    ).all()


def _validate_roles(columns: list[SchemaColumnUpdate], task_type: TaskType) -> None:
    problems: list[str] = []
    roles = Counter(c.role for c in columns)

    if roles[ColumnRole.target] != 1:
        problems.append("Exactly one column must have role 'target'")
    for r in (ColumnRole.id, ColumnRole.time, ColumnRole.prediction):
        if roles[r] > 1:
            problems.append(f"At most one column can have role '{r.value}'")

    expected_target = (
        ColumnDtype.numeric if task_type == TaskType.regression
        else ColumnDtype.categorical
    )
    for c in columns:
        if c.role == ColumnRole.target and c.dtype != expected_target:
            problems.append(
                f"Target '{c.name}' must be {expected_target.value} "
                f"for a {task_type.value} project"
            )
        if c.role == ColumnRole.time and c.dtype != ColumnDtype.datetime:
            problems.append(f"Time column '{c.name}' must have dtype 'datetime'")
        if c.role == ColumnRole.id and c.dtype != ColumnDtype.id:
            problems.append(f"ID column '{c.name}' must have dtype 'id'")

    if problems:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=problems)


@router.post("/reference", response_model=list[SchemaColumnProposal])
def upload_reference(
    project_id: UUID,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    project = get_owned_project(project_id, user, db)
    if project.status == ProjectStatus.ready:
        raise HTTPException(status.HTTP_409_CONFLICT, "Project is already profiled")

    raw = file.file.read(settings.max_upload_bytes + 1)
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File too large")

    try:
        df = read_csv(raw)
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e))

    inferred = infer_schema(df)  # list of objects with .name, .dtype, .reason
    save_reference(project.id, df)

    # re-uploading replaces any earlier proposed schema
    db.execute(delete(SchemaColumn).where(SchemaColumn.project_id == project.id))
    proposals = []
    for i, c in enumerate(inferred):
        role = ColumnRole.id if c.dtype == ColumnDtype.id else ColumnRole.feature
        db.add(SchemaColumn(
            project_id=project.id, name=c.name, dtype=c.dtype, role=role, position=i,
        ))
        proposals.append(SchemaColumnProposal(
            name=c.name, dtype=c.dtype, role=role, position=i, reason=c.reason,
        ))

    project.status = ProjectStatus.schema_pending
    db.commit()
    return proposals


@router.get("/schema", response_model=list[SchemaColumnRead])
def get_schema(
    project_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    project = get_owned_project(project_id, user, db)
    return _columns(db, project.id)


@router.put("/schema", response_model=list[SchemaColumnRead])
def update_schema(
    project_id: UUID,
    payload: SchemaUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    project = get_owned_project(project_id, user, db)
    if project.status != ProjectStatus.schema_pending:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Cannot edit schema while project is '{project.status.value}'",
        )

    existing = {c.name: c for c in _columns(db, project.id)}
    names = [c.name for c in payload.columns]
    if len(set(names)) != len(names):
        raise HTTPException(422, "Duplicate column names in request")
    if set(names) != set(existing):
        raise HTTPException(
            422,
            detail={
                "missing": sorted(set(existing) - set(names)),
                "unknown": sorted(set(names) - set(existing)),
            },
        )

    _validate_roles(payload.columns, project.task_type)

    for c in payload.columns:
        existing[c.name].dtype = c.dtype
        existing[c.name].role = c.role
    db.commit()
    return _columns(db, project.id)


def _profile_response(profile: ReferenceProfile, db: Session) -> ReferenceProfileRead:
    stats = db.scalars(
        select(ColumnStat).where(ColumnStat.profile_id == profile.id)
    ).all()
    return ReferenceProfileRead(
        id=profile.id,
        n_rows=profile.n_rows,
        created_at=profile.created_at,
        columns=[ColumnStatRead(column_name=s.column_name, stats=s.stats) for s in stats],
    )


@router.post(
    "/profile", response_model=ReferenceProfileRead, status_code=status.HTTP_201_CREATED
)
def create_profile(
    project_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    project = get_owned_project(project_id, user, db)
    if project.status != ProjectStatus.schema_pending:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Cannot profile while project is '{project.status.value}'",
        )

    df = load_reference(project.id)
    stats = compute_reference_profile(df, _columns(db, project.id))  # {col: dict}
    sample_path = save_sample(project.id, df)

    profile = ReferenceProfile(
        project_id=project.id, n_rows=len(df), sample_path=str(sample_path)
    )
    db.add(profile)
    db.flush()  # populates profile.id without committing
    db.add_all(
        ColumnStat(profile_id=profile.id, column_name=name, stats=s)
        for name, s in stats.items()
    )
    project.status = ProjectStatus.ready
    db.commit()  # profile, stats and status change succeed or fail together
    db.refresh(profile)
    return _profile_response(profile, db)


@router.get("/profile", response_model=ReferenceProfileRead)
def get_profile(
    project_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    project = get_owned_project(project_id, user, db)
    profile = db.scalar(
        select(ReferenceProfile)
        .where(ReferenceProfile.project_id == project.id)
        .order_by(ReferenceProfile.created_at.desc())
    )
    if profile is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No profile yet")
    return _profile_response(profile, db)