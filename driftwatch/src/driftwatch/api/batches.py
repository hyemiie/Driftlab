import hashlib
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import pandas as pd
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from driftwatch.api.deps import get_current_user, get_owned_project
from driftwatch.api.reference import _columns
from driftwatch.config import settings
from driftwatch.db.models import (
    Batch,
    ColumnRole,
    ColumnStat,
    ProjectStatus,
    ReferenceProfile,
    User,
    Window,
    WindowColumnSummary,
)
from driftwatch.db.session import get_db
from driftwatch.services.ingestion import read_csv
from driftwatch.services.validation import validate_against_schema
from driftwatch.services.window_profile import (
    CategoricalRef,
    ColumnRef,
    NumericRef,
    WindowSummary,
    merge_columns,
    profile_window,
    summary_from_dict,
    summary_to_dict,
)
from driftwatch.services.windowing import (
    assign_windows,
    window_end_for,
)

router = APIRouter(prefix="/projects/{project_id}", tags=["batches"])


def _load_reference(
    db: Session,
    project_id: UUID,
) -> tuple[ReferenceProfile, dict[str, dict[str, Any]], dict[str, ColumnRef]]:
    profile = db.scalar(
        select(ReferenceProfile)
        .where(ReferenceProfile.project_id == project_id)
        .order_by(ReferenceProfile.created_at.desc(), ReferenceProfile.id.desc())
        .limit(1)
    )
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Create a reference profile before uploading production data.",
        )

    rows = db.scalars(
        select(ColumnStat).where(ColumnStat.profile_id == profile.id)
    ).all()

    stats_by_column = {row.column_name: row.stats for row in rows}
    reference: dict[str, ColumnRef] = {}

    for row in rows:
        stats = row.stats

        # Only profile input features. Targets and identifiers are not
        # distribution summaries for production input data.
        if stats.get("role") != "feature":
            continue

        if stats.get("kind") == "numeric":
            edges = stats.get("bin_edges") or []
            if len(edges) >= 2:
                reference[row.column_name] = NumericRef(
                    tuple(float(edge) for edge in edges)
                )
        elif stats.get("kind") == "categorical":
            categories = stats.get("categories") or []
            reference[row.column_name] = CategoricalRef(
                tuple(str(category["value"]) for category in categories)
            )

    return profile, stats_by_column, reference


def _batch_response(
    batch: Batch,
    *,
    duplicate: bool,
    windows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "batch_id": str(batch.id),
        "duplicate": duplicate,
        "n_rows": batch.n_rows,
        "warnings": batch.warnings,
        "windows": windows or [],
    }


@router.post("/batches", status_code=status.HTTP_201_CREATED)
def upload_batch(
    project_id: UUID,
    response: Response,
    file: UploadFile = File(...),
    window_size: str | None = Query(default=None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    project = get_owned_project(project_id, user, db)

    if project.status != ProjectStatus.ready:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Profile the reference data before uploading production data.",
        )

    raw = file.file.read(settings.max_upload_bytes + 1)
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large.",
        )

    file_hash = hashlib.sha256(raw).hexdigest()

    # Return an existing batch for an exact retry.
    existing_batch = db.scalar(
        select(Batch).where(
            Batch.project_id == project.id,
            Batch.file_hash == file_hash,
        )
    )
    if existing_batch is not None:
        response.status_code = status.HTTP_200_OK
        return _batch_response(existing_batch, duplicate=True)

    try:
        df = read_csv(raw, min_rows=1)
        columns = _columns(db, project.id)

        time_columns = [
            column.name
            for column in columns
            if column.role == ColumnRole.time
        ]
        if len(time_columns) > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The project schema has more than one time column.",
            )

        time_col = time_columns[0] if time_columns else None

        requested_window_size = (
            window_size
            or project.window_size
            or ("day" if time_col is not None else "batch")
        )

        if requested_window_size not in {"day", "week", "batch"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="window_size must be day, week, or batch.",
            )

        if time_col is None and requested_window_size != "batch":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Projects without a time column must use batch windows.",
            )

        if (
            project.window_size is not None
            and project.window_size != requested_window_size
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Project uses '{project.window_size}' windows. "
                    f"Cannot change to '{requested_window_size}'."
                ),
            )

        profile, reference_stats, reference = _load_reference(db, project.id)
        findings = validate_against_schema(df, columns, reference_stats)

        errors = [finding for finding in findings if finding.severity == "error"]
        if errors:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=[asdict(finding) for finding in errors],
            )

        uploaded_at = datetime.now(timezone.utc)
        windowed_df, window_findings = assign_windows(
            df=df,
            time_col=time_col,
            window_size=requested_window_size,
            uploaded_at=uploaded_at,
        )

        if windowed_df.empty:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No rows could be assigned to a window.",
            )

        warnings = [
            asdict(finding)
            for finding in findings
            if finding.severity == "warning"
        ]
        warnings.extend(asdict(finding) for finding in window_findings)

        batch = Batch(
            id=uuid.uuid4(),
            project_id=project.id,
            filename=file.filename or "upload.csv",
            file_hash=file_hash,
            n_rows=len(windowed_df),
            warnings=warnings,
        )

        # The unique constraint makes simultaneous retries safe.
        inserted_batch_id = db.scalar(
            pg_insert(Batch)
            .values(
                id=batch.id,
                project_id=batch.project_id,
                filename=batch.filename,
                file_hash=batch.file_hash,
                n_rows=batch.n_rows,
                warnings=batch.warnings,
            )
            .on_conflict_do_nothing(
                index_elements=["project_id", "file_hash"]
            )
            .returning(Batch.id)
        )

        if inserted_batch_id is None:
            db.rollback()
            existing_batch = db.scalar(
                select(Batch).where(
                    Batch.project_id == project.id,
                    Batch.file_hash == file_hash,
                )
            )
            if existing_batch is None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="The batch was uploaded concurrently. Retry the request.",
                )
            response.status_code = status.HTTP_200_OK
            return _batch_response(existing_batch, duplicate=True)

        starts = sorted(
            {
                pd.Timestamp(value).to_pydatetime()
                for value in windowed_df["window_start"].drop_duplicates()
            }
        )

        db.execute(
            pg_insert(Window)
            .values(
                [
                    {
                        "id": uuid.uuid4(),
                        "project_id": project.id,
                        "reference_profile_id": profile.id,
                        "window_size": requested_window_size,
                        "window_start": start,
                        "window_end": window_end_for(
                            pd.Timestamp(start),
                            requested_window_size,
                        ).to_pydatetime(),
                        "n_rows": 0,
                    }
                    for start in starts
                ]
            )
            .on_conflict_do_nothing(
                index_elements=[
                    "project_id",
                    "window_size",
                    "window_start",
                ]
            )
        )

        # Lock windows in a consistent order before changing summaries.
        windows = db.scalars(
            select(Window)
            .where(
                Window.project_id == project.id,
                Window.window_size == requested_window_size,
                Window.window_start.in_(starts),
            )
            .order_by(Window.id)
            .with_for_update()
        ).all()

        window_by_start = {window.window_start: window for window in windows}

        for window in windows:
            if window.reference_profile_id is None:
                window.reference_profile_id = profile.id
            elif window.reference_profile_id != profile.id:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=(
                        "A window was created with a different reference profile."
                    ),
                )

        per_window: dict[UUID, WindowSummary] = {}

        for start, group in windowed_df.groupby("window_start", sort=True):
            start_key = pd.Timestamp(start).to_pydatetime()
            window = window_by_start[start_key]

            feature_frame = group.drop(columns=["window_start"])
            per_window[window.id] = profile_window(feature_frame, reference)

        stored_rows = db.scalars(
            select(WindowColumnSummary).where(
                WindowColumnSummary.window_id.in_(list(per_window))
            )
        ).all()
        stored = {
            (row.window_id, row.column_name): row
            for row in stored_rows
        }

        summary_upserts = []
        response_windows = []

        for window in windows:
            new_summary = per_window.get(window.id)
            if new_summary is None:
                continue

            window.n_rows += new_summary.n_rows

            response_windows.append(
                {
                    "window_id": str(window.id),
                    "window_start": window.window_start.isoformat(),
                    "n_rows": window.n_rows,
                }
            )

            for column_name, new_column_summary in new_summary.columns.items():
                old_row = stored.get((window.id, column_name))
                merged_summary = (
                    merge_columns(
                        summary_from_dict(old_row.summary),
                        new_column_summary,
                    )
                    if old_row is not None
                    else new_column_summary
                )

                summary_upserts.append(
                    {
                        "window_id": window.id,
                        "column_name": column_name,
                        "kind": merged_summary.kind,
                        "summary": summary_to_dict(merged_summary),
                    }
                )

        if summary_upserts:
            statement = pg_insert(WindowColumnSummary).values(summary_upserts)
            db.execute(
                statement.on_conflict_do_update(
                    index_elements=["window_id", "column_name"],
                    set_={
                        "kind": statement.excluded.kind,
                        "summary": statement.excluded.summary,
                    },
                )
            )

        if project.window_size is None:
            project.window_size = requested_window_size

        db.commit()

        return _batch_response(
            batch,
            duplicate=False,
            windows=response_windows,
        )

    except HTTPException:
        db.rollback()
        raise
    except ValueError as error:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
    except Exception:
        db.rollback()
        raise